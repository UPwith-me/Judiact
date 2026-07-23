from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from PIL import Image

from .clean_reproduction import policy_system_prompt
from .constants import INTENT_TO_ID, STAGES
from .io_utils import read_jsonl
from .schema import normalize_record


SENTINELS = (*[f"<DIAG_{stage}>" for stage in STAGES], "<DIAG_DECISION>")
SENTINEL_BLOCK = "\n".join(("Diagnostic state slots:", *SENTINELS))


def export_hidden_cache(
    *,
    input_jsonl: str,
    model_path: str,
    output_pt: str,
    adapter_path: str | None = None,
    batch_size: int = 8,
    min_pixels: int = 200704,
    max_pixels: int = 401408,
    pooling: str = "last4_mean",
    image_mode: str = "normal",
    max_samples: int = 0,
    log_every_batches: int = 10,
) -> dict[str, Any]:
    import torch

    if pooling not in {"last", "last4_mean"}:
        raise ValueError("pooling must be last or last4_mean")
    if image_mode not in {"normal", "drop", "zero"}:
        raise ValueError("image_mode must be normal, drop, or zero")
    input_sha256 = _sha256_file(input_jsonl)
    rows = read_jsonl(input_jsonl)
    if max_samples > 0:
        rows = rows[:max_samples]
    if not rows:
        raise ValueError(f"empty hidden export input: {input_jsonl}")

    model, processor = _load_model(
        model_path=model_path,
        adapter_path=adapter_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    block_ids, sentinel_token_indices = _sentinel_layout(processor.tokenizer)

    stage_chunks: list[torch.Tensor] = []
    decision_chunks: list[torch.Tensor] = []
    labels = _empty_label_cache()
    metadata = _empty_metadata_cache()
    started = time.time()
    processed = 0
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(rows), batch_size), 1):
            batch_rows = rows[start : start + batch_size]
            encoded, positions = encode_hidden_batch(
                processor,
                batch_rows,
                block_ids=block_ids,
                sentinel_token_indices=sentinel_token_indices,
                image_mode=image_mode,
            )
            encoded = _move_batch(encoded, model.device)
            output = model(**encoded, output_hidden_states=True, use_cache=False, return_dict=True)
            hidden = _pool_hidden_states(output.hidden_states, pooling=pooling)
            position_tensor = torch.tensor(positions, device=hidden.device, dtype=torch.long)
            row_index = torch.arange(hidden.shape[0], device=hidden.device).unsqueeze(1)
            sentinel_hidden = hidden[row_index, position_tensor]
            stage_chunks.append(sentinel_hidden[:, :5].to(dtype=torch.float16, device="cpu"))
            decision_chunks.append(sentinel_hidden[:, 5].to(dtype=torch.float16, device="cpu"))
            _append_labels(labels, batch_rows)
            _append_metadata(metadata, batch_rows)
            processed += len(batch_rows)
            if batch_index == 1 or batch_index % log_every_batches == 0 or processed == len(rows):
                elapsed = time.time() - started
                rate = processed / max(elapsed, 1e-6)
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "total": len(rows),
                            "rows_per_sec": round(rate, 3),
                            "eta_sec": round((len(rows) - processed) / max(rate, 1e-6), 1),
                            "max_gpu_memory_gb": round(
                                torch.cuda.max_memory_allocated() / 1024**3, 3
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

    if _sha256_file(input_jsonl) != input_sha256:
        raise RuntimeError("hidden export input changed while the cache was being generated")
    cache = {
        "format": "causal_diagsafe_hidden_cache_v2",
        "stage_hidden": torch.cat(stage_chunks, dim=0),
        "decision_hidden": torch.cat(decision_chunks, dim=0),
        "labels": _tensorize_labels(labels),
        "metadata": metadata,
        "export": {
            "input_jsonl": input_jsonl,
            "input_sha256": input_sha256,
            "model_path": model_path,
            "adapter_path": adapter_path,
            "pooling": pooling,
            "image_mode": image_mode,
            "max_samples": max_samples,
            "row_count": len(rows),
            "hidden_size": int(stage_chunks[0].shape[-1]),
            "sentinels": list(SENTINELS),
            "sentinel_block_token_count": len(block_ids),
            "sentinel_terminal_token_indices": list(sentinel_token_indices),
            "prompt_contract_version": "model_visible_policy_response_v2",
            "elapsed_sec": round(time.time() - started, 2),
        },
    }
    target = Path(output_pt)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    torch.save(cache, temporary)
    temporary.replace(target)
    return cache["export"]


def encode_hidden_batch(
    processor: Any,
    rows: list[dict[str, Any]],
    *,
    block_ids: list[int],
    sentinel_token_indices: list[int],
    image_mode: str,
) -> tuple[dict[str, Any], list[list[int]]]:
    from qwen_vl_utils import process_vision_info

    prompt_texts: list[str] = []
    message_batch: list[list[dict[str, Any]]] = []
    opened_images: list[Image.Image] = []
    try:
        for row in rows:
            record = normalize_record(row)
            query = record.query.strip()
            if not query:
                raise ValueError(f"{row.get('id')}: missing query")
            image_paths = list(record.images)
            user_content: list[dict[str, Any]] = []
            if image_mode != "drop":
                for image_path in image_paths:
                    with Image.open(str(image_path)) as source:
                        image = source.convert("RGB")
                    if image_mode == "zero":
                        size = image.size
                        image.close()
                        image = Image.new("RGB", size, color=(0, 0, 0))
                    opened_images.append(image)
                    user_content.append({"type": "image", "image": image})
            visible_parts = [f"User query:\n{query}"]
            if record.response_candidate:
                visible_parts.append(
                    f"Candidate response to audit:\n{record.response_candidate}"
                )
            visible_parts.append(SENTINEL_BLOCK)
            user_content.append({"type": "text", "text": "\n\n".join(visible_parts)})
            messages = [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": policy_system_prompt(
                                record.policy_text,
                                supplied=record.policy_text_supplied,
                            ),
                        }
                    ],
                },
                {"role": "user", "content": user_content},
            ]
            message_batch.append(messages)
            prompt_texts.append(
                processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

        image_inputs: list[Any] = []
        video_inputs: list[Any] = []
        for messages in message_batch:
            row_images, row_videos = process_vision_info(messages)
            if row_images:
                image_inputs.extend(row_images)
            if row_videos:
                video_inputs.extend(row_videos)
        encoded = processor(
            text=prompt_texts,
            images=image_inputs or None,
            videos=video_inputs or None,
            padding=True,
            return_tensors="pt",
        )
        positions = [
            _locate_sentinel_positions(
                token_row.tolist(),
                block_ids=block_ids,
                sentinel_token_indices=sentinel_token_indices,
            )
            for token_row in encoded["input_ids"]
        ]
        return encoded, positions
    finally:
        for image in opened_images:
            image.close()


def _load_model(
    *, model_path: str, adapter_path: str | None, min_pixels: int, max_pixels: int
) -> tuple[Any, Any]:
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    if adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    else:
        model = base
    model.to("cuda:0")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels
    return model, processor


def _pool_hidden_states(hidden_states: tuple[Any, ...], *, pooling: str) -> Any:
    import torch

    if pooling == "last":
        return hidden_states[-1]
    return torch.stack(hidden_states[-4:], dim=0).mean(dim=0)


def _find_last_subsequence(values: list[int], needle: list[int]) -> int:
    for start in range(len(values) - len(needle), -1, -1):
        if values[start : start + len(needle)] == needle:
            return start
    raise ValueError(f"diagnostic sentinel token sequence not found: {needle}")


def _sentinel_layout(tokenizer: Any) -> tuple[list[int], list[int]]:
    encoded = tokenizer(
        SENTINEL_BLOCK,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = list(encoded["input_ids"])
    offsets = list(encoded["offset_mapping"])
    if not token_ids or len(token_ids) != len(offsets):
        raise RuntimeError("tokenizer did not return a valid diagnostic block layout")
    token_indices: list[int] = []
    search_start = 0
    for sentinel in SENTINELS:
        marker_start = SENTINEL_BLOCK.index(sentinel, search_start)
        marker_end = marker_start + len(sentinel)
        token_index = next(
            (
                index
                for index, (start, end) in enumerate(offsets)
                if start < marker_end <= end
            ),
            None,
        )
        if token_index is None:
            raise RuntimeError(f"could not map sentinel to token: {sentinel}")
        token_indices.append(token_index)
        search_start = marker_end
    return token_ids, token_indices


def _locate_sentinel_positions(
    values: list[int],
    *,
    block_ids: list[int],
    sentinel_token_indices: list[int],
) -> list[int]:
    max_drop = min(8, len(block_ids) - 1)
    for drop in range(max_drop + 1):
        anchor = block_ids[drop:]
        try:
            match = _find_last_subsequence(values, anchor)
        except ValueError:
            continue
        positions = [match + token_index - drop for token_index in sentinel_token_indices]
        if all(0 <= position < len(values) for position in positions):
            return positions
    raise ValueError("diagnostic sentinel block could not be located after boundary normalization")


def _empty_label_cache() -> dict[str, list[Any]]:
    return {
        "decision": [],
        "decision_mask": [],
        "policy_action": [],
        "policy_action_mask": [],
        "benign": [],
        "benign_mask": [],
        "intent": [],
        "intent_mask": [],
        "R": [],
        "R_mask": [],
        "E": [],
        "E_mask": [],
        "F": [],
        "F_mask": [],
        "target_stage": [],
    }


def _empty_metadata_cache() -> dict[str, list[Any]]:
    return {
        "id": [],
        "group_id": [],
        "source_dataset": [],
        "modality_scope": [],
        "role": [],
        "changed_factor": [],
        "policy_condition": [],
        "has_response_candidate": [],
        "policy_text_supplied": [],
    }


def _append_labels(cache: dict[str, list[Any]], rows: list[dict[str, Any]]) -> None:
    unknown_intent = INTENT_TO_ID["unknown"]
    for row in rows:
        targets: dict[str, Any] = (
            row["targets"] if isinstance(row.get("targets"), dict) else {}
        )
        stage: dict[str, Any] = (
            targets["stage"] if isinstance(targets.get("stage"), dict) else {}
        )
        cache["decision"].append(int(targets.get("decision", 0)))
        cache["decision_mask"].append(int(targets.get("decision_mask", 1)))
        cache["policy_action"].append(int(targets.get("policy_action", 0)))
        cache["policy_action_mask"].append(int(targets.get("policy_action_mask", 0)))
        cache["benign"].append(int(targets.get("benign", 0)))
        cache["benign_mask"].append(int(targets.get("benign_mask", 0)))
        cache["intent"].append(int(targets.get("intent", unknown_intent)))
        cache["intent_mask"].append(int(targets.get("intent_mask", 0)))
        for key in ("R", "R_mask", "E", "E_mask", "F", "F_mask"):
            values = list(stage.get(key) or [])
            if len(values) != len(STAGES):
                raise ValueError(f"{row.get('id')}: targets.stage.{key} must have five values")
            cache[key].append([int(value) for value in values])
        cache["target_stage"].append(int(targets.get("target_stage_id", -1)))


def _append_metadata(cache: dict[str, list[Any]], rows: list[dict[str, Any]]) -> None:
    for row in rows:
        targets: dict[str, Any] = (
            row["targets"] if isinstance(row.get("targets"), dict) else {}
        )
        metadata: dict[str, Any] = (
            row["metadata"] if isinstance(row.get("metadata"), dict) else {}
        )
        record = normalize_record(row)
        cache["id"].append(str(row.get("id") or ""))
        cache["group_id"].append(str(metadata.get("group_id") or ""))
        cache["source_dataset"].append(str(metadata.get("source_dataset") or "unknown"))
        cache["modality_scope"].append(str(metadata.get("modality_scope") or "unknown"))
        cache["role"].append(str(targets.get("role") or "unknown"))
        cache["changed_factor"].append(str(targets.get("changed_factor") or "unknown"))
        cache["policy_condition"].append(record.policy_condition)
        cache["has_response_candidate"].append(bool(record.response_candidate))
        cache["policy_text_supplied"].append(record.policy_text_supplied)


def _tensorize_labels(labels: dict[str, list[Any]]) -> dict[str, Any]:
    import torch

    float_keys = {
        "R",
        "R_mask",
        "F",
        "F_mask",
        "decision_mask",
        "policy_action_mask",
        "benign",
        "benign_mask",
        "intent_mask",
    }
    return {
        key: torch.tensor(values, dtype=torch.float32 if key in float_keys else torch.long)
        for key, values in labels.items()
    }


def _move_batch(batch: dict[str, Any], device: Any) -> dict[str, Any]:
    import torch

    moved: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            value = value.to(device, non_blocking=True)
            if key == "pixel_values":
                value = value.to(torch.bfloat16)
        moved[key] = value
    return moved


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
