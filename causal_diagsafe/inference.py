from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .parser import parse_prediction
from .training.multimodal_sft import read_clean_sft_rows


def run_clean_adapter_inference(
    *,
    input_jsonl: str,
    model_path: str,
    adapter_path: str | None,
    output_jsonl: str,
    batch_size: int = 8,
    max_new_tokens: int = 192,
    min_pixels: int = 200704,
    max_pixels: int = 401408,
    log_every_batches: int = 10,
    max_samples: int = 0,
) -> dict[str, Any]:
    import torch

    rows = read_clean_sft_rows(input_jsonl)
    if max_samples > 0:
        rows = rows[:max_samples]
    model, processor = _load_inference_model(
        model_path=model_path,
        adapter_path=adapter_path,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )
    output_path = Path(output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    invalid_json = 0
    processed = 0
    with output_path.open("w", encoding="utf-8") as handle, torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(rows), batch_size), 1):
            batch_rows = rows[start : start + batch_size]
            batch = encode_inference_batch(processor, batch_rows)
            batch = _move_batch(batch, model.device)
            prompt_width = batch["input_ids"].shape[1]
            generated = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
            decoded = processor.batch_decode(
                generated[:, prompt_width:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )
            for row, raw_output in zip(batch_rows, decoded):
                parsed = parse_prediction(raw_output)
                invalid_json += int(not parsed.valid_json)
                result = {
                    "id": row["id"],
                    "model_output": raw_output,
                    "valid_json": parsed.valid_json,
                    "final_decision": parsed.final_decision,
                    "policy_id": parsed.policy_id,
                    "stage_states": parsed.stage_states.to_dict(),
                    "legacy_stage_labels": parsed.legacy_stage_labels,
                    "parse_error": parsed.parse_error,
                    "adapter_path": adapter_path,
                }
                handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            processed += len(batch_rows)
            if batch_index == 1 or batch_index % log_every_batches == 0 or processed == len(rows):
                elapsed = time.time() - started
                rate = processed / max(elapsed, 1e-6)
                eta = (len(rows) - processed) / max(rate, 1e-6)
                print(
                    json.dumps(
                        {
                            "processed": processed,
                            "total": len(rows),
                            "rows_per_sec": round(rate, 3),
                            "eta_sec": round(eta, 1),
                            "invalid_json": invalid_json,
                            "max_gpu_memory_gb": round(
                                torch.cuda.max_memory_allocated() / 1024**3,
                                3,
                            ),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    return {
        "format": "clean_adapter_inference_summary_v1",
        "input_jsonl": input_jsonl,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "output_jsonl": output_jsonl,
        "row_count": len(rows),
        "invalid_json_count": invalid_json,
        "elapsed_sec": round(time.time() - started, 2),
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "max_samples": max_samples,
    }


def encode_inference_batch(processor: Any, rows: list[dict[str, Any]]) -> dict[str, Any]:
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    prompt_texts: list[str] = []
    full_message_batch: list[list[dict[str, Any]]] = []
    opened_images: list[Image.Image] = []
    try:
        for row in rows:
            messages = row["messages"]
            user_content: list[dict[str, Any]] = []
            for image_path in row.get("images", []):
                image = Image.open(str(image_path)).convert("RGB")
                opened_images.append(image)
                user_content.append({"type": "image", "image": image})
            user_content.append({"type": "text", "text": str(messages[1]["content"])})
            prompt_messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": str(messages[0]["content"])}],
                },
                {"role": "user", "content": user_content},
            ]
            full_message_batch.append(prompt_messages)
            prompt_texts.append(
                processor.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        image_inputs: list[Any] = []
        video_inputs: list[Any] = []
        for prompt_messages in full_message_batch:
            row_images, row_videos = process_vision_info(prompt_messages)
            if row_images:
                image_inputs.extend(row_images)
            if row_videos:
                video_inputs.extend(row_videos)
        return processor(
            text=prompt_texts,
            images=image_inputs or None,
            videos=video_inputs or None,
            padding=True,
            return_tensors="pt",
        )
    finally:
        for image in opened_images:
            image.close()


def _load_inference_model(
    *,
    model_path: str,
    adapter_path: str | None,
    min_pixels: int,
    max_pixels: int,
) -> tuple[Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False) if adapter_path else base
    model.to("cuda:0")
    model.eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels
    return model, processor


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
