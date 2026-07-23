from __future__ import annotations

import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LORA_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class CleanSFTConfig:
    train_jsonl: str
    eval_jsonl: str | None
    model: str
    output_dir: str
    init_adapter: str | None = None
    epochs: int = 1
    max_steps: int = 0
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.03
    lora_r: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    max_length: int = 1536
    min_pixels: int = 200704
    max_pixels: int = 401408
    seed: int = 42
    log_every: int = 5
    eval_every: int = 0
    max_eval_samples: int = 0
    gradient_clip: float = 1.0
    attn_implementation: str = "sdpa"


def read_clean_sft_rows(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"empty training manifest: {path}")
    return rows


def validate_training_manifest(
    path: str | Path,
    *,
    model_path: str | Path,
) -> dict[str, Any]:
    rows = read_clean_sft_rows(path)
    issues: list[str] = []
    missing_images: list[dict[str, str]] = []
    decisions: dict[str, int] = {}
    image_rows = 0
    image_count = 0
    assistant_char_lengths: list[int] = []
    ids: set[str] = set()

    for index, row in enumerate(rows):
        row_id = str(row.get("id") or "")
        if not row_id:
            issues.append(f"row[{index}]:missing_id")
        elif row_id in ids:
            issues.append(f"{row_id}:duplicate_id")
        ids.add(row_id)
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            issues.append(f"{row_id or index}:messages_must_have_three_turns")
            continue
        expected_roles = ("system", "user", "assistant")
        roles = tuple(str(message.get("role") or "") for message in messages)
        if roles != expected_roles:
            issues.append(f"{row_id or index}:invalid_roles:{roles}")
        target = row.get("training_target")
        if not isinstance(target, dict):
            issues.append(f"{row_id or index}:missing_training_target")
            continue
        decision = str(target.get("final_decision") or "")
        if decision not in {"allow", "warn", "review", "refuse"}:
            issues.append(f"{row_id or index}:invalid_decision:{decision}")
        decisions[decision] = decisions.get(decision, 0) + 1
        assistant_char_lengths.append(len(str(messages[2].get("content") or "")))
        images = row.get("images")
        if not isinstance(images, list):
            issues.append(f"{row_id or index}:images_must_be_list")
            images = []
        if images:
            image_rows += 1
        image_count += len(images)
        for image_path in images:
            path_obj = Path(str(image_path))
            if not path_obj.is_file():
                missing_images.append({"id": row_id, "image": str(path_obj)})

    model_obj = Path(model_path)
    if not model_obj.exists():
        issues.append(f"model_path_missing:{model_obj}")
    return {
        "format": "clean_multimodal_sft_preflight_v1",
        "manifest": str(path),
        "model": str(model_path),
        "row_count": len(rows),
        "decision": decisions,
        "image_rows": image_rows,
        "image_count": image_count,
        "missing_image_count": len(missing_images),
        "missing_image_examples": missing_images[:50],
        "assistant_chars": {
            "min": min(assistant_char_lengths, default=0),
            "max": max(assistant_char_lengths, default=0),
            "mean": (
                sum(assistant_char_lengths) / len(assistant_char_lengths)
                if assistant_char_lengths
                else 0.0
            ),
        },
        "issue_count": len(issues) + len(missing_images),
        "issues": issues[:100],
    }


def train_clean_multimodal_sft(config: CleanSFTConfig) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_cosine_schedule_with_warmup

    _seed_everything(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = read_clean_sft_rows(config.train_jsonl)
    eval_rows = read_clean_sft_rows(config.eval_jsonl) if config.eval_jsonl else []
    if config.max_eval_samples > 0:
        eval_rows = eval_rows[: config.max_eval_samples]

    print("[train] loading model and processor", flush=True)
    model, processor = _load_model_and_processor(config)
    train_loader = DataLoader(
        train_rows,
        batch_size=config.micro_batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda rows: rows,
        generator=torch.Generator().manual_seed(config.seed),
    )
    eval_loader = (
        DataLoader(
            eval_rows,
            batch_size=config.micro_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=lambda rows: rows,
        )
        if eval_rows
        else None
    )

    updates_per_epoch = math.ceil(len(train_loader) / config.gradient_accumulation_steps)
    total_steps = config.max_steps or updates_per_epoch * config.epochs
    if total_steps <= 0:
        raise ValueError("training requires at least one optimizer step")
    warmup_steps = int(total_steps * config.warmup_ratio)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer_kwargs: dict[str, Any] = {
        "lr": config.learning_rate,
        "weight_decay": config.weight_decay,
    }
    if torch.cuda.is_available():
        optimizer_kwargs["fused"] = True
    try:
        optimizer = torch.optim.AdamW(trainable, **optimizer_kwargs)
    except (RuntimeError, TypeError):
        optimizer_kwargs.pop("fused", None)
        optimizer = torch.optim.AdamW(trainable, **optimizer_kwargs)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    run_manifest = {
        "format": "clean_multimodal_sft_run_v1",
        "config": config.__dict__,
        "train_rows": len(train_rows),
        "eval_rows": len(eval_rows),
        "updates_per_epoch": updates_per_epoch,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "claim_boundary": (
            "Clean final-decision multimodal SFT baseline. It does not establish causal stage "
            "diagnosis and does not consume stage, role, source, or manual evidence in the prompt."
        ),
    }
    _write_json(output_dir / "run_manifest.json", run_manifest)
    log_path = output_dir / "train_metrics.jsonl"
    optimizer.zero_grad(set_to_none=True)
    model.train()
    global_step = 0
    accumulation_count = 0
    running_loss = 0.0
    running_microbatches = 0
    running_target_tokens = 0
    best_eval_loss = float("inf")
    started = time.time()

    print(
        f"[train] rows={len(train_rows)} total_steps={total_steps} "
        f"micro_batch={config.micro_batch_size} grad_accum={config.gradient_accumulation_steps}",
        flush=True,
    )
    while global_step < total_steps:
        for batch_index, rows in enumerate(train_loader):
            batch, batch_meta = encode_multimodal_batch(
                processor,
                rows,
                max_length=config.max_length,
            )
            batch = _move_batch(batch, model.device)
            outputs = model(**batch)
            raw_loss = outputs.loss
            (raw_loss / config.gradient_accumulation_steps).backward()
            accumulation_count += 1
            running_loss += float(raw_loss.detach().cpu())
            running_microbatches += 1
            running_target_tokens += batch_meta["target_token_count"]
            is_update = accumulation_count == config.gradient_accumulation_steps
            is_last_batch = batch_index + 1 == len(train_loader)
            if not is_update and not is_last_batch:
                continue

            if accumulation_count < config.gradient_accumulation_steps:
                correction = config.gradient_accumulation_steps / accumulation_count
                for parameter in trainable:
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)
            torch.nn.utils.clip_grad_norm_(trainable, config.gradient_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            accumulation_count = 0
            global_step += 1

            if global_step == 1 or global_step % config.log_every == 0:
                elapsed = time.time() - started
                rate = global_step / max(elapsed, 1e-6)
                eta = (total_steps - global_step) / max(rate, 1e-6)
                log_row = {
                    "step": global_step,
                    "total_steps": total_steps,
                    "loss": running_loss / max(running_microbatches, 1),
                    "target_tokens_since_log": running_target_tokens,
                    "learning_rate": scheduler.get_last_lr()[0],
                    "elapsed_sec": round(elapsed, 2),
                    "eta_sec": round(eta, 2),
                    "max_gpu_memory_gb": round(
                        torch.cuda.max_memory_allocated() / 1024**3,
                        3,
                    ),
                }
                _append_jsonl(log_path, log_row)
                print(json.dumps(log_row, ensure_ascii=False), flush=True)
                running_loss = 0.0
                running_microbatches = 0
                running_target_tokens = 0

            should_eval = (
                eval_loader is not None
                and (
                    (config.eval_every > 0 and global_step % config.eval_every == 0)
                    or global_step == total_steps
                )
            )
            if should_eval:
                eval_metrics = evaluate_loss(
                    model,
                    processor,
                    eval_loader,
                    max_length=config.max_length,
                )
                eval_metrics["step"] = global_step
                _append_jsonl(output_dir / "eval_metrics.jsonl", eval_metrics)
                print(json.dumps({"eval": eval_metrics}, ensure_ascii=False), flush=True)
                if eval_metrics["loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["loss"]
                    _save_adapter(model, processor, output_dir / "best")
                model.train()

            if global_step >= total_steps:
                break

    _save_adapter(model, processor, output_dir / "last")
    summary = {
        **run_manifest,
        "completed_steps": global_step,
        "elapsed_sec": round(time.time() - started, 2),
        "best_eval_loss": None if best_eval_loss == float("inf") else best_eval_loss,
        "last_adapter": str(output_dir / "last"),
        "best_adapter": str(output_dir / "best") if best_eval_loss < float("inf") else None,
    }
    _write_json(output_dir / "training_summary.json", summary)
    print(f"[train] complete last_adapter={output_dir / 'last'}", flush=True)
    return summary


def encode_multimodal_batch(
    processor: Any,
    rows: list[dict[str, Any]],
    *,
    max_length: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from PIL import Image
    from qwen_vl_utils import process_vision_info

    prompt_texts: list[str] = []
    full_texts: list[str] = []
    opened_images: list[Image.Image] = []
    prompt_messages_batch: list[list[dict[str, Any]]] = []
    full_messages_batch: list[list[dict[str, Any]]] = []

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
            full_messages = prompt_messages + [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": str(messages[2]["content"])}],
                }
            ]
            prompt_messages_batch.append(prompt_messages)
            full_messages_batch.append(full_messages)
            prompt_texts.append(
                processor.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            full_texts.append(
                processor.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

        image_inputs: list[Any] = []
        video_inputs: list[Any] = []
        for full_messages in full_messages_batch:
            row_images, row_videos = process_vision_info(full_messages)
            if row_images:
                image_inputs.extend(row_images)
            if row_videos:
                video_inputs.extend(row_videos)
        processor_kwargs = {
            "images": image_inputs or None,
            "videos": video_inputs or None,
            "padding": True,
            "truncation": True,
            "max_length": max_length,
            "return_tensors": "pt",
        }
        full_inputs = processor(text=full_texts, **processor_kwargs)
        prompt_inputs = processor(text=prompt_texts, **processor_kwargs)
        labels = full_inputs["input_ids"].clone()
        attention = full_inputs["attention_mask"]
        prompt_attention = prompt_inputs["attention_mask"]
        target_token_count = 0
        target_token_counts: list[int] = []
        sequence_lengths: list[int] = []
        for index in range(labels.shape[0]):
            full_length = int(attention[index].sum().item())
            prompt_length = int(prompt_attention[index].sum().item())
            if prompt_length >= full_length:
                raise ValueError(
                    f"{rows[index].get('id')}: assistant target was fully truncated; "
                    f"prompt={prompt_length}, full={full_length}, max_length={max_length}"
                )
            valid_positions = torch.nonzero(attention[index], as_tuple=False).flatten()
            full_start = int(valid_positions[0].item())
            full_end = int(valid_positions[-1].item()) + 1
            labels[index, : full_start + prompt_length] = -100
            labels[index, full_end:] = -100
            row_target_tokens = full_length - prompt_length
            target_token_count += row_target_tokens
            target_token_counts.append(row_target_tokens)
            sequence_lengths.append(full_length)
        labels[attention == 0] = -100
        full_inputs["labels"] = labels
        return full_inputs, {
            "target_token_count": target_token_count,
            "target_token_counts": target_token_counts,
            "sequence_lengths": sequence_lengths,
        }
    finally:
        for image in opened_images:
            image.close()


def evaluate_loss(
    model: Any,
    processor: Any,
    loader: Any,
    *,
    max_length: int,
) -> dict[str, Any]:
    import torch

    model.eval()
    loss_sum = 0.0
    batch_count = 0
    target_tokens = 0
    with torch.inference_mode():
        for rows in loader:
            batch, metadata = encode_multimodal_batch(processor, rows, max_length=max_length)
            batch = _move_batch(batch, model.device)
            loss_sum += float(model(**batch).loss.detach().cpu())
            batch_count += 1
            target_tokens += metadata["target_token_count"]
    return {
        "loss": loss_sum / max(batch_count, 1),
        "batch_count": batch_count,
        "target_token_count": target_tokens,
    }


def _load_model_and_processor(config: CleanSFTConfig) -> tuple[Any, Any]:
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model,
        torch_dtype=torch.bfloat16,
        attn_implementation=config.attn_implementation,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    if config.init_adapter:
        model = PeftModel.from_pretrained(model, config.init_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias="none",
            target_modules=list(LORA_TARGET_MODULES),
        )
        model = get_peft_model(model, lora_config)
    model.to("cuda:0")
    model.print_trainable_parameters()
    processor = AutoProcessor.from_pretrained(config.model, trust_remote_code=True)
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = config.min_pixels
        processor.image_processor.max_pixels = config.max_pixels
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
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


def _save_adapter(model: Any, processor: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    processor.save_pretrained(path)


def _seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, ensure_ascii=False) + "\n")
