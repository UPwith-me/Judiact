from __future__ import annotations

import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

from ..inference import encode_inference_batch
from ..io_utils import read_jsonl
from ..parser import parse_prediction
from .multimodal_sft import encode_multimodal_batch


@dataclass(frozen=True)
class RoutedOPDConfig:
    train_jsonl: str
    model: str
    student_init_adapter: str
    safety_teacher_adapter: str
    helpful_teacher_adapter: str
    output_dir: str
    mode: str = "multimodal"
    max_steps: int = 64
    batch_size: int = 4
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.05
    max_length: int = 1536
    rollout_tokens: int = 96
    top_k: int = 16
    opd_weight: float = 1.0
    anchor_sft_weight: float = 0.35
    repair_sft_weight: float = 0.70
    margin_weight: float = 0.55
    margin_beta: float = 1.0
    margin: float = 0.05
    min_pixels: int = 200704
    max_pixels: int = 401408
    seed: int = 127
    log_every: int = 2


def train_routed_opd(config: RoutedOPDConfig) -> dict[str, Any]:
    import torch
    from torch import nn

    if config.mode not in {"text_only", "multimodal"}:
        raise ValueError("mode must be text_only or multimodal")
    if config.max_steps <= 0 or config.batch_size <= 0:
        raise ValueError("max_steps and batch_size must be positive")
    if torch.cuda.device_count() < 2:
        raise RuntimeError("routed OPD requires two visible GPUs: student cuda:0, teacher cuda:1")

    rows = _load_rows(config.train_jsonl, mode=config.mode)
    route_rows = {
        "safety": [row for row in rows if _teacher_route(row) == "safety"],
        "helpful": [row for row in rows if _teacher_route(row) == "helpful"],
    }
    role_rows = {
        role: [row for row in rows if _boundary_role(row) == role]
        for role in ("unsafe", "safe", "helpful")
    }
    if not all(route_rows.values()):
        raise ValueError("both safety and helpful routed rows are required")
    if not all(role_rows.values()):
        raise ValueError("unsafe, safe, and helpful calibration roles are all required")

    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    student, teacher, processor = _load_models(config)
    trainable = [parameter for parameter in student.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    total_updates = math.ceil(config.max_steps / config.gradient_accumulation_steps)
    warmup = max(1, int(total_updates * config.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        _cosine_schedule(warmup=warmup, total=total_updates),
    )
    batches = _balanced_route_batches(
        role_rows,
        batch_size=config.batch_size,
        steps=config.max_steps,
        seed=config.seed,
    )

    manifest = {
        "format": "clean_routed_opd_run_v1",
        "config": asdict(config),
        "row_count": len(rows),
        "route_counts": {key: len(value) for key, value in route_rows.items()},
        "role_counts": {key: len(value) for key, value in role_rows.items()},
        "visible_cuda_devices": str(torch.cuda.device_count()),
        "claim_boundary": (
            "Decision-routed teacher distillation on role-balanced calibration rows. "
            "Calibration groups are not causal pairs and this run does not establish stage causality."
        ),
    }
    _write_json(output_dir / "run_manifest.json", manifest)
    student.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    running: dict[str, float] = {key: 0.0 for key in ("total", "opd", "sft", "margin")}
    route_counter = {"safety": 0, "helpful": 0}
    invalid_rollouts = 0
    wrong_decision_rollouts = 0
    update = 0
    running_steps = 0

    for step, (route, batch_rows) in enumerate(batches, 1):
        route_counter[route] += len(batch_rows)
        completions = _generate_rollouts(
            student,
            processor,
            batch_rows,
            max_new_tokens=config.rollout_tokens,
        )
        parsed = [parse_prediction(text) for text in completions]
        invalid_rollouts += sum(not item.valid_json for item in parsed)
        gold_decisions = [_gold_decision(row) for row in batch_rows]
        wrong_mask = torch.tensor(
            [
                (not item.valid_json) or item.final_decision != gold
                for item, gold in zip(parsed, gold_decisions)
            ],
            device="cuda:0",
            dtype=torch.float32,
        )
        wrong_decision_rollouts += int(wrong_mask.sum().item())

        rollout_rows = _replace_completions(batch_rows, completions)
        rollout_batch, _ = encode_multimodal_batch(
            processor,
            rollout_rows,
            max_length=config.max_length,
        )
        rollout_student = _move_batch(rollout_batch, torch.device("cuda:0"))
        labels = rollout_student.pop("labels")
        student_logits = student(**rollout_student, use_cache=False).logits
        kl_mask = _completion_prediction_mask(labels)
        teacher_indices, teacher_log_probs = _teacher_topk_targets(
            teacher,
            route,
            rollout_batch,
            top_k=config.top_k,
        )
        opd_loss = _topk_distillation_loss(
            student_logits,
            teacher_indices.to("cuda:0"),
            teacher_log_probs.to("cuda:0"),
            kl_mask,
        )
        del student_logits, rollout_student, teacher_indices, teacher_log_probs

        gold_batch, _ = encode_multimodal_batch(
            processor,
            batch_rows,
            max_length=config.max_length,
        )
        gold_batch = _move_batch(gold_batch, torch.device("cuda:0"))
        gold_labels = gold_batch["labels"]
        gold_logits = student(**{key: value for key, value in gold_batch.items() if key != "labels"}).logits
        chosen_logprob, per_row_nll = _completion_scores(gold_logits, gold_labels)
        sft_weights = config.anchor_sft_weight + config.repair_sft_weight * wrong_mask
        sft_loss = (per_row_nll * sft_weights).sum() / sft_weights.sum().clamp_min(1e-6)

        rejected_rows = _replace_completions(
            batch_rows,
            [_rejected_completion(row) for row in batch_rows],
        )
        rejected_batch, _ = encode_multimodal_batch(
            processor,
            rejected_rows,
            max_length=config.max_length,
        )
        rejected_batch = _move_batch(rejected_batch, torch.device("cuda:0"))
        rejected_labels = rejected_batch["labels"]
        rejected_logits = student(
            **{key: value for key, value in rejected_batch.items() if key != "labels"}
        ).logits
        rejected_logprob, _ = _completion_scores(rejected_logits, rejected_labels)
        margin_loss = torch.nn.functional.softplus(
            config.margin_beta * (rejected_logprob - chosen_logprob + config.margin)
        ).mean()

        loss = (
            config.opd_weight * opd_loss
            + sft_loss
            + config.margin_weight * margin_loss
        ) / config.gradient_accumulation_steps
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite routed OPD loss at step {step}")
        loss.backward()
        nn.utils.clip_grad_norm_(trainable, 1.0)
        if step % config.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            update += 1

        current = {
            "total": float(loss.detach().cpu()) * config.gradient_accumulation_steps,
            "opd": float(opd_loss.detach().cpu()),
            "sft": float(sft_loss.detach().cpu()),
            "margin": float(margin_loss.detach().cpu()),
        }
        for key, value in current.items():
            running[key] += value
        running_steps += 1
        if step == 1 or step % config.log_every == 0 or step == config.max_steps:
            elapsed = time.time() - started
            divisor = max(running_steps, 1)
            row = {
                "step": step,
                "max_steps": config.max_steps,
                "update": update,
                "route": route,
                "loss": round(running["total"] / divisor, 6),
                "opd_loss": round(running["opd"] / divisor, 6),
                "sft_loss": round(running["sft"] / divisor, 6),
                "margin_loss": round(running["margin"] / divisor, 6),
                "invalid_rollouts": invalid_rollouts,
                "wrong_decision_rollouts": wrong_decision_rollouts,
                "learning_rate": scheduler.get_last_lr()[0],
                "elapsed_sec": round(elapsed, 2),
                "eta_sec": round(elapsed / step * (config.max_steps - step), 1),
                "student_max_memory_gb": round(torch.cuda.max_memory_allocated(0) / 1024**3, 3),
                "teacher_max_memory_gb": round(torch.cuda.max_memory_allocated(1) / 1024**3, 3),
            }
            _append_jsonl(output_dir / "train_metrics.jsonl", row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            running = {key: 0.0 for key in running}
            running_steps = 0

    _save_adapter(student, processor, output_dir / "last")
    summary = {
        **manifest,
        "completed_steps": config.max_steps,
        "optimizer_updates": update,
        "route_examples_seen": route_counter,
        "invalid_rollouts": invalid_rollouts,
        "wrong_decision_rollouts": wrong_decision_rollouts,
        "elapsed_sec": round(time.time() - started, 2),
        "last_adapter": str(output_dir / "last"),
    }
    _write_json(output_dir / "training_summary.json", summary)
    return summary


def _load_rows(path: str, *, mode: str) -> list[dict[str, Any]]:
    rows = read_jsonl(path)
    clean: list[dict[str, Any]] = []
    for row in rows:
        messages = row.get("messages")
        target = row.get("training_target")
        if not isinstance(messages, list) or len(messages) != 3 or not isinstance(target, dict):
            raise ValueError(f"{row.get('id')}: invalid clean OPD row")
        images = list(row.get("images") or [])
        if mode == "text_only" and images:
            continue
        for image_path in images:
            if not Path(str(image_path)).is_file():
                raise FileNotFoundError(f"{row.get('id')}: missing image {image_path}")
        clean.append(row)
    if not clean:
        raise ValueError(f"no rows available for mode={mode}")
    return clean


def _load_models(config: RoutedOPDConfig) -> tuple[Any, Any, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    student_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    student = PeftModel.from_pretrained(
        student_base,
        config.student_init_adapter,
        is_trainable=True,
    )
    student.config.use_cache = False
    student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    student.enable_input_require_grads()
    student.to("cuda:0")

    teacher_base = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        config.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    teacher = PeftModel.from_pretrained(
        teacher_base,
        config.safety_teacher_adapter,
        adapter_name="safety",
        is_trainable=False,
    )
    teacher.load_adapter(config.helpful_teacher_adapter, adapter_name="helpful", is_trainable=False)
    teacher.to("cuda:1")
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    processor = AutoProcessor.from_pretrained(config.model, trust_remote_code=True)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = config.min_pixels
        processor.image_processor.max_pixels = config.max_pixels
    return student, teacher, processor


def _balanced_route_batches(
    roles: dict[str, list[dict[str, Any]]],
    *,
    batch_size: int,
    steps: int,
    seed: int,
) -> list[tuple[str, list[dict[str, Any]]]]:
    rng = random.Random(seed)
    pools = {key: list(value) for key, value in roles.items()}
    for pool in pools.values():
        rng.shuffle(pool)
    offsets = {"unsafe": 0, "safe": 0, "helpful": 0}
    result: list[tuple[str, list[dict[str, Any]]]] = []
    role_order = ("unsafe", "safe", "helpful")
    for step in range(steps):
        role = role_order[step % len(role_order)]
        route = "safety" if role == "unsafe" else "helpful"
        pool = pools[role]
        if offsets[role] + batch_size > len(pool):
            rng.shuffle(pool)
            offsets[role] = 0
        start = offsets[role]
        batch = pool[start : start + batch_size]
        if len(batch) < batch_size:
            batch = batch + pool[: batch_size - len(batch)]
        offsets[role] += batch_size
        result.append((route, batch))
    return result


def _generate_rollouts(
    student: Any,
    processor: Any,
    rows: list[dict[str, Any]],
    *,
    max_new_tokens: int,
) -> list[str]:
    import torch

    was_training = student.training
    had_gradient_checkpointing = bool(getattr(student, "is_gradient_checkpointing", False))
    student.eval()
    if had_gradient_checkpointing:
        student.gradient_checkpointing_disable()
    try:
        batch = encode_inference_batch(processor, rows)
        batch = _move_batch(batch, torch.device("cuda:0"))
        prompt_width = batch["input_ids"].shape[1]
        with torch.inference_mode():
            generated = student.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )
        completions = processor.batch_decode(
            generated[:, prompt_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    finally:
        if had_gradient_checkpointing:
            student.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        if was_training:
            student.train()
    return completions


def _teacher_topk_targets(
    teacher: Any,
    route: str,
    cpu_batch: dict[str, Any],
    *,
    top_k: int,
) -> tuple[Any, Any]:
    import torch

    teacher.set_adapter(route)
    batch = _move_batch(
        {key: value for key, value in cpu_batch.items() if key != "labels"},
        torch.device("cuda:1"),
    )
    with torch.inference_mode():
        logits = teacher(**batch, use_cache=False).logits.float()
        values, indices = torch.topk(logits, k=min(top_k, logits.shape[-1]), dim=-1)
        log_probs = torch.log_softmax(values, dim=-1)
    return indices.cpu(), log_probs.to(dtype=torch.float16, device="cpu")


def _topk_distillation_loss(
    student_logits: Any,
    teacher_indices: Any,
    teacher_log_probs: Any,
    mask: Any,
) -> Any:
    import torch

    selected = torch.gather(student_logits.float(), dim=-1, index=teacher_indices.long())
    student_log_probs = torch.log_softmax(selected, dim=-1)
    teacher_log_probs = teacher_log_probs.float()
    token_kl = (
        teacher_log_probs.exp() * (teacher_log_probs - student_log_probs)
    ).sum(dim=-1)
    weights = mask.float()
    return (token_kl * weights).sum() / weights.sum().clamp_min(1.0)


def _completion_prediction_mask(labels: Any) -> Any:
    import torch

    mask = torch.zeros_like(labels, dtype=torch.float32)
    mask[:, :-1] = labels[:, 1:].ne(-100).float()
    return mask


def _completion_scores(logits: Any, labels: Any) -> tuple[Any, Any]:
    import torch

    shift_logits = logits[:, :-1].float()
    shift_labels = labels[:, 1:].long()
    mask = shift_labels.ne(-100)
    safe = shift_labels.clamp_min(0)
    token_logprob = torch.log_softmax(shift_logits, dim=-1).gather(
        -1, safe.unsqueeze(-1)
    ).squeeze(-1)
    denominator = mask.sum(dim=1).clamp_min(1).float()
    mean_logprob = (token_logprob * mask.float()).sum(dim=1) / denominator
    return mean_logprob, -mean_logprob


def _replace_completions(
    rows: list[dict[str, Any]], completions: list[str]
) -> list[dict[str, Any]]:
    replaced: list[dict[str, Any]] = []
    for row, completion in zip(rows, completions):
        clone = dict(row)
        messages = [dict(message) for message in row["messages"]]
        messages[2]["content"] = completion
        clone["messages"] = messages
        replaced.append(clone)
    return replaced


def _rejected_completion(row: dict[str, Any]) -> str:
    target = row["training_target"]
    gold = str(target["final_decision"])
    rejected = "allow" if gold in {"refuse", "review"} else "refuse"
    payload = {
        "final_decision": rejected,
        "policy_id": "P_BENIGN" if rejected == "allow" else str(target["policy_id"]),
        "safe_response_behavior": (
            "answer_normally" if rejected == "allow" else "refuse_with_safe_alternative"
        ),
        "safe_response_policy": (
            "Answer the request directly."
            if rejected == "allow"
            else "Refuse the request and redirect to a safe alternative."
        ),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _teacher_route(row: dict[str, Any]) -> str:
    control = _nested_dict(row, "training_control")
    role = str(control.get("role") or control.get("source_role") or "")
    return "safety" if role == "unsafe" or _gold_decision(row) in {"refuse", "review"} else "helpful"


def _boundary_role(row: dict[str, Any]) -> str:
    control = _nested_dict(row, "training_control")
    role = str(control.get("role") or control.get("source_role") or "")
    if role in {"unsafe", "safe", "helpful"}:
        return role
    decision = _gold_decision(row)
    if decision in {"refuse", "review"}:
        return "unsafe"
    return "helpful" if decision == "allow" else "safe"


def _gold_decision(row: dict[str, Any]) -> str:
    target = _nested_dict(row, "training_target")
    return str(target.get("final_decision") or "review")


def _nested_dict(row: dict[str, Any], key: str) -> dict[str, Any]:
    value = row.get(key)
    return value if isinstance(value, dict) else {}


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


def _cosine_schedule(*, warmup: int, total: int) -> Any:
    def schedule(step: int) -> float:
        if step < warmup:
            return float(step + 1) / warmup
        progress = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))

    return schedule


def _save_adapter(model: Any, processor: Any, path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    processor.save_pretrained(path)


def _write_json(path: Path, obj: dict[str, Any]) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _append_jsonl(path: Path, obj: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(obj, ensure_ascii=False) + "\n")
