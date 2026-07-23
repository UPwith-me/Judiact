from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from ..training.multimodal_sft import (
    CleanSFTConfig,
    encode_multimodal_batch,
    read_clean_sft_rows,
    train_clean_multimodal_sft,
    validate_training_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train B1/B2/B3 clean Qwen2.5-VL LoRA with real image inputs."
    )
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--eval-jsonl")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--init-adapter")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--max-length", type=int, default=1536)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=5)
    parser.add_argument("--eval-every", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=0)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--processor-smoke",
        action="store_true",
        help="During dry-run, encode a mixed two-row batch on CPU without loading the model.",
    )
    parser.add_argument(
        "--processor-audit-all",
        action="store_true",
        help="During dry-run, encode every training row on CPU and fail on target truncation.",
    )
    parser.add_argument("--processor-audit-batch-size", type=int, default=2)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.dry_run:
        print("[1/2] validating all manifest rows and image paths", flush=True)
        report = validate_training_manifest(args.train_jsonl, model_path=args.model)
        if args.eval_jsonl:
            print("[2/2] validating evaluation manifest", flush=True)
            report["eval"] = validate_training_manifest(args.eval_jsonl, model_path=args.model)
            report["issue_count"] += report["eval"]["issue_count"]
        else:
            print("[2/2] no evaluation manifest requested", flush=True)
        if (args.processor_smoke or args.processor_audit_all) and report["issue_count"] == 0:
            print("[processor] loading processor", flush=True)
            from transformers import AutoProcessor

            processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
            if hasattr(processor, "image_processor"):
                processor.image_processor.min_pixels = args.min_pixels
                processor.image_processor.max_pixels = args.max_pixels
            rows = read_clean_sft_rows(args.train_jsonl)
        if args.processor_smoke and report["issue_count"] == 0:
            print("[processor-smoke] encoding a mixed batch", flush=True)
            image_row = next((row for row in rows if row.get("images")), None)
            text_row = next((row for row in rows if not row.get("images")), None)
            smoke_rows = [row for row in (image_row, text_row) if row is not None]
            batch, metadata = encode_multimodal_batch(
                processor,
                smoke_rows,
                max_length=args.max_length,
            )
            report["processor_smoke"] = {
                "row_ids": [row["id"] for row in smoke_rows],
                "tensor_shapes": {
                    key: list(value.shape)
                    for key, value in batch.items()
                    if hasattr(value, "shape")
                },
                **metadata,
            }
            print(json.dumps(report["processor_smoke"], ensure_ascii=False), flush=True)
        if args.processor_audit_all and report["issue_count"] == 0:
            if args.processor_audit_batch_size <= 0:
                raise ValueError("processor audit batch size must be positive")
            print(f"[processor-audit] encoding all {len(rows)} rows", flush=True)
            sequence_lengths: list[int] = []
            target_token_counts: list[int] = []
            batch_size = args.processor_audit_batch_size
            for start in range(0, len(rows), batch_size):
                _, metadata = encode_multimodal_batch(
                    processor,
                    rows[start : start + batch_size],
                    max_length=args.max_length,
                )
                sequence_lengths.extend(metadata["sequence_lengths"])
                target_token_counts.extend(metadata["target_token_counts"])
                completed = min(start + batch_size, len(rows))
                if completed == len(rows) or completed % 100 == 0:
                    print(
                        f"[processor-audit] {completed}/{len(rows)} rows complete",
                        flush=True,
                    )
            report["processor_audit_all"] = {
                "row_count": len(rows),
                "batch_size": batch_size,
                "sequence_tokens": _distribution(sequence_lengths),
                "target_tokens": _distribution(target_token_counts),
                "target_truncation_count": 0,
            }
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dry_run_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        if report["issue_count"]:
            raise SystemExit(2)
        return

    config = CleanSFTConfig(
        train_jsonl=args.train_jsonl,
        eval_jsonl=args.eval_jsonl,
        model=args.model,
        output_dir=args.output_dir,
        init_adapter=args.init_adapter,
        epochs=args.epochs,
        max_steps=args.max_steps,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        max_length=args.max_length,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        seed=args.seed,
        log_every=args.log_every,
        eval_every=args.eval_every,
        max_eval_samples=args.max_eval_samples,
        gradient_clip=args.gradient_clip,
        attn_implementation=args.attn_implementation,
    )
    train_clean_multimodal_sft(config)


def _distribution(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    if not ordered:
        return {"min": 0, "p50": 0, "p95": 0, "max": 0, "mean": 0.0}
    return {
        "min": ordered[0],
        "p50": ordered[len(ordered) // 2],
        "p95": ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))],
        "max": ordered[-1],
        "mean": statistics.fmean(ordered),
    }


if __name__ == "__main__":
    main()
