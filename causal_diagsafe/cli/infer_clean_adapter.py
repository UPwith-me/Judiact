from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..inference import run_clean_adapter_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Qwen2.5-VL LoRA inference.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output-jsonl", required=True)
    parser.add_argument("--summary-json", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    summary = run_clean_adapter_inference(
        input_jsonl=args.input_jsonl,
        model_path=args.model,
        adapter_path=args.adapter,
        output_jsonl=args.output_jsonl,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        max_samples=args.max_samples,
    )
    target = Path(args.summary_json)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
