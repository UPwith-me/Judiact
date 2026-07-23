from __future__ import annotations

import argparse
import json

from ..hidden_export import export_hidden_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Export compact sentinel hidden-state caches.")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--output-pt", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=200704)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--pooling", choices=["last", "last4_mean"], default="last4_mean")
    parser.add_argument("--image-mode", choices=["normal", "drop", "zero"], default="normal")
    parser.add_argument("--max-samples", type=int, default=0)
    args = parser.parse_args()

    summary = export_hidden_cache(
        input_jsonl=args.input_jsonl,
        model_path=args.model,
        adapter_path=args.adapter,
        output_pt=args.output_pt,
        batch_size=args.batch_size,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        pooling=args.pooling,
        image_mode=args.image_mode,
        max_samples=args.max_samples,
    )
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
