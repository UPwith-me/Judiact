from __future__ import annotations

import argparse
from pathlib import Path

from ..controls import render_cpu_controls_markdown, run_cpu_controls
from ..io_utils import ensure_out_dir, read_jsonl, write_json, write_text


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pre-GPU metadata and gold-router controls.")
    parser.add_argument("--prepared-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--router-epochs", type=int, default=50)
    args = parser.parse_args()

    prepared = Path(args.prepared_dir)
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    print("[1/3] loading prepared train/dev manifests", flush=True)
    train_rows = read_jsonl(prepared / "train_algorithm_manifest.jsonl")
    dev_rows = read_jsonl(prepared / "dev_algorithm_manifest.jsonl")
    dev_pairs = read_jsonl(prepared / "dev_causal_pairs.jsonl")
    print("[2/3] running metadata and gold-router controls on CPU", flush=True)
    summary = run_cpu_controls(
        train_rows,
        dev_rows,
        dev_pairs,
        seeds=seeds,
        router_epochs=args.router_epochs,
    )
    print("[3/3] writing reports", flush=True)
    out_dir = ensure_out_dir(args.out_dir)
    write_json(out_dir / "cpu_controls_summary.json", summary)
    write_text(out_dir / "cpu_controls_summary.md", render_cpu_controls_markdown(summary))
    print(f"CPU controls complete: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
