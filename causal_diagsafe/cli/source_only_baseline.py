from __future__ import annotations

import argparse

from ..io_utils import ensure_out_dir, read_jsonl, write_json, write_text
from ..source_only import DEFAULT_METADATA_FIELDS, run_metadata_baseline


def main() -> None:
    parser = argparse.ArgumentParser(description="Run source-only / metadata-only control baseline.")
    parser.add_argument("--train", required=True, help="Training JSONL records.")
    parser.add_argument("--eval", required=True, help="Evaluation JSONL records.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--fields",
        default=",".join(DEFAULT_METADATA_FIELDS),
        help="Comma-separated metadata fields used as shortcut features.",
    )
    args = parser.parse_args()

    fields = tuple(field.strip() for field in args.fields.split(",") if field.strip())
    summary = run_metadata_baseline(read_jsonl(args.train), read_jsonl(args.eval), fields=fields)
    out_dir = ensure_out_dir(args.out_dir)
    write_json(out_dir / "source_only_baseline.json", summary)
    write_text(
        out_dir / "source_only_baseline.md",
        "# Source-only / Metadata-only Baseline\n\n"
        f"- train_count: {summary['train_count']}\n"
        f"- eval_count: {summary['eval_count']}\n"
        f"- fields: {', '.join(summary['fields'])}\n"
        f"- accuracy: {summary['decision_metrics']['accuracy']:.4f}\n"
        f"- harmful_allow: {summary['decision_metrics']['harmful_allow_rate']:.4f}\n"
        f"- over_refusal: {summary['decision_metrics']['over_refusal_rate']:.4f}\n\n"
        f"{summary['interpretation']}\n",
    )
    print(f"source-only baseline complete: {out_dir}")


if __name__ == "__main__":
    main()

