from __future__ import annotations

import argparse
from typing import Any

from ..io_utils import ensure_out_dir, read_jsonl, write_csv, write_json, write_text
from ..metrics import (
    bucket_metrics,
    decision_metrics,
    normalize_records_and_predictions,
    policy_metrics,
    stage_collapse_metrics,
    stage_state_metrics,
)
from ..report import render_eval_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Stage-0 / CausalDiagSafe predictions.")
    parser.add_argument("--gold", required=True, help="Gold JSONL records.")
    parser.add_argument("--pred", required=True, help="Prediction JSONL records.")
    parser.add_argument("--out-dir", required=True, help="Output report directory.")
    args = parser.parse_args()

    gold_rows = read_jsonl(args.gold)
    pred_rows = read_jsonl(args.pred)
    records, predictions = normalize_records_and_predictions(gold_rows, pred_rows)
    bucket_rows = bucket_metrics(records, predictions)
    summary: dict[str, Any] = {
        "decision_metrics": decision_metrics(records, predictions),
        "policy_metrics": policy_metrics(records, predictions),
        "stage_state_metrics": stage_state_metrics(records, predictions),
        "stage_collapse_metrics": stage_collapse_metrics(predictions),
        "bucket_metrics": bucket_rows,
    }
    out_dir = ensure_out_dir(args.out_dir)
    write_json(out_dir / "eval_summary.json", summary)
    write_csv(out_dir / "bucket_metrics.csv", bucket_rows)
    write_text(out_dir / "eval_report.md", render_eval_markdown(summary))
    print(f"evaluation complete: {out_dir}")


if __name__ == "__main__":
    main()
