from __future__ import annotations

import argparse

from ..group_audit import audit_groups
from ..io_utils import ensure_out_dir, read_jsonl, write_json, write_text
from ..report import render_group_audit_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Run strict intervention group audit.")
    parser.add_argument("--gold", required=True, help="Gold causal group JSONL records.")
    parser.add_argument("--pred", required=True, help="Prediction JSONL records.")
    parser.add_argument("--out-dir", required=True, help="Output report directory.")
    args = parser.parse_args()

    summary = audit_groups(read_jsonl(args.gold), read_jsonl(args.pred))
    out_dir = ensure_out_dir(args.out_dir)
    write_json(out_dir / "group_audit_summary.json", summary)
    write_text(out_dir / "group_audit_report.md", render_group_audit_markdown(summary))
    print(f"group audit complete: {out_dir}")


if __name__ == "__main__":
    main()

