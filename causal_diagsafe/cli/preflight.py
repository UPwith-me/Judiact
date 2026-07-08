from __future__ import annotations

import argparse

from ..io_utils import ensure_out_dir, read_jsonl, write_json, write_text
from ..preflight import preflight_rows, render_preflight_markdown


def main() -> None:
    parser = argparse.ArgumentParser(description="Run CausalDiagSafe-VL schema/data preflight.")
    parser.add_argument("--input", required=True, help="Input JSONL records.")
    parser.add_argument("--out-dir", required=True, help="Output report directory.")
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    summary = preflight_rows(rows)
    out_dir = ensure_out_dir(args.out_dir)
    write_json(out_dir / "preflight_summary.json", summary)
    write_text(out_dir / "preflight_report.md", render_preflight_markdown(summary))
    print(f"preflight complete: {out_dir}")


if __name__ == "__main__":
    main()

