from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..algorithm_data import (
    build_causal_pairs,
    build_release_examples,
    render_label_bridge_markdown,
    render_pair_markdown,
)
from ..io_utils import ensure_out_dir, write_json, write_jsonl, write_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build masked R/E/F algorithm manifests and strict causal pair candidates."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--release", default="combined_manual_gold_v1")
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("config root must be an object")

    print("[1/4] loading and joining release rows", flush=True)
    by_split, bridge_report = build_release_examples(config, release_name=args.release)
    out_dir = ensure_out_dir(args.out_dir)

    print("[2/4] writing algorithm manifests", flush=True)
    for split, examples in by_split.items():
        write_jsonl(out_dir / f"{split}_algorithm_manifest.jsonl", (e.to_dict() for e in examples))

    print("[3/4] building strict group pairs", flush=True)
    pair_reports = {}
    for split, examples in by_split.items():
        pairs, pair_report = build_causal_pairs(examples)
        write_jsonl(out_dir / f"{split}_causal_pairs.jsonl", pairs)
        pair_reports[split] = pair_report

    print("[4/4] writing reports", flush=True)
    write_json(out_dir / "label_bridge_report.json", bridge_report)
    write_text(out_dir / "label_bridge_report.md", render_label_bridge_markdown(bridge_report))
    write_json(out_dir / "causal_pair_report.json", pair_reports)
    pair_md = "# Causal Pair Reports\n\n" + "\n".join(
        f"## {split}\n\n{render_pair_markdown(report)}" for split, report in pair_reports.items()
    )
    write_text(out_dir / "causal_pair_report.md", pair_md)
    summary = {
        "release": args.release,
        "split_counts": {split: len(examples) for split, examples in by_split.items()},
        "pair_counts": {split: report["pair_count"] for split, report in pair_reports.items()},
        "bridge_issue_count": bridge_report["issue_count"],
    }
    write_json(out_dir / "prepare_summary.json", summary)
    print(f"prepare complete: {summary} out={out_dir}", flush=True)
    if bridge_report["issue_count"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
