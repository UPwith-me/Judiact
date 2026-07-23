from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..algorithm_data import build_release_examples
from ..clean_reproduction import (
    build_clean_reproduction_bundle,
    build_clean_sft_row,
    validate_clean_rows,
)
from ..io_utils import ensure_out_dir, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare clean B1-B5 manifests from the frozen manually reviewed train split."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--release", default="combined_manual_gold_v1")
    parser.add_argument("--max-calibration-groups", type=int, default=512)
    parser.add_argument("--seed", type=int, default=83)
    args = parser.parse_args()

    with Path(args.config).open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("config root must be an object")

    print("[1/5] loading frozen release and label overlay", flush=True)
    by_split, bridge_report = build_release_examples(config, release_name=args.release)
    if bridge_report["issue_count"]:
        raise RuntimeError(
            f"label bridge has {bridge_report['issue_count']} issue(s); clean training export aborted"
        )

    print("[2/5] building clean joint and teacher manifests", flush=True)
    bundle = build_clean_reproduction_bundle(
        by_split["train"],
        max_calibration_groups=args.max_calibration_groups,
        seed=args.seed,
    )
    outputs = {
        "b1_clean_joint_sft_train.jsonl": bundle.joint_sft,
        "b1_clean_joint_sft_dev.jsonl": tuple(
            build_clean_sft_row(example, dataset_role="joint_sft_eval")
            for example in by_split["dev"]
        ),
        "b1_clean_joint_sft_test.jsonl": tuple(
            build_clean_sft_row(example, dataset_role="joint_sft_frozen_test")
            for example in by_split["test"]
        ),
        "b2_clean_safety_teacher_train.jsonl": bundle.safety_teacher,
        "b3_clean_helpful_teacher_train.jsonl": bundle.helpful_teacher,
        "b4_e20_clean_exact_textonly_train.jsonl": bundle.text_only_exact,
        "b45_clean_calibration_rows.jsonl": bundle.calibration_rows,
    }

    print("[3/5] validating prompt leakage and target contracts", flush=True)
    validation: dict[str, list[str]] = {}
    for name, rows in outputs.items():
        validation[name] = validate_clean_rows(rows)
    issue_count = sum(len(issues) for issues in validation.values())
    if issue_count:
        for name, issues in validation.items():
            if issues:
                print(f"[invalid] {name}: {issues[:10]}", flush=True)
        raise RuntimeError(f"clean reproduction validation failed: {issue_count} issue(s)")

    print("[4/5] writing five training manifests and calibration groups", flush=True)
    out_dir = ensure_out_dir(args.out_dir)
    for name, rows in outputs.items():
        write_jsonl(out_dir / name, rows)
    write_jsonl(out_dir / "b45_clean_calibration_groups.jsonl", bundle.calibration_groups)

    print("[5/5] writing reproducibility report", flush=True)
    report = {
        **bundle.report,
        "release": args.release,
        "validation": {name: {"issue_count": len(issues)} for name, issues in validation.items()},
        "bridge_version": bridge_report.get("bridge_version"),
        "source_bridge_issue_count": bridge_report.get("issue_count"),
        "output_files": sorted([*outputs, "b45_clean_calibration_groups.jsonl"]),
    }
    write_json(out_dir / "clean_reproduction_report.json", report)
    print(
        f"clean reproduction ready: train={len(bundle.joint_sft)} "
        f"safety={len(bundle.safety_teacher)} helpful={len(bundle.helpful_teacher)} "
        f"calibration_groups={len(bundle.calibration_groups)} out={out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
