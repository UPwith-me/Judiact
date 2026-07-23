from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any


METRICS = (
    "stage_macro_f1",
    "decision_accuracy",
    "decision_macro_f1",
    "harmful_allow_rate",
    "over_refusal_rate",
    "target_stage_direction_accuracy",
    "target_stage_flip_rate",
    "target_stage_contrast",
    "non_target_invariance_rate",
    "pair_boundary_success",
    "all_pass_rate",
    "multi_fail_rate",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate H1 probe seeds and controls.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--pair-audit-json")
    args = parser.parse_args()

    pair_audit = None
    pair_audit_valid = False
    if args.pair_audit_json:
        pair_audit = json.loads(Path(args.pair_audit_json).read_text(encoding="utf-8"))
        pair_audit_valid = bool(pair_audit.get("valid_for_causal_evaluation"))

    runs = []
    for path in sorted(Path(args.root).glob("*/h1_probe_metrics.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        history = report.get("history") or []
        if not history:
            continue
        best = max(
            history,
            key=lambda row: float(row.get("stage_macro_f1", 0.0))
            + 0.25 * float(row.get("decision_macro_f1", 0.0)),
        )
        run = path.parent.name
        family = re.sub(r"_s(?:11|23|37)$", "", run)
        runs.append({"run": run, "family": family, "best_epoch": best.get("epoch"), **best})
    if not runs:
        raise FileNotFoundError(f"no H1 reports under {args.root}")

    families: dict[str, list[dict[str, Any]]] = {}
    for row in runs:
        families.setdefault(row["family"], []).append(row)
    aggregates = []
    for family, family_rows in sorted(families.items()):
        aggregate: dict[str, Any] = {"family": family, "seed_count": len(family_rows)}
        for metric in METRICS:
            values = [float(row.get(metric, 0.0)) for row in family_rows]
            aggregate[f"{metric}_mean"] = statistics.fmean(values)
            aggregate[f"{metric}_std"] = statistics.pstdev(values) if len(values) > 1 else 0.0
        aggregate["h1_gate_pass"] = (
            pair_audit_valid
            and aggregate["target_stage_flip_rate_mean"] >= 0.35
            and aggregate["target_stage_direction_accuracy_mean"] >= 0.60
            and aggregate["non_target_invariance_rate_mean"] >= 0.75
            and aggregate["all_pass_rate_mean"] <= 0.70
            and aggregate["multi_fail_rate_mean"] <= 0.30
        )
        aggregates.append(aggregate)

    by_family = {row["family"]: row for row in aggregates}
    b5 = by_family.get("b5_clean_linear")
    shuffled = by_family.get("b5_clean_shuffle_linear")
    shuffle_drop = None
    if b5 and shuffled:
        shuffle_drop = (
            b5["stage_macro_f1_mean"] - shuffled["stage_macro_f1_mean"]
        )
    result = {
        "format": "h1_probe_aggregate_v1",
        "runs": runs,
        "families": aggregates,
        "b5_stage_macro_f1_shuffle_drop": shuffle_drop,
        "pair_audit_valid": pair_audit_valid,
        "pair_audit": pair_audit,
        "claim_boundary": (
            "Strict pair counts are small; confidence intervals are not submission-grade. "
            "Failure of the H1 gate blocks H2/IICT rather than being hidden by static accuracy."
        ),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    Path(args.out_md).write_text(_render_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False), flush=True)


def _render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# H1 冻结表示探针结果",
        "",
        "| family | seeds | stage macro-F1 | target flip | target direction | non-target inv | all-pass | gate |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in result["families"]:
        lines.append(
            "| {family} | {seed_count} | {stage:.4f} +/- {stage_std:.4f} | {flip:.4f} | "
            "{direction:.4f} | {invariance:.4f} | {all_pass:.4f} | {gate} |".format(
                family=row["family"],
                seed_count=row["seed_count"],
                stage=row["stage_macro_f1_mean"],
                stage_std=row["stage_macro_f1_std"],
                flip=row["target_stage_flip_rate_mean"],
                direction=row["target_stage_direction_accuracy_mean"],
                invariance=row["non_target_invariance_rate_mean"],
                all_pass=row["all_pass_rate_mean"],
                gate="PASS" if row["h1_gate_pass"] else "FAIL",
            )
        )
    lines.extend(
        [
            "",
            f"- 严格 pair 输入干预约束有效: {result.get('pair_audit_valid')}",
            f"- B5 标签打乱后的 stage macro-F1 降幅: {result.get('b5_stage_macro_f1_shuffle_drop')}",
            "- H1 gate 未通过时保持 H2/IICT 冻结。",
            "- strict pair 数量较少，结果只用于路线判定。",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
