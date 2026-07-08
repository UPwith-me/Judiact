from __future__ import annotations

from typing import Any


def render_eval_markdown(summary: dict[str, Any]) -> str:
    decision = summary.get("decision_metrics", {})
    collapse = summary.get("stage_collapse_metrics", {})
    lines = [
        "# CausalDiagSafe-VL 评测报告",
        "",
        "## Final Decision",
        "",
        f"- count: {decision.get('count', 0)}",
        f"- accuracy: {_pct(decision.get('accuracy'))}",
        f"- macro_f1: {_pct(decision.get('macro_f1'))}",
        f"- harmful_allow: {_pct(decision.get('harmful_allow_rate'))}",
        f"- under_refusal: {_pct(decision.get('under_refusal_rate'))}",
        f"- over_refusal: {_pct(decision.get('over_refusal_rate'))}",
        f"- json_valid_rate: {_pct(decision.get('json_valid_rate'))}",
        f"- legacy_stage_output_rate: {_pct(decision.get('legacy_stage_output_rate'))}",
        "",
        "## Stage Collapse",
        "",
        f"- all_clear_rate: {_pct(collapse.get('all_clear_rate'))}",
        f"- all_risk_rate: {_pct(collapse.get('all_risk_rate'))}",
        f"- multi_risk_rate: {_pct(collapse.get('multi_risk_rate'))}",
        f"- unknown_stage_rate: {_pct(collapse.get('unknown_stage_rate'))}",
        "",
        "## Bucket 指标",
        "",
        "| bucket | count | acc | macro_f1 | harmful_allow | over_refusal |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in (summary.get("bucket_metrics") or [])[:80]:
        lines.append(
            "| {bucket} | {count} | {acc} | {macro} | {ha} | {orr} |".format(
                bucket=str(row.get("bucket", "")).replace("|", "/"),
                count=row.get("count", 0),
                acc=_pct(row.get("accuracy")),
                macro=_pct(row.get("macro_f1")),
                ha=_pct(row.get("harmful_allow_rate")),
                orr=_pct(row.get("over_refusal_rate")),
            )
        )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- legacy stage output 只能用于 collapse 审计，不能作为 CausalDiagSafe-VL 的 R/E/F 诊断证据。",
            "- harmful-only benchmark 上 accuracy 高时，必须同时查看 over-refusal 与 hard benign 分桶。",
        ]
    )
    return "\n".join(lines) + "\n"


def render_group_audit_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# CausalDiagSafe-VL Strict Group Audit",
        "",
        "## 核心指标",
        "",
        f"- group_count: {summary.get('group_count', 0)}",
        f"- complete_group_count: {summary.get('complete_group_count', 0)}",
        f"- boundary_success_rate: {_pct(summary.get('boundary_success_rate'))}",
        f"- target_stage_flip_rate: {_pct(summary.get('target_stage_flip_rate'))}",
        f"- non_target_invariance_rate: {_pct(summary.get('non_target_invariance_rate'))}",
        f"- evidence_flip_rate: {_pct(summary.get('evidence_flip_rate'))}",
        f"- policy_leakage_rate: {_pct(summary.get('policy_leakage_rate'))}",
        "",
        "## Role 指标",
        "",
        f"- unsafe_refuse_rate: {_pct(summary.get('unsafe_refuse_rate'))}",
        f"- safe_not_refuse_rate: {_pct(summary.get('safe_not_refuse_rate'))}",
        f"- helpful_not_refuse_rate: {_pct(summary.get('helpful_not_refuse_rate'))}",
        "",
        "## 按 target_stage",
        "",
        "| stage | count | boundary | target_flip | non_target_inv |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.get("by_target_stage", []):
        lines.append(
            f"| {row.get('target_stage')} | {row.get('count')} | {_pct(row.get('boundary_success_rate'))} "
            f"| {_pct(row.get('target_stage_flip_rate'))} | {_pct(row.get('non_target_invariance_rate'))} |"
        )
    failed = summary.get("failed_group_examples") or []
    if failed:
        lines.extend(["", "## 失败组样例", ""])
        for row in failed[:20]:
            lines.append(
                f"- {row.get('group_id')}: boundary={row.get('boundary_success')}, "
                f"target_flip={row.get('target_stage_flip')}, non_target_inv={row.get('non_target_invariant')}"
            )
    return "\n".join(lines) + "\n"


def _pct(value: Any) -> str:
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "0.00%"

