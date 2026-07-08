from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .constants import STAGES
from .schema import DiagRecord, normalize_record, validate_record


def preflight_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = [normalize_record(row) for row in rows]
    validation_errors = [
        {"id": normalize_record(row).sample_id, "errors": validate_record(row)}
        for row in rows
        if validate_record(row)
    ]
    return {
        "format": "causal_diagsafe_preflight_v0.1",
        "count": len(records),
        "validation_error_count": len(validation_errors),
        "validation_error_examples": validation_errors[:50],
        "distributions": _distributions(records),
        "coverage": _coverage(records),
        "source_confound_risks": _source_confound_risks(records),
        "recommendations": _recommendations(records, validation_errors),
    }


def _distributions(records: list[DiagRecord]) -> dict[str, dict[str, int]]:
    fields = (
        "final_decision",
        "source_dataset",
        "modality_scope",
        "harm_category",
        "attack_mechanism",
        "target_stage",
        "role",
        "changed_factor",
        "policy_condition",
    )
    return {field: dict(Counter(str(getattr(r, field) or "unknown") for r in records)) for field in fields}


def _coverage(records: list[DiagRecord]) -> dict[str, Any]:
    stage_cov: dict[str, dict[str, int]] = {}
    for stage in STAGES:
        r_count = sum(1 for row in records if row.stage_states.states[stage].R is not None)
        e_count = sum(1 for row in records if row.stage_states.states[stage].E is not None)
        f_count = sum(1 for row in records if row.stage_states.states[stage].F is not None)
        stage_cov[stage] = {"R": r_count, "E": e_count, "F": f_count}
    groups = Counter(r.group_id for r in records if r.group_id)
    return {
        "records_with_image_rate": _rate(sum(1 for r in records if r.images), len(records)),
        "records_with_group_id_rate": _rate(sum(1 for r in records if r.group_id), len(records)),
        "group_count": len(groups),
        "stage_label_coverage": stage_cov,
    }


def _source_confound_risks(records: list[DiagRecord]) -> list[dict[str, Any]]:
    by_modality: dict[str, Counter[str]] = defaultdict(Counter)
    for row in records:
        by_modality[row.modality_scope][row.source_dataset] += 1
    risks: list[dict[str, Any]] = []
    for modality, counts in sorted(by_modality.items()):
        total = sum(counts.values())
        if not total:
            continue
        source, count = counts.most_common(1)[0]
        dominance = count / total
        if dominance >= 0.8 and total >= 20:
            risks.append(
                {
                    "modality_scope": modality,
                    "dominant_source": source,
                    "dominance_rate": dominance,
                    "count": total,
                    "risk": "source-modality confound",
                }
            )
    return risks


def _recommendations(records: list[DiagRecord], errors: list[dict[str, Any]]) -> list[str]:
    recs: list[str] = []
    if errors:
        recs.append("先修复 schema validation error，再进入训练。")
    dist = Counter(r.target_stage for r in records)
    known_stage_total = sum(count for stage, count in dist.items() if stage in STAGES)
    if known_stage_total:
        max_stage, max_count = max(((s, c) for s, c in dist.items() if s in STAGES), key=lambda x: x[1])
        if max_count / known_stage_total > 0.75:
            recs.append(f"target_stage 过度集中在 {max_stage}，IICT 训练会退化成单阶段校准。")
    if sum(1 for r in records if r.final_decision == "refuse") / max(len(records), 1) > 0.75:
        recs.append("refuse 标签占比过高，accuracy 会奖励拒绝先验，必须同时报告 over-refusal。")
    if _source_confound_risks(records):
        recs.append("存在 source-modality confound，训练和评测都需要 source-held-out 或 source-only baseline。")
    if not recs:
        recs.append("未发现阻断性 preflight 问题，可以进入 smoke training/evaluation。")
    return recs


def render_preflight_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# CausalDiagSafe-VL Preflight 报告",
        "",
        f"- count: {summary.get('count', 0)}",
        f"- validation_error_count: {summary.get('validation_error_count', 0)}",
        "",
        "## 关键分布",
        "",
    ]
    distributions = summary.get("distributions", {})
    for field in ("final_decision", "source_dataset", "modality_scope", "target_stage", "changed_factor"):
        lines.append(f"### {field}")
        for key, value in sorted((distributions.get(field) or {}).items(), key=lambda kv: str(kv[0])):
            lines.append(f"- {key}: {value}")
        lines.append("")
    lines.extend(["## Source Confound 风险", ""])
    risks = summary.get("source_confound_risks") or []
    if risks:
        for risk in risks:
            lines.append(
                f"- {risk['modality_scope']}: {risk['dominant_source']} "
                f"dominance={risk['dominance_rate']:.3f}, count={risk['count']}"
            )
    else:
        lines.append("- 未触发 0.8 dominance 阈值。")
    lines.extend(["", "## 建议", ""])
    for rec in summary.get("recommendations", []):
        lines.append(f"- {rec}")
    return "\n".join(lines) + "\n"


def _rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0

