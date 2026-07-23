from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

from .constants import (
    DECISION_TO_ID,
    EVIDENCE_TO_ID,
    ID_TO_EVIDENCE,
    INTENT_TO_ID,
    STAGES,
    STAGE_TO_ID,
)
from .io_utils import read_jsonl
from .schema import DiagRecord, StageStateBundle, normalize_record


BRIDGE_VERSION = "crossmodal_requirement_evidence_partial_v2"
EXPECTED_FACTOR_BY_STAGE = {
    "S1": frozenset({"visual_evidence", "ocr_text"}),
    "S2": frozenset({"image_text_binding"}),
    "S3": frozenset({"intent"}),
    "S4": frozenset({"policy_rule"}),
    "S5": frozenset({"response_execution"}),
}


@dataclass(frozen=True)
class StageSupervision:
    R: tuple[int, ...]
    R_mask: tuple[int, ...]
    E: tuple[int, ...]
    E_mask: tuple[int, ...]
    F: tuple[int, ...]
    F_mask: tuple[int, ...]
    source: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "R": list(self.R),
            "R_mask": list(self.R_mask),
            "E": list(self.E),
            "E_mask": list(self.E_mask),
            "F": list(self.F),
            "F_mask": list(self.F_mask),
            "source": list(self.source),
        }


@dataclass(frozen=True)
class AlgorithmExample:
    sample_id: str
    split: str
    query: str
    images: tuple[str, ...]
    response_candidate: str
    policy_text: str
    policy_text_supplied: bool
    policy_condition: str
    final_decision: str
    policy_id: str
    fine_policy_id: str
    gold_response_policy: str
    policy_action: int
    policy_action_mask: int
    benign: int
    benign_mask: int
    intent: int
    intent_mask: int
    role: str
    source_role: str
    target_stage: str
    changed_factor: str
    modality_scope: str
    source_dataset: str
    group_id: str
    stable_non_target_mask: tuple[int, ...]
    stage: StageSupervision
    evidence_by_stage: dict[str, tuple[dict[str, str], ...]] = field(default_factory=dict)
    label_provenance: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        stage_states = {
            stage: {
                "R": self.stage.R[index] if self.stage.R_mask[index] else None,
                "E": ID_TO_EVIDENCE[self.stage.E[index]] if self.stage.E_mask[index] else None,
                "F": self.stage.F[index] if self.stage.F_mask[index] else None,
            }
            for index, stage in enumerate(STAGES)
        }
        model_input: dict[str, Any] = {
            "query": self.query,
            "images": list(self.images),
        }
        if self.response_candidate:
            model_input["response_candidate"] = self.response_candidate
        if self.policy_text_supplied:
            model_input["policy_text"] = self.policy_text
        return {
            "id": self.sample_id,
            "split": self.split,
            "policy_condition": self.policy_condition,
            "model_input": model_input,
            "targets": {
                "decision": DECISION_TO_ID[self.final_decision],
                "final_decision": self.final_decision,
                "policy_id": self.policy_id,
                "fine_policy_id": self.fine_policy_id,
                "gold_response_policy": self.gold_response_policy,
                "policy_action": self.policy_action,
                "policy_action_mask": self.policy_action_mask,
                "policy_action_source": (
                    "explicit_manual" if self.policy_action_mask else "unknown_independent_label"
                ),
                "benign": self.benign,
                "benign_mask": self.benign_mask,
                "benign_source": (
                    "explicit_manual" if self.benign_mask else "unknown_independent_label"
                ),
                "intent": self.intent,
                "intent_mask": self.intent_mask,
                "intent_source": (
                    "explicit_manual" if self.intent_mask else "unknown_independent_label"
                ),
                "role": self.role,
                "target_stage": self.target_stage,
                "target_stage_id": STAGE_TO_ID.get(self.target_stage, -1),
                "changed_factor": self.changed_factor,
                "stable_non_target_mask": list(self.stable_non_target_mask),
                "stage": self.stage.to_dict(),
            },
            "metadata": {
                "modality_scope": self.modality_scope,
                "source_dataset": self.source_dataset,
                "group_id": self.group_id,
                "source_role": self.source_role,
                "label_provenance": self.label_provenance,
            },
            "stage_states": stage_states,
            "audit_supervision": {
                "evidence_by_stage": {
                    stage: list(items) for stage, items in self.evidence_by_stage.items()
                }
            },
        }


def load_group_registry(path: str | Path) -> tuple[dict[str, str], dict[str, str]]:
    group_by_id: dict[str, str] = {}
    split_by_id: dict[str, str] = {}
    for row in read_jsonl(path):
        group_id = str(row.get("group_id") or "")
        split = str(row.get("split") or "")
        candidate_ids = row.get("candidate_ids")
        if not group_id or not isinstance(candidate_ids, list):
            raise ValueError(f"invalid group registry row: {row}")
        for candidate_id in candidate_ids:
            sample_id = str(candidate_id)
            old_group = group_by_id.get(sample_id)
            if old_group is not None and old_group != group_id:
                raise ValueError(f"{sample_id} appears in multiple groups")
            group_by_id[sample_id] = group_id
            split_by_id[sample_id] = split
    return group_by_id, split_by_id


def build_algorithm_examples(
    input_rows: list[dict[str, Any]],
    *,
    split: str,
    audit_rows: list[dict[str, Any]] | None = None,
    overlay_rows: list[dict[str, Any]] | None = None,
    group_by_id: dict[str, str] | None = None,
    registry_split_by_id: dict[str, str] | None = None,
) -> tuple[list[AlgorithmExample], list[str]]:
    audit_by_id = _index_rows(audit_rows or [])
    overlay_by_id = _index_rows(overlay_rows or [])
    group_by_id = group_by_id or {}
    registry_split_by_id = registry_split_by_id or {}
    issues: list[str] = []
    examples: list[AlgorithmExample] = []

    for row in input_rows:
        record = normalize_record(row)
        if not record.sample_id:
            issues.append("row with missing id")
            continue
        audit = audit_by_id.get(record.sample_id, {})
        overlay = overlay_by_id.get(record.sample_id, {})
        if audit:
            _append_consistency_issues(record, normalize_record(audit), issues)
        registry_split = registry_split_by_id.get(record.sample_id)
        if registry_split and registry_split != split:
            issues.append(
                f"{record.sample_id}: registry split={registry_split}, requested split={split}"
            )

        explicit_row = _find_explicit_stage_row(row, overlay, audit)
        supervision, provenance = _build_stage_supervision(record, audit, explicit_row)
        evidence = _collect_evidence(record, audit, explicit_row)
        targets: dict[str, Any] = (
            row["targets"] if isinstance(row.get("targets"), dict) else {}
        )
        audit_targets: dict[str, Any] = (
            audit["targets"] if isinstance(audit.get("targets"), dict) else {}
        )
        fine_policy_id = str(
            targets.get("fine_policy_id")
            or row.get("fine_policy_id")
            or audit_targets.get("fine_policy_id")
            or audit.get("fine_policy_id")
            or record.policy_id
        )
        gold_response_policy = str(
            targets.get("gold_response_policy")
            or row.get("gold_response_policy")
            or audit_targets.get("gold_response_policy")
            or audit.get("gold_response_policy")
            or ""
        )
        auxiliary, auxiliary_provenance = _build_auxiliary_supervision(row, overlay, audit)
        group_id = group_by_id.get(record.sample_id, record.group_id)
        if not group_id:
            issues.append(f"{record.sample_id}: missing group_id")

        examples.append(
            AlgorithmExample(
                sample_id=record.sample_id,
                split=split,
                query=record.query,
                images=record.images,
                response_candidate=record.response_candidate,
                policy_text=record.policy_text,
                policy_text_supplied=record.policy_text_supplied,
                policy_condition=record.policy_condition,
                final_decision=record.final_decision,
                policy_id=record.policy_id,
                fine_policy_id=fine_policy_id,
                gold_response_policy=gold_response_policy,
                policy_action=auxiliary["policy_action"],
                policy_action_mask=auxiliary["policy_action_mask"],
                benign=auxiliary["benign"],
                benign_mask=auxiliary["benign_mask"],
                intent=auxiliary["intent"],
                intent_mask=auxiliary["intent_mask"],
                role=record.role,
                source_role=record.role,
                target_stage=record.target_stage,
                changed_factor=record.changed_factor,
                modality_scope=record.modality_scope,
                source_dataset=record.source_dataset,
                group_id=group_id,
                stable_non_target_mask=tuple(
                    int(bool(record.mutable_non_target_mask.get(stage, 1)))
                    for stage in STAGES
                ),
                stage=supervision,
                evidence_by_stage=evidence,
                label_provenance={**provenance, "auxiliary": auxiliary_provenance},
            )
        )
    return examples, issues


def build_release_examples(
    config: dict[str, Any],
    *,
    release_name: str = "combined_manual_gold_v1",
) -> tuple[dict[str, list[AlgorithmExample]], dict[str, Any]]:
    releases = config.get("releases")
    if not isinstance(releases, dict) or release_name not in releases:
        raise ValueError(f"release not found: {release_name}")
    release = releases[release_name]
    if not isinstance(release, dict):
        raise ValueError(f"invalid release spec: {release_name}")
    group_registry = release.get("group_registry")
    if not group_registry:
        raise ValueError(f"{release_name} requires group_registry")
    group_by_id, registry_split_by_id = load_group_registry(str(group_registry))

    overlay_rows: list[dict[str, Any]] = []
    overlay_path = release.get("stage_state_overlay")
    if overlay_path:
        overlay_rows = read_jsonl(str(overlay_path))

    by_split: dict[str, list[AlgorithmExample]] = {}
    all_issues: list[str] = []
    for split in ("train", "dev", "test"):
        split_spec = release.get("splits", {}).get(split)
        if not isinstance(split_spec, dict):
            raise ValueError(f"{release_name}/{split}: missing split spec")
        input_rows = read_jsonl(split_spec["input"])
        audit_rows = read_jsonl(split_spec["audit"]) if split_spec.get("audit") else []
        examples, issues = build_algorithm_examples(
            input_rows,
            split=split,
            audit_rows=audit_rows,
            overlay_rows=overlay_rows,
            group_by_id=group_by_id,
            registry_split_by_id=registry_split_by_id,
        )
        by_split[split] = examples
        all_issues.extend(f"{split}:{issue}" for issue in issues)

    report = build_label_bridge_report(by_split, issues=all_issues)
    return by_split, report


def build_label_bridge_report(
    by_split: dict[str, list[AlgorithmExample]],
    *,
    issues: list[str] | None = None,
) -> dict[str, Any]:
    split_reports: dict[str, Any] = {}
    for split, examples in by_split.items():
        coverage = {
            stage: {
                key: sum(getattr(example.stage, f"{key}_mask")[idx] for example in examples)
                for key in ("R", "E", "F")
            }
            for idx, stage in enumerate(STAGES)
        }
        source_counts = Counter(
            source
            for example in examples
            for source in example.stage.source
            if source != "unknown"
        )
        contradictions = _label_contradictions(examples)
        split_reports[split] = {
            "count": len(examples),
            "modality_scope": dict(Counter(e.modality_scope for e in examples)),
            "decision": dict(Counter(e.final_decision for e in examples)),
            "role": dict(Counter(e.role or "unknown" for e in examples)),
            "source_role": dict(Counter(e.source_role or "unknown" for e in examples)),
            "target_stage": dict(Counter(e.target_stage for e in examples)),
            "changed_factor": dict(Counter(e.changed_factor for e in examples)),
            "label_source": dict(source_counts),
            "stage_label_coverage": coverage,
            "missing_group_id": sum(1 for e in examples if not e.group_id),
            "contradiction_count": len(contradictions),
            "contradiction_examples": contradictions[:50],
        }
    issue_list = issues or []
    return {
        "format": "causal_diagsafe_label_bridge_report_v1",
        "bridge_version": BRIDGE_VERSION,
        "splits": split_reports,
        "issue_count": len(issue_list),
        "issues": issue_list[:200],
    }


def build_causal_pairs(
    examples: Iterable[AlgorithmExample],
    *,
    include_safe_helpful: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[str, list[AlgorithmExample]] = defaultdict(list)
    for example in examples:
        if example.group_id:
            grouped[example.group_id].append(example)

    pairs: list[dict[str, Any]] = []
    role_pair_specs = [("unsafe", "safe"), ("unsafe", "helpful")]
    if include_safe_helpful:
        role_pair_specs.append(("safe", "helpful"))
    group_role_counts: Counter[str] = Counter()
    unpaired: Counter[str] = Counter()

    for group_id, group_rows in sorted(grouped.items()):
        roles = {row.role for row in group_rows}
        group_role_counts["+".join(sorted(role for role in roles if role)) or "unknown"] += 1
        for left_role, right_role in role_pair_specs:
            left = [row for row in group_rows if row.role == left_role]
            right = [row for row in group_rows if row.role == right_role]
            candidates = [
                (a, b)
                for a in left
                for b in right
                if a.target_stage == b.target_stage
                and a.changed_factor == b.changed_factor
                and a.target_stage in STAGES
            ]
            comparable = [
                (a, b)
                for a, b in candidates
                if _pair_rejection_reason(a, b) is None
            ]
            matched = _greedy_match(comparable)
            if candidates and not comparable:
                reasons = {
                    reason
                    for a, b in candidates
                    if (reason := _pair_rejection_reason(a, b)) is not None
                }
                for reason in sorted(reasons):
                    unpaired[f"{left_role}_{right_role}:{reason}"] += 1
            elif not matched and left and right:
                unpaired[f"{left_role}_{right_role}:stage_or_factor_mismatch"] += 1
            for a, b, score in matched:
                target_idx = STAGE_TO_ID[a.target_stage]
                target_direction, target_direction_mask = _target_direction(
                    a,
                    b,
                    target_idx,
                )
                evidence_direction, evidence_direction_mask = _evidence_direction(
                    a,
                    b,
                    target_idx,
                )
                pairs.append(
                    {
                        "pair_id": f"pair_{len(pairs)+1:07d}",
                        "group_id": group_id,
                        "split": a.split,
                        "left_id": a.sample_id,
                        "right_id": b.sample_id,
                        "left_role": a.role,
                        "right_role": b.role,
                        "pair_type": f"{left_role}_{right_role}",
                        "target_stage": a.target_stage,
                        "target_stage_id": target_idx,
                        "changed_factor": a.changed_factor,
                        "decision_relation": (
                            "flip" if a.final_decision != b.final_decision else "invariant"
                        ),
                        "decision_boundary_crossed": int(
                            (a.final_decision == "refuse")
                            != (b.final_decision == "refuse")
                        ),
                        "target_direction": target_direction,
                        "target_direction_mask": target_direction_mask,
                        "evidence_direction": evidence_direction,
                        "evidence_direction_mask": evidence_direction_mask,
                        "stable_non_target_mask": [
                            int(
                                index != target_idx
                                and a.stable_non_target_mask[index]
                                and b.stable_non_target_mask[index]
                            )
                            for index in range(len(STAGES))
                        ],
                        "left_decision": a.final_decision,
                        "right_decision": b.final_decision,
                        "left_modality": a.modality_scope,
                        "right_modality": b.modality_scope,
                        "image_overlap": int(bool(set(a.images) & set(b.images))),
                        "query_similarity": _query_similarity(a.query, b.query),
                        "match_score": score,
                        "provenance": {
                            "method": "strict_group_stage_factor_greedy_v1",
                            "group_registry_required": True,
                        },
                    }
                )

    report = {
        "format": "causal_diagsafe_pair_report_v1",
        "group_count": len(grouped),
        "group_role_composition": dict(group_role_counts),
        "pair_count": len(pairs),
        "pair_type": dict(Counter(pair["pair_type"] for pair in pairs)),
        "target_stage": dict(Counter(pair["target_stage"] for pair in pairs)),
        "changed_factor": dict(Counter(pair["changed_factor"] for pair in pairs)),
        "decision_relation": dict(Counter(pair["decision_relation"] for pair in pairs)),
        "target_direction_supervised_count": sum(
            int(pair["target_direction_mask"]) for pair in pairs
        ),
        "evidence_direction_supervised_count": sum(
            int(pair["evidence_direction_mask"]) for pair in pairs
        ),
        "cross_modal_pair_count": sum(
            1
            for pair in pairs
            if "cross_modal" in {pair["left_modality"], pair["right_modality"]}
        ),
        "image_overlap_pair_count": sum(pair["image_overlap"] for pair in pairs),
        "unpaired_reasons": dict(unpaired),
    }
    return pairs, report


def render_label_bridge_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CausalDiagSafe-VL Label Bridge 报告",
        "",
        f"- bridge_version: {report.get('bridge_version')}",
        f"- issue_count: {report.get('issue_count', 0)}",
        "",
    ]
    for split, summary in report.get("splits", {}).items():
        lines.extend([f"## {split}", "", f"- count: {summary.get('count', 0)}"])
        lines.append(f"- modality_scope: {summary.get('modality_scope', {})}")
        lines.append(f"- decision: {summary.get('decision', {})}")
        lines.append(f"- target_stage: {summary.get('target_stage', {})}")
        lines.append(f"- label_source: {summary.get('label_source', {})}")
        lines.append(f"- missing_group_id: {summary.get('missing_group_id', 0)}")
        lines.append(f"- contradiction_count: {summary.get('contradiction_count', 0)}")
        lines.extend(["", "### Stage coverage", ""])
        for stage, counts in summary.get("stage_label_coverage", {}).items():
            lines.append(f"- {stage}: R={counts['R']}, E={counts['E']}, F={counts['F']}")
        lines.append("")
    return "\n".join(lines) + "\n"


def render_pair_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# CausalDiagSafe-VL Causal Pair 报告",
        "",
        f"- group_count: {report.get('group_count', 0)}",
        f"- pair_count: {report.get('pair_count', 0)}",
        f"- pair_type: {report.get('pair_type', {})}",
        f"- target_stage: {report.get('target_stage', {})}",
        f"- changed_factor: {report.get('changed_factor', {})}",
        f"- decision_relation: {report.get('decision_relation', {})}",
        f"- cross_modal_pair_count: {report.get('cross_modal_pair_count', 0)}",
        f"- image_overlap_pair_count: {report.get('image_overlap_pair_count', 0)}",
        f"- unpaired_reasons: {report.get('unpaired_reasons', {})}",
        "",
        "## Group role composition",
        "",
    ]
    for key, value in sorted(report.get("group_role_composition", {}).items()):
        lines.append(f"- {key}: {value}")
    return "\n".join(lines) + "\n"


def _build_stage_supervision(
    record: DiagRecord,
    audit: dict[str, Any],
    explicit_row: dict[str, Any] | None,
) -> tuple[StageSupervision, dict[str, Any]]:
    r = [0] * len(STAGES)
    r_mask = [0] * len(STAGES)
    e = [EVIDENCE_TO_ID["uncertain"]] * len(STAGES)
    e_mask = [0] * len(STAGES)
    f = [0] * len(STAGES)
    f_mask = [0] * len(STAGES)
    source = ["unknown"] * len(STAGES)

    if explicit_row is not None:
        bundle: StageStateBundle = normalize_record(explicit_row).stage_states.normalized()
        for index, stage in enumerate(STAGES):
            state = bundle.states[stage]
            if state.R is not None:
                r[index] = state.R
                r_mask[index] = 1
            if state.E is not None:
                e[index] = EVIDENCE_TO_ID[state.E]
                e_mask[index] = 1
            if state.F is not None:
                f[index] = state.F
                f_mask[index] = 1
            if r_mask[index] or e_mask[index] or f_mask[index]:
                source[index] = "explicit_manual"

    target_index = STAGE_TO_ID.get(record.target_stage)
    bridge_used = False
    evidence_fields_used: list[str] = []
    if record.modality_scope == "cross_modal" and target_index is not None:
        if not r_mask[target_index]:
            r[target_index] = 1
            r_mask[target_index] = 1
            bridge_used = True
        if not e_mask[target_index]:
            evidence_items, evidence_fields_used = _target_evidence(record.target_stage, audit)
            if evidence_items:
                e[target_index] = EVIDENCE_TO_ID["present"]
                e_mask[target_index] = 1
                bridge_used = True
        if bridge_used:
            source[target_index] = "derived_partial"

    provenance = {
        "bridge_version": BRIDGE_VERSION if bridge_used else None,
        "explicit_stage_source": _row_id(explicit_row or {}),
        "audit_source": _row_id(audit),
        "target_stage": record.target_stage,
        "manual_evidence_fields_used": evidence_fields_used,
        "source_role": record.role,
        "source_role_used_for_F": False,
        "target_F_rule": "never_derived_from_decision_or_role",
    }
    return (
        StageSupervision(
            R=tuple(r),
            R_mask=tuple(r_mask),
            E=tuple(e),
            E_mask=tuple(e_mask),
            F=tuple(f),
            F_mask=tuple(f_mask),
            source=tuple(source),
        ),
        provenance,
    )


def _collect_evidence(
    record: DiagRecord,
    audit: dict[str, Any],
    explicit_row: dict[str, Any] | None,
) -> dict[str, tuple[dict[str, str], ...]]:
    collected: dict[str, list[dict[str, str]]] = {stage: [] for stage in STAGES}
    if explicit_row is not None:
        raw_states = explicit_row.get("stage_states")
        if isinstance(raw_states, dict):
            for stage in STAGES:
                state = raw_states.get(stage)
                if not isinstance(state, dict):
                    continue
                evidence = state.get("evidence")
                if isinstance(evidence, list):
                    for item in evidence:
                        text = str(item.get("text") if isinstance(item, dict) else item).strip()
                        if text:
                            collected[stage].append({"field": "stage_states", "text": text})

    target_items, _ = _target_evidence(record.target_stage, audit)
    for item in target_items:
        collected[record.target_stage].append(item)
    return {stage: tuple(items) for stage, items in collected.items() if items}


def _build_auxiliary_supervision(
    row: dict[str, Any],
    overlay: dict[str, Any],
    audit: dict[str, Any],
) -> tuple[dict[str, int], dict[str, str | None]]:
    policy_value, policy_source = _masked_target_value(
        (overlay, audit, row),
        value_key="policy_action",
        mask_key="policy_action_mask",
    )
    benign_value, benign_source = _masked_target_value(
        (overlay, audit, row),
        value_key="benign",
        mask_key="benign_mask",
    )
    intent_value, intent_source = _masked_target_value(
        (overlay, audit, row),
        value_key="intent",
        mask_key="intent_mask",
    )

    policy_action = _decision_id(policy_value)
    benign = _binary_id(benign_value)
    intent = _intent_id(intent_value)
    values = {
        "policy_action": policy_action if policy_action is not None else 0,
        "policy_action_mask": int(policy_action is not None),
        "benign": benign if benign is not None else 0,
        "benign_mask": int(benign is not None),
        "intent": intent if intent is not None else INTENT_TO_ID["unknown"],
        "intent_mask": int(intent is not None),
    }
    provenance = {
        "policy_action": policy_source if policy_action is not None else None,
        "benign": benign_source if benign is not None else None,
        "intent": intent_source if intent is not None else None,
    }
    return values, provenance


def _masked_target_value(
    rows: Iterable[dict[str, Any]],
    *,
    value_key: str,
    mask_key: str,
) -> tuple[Any, str | None]:
    for candidate in rows:
        if not isinstance(candidate, dict):
            continue
        targets: dict[str, Any] = (
            candidate["targets"]
            if isinstance(candidate.get("targets"), dict)
            else {}
        )
        raw_mask = targets.get(mask_key) if mask_key in targets else candidate.get(mask_key)
        if _binary_id(raw_mask) != 1:
            continue
        value = targets.get(value_key) if value_key in targets else candidate.get(value_key)
        return value, _row_id(candidate) or "inline_explicit"
    return None, None


def _decision_id(value: Any) -> int | None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in DECISION_TO_ID:
            return DECISION_TO_ID[normalized]
    if isinstance(value, (int, float)) and int(value) in range(len(DECISION_TO_ID)):
        return int(value)
    return None


def _intent_id(value: Any) -> int | None:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in INTENT_TO_ID:
            return INTENT_TO_ID[normalized]
    if isinstance(value, (int, float)) and int(value) in range(len(INTENT_TO_ID)):
        return int(value)
    return None


def _binary_id(value: Any) -> int | None:
    if isinstance(value, str):
        value = value.strip().lower()
    if value in (True, 1, 1.0, "1", "true", "yes"):
        return 1
    if value in (False, 0, 0.0, "0", "false", "no"):
        return 0
    return None


def _target_evidence(
    stage: str,
    audit: dict[str, Any],
) -> tuple[list[dict[str, str]], list[str]]:
    field_map = {
        "S1": ("manual_visual_evidence",),
        "S2": ("manual_binding_evidence", "manual_visual_evidence", "manual_text_evidence"),
        "S3": ("manual_text_evidence",),
        "S4": ("manual_text_evidence",),
        "S5": ("gold_response_policy", "manual_text_evidence"),
    }
    items: list[dict[str, str]] = []
    used: list[str] = []
    for field_name in field_map.get(stage, ()):
        text = str(audit.get(field_name) or "").strip()
        if text and text.lower() not in {"none", "n/a", "na", "null", "unknown"}:
            items.append({"field": field_name, "text": text})
            used.append(field_name)
    return items, used


def _find_explicit_stage_row(*rows: dict[str, Any]) -> dict[str, Any] | None:
    for row in rows:
        if not isinstance(row, dict):
            continue
        if isinstance(row.get("stage_states"), dict):
            return row
        if isinstance(row.get("stage_diagnosis"), dict):
            return row
        if isinstance(row.get("failure_or_risk_labels"), dict):
            return row
    return None


def _index_rows(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = _row_id(row)
        if not sample_id:
            continue
        if sample_id in indexed:
            raise ValueError(f"duplicate id: {sample_id}")
        indexed[sample_id] = row
    return indexed


def _row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("sample_id") or row.get("candidate_id") or "")


def _append_consistency_issues(
    record: DiagRecord,
    audit: DiagRecord,
    issues: list[str],
) -> None:
    for field_name in (
        "final_decision",
        "query",
        "images",
        "response_candidate",
        "policy_text",
        "policy_text_supplied",
        "policy_id",
        "policy_condition",
        "target_stage",
        "role",
        "changed_factor",
    ):
        if getattr(record, field_name) != getattr(audit, field_name):
            issues.append(f"{record.sample_id}: input/audit mismatch:{field_name}")


def _label_contradictions(examples: list[AlgorithmExample]) -> list[str]:
    contradictions: list[str] = []
    for example in examples:
        if example.modality_scope == "text_only" and example.target_stage in {"S1", "S2"}:
            contradictions.append(f"{example.sample_id}: text_only target={example.target_stage}")
        if example.modality_scope == "cross_modal" and not example.images:
            contradictions.append(f"{example.sample_id}: cross_modal without image")
        if example.changed_factor == "policy_rule" and not example.policy_text_supplied:
            contradictions.append(
                f"{example.sample_id}: policy_rule intervention without model-visible policy_text"
            )
        if example.changed_factor == "response_execution" and not example.response_candidate:
            contradictions.append(
                f"{example.sample_id}: response_execution intervention without candidate response"
            )
        expected = EXPECTED_FACTOR_BY_STAGE.get(example.target_stage)
        if expected and example.changed_factor not in expected:
            contradictions.append(
                f"{example.sample_id}: {example.target_stage} factor={example.changed_factor}, "
                f"expected one of {sorted(expected)}"
            )
        if example.role == "unsafe" and example.final_decision != "refuse":
            contradictions.append(
                f"{example.sample_id}: unsafe role with decision={example.final_decision}"
            )
        if example.role in {"safe", "helpful"} and example.final_decision == "refuse":
            contradictions.append(f"{example.sample_id}: {example.role} role with refuse")
    return contradictions


def _greedy_match(
    candidates: list[tuple[AlgorithmExample, AlgorithmExample]],
) -> list[tuple[AlgorithmExample, AlgorithmExample, float]]:
    scored = sorted(
        ((a, b, _pair_score(a, b)) for a, b in candidates),
        key=lambda item: (-item[2], item[0].sample_id, item[1].sample_id),
    )
    used_left: set[str] = set()
    used_right: set[str] = set()
    selected: list[tuple[AlgorithmExample, AlgorithmExample, float]] = []
    for left, right, score in scored:
        if left.sample_id in used_left or right.sample_id in used_right:
            continue
        used_left.add(left.sample_id)
        used_right.add(right.sample_id)
        selected.append((left, right, score))
    return selected


def _pair_score(a: AlgorithmExample, b: AlgorithmExample) -> float:
    score = 0.0
    score += 2.0 if a.modality_scope == b.modality_scope else 0.0
    score += 1.0 if a.source_dataset == b.source_dataset else 0.0
    score += 2.0 if set(a.images) & set(b.images) else 0.0
    if a.target_stage != "S4":
        score += 1.0 if a.policy_id == b.policy_id else 0.0
    score += 3.0 * _query_similarity(a.query, b.query)
    return round(score, 6)


def _causal_comparable(a: AlgorithmExample, b: AlgorithmExample) -> bool:
    image_overlap = bool(set(a.images) & set(b.images))
    query_similarity = _query_similarity(a.query, b.query)
    return image_overlap or query_similarity >= 0.72


def _pair_rejection_reason(
    left: AlgorithmExample,
    right: AlgorithmExample,
) -> str | None:
    if not _role_decision_consistent(left) or not _role_decision_consistent(right):
        return "role_decision_conflict"
    if not _causal_comparable(left, right):
        return "no_shared_or_similar_modality"
    if not _intervention_is_model_visible(left, right):
        return "intervention_not_model_visible"
    return None


def _query_similarity(left: str, right: str) -> float:
    return round(SequenceMatcher(None, _normalize_text(left), _normalize_text(right)).ratio(), 6)


def _normalize_text(value: str) -> str:
    return " ".join(str(value).lower().split())


def _target_direction(
    left: AlgorithmExample,
    right: AlgorithmExample,
    target_index: int,
) -> tuple[int, int]:
    if not left.stage.F_mask[target_index] or not right.stage.F_mask[target_index]:
        return 0, 0
    delta = left.stage.F[target_index] - right.stage.F[target_index]
    if delta == 0:
        return 0, 0
    return (1 if delta > 0 else -1), 1


def _evidence_direction(
    left: AlgorithmExample,
    right: AlgorithmExample,
    target_index: int,
) -> tuple[int, int]:
    if not left.stage.E_mask[target_index] or not right.stage.E_mask[target_index]:
        return 0, 0
    present = EVIDENCE_TO_ID["present"]
    delta = int(left.stage.E[target_index] == present) - int(
        right.stage.E[target_index] == present
    )
    if delta == 0:
        return 0, 0
    return (1 if delta > 0 else -1), 1


def _intervention_is_model_visible(
    left: AlgorithmExample,
    right: AlgorithmExample,
) -> bool:
    query_equal = _normalize_text(left.query) == _normalize_text(right.query)
    images_equal = left.images == right.images
    response_equal = left.response_candidate == right.response_candidate
    policy_equal = (
        left.policy_text_supplied == right.policy_text_supplied
        and left.policy_text == right.policy_text
    )
    factor = left.changed_factor
    if factor in {"visual_evidence", "ocr_text"}:
        return query_equal and not images_equal and response_equal and policy_equal
    if factor == "image_text_binding":
        return (query_equal != images_equal) and response_equal and policy_equal
    if factor == "intent":
        return not query_equal and images_equal and response_equal and policy_equal
    if factor == "policy_rule":
        return query_equal and images_equal and response_equal and not policy_equal
    if factor == "response_execution":
        return query_equal and images_equal and not response_equal and policy_equal
    return False


def _role_decision_consistent(example: AlgorithmExample) -> bool:
    if example.role == "unsafe":
        return example.final_decision == "refuse"
    if example.role in {"safe", "helpful"}:
        return example.final_decision in {"allow", "warn"}
    return False
