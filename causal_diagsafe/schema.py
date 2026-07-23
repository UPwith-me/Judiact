from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .constants import (
    CHANGED_FACTORS,
    DECISIONS,
    EVIDENCE_STATES,
    GROUP_ROLES,
    POLICY_CONDITIONS,
    STAGES,
)


@dataclass(frozen=True)
class StageState:
    """R/E/F stage-state label or prediction."""

    R: int | None = None
    E: str | None = None
    F: int | None = None
    evidence: tuple[dict[str, Any], ...] = ()

    def normalized(self) -> "StageState":
        r = _norm_binary(self.R)
        f = _norm_binary(self.F)
        e = None if self.E is None else str(self.E).lower()
        if e is not None and e not in EVIDENCE_STATES:
            e = "uncertain"
        evidence = tuple(item for item in self.evidence if isinstance(item, dict))
        return StageState(R=r, E=e, F=f, evidence=evidence)

    def to_dict(self) -> dict[str, Any]:
        norm = self.normalized()
        return {"R": norm.R, "E": norm.E, "F": norm.F, "evidence": list(norm.evidence)}


@dataclass(frozen=True)
class StageStateBundle:
    states: dict[str, StageState] = field(default_factory=dict)

    def normalized(self) -> "StageStateBundle":
        return StageStateBundle(
            states={stage: self.states.get(stage, StageState()).normalized() for stage in STAGES}
        )

    def to_dict(self) -> dict[str, Any]:
        return {stage: state.to_dict() for stage, state in self.normalized().states.items()}

    def f_vector(self) -> tuple[int | None, ...]:
        norm = self.normalized()
        return tuple(norm.states[stage].F for stage in STAGES)

    def e_vector(self) -> tuple[str | None, ...]:
        norm = self.normalized()
        return tuple(norm.states[stage].E for stage in STAGES)


@dataclass(frozen=True)
class DiagRecord:
    sample_id: str
    final_decision: str
    query: str = ""
    images: tuple[str, ...] = ()
    response_candidate: str = ""
    policy_text: str = ""
    policy_text_supplied: bool = False
    source_dataset: str = "unknown"
    modality_scope: str = "unknown"
    harm_category: str = "unknown"
    attack_mechanism: str = "unknown"
    target_stage: str = "unknown"
    group_id: str = ""
    role: str = ""
    changed_factor: str = "unknown"
    policy_id: str = "unknown"
    policy_condition: str = "unknown"
    stage_states: StageStateBundle = field(default_factory=StageStateBundle)
    mutable_non_target_mask: dict[str, int] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.sample_id,
            "final_decision": self.final_decision,
            "query": self.query,
            "images": list(self.images),
            "response_candidate": self.response_candidate,
            "policy_text": self.policy_text,
            "policy_text_supplied": self.policy_text_supplied,
            "source_dataset": self.source_dataset,
            "modality_scope": self.modality_scope,
            "harm_category": self.harm_category,
            "attack_mechanism": self.attack_mechanism,
            "target_stage": self.target_stage,
            "group_id": self.group_id,
            "role": self.role,
            "changed_factor": self.changed_factor,
            "policy_id": self.policy_id,
            "policy_condition": self.policy_condition,
            "stage_states": self.stage_states.to_dict(),
            "mutable_non_target_mask": dict(self.mutable_non_target_mask),
        }


def normalize_record(row: dict[str, Any]) -> DiagRecord:
    """Normalize historical and new DiagSafe rows into the new algorithm schema."""

    model_input = _as_dict(row.get("model_input"))
    targets = _as_dict(row.get("targets"))
    metadata = _as_dict(row.get("metadata"))
    stimulus = _as_dict(row.get("stimulus"))
    meta = _as_dict(row.get("meta"))
    pre = _as_dict(row.get("pre_annotation"))
    controls = _as_dict(row.get("controls"))
    group = _as_dict(row.get("group"))
    policy_condition = (
        row.get("policy_condition")
        or targets.get("policy_condition")
        or metadata.get("policy_condition")
        or group.get("policy_condition")
    )
    if isinstance(policy_condition, dict):
        policy_condition = policy_condition.get("deck_type") or policy_condition.get("condition")

    final_decision = _norm_decision(
        row.get("expected_final_decision")
        or targets.get("final_decision")
        or row.get("final_decision")
        or pre.get("gold_decision")
        or row.get("gold_decision")
    )
    images = _norm_images(
        model_input.get("images")
        if "images" in model_input
        else row.get("images", stimulus.get("images", []))
    )
    sample_id = str(row.get("sample_id") or row.get("id") or row.get("uid") or "")
    policy_text_supplied = (
        "policy_text" in model_input
        or "policy_deck" in model_input
        or bool(row.get("policy_text_supplied"))
        or "policy_text" in row
        or "policy_deck" in row
    )
    target_stage = _norm_stage(
        targets.get("target_stage")
        or row.get("target_stage")
        or meta.get("target_stage")
        or group.get("target_stage")
    )
    mutable_mask = _norm_mutable_mask(
        targets.get("stable_non_target_mask")
        or row.get("stable_non_target_mask")
        or group.get("stable_non_target_mask")
        or row.get("mutable_non_target_mask")
        or group.get("mutable_non_target_mask")
        or row.get("non_target_stages")
    )
    return DiagRecord(
        sample_id=sample_id,
        final_decision=final_decision,
        query=str(
            model_input.get("query")
            or row.get("query")
            or stimulus.get("text_query")
            or row.get("text")
            or ""
        ),
        images=images,
        response_candidate=str(
            model_input.get("response_candidate")
            or row.get("response_candidate")
            or row.get("candidate_response")
            or ""
        ),
        policy_text=_normalize_policy_text(
            model_input.get("policy_text")
            if "policy_text" in model_input
            else (
                model_input.get("policy_deck")
                if "policy_deck" in model_input
                else (
                    row.get("policy_text")
                    if "policy_text" in row
                    else row.get("policy_deck")
                )
            )
        ),
        policy_text_supplied=policy_text_supplied,
        source_dataset=str(
            metadata.get("source_dataset")
            or row.get("source_dataset")
            or meta.get("source_dataset")
            or "unknown"
        ),
        modality_scope=str(
            row.get("modality_scope")
            or metadata.get("modality_scope")
            or meta.get("modality_scope")
            or _infer_modality(images)
        ),
        harm_category=str(
            row.get("harm_category")
            or metadata.get("harm_category")
            or meta.get("harm_category")
            or "unknown"
        ),
        attack_mechanism=str(
            row.get("attack_mechanism")
            or metadata.get("attack_mechanism")
            or meta.get("attack_mechanism")
            or "unknown"
        ),
        target_stage=target_stage,
        group_id=str(
            row.get("group_id")
            or metadata.get("group_id")
            or group.get("group_id")
            or ""
        ),
        role=_norm_role(targets.get("role") or row.get("role") or group.get("role")),
        changed_factor=_norm_changed_factor(
            targets.get("changed_factor")
            or row.get("changed_factor")
            or group.get("changed_factor")
        ),
        policy_id=str(
            row.get("expected_policy_id")
            or targets.get("policy_id")
            or row.get("policy_id")
            or pre.get("policy_id")
            or "unknown"
        ),
        policy_condition=_norm_policy_condition(policy_condition),
        stage_states=_extract_stage_states(row),
        mutable_non_target_mask=mutable_mask,
        raw=row,
    )


def validate_record(row: dict[str, Any]) -> list[str]:
    record = normalize_record(row)
    errors: list[str] = []
    if not record.sample_id:
        errors.append("id:missing")
    if record.final_decision not in DECISIONS:
        errors.append(f"final_decision:invalid:{record.final_decision}")
    if record.target_stage != "unknown" and record.target_stage not in STAGES:
        errors.append(f"target_stage:invalid:{record.target_stage}")
    if record.role and record.role not in GROUP_ROLES:
        errors.append(f"role:invalid:{record.role}")
    if record.changed_factor not in CHANGED_FACTORS:
        errors.append(f"changed_factor:invalid:{record.changed_factor}")
    if record.policy_condition not in POLICY_CONDITIONS:
        errors.append(f"policy_condition:invalid:{record.policy_condition}")
    states = record.stage_states.normalized().states
    for stage, state in states.items():
        if state.R not in (0, 1, None):
            errors.append(f"{stage}.R:invalid")
        if state.E not in (*EVIDENCE_STATES, None):
            errors.append(f"{stage}.E:invalid")
        if state.F not in (0, 1, None):
            errors.append(f"{stage}.F:invalid")
    return errors


def _extract_stage_states(row: dict[str, Any]) -> StageStateBundle:
    for key in ("stage_states", "stage_diagnosis", "stage_state"):
        value = row.get(key)
        if isinstance(value, dict):
            return StageStateBundle(
                {
                    stage: _stage_state_from_any(value.get(stage))
                    for stage in STAGES
                }
            ).normalized()
    if isinstance(row.get("failure_or_risk_labels"), dict):
        risk = row["failure_or_risk_labels"]
        req = _as_dict(row.get("requirement_labels"))
        ev = _as_dict(row.get("evidence_labels"))
        return StageStateBundle(
            {
                stage: StageState(
                    R=_norm_binary(req.get(stage)),
                    E=_norm_evidence(ev.get(stage)),
                    F=_norm_binary(risk.get(stage)),
                )
                for stage in STAGES
            }
        ).normalized()
    # Legacy generated trace compatibility. This is only for audit, not causal supervision.
    legacy = row.get("stage_labels")
    if isinstance(legacy, dict):
        return StageStateBundle(
            {
                stage: StageState(R=None, E="uncertain", F=_legacy_stage_to_f(legacy.get(stage)))
                for stage in STAGES
            }
        ).normalized()
    return StageStateBundle().normalized()


def _stage_state_from_any(value: Any) -> StageState:
    if isinstance(value, dict):
        return StageState(
            R=_norm_binary(value.get("R", value.get("requirement"))),
            E=_norm_evidence(value.get("E", value.get("evidence"))),
            F=_norm_binary(value.get("F", value.get("risk", value.get("failure")))),
            evidence=tuple(value.get("evidence", []) if isinstance(value.get("evidence"), list) else ()),
        )
    if isinstance(value, str):
        return StageState(R=None, E="uncertain", F=_legacy_stage_to_f(value))
    return StageState()


def _norm_decision(value: Any) -> str:
    decision = str(value or "review").strip().lower()
    aliases = {
        "safe": "allow",
        "unsafe": "refuse",
        "refusal": "refuse",
        "reject": "refuse",
        "deny": "refuse",
        "caution": "warn",
        "warning": "warn",
        "needs_review": "review",
    }
    decision = aliases.get(decision, decision)
    return decision if decision in DECISIONS else "review"


def _norm_stage(value: Any) -> str:
    stage = str(value or "unknown").strip().upper()
    return stage if stage in STAGES else "unknown"


def _norm_role(value: Any) -> str:
    role = str(value or "").strip().lower()
    return role if role in GROUP_ROLES else ""


def _norm_changed_factor(value: Any) -> str:
    factor = str(value or "unknown").strip().lower()
    aliases = {
        "visual_perception": "visual_evidence",
        "visual_grounding": "visual_evidence",
        "binding": "image_text_binding",
    }
    factor = aliases.get(factor, factor)
    return factor if factor in CHANGED_FACTORS else "unknown"


def _norm_policy_condition(value: Any) -> str:
    condition = str(value or "unknown").strip().lower()
    return condition if condition in POLICY_CONDITIONS else "unknown"


def _normalize_policy_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    if isinstance(value, dict):
        return "\n".join(
            f"{key}: {item}"
            for key, item in value.items()
            if str(item).strip()
        )
    return str(value).strip()


def _norm_images(value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value else ()
    if isinstance(value, list):
        return tuple(str(item) for item in value if str(item or ""))
    return ()


def _infer_modality(images: tuple[str, ...]) -> str:
    return "cross_modal" if images else "text_only"


def _norm_binary(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if value in (0, 1):
            return int(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "present", "fail", "risk", "activated"}:
        return 1
    if text in {"0", "false", "no", "absent", "pass", "safe", "inactive"}:
        return 0
    return None


def _norm_evidence(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return "present" if value else "absent"
    text = str(value).strip().lower()
    aliases = {"yes": "present", "no": "absent", "unknown": "uncertain"}
    text = aliases.get(text, text)
    return text if text in EVIDENCE_STATES else "uncertain"


def _legacy_stage_to_f(value: Any) -> int | None:
    text = str(value or "").strip().lower()
    if text == "fail":
        return 1
    if text == "pass":
        return 0
    return None


def _norm_mutable_mask(value: Any) -> dict[str, int]:
    if isinstance(value, dict):
        return {stage: int(bool(value.get(stage, 0))) for stage in STAGES}
    if isinstance(value, list):
        stable = {str(item).upper() for item in value}
        return {stage: int(stage in stable) for stage in STAGES}
    return {stage: 1 for stage in STAGES}


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}
