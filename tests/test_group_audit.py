from __future__ import annotations

from causal_diagsafe.group_audit import audit_groups


def test_group_audit_scores_boundary_and_target_flip() -> None:
    gold = [
        _row("g1_u", "g1", "unsafe", "refuse", "S2"),
        _row("g1_s", "g1", "safe", "allow", "S2"),
        _row("g1_h", "g1", "helpful", "allow", "S2"),
    ]
    pred = [
        _pred("g1_u", "refuse", {"S2": 1, "S1": 0, "S3": 0, "S4": 0, "S5": 0}),
        _pred("g1_s", "allow", {"S2": 0, "S1": 0, "S3": 0, "S4": 0, "S5": 0}),
        _pred("g1_h", "warn", {"S2": 0, "S1": 0, "S3": 0, "S4": 0, "S5": 0}),
    ]
    summary = audit_groups(gold, pred)
    assert summary["boundary_success_rate"] == 1.0
    assert summary["target_stage_flip_rate"] == 1.0
    assert summary["non_target_invariance_rate"] == 1.0


def test_group_audit_flags_policy_leakage() -> None:
    gold = [
        _row("g1_u", "g1", "unsafe", "refuse", "S4", policy_condition="wrong"),
        _row("g1_s", "g1", "safe", "allow", "S4"),
        _row("g1_h", "g1", "helpful", "allow", "S4"),
    ]
    pred = [
        _pred("g1_u", "refuse", {"S4": 1}, policy_id="gold_policy"),
        _pred("g1_s", "allow", {"S4": 0}),
        _pred("g1_h", "allow", {"S4": 0}),
    ]
    summary = audit_groups(gold, pred)
    assert summary["policy_leakage_rate"] == 1.0


def _row(
    sample_id: str,
    group_id: str,
    role: str,
    decision: str,
    target_stage: str,
    policy_condition: str = "correct",
) -> dict:
    return {
        "id": sample_id,
        "group_id": group_id,
        "role": role,
        "final_decision": decision,
        "target_stage": target_stage,
        "policy_condition": policy_condition,
        "mutable_non_target_mask": {"S1": 1, "S2": 1, "S3": 1, "S4": 1, "S5": 1},
    }


def _pred(sample_id: str, decision: str, f_by_stage: dict[str, int], policy_id: str = "unknown") -> dict:
    return {
        "id": sample_id,
        "final_decision": decision,
        "policy_id": policy_id,
        "stage_diagnosis": {
            stage: {"R": 1, "E": "present", "F": f_by_stage.get(stage, 0)}
            for stage in ("S1", "S2", "S3", "S4", "S5")
        },
    }

