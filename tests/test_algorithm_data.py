from __future__ import annotations

from causal_diagsafe.algorithm_data import build_algorithm_examples, build_causal_pairs
from causal_diagsafe.constants import EVIDENCE_TO_ID, STAGE_TO_ID


def _nested_row(
    sample_id: str,
    *,
    role: str,
    decision: str,
    modality: str = "cross_modal",
) -> dict:
    images = ["/images/a.jpg"] if modality == "cross_modal" else []
    return {
        "id": sample_id,
        "modality_scope": modality,
        "model_input": {"query": f"query {sample_id}", "images": images},
        "targets": {
            "final_decision": decision,
            "policy_id": "P_TEST",
            "role": role,
            "target_stage": "S2" if modality == "cross_modal" else "S3",
            "changed_factor": "image_text_binding" if modality == "cross_modal" else "intent",
        },
        "metadata": {"source_dataset": "manual"},
    }


def test_crossmodal_partial_bridge_only_supervises_target_stage() -> None:
    row = _nested_row("cm-1", role="unsafe", decision="refuse")
    audit = {
        "id": "cm-1",
        "query": "query cm-1",
        "images": ["/images/a.jpg"],
        "final_decision": "refuse",
        "policy_id": "P_TEST",
        "role": "unsafe",
        "target_stage": "S2",
        "changed_factor": "image_text_binding",
        "manual_binding_evidence": "The text meaning depends on the visible context.",
    }
    examples, issues = build_algorithm_examples(
        [row], split="train", audit_rows=[audit], group_by_id={"cm-1": "g1"}
    )
    assert not issues
    example = examples[0]
    target = STAGE_TO_ID["S2"]
    assert example.stage.R[target] == 1
    assert example.stage.R_mask == (0, 1, 0, 0, 0)
    assert example.stage.E[target] == EVIDENCE_TO_ID["present"]
    assert example.stage.E_mask == (0, 1, 0, 0, 0)
    assert example.stage.F[target] == 0
    assert example.stage.F_mask == (0, 0, 0, 0, 0)
    assert example.stage.source[target] == "derived_partial"
    assert example.label_provenance["target_F_rule"] == "never_derived_from_decision_or_role"
    assert example.policy_action_mask == 0


def test_textonly_overlay_preserves_explicit_stage_states() -> None:
    row = _nested_row("tx-1", role="safe", decision="warn", modality="text_only")
    overlay = dict(row)
    overlay["stage_states"] = {
        "S1": {"R": 0, "E": "absent", "F": 0},
        "S2": {"R": 0, "E": "absent", "F": 0},
        "S3": {"R": 1, "E": "present", "F": 1},
        "S4": {"R": 1, "E": "present", "F": 1},
        "S5": {"R": 1, "E": "present", "F": 0},
    }
    examples, issues = build_algorithm_examples(
        [row], split="train", overlay_rows=[overlay], group_by_id={"tx-1": "g1"}
    )
    assert not issues
    example = examples[0]
    assert example.stage.R_mask == (1, 1, 1, 1, 1)
    assert example.stage.E_mask == (1, 1, 1, 1, 1)
    assert example.stage.F_mask == (1, 1, 1, 1, 1)
    assert set(example.stage.source) == {"explicit_manual"}


def test_auxiliary_targets_require_explicit_masks() -> None:
    row = _nested_row("tx-aux", role="safe", decision="warn", modality="text_only")
    row["targets"].update(
        {
            "policy_action": "warn",
            "policy_action_mask": 1,
            "intent": "benign_educational",
            "intent_mask": 1,
            "benign": 1,
            "benign_mask": 1,
        }
    )
    examples, issues = build_algorithm_examples(
        [row],
        split="train",
        group_by_id={"tx-aux": "g1"},
    )
    assert not issues
    exported = examples[0].to_dict()["targets"]
    assert exported["policy_action_mask"] == 1
    assert exported["intent_mask"] == 1
    assert exported["benign_mask"] == 1


def test_pair_builder_requires_same_group_stage_and_factor() -> None:
    rows = [
        _nested_row("unsafe", role="unsafe", decision="refuse"),
        _nested_row("safe", role="safe", decision="warn"),
        _nested_row("helpful", role="helpful", decision="allow"),
    ]
    audits = []
    for row in rows:
        record = row["targets"]
        audits.append(
            {
                "id": row["id"],
                "query": row["model_input"]["query"],
                "images": row["model_input"]["images"],
                "final_decision": record["final_decision"],
                "policy_id": record["policy_id"],
                "role": record["role"],
                "target_stage": record["target_stage"],
                "changed_factor": record["changed_factor"],
                "manual_binding_evidence": "binding evidence",
            }
        )
    examples, issues = build_algorithm_examples(
        rows,
        split="train",
        audit_rows=audits,
        group_by_id={row["id"]: "g1" for row in rows},
    )
    assert not issues
    pairs, report = build_causal_pairs(examples)
    assert report["pair_count"] == 3
    assert {pair["pair_type"] for pair in pairs} == {
        "unsafe_safe",
        "unsafe_helpful",
        "safe_helpful",
    }
    assert all(pair["target_stage"] == "S2" for pair in pairs)
    assert all(pair["target_direction_mask"] == 0 for pair in pairs)


def test_source_role_is_preserved_and_does_not_create_target_risk() -> None:
    row = _nested_row("cm-refuse", role="safe", decision="refuse")
    audit = {
        "id": "cm-refuse",
        "query": "query cm-refuse",
        "images": ["/images/a.jpg"],
        "final_decision": "refuse",
        "policy_id": "P_TEST",
        "role": "safe",
        "target_stage": "S2",
        "changed_factor": "image_text_binding",
        "manual_binding_evidence": "binding evidence",
    }
    examples, issues = build_algorithm_examples(
        [row],
        split="train",
        audit_rows=[audit],
        group_by_id={"cm-refuse": "g1"},
    )
    assert not issues
    example = examples[0]
    assert example.source_role == "safe"
    assert example.role == "safe"
    target = STAGE_TO_ID["S2"]
    assert example.stage.F[target] == 0
    assert example.stage.F_mask[target] == 0
    assert example.label_provenance["source_role_used_for_F"] is False


def test_pair_direction_requires_explicit_target_f_labels() -> None:
    left = _nested_row("left", role="unsafe", decision="refuse")
    right = _nested_row("right", role="safe", decision="warn")
    for row, target_f in ((left, 1), (right, 0)):
        row["model_input"]["query"] = "same query"
        row["stage_states"] = {
            stage: {
                "R": int(stage == "S2"),
                "E": "present" if stage == "S2" else "absent",
                "F": target_f if stage == "S2" else 0,
            }
            for stage in ("S1", "S2", "S3", "S4", "S5")
        }
    right["model_input"]["images"] = ["/images/b.jpg"]
    examples, issues = build_algorithm_examples(
        [left, right],
        split="train",
        group_by_id={"left": "g1", "right": "g1"},
    )
    assert not issues
    pairs, _ = build_causal_pairs(examples, include_safe_helpful=False)
    assert len(pairs) == 1
    assert pairs[0]["target_direction"] == 1
    assert pairs[0]["target_direction_mask"] == 1


def test_pair_builder_rejects_group_members_without_shared_causal_context() -> None:
    left = _nested_row("left", role="unsafe", decision="refuse", modality="text_only")
    right = _nested_row("right", role="safe", decision="warn", modality="text_only")
    left["model_input"]["query"] = "one completely unrelated request"
    right["model_input"]["query"] = "a different topic with no shared content"
    examples, issues = build_algorithm_examples(
        [left, right],
        split="train",
        group_by_id={"left": "g1", "right": "g1"},
    )
    assert not issues
    pairs, report = build_causal_pairs(examples)
    assert pairs == []
    assert report["unpaired_reasons"]["unsafe_safe:no_shared_or_similar_modality"] == 1
