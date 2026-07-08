from __future__ import annotations

from causal_diagsafe.metrics import decision_metrics, stage_collapse_metrics, stage_state_metrics
from causal_diagsafe.parser import parse_prediction
from causal_diagsafe.schema import normalize_record


def test_parse_new_stage_diagnosis() -> None:
    pred = parse_prediction(
        {
            "id": "x1",
            "final_decision": "refuse",
            "policy_id": "p1",
            "stage_diagnosis": {
                "S1": {"R": 1, "E": "present", "F": 1},
                "S2": {"R": 1, "E": "present", "F": 0},
                "S3": {"R": 1, "E": "present", "F": 1},
                "S4": {"R": 1, "E": "present", "F": 1},
                "S5": {"R": 0, "E": "absent", "F": 0},
            },
        }
    )
    assert pred.valid_json
    assert pred.final_decision == "refuse"
    assert pred.stage_states.states["S1"].F == 1
    assert not pred.legacy_stage_labels


def test_parse_legacy_stage_labels_marked_legacy() -> None:
    pred = parse_prediction(
        '{"final_decision":"allow","stage_labels":{"S1":"pass","S2":"pass","S3":"pass","S4":"pass","S5":"pass"}}'
    )
    assert pred.final_decision == "allow"
    assert pred.legacy_stage_labels
    assert pred.stage_states.f_vector() == (0, 0, 0, 0, 0)


def test_decision_and_stage_metrics() -> None:
    records = [
        normalize_record({"id": "a", "final_decision": "refuse"}),
        normalize_record({"id": "b", "final_decision": "allow"}),
    ]
    preds = [
        parse_prediction({"id": "a", "final_decision": "allow"}),
        parse_prediction({"id": "b", "final_decision": "refuse"}),
    ]
    metrics = decision_metrics(records, preds)
    assert metrics["harmful_allow_rate"] == 1.0
    assert metrics["over_refusal_rate"] == 1.0


def test_stage_collapse_metrics_all_clear() -> None:
    preds = [
        parse_prediction(
            {
                "id": "x",
                "final_decision": "allow",
                "stage_diagnosis": {stage: {"F": 0, "E": "absent"} for stage in ("S1", "S2", "S3", "S4", "S5")},
            }
        )
    ]
    metrics = stage_collapse_metrics(preds)
    assert metrics["all_clear_rate"] == 1.0


def test_stage_state_metrics_binary_and_evidence() -> None:
    gold = [
        normalize_record(
            {
                "id": "x",
                "final_decision": "refuse",
                "stage_states": {"S1": {"R": 1, "E": "present", "F": 1}},
            }
        )
    ]
    pred = [
        parse_prediction(
            {
                "id": "x",
                "final_decision": "refuse",
                "stage_diagnosis": {"S1": {"R": 1, "E": "present", "F": 1}},
            }
        )
    ]
    metrics = stage_state_metrics(gold, pred)
    assert metrics["S1.F"]["accuracy"] == 1.0
    assert metrics["S1.E"]["accuracy"] == 1.0

