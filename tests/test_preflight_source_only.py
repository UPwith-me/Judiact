from __future__ import annotations

from causal_diagsafe.preflight import preflight_rows
from causal_diagsafe.source_only import run_metadata_baseline
from causal_diagsafe.gold_router import train_gold_router


def test_preflight_detects_source_modality_confound() -> None:
    rows = [
        {"id": f"a{i}", "final_decision": "refuse", "source_dataset": "SPA-VL", "modality_scope": "cross_modal"}
        for i in range(25)
    ]
    summary = preflight_rows(rows)
    assert summary["source_confound_risks"]


def test_source_only_baseline_runs() -> None:
    train = [
        {"id": "a", "final_decision": "refuse", "source_dataset": "x", "modality_scope": "text_only"},
        {"id": "b", "final_decision": "allow", "source_dataset": "y", "modality_scope": "cross_modal"},
    ]
    summary = run_metadata_baseline(train, train, fields=("source_dataset",))
    assert summary["decision_metrics"]["accuracy"] == 1.0


def test_gold_router_runs_on_gold_states() -> None:
    rows = [
        {
            "id": "u",
            "final_decision": "refuse",
            "target_stage": "S4",
            "stage_states": {"S4": {"R": 1, "E": "present", "F": 1}},
        },
        {
            "id": "s",
            "final_decision": "allow",
            "target_stage": "S4",
            "stage_states": {"S4": {"R": 1, "E": "present", "F": 0}},
        },
    ]
    _, summary = train_gold_router(rows, rows, epochs=2, lr=0.01, batch_size=2, device="cpu")
    assert summary["decision_metrics"]["count"] == 2
