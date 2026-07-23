from __future__ import annotations

import json

import torch

from causal_diagsafe.training.routed_opd import (
    _balanced_route_batches,
    _completion_prediction_mask,
    _rejected_completion,
)


def _row(sample_id: str, role: str, decision: str) -> dict:
    return {
        "id": sample_id,
        "training_control": {"role": role},
        "training_target": {"final_decision": decision, "policy_id": "P_VIOLENCE"},
    }


def test_balanced_batches_alternate_teacher_routes() -> None:
    roles = {
        "unsafe": [_row("u1", "unsafe", "refuse"), _row("u2", "unsafe", "refuse")],
        "safe": [_row("s1", "safe", "warn"), _row("s2", "safe", "warn")],
        "helpful": [_row("h1", "helpful", "allow"), _row("h2", "helpful", "allow")],
    }
    batches = _balanced_route_batches(roles, batch_size=1, steps=6, seed=3)
    assert [route for route, _ in batches] == [
        "safety",
        "helpful",
        "helpful",
        "safety",
        "helpful",
        "helpful",
    ]


def test_completion_prediction_mask_is_shifted_for_causal_logits() -> None:
    labels = torch.tensor([[-100, -100, 7, 8, -100]])
    mask = _completion_prediction_mask(labels)
    assert mask.tolist() == [[0.0, 1.0, 1.0, 0.0, 0.0]]


def test_rejected_completion_flips_decision_boundary() -> None:
    unsafe = json.loads(_rejected_completion(_row("u", "unsafe", "refuse")))
    helpful = json.loads(_rejected_completion(_row("h", "helpful", "allow")))
    assert unsafe["final_decision"] == "allow"
    assert helpful["final_decision"] == "refuse"
