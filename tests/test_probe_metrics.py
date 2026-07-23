from __future__ import annotations

import torch

from causal_diagsafe.cli.train_probe import _pair_metrics, _shuffle_stage_labels


def test_pair_metrics_require_target_direction_and_boundary() -> None:
    predictions = [
        {
            "id": "u",
            "final_decision": "refuse",
            "F_probability": [0.8, 0.1, 0.1, 0.1, 0.1],
            "R_probability": [1.0, 0.1, 0.1, 0.1, 0.1],
            "E_probability": [[0.0, 1.0, 0.0]] * 5,
            "F_prediction": [1, 0, 0, 0, 0],
        },
        {
            "id": "s",
            "final_decision": "allow",
            "F_probability": [0.2, 0.1, 0.1, 0.1, 0.1],
            "R_probability": [1.0, 0.1, 0.1, 0.1, 0.1],
            "E_probability": [[0.0, 1.0, 0.0]] * 5,
            "F_prediction": [0, 0, 0, 0, 0],
        },
    ]
    pairs = [
        {
            "left_id": "u",
            "right_id": "s",
            "left_role": "unsafe",
            "right_role": "safe",
            "target_stage_id": 0,
            "target_direction": 1,
            "target_direction_mask": 1,
            "stable_non_target_mask": [0, 1, 1, 1, 1],
        }
    ]
    metrics = _pair_metrics(predictions, pairs, tolerance=0.05)
    assert metrics["target_stage_flip_rate"] == 1.0
    assert metrics["non_target_invariance_rate"] == 1.0
    assert metrics["pair_boundary_success"] == 1.0


def test_safe_helpful_pair_is_not_counted_as_boundary_or_target_flip() -> None:
    predictions = [
        {
            "id": "s",
            "final_decision": "allow",
            "F_probability": [0.1] * 5,
            "R_probability": [0.1] * 5,
            "E_probability": [[1.0, 0.0, 0.0]] * 5,
            "F_prediction": [0] * 5,
        },
        {
            "id": "h",
            "final_decision": "warn",
            "F_probability": [0.1] * 5,
            "R_probability": [0.1] * 5,
            "E_probability": [[1.0, 0.0, 0.0]] * 5,
            "F_prediction": [0] * 5,
        },
    ]
    pairs = [
        {
            "left_id": "s",
            "right_id": "h",
            "left_role": "safe",
            "right_role": "helpful",
            "target_stage_id": 2,
            "target_direction": 0,
            "target_direction_mask": 0,
            "stable_non_target_mask": [1, 1, 0, 1, 1],
        }
    ]
    metrics = _pair_metrics(predictions, pairs, tolerance=0.05)
    assert metrics["strict_pair_count"] == 1
    assert metrics["target_supervised_pair_count"] == 0
    assert metrics["boundary_pair_count"] == 0
    assert metrics["target_stage_flip_rate"] == 0.0
    assert metrics["pair_boundary_success"] == 0.0


def test_stage_shuffle_preserves_masks_and_value_counts() -> None:
    cache = {
        "labels": {
            "R": torch.tensor([[0, 0, 1, 0, 1], [1, 1, 0, 1, 0], [1, 0, 1, 0, 1]]),
            "R_mask": torch.ones(3, 5),
            "E": torch.tensor([[0, 1, 2, 0, 1], [1, 1, 0, 1, 2], [2, 0, 1, 2, 0]]),
            "E_mask": torch.ones(3, 5),
            "F": torch.tensor([[0, 0, 1, 0, 1], [1, 1, 0, 1, 0], [1, 0, 1, 0, 1]]),
            "F_mask": torch.ones(3, 5),
        },
        "metadata": {"id": ["a", "b", "c"]},
    }
    shuffled = _shuffle_stage_labels(cache, seed=7)
    assert torch.equal(shuffled["labels"]["R_mask"], cache["labels"]["R_mask"])
    for key in ("R", "E", "F"):
        for stage in range(5):
            assert sorted(shuffled["labels"][key][:, stage].tolist()) == sorted(
                cache["labels"][key][:, stage].tolist()
            )
