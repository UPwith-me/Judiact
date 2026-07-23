from __future__ import annotations

import pytest

from causal_diagsafe.hidden_export import _append_labels, _find_last_subsequence


def test_find_last_subsequence_uses_last_occurrence() -> None:
    assert _find_last_subsequence([1, 2, 3, 2, 3, 4], [2, 3]) == 3
    with pytest.raises(ValueError):
        _find_last_subsequence([1, 2, 3], [4])


def test_append_labels_preserves_masks_and_unknown_intent() -> None:
    cache = {
        key: []
        for key in (
            "decision",
            "decision_mask",
            "policy_action",
            "policy_action_mask",
            "benign",
            "benign_mask",
            "intent",
            "intent_mask",
            "R",
            "R_mask",
            "E",
            "E_mask",
            "F",
            "F_mask",
            "target_stage",
        )
    }
    row = {
        "id": "x",
        "targets": {
            "decision": 1,
            "policy_action": 1,
            "policy_action_mask": 1,
            "target_stage_id": 2,
            "stage": {
                "R": [0, 0, 1, 0, 0],
                "R_mask": [0, 0, 1, 0, 0],
                "E": [2, 2, 1, 2, 2],
                "E_mask": [0, 0, 1, 0, 0],
                "F": [0, 0, 1, 0, 0],
                "F_mask": [0, 0, 1, 0, 0],
            },
        },
    }
    _append_labels(cache, [row])
    assert cache["decision"] == [1]
    assert cache["decision_mask"] == [1]
    assert cache["policy_action_mask"] == [1]
    assert cache["target_stage"] == [2]
    assert cache["R_mask"] == [[0, 0, 1, 0, 0]]
    assert cache["intent_mask"] == [0]
