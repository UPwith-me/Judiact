from __future__ import annotations

from causal_diagsafe.constants import DECISION_TO_ID, INTENT_TO_ID
from causal_diagsafe.gold_router import build_gold_router_tensors
from causal_diagsafe.schema import normalize_record


def test_no_policy_gold_router_does_not_reconstruct_intent_or_benign_from_decision() -> None:
    row = {
        "id": "sample-1",
        "model_input": {"query": "query", "images": []},
        "targets": {
            "final_decision": "refuse",
            "policy_action": DECISION_TO_ID["refuse"],
            "role": "unsafe",
            "benign": 0,
            "benign_mask": 0,
            "benign_source": "unknown_independent_label",
        },
    }
    tensors = build_gold_router_tensors(
        [normalize_record(row)],
        ablations={"policy_action"},
    )
    assert tensors["p_intent"][0, INTENT_TO_ID["unknown"]].item() == 1.0
    assert tensors["p_benign"][0].item() == 0.5
    assert tensors["p_policy_action"][0].tolist() == [0.25, 0.25, 0.25, 0.25]
    assert tensors["policy_action_mask"][0].item() == 0.0
    assert tensors["response_present"][0].item() == 0.0


def test_policy_action_is_used_only_with_an_explicit_mask() -> None:
    row = {
        "id": "sample-2",
        "model_input": {"query": "query", "images": []},
        "targets": {
            "final_decision": "refuse",
            "policy_action": DECISION_TO_ID["warn"],
            "policy_action_mask": 1,
        },
    }
    tensors = build_gold_router_tensors([normalize_record(row)])
    assert tensors["p_policy_action"][0, DECISION_TO_ID["warn"]].item() == 1.0
    assert tensors["policy_action_mask"][0].item() == 1.0


def test_candidate_response_availability_is_explicit() -> None:
    row = {
        "id": "sample-3",
        "model_input": {
            "query": "query",
            "images": [],
            "response_candidate": "candidate output",
        },
        "targets": {"final_decision": "review"},
    }
    tensors = build_gold_router_tensors([normalize_record(row)])
    assert tensors["response_present"][0].item() == 1.0
