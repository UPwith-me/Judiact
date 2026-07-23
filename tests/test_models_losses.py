from __future__ import annotations

import torch

from causal_diagsafe.constants import DECISION_TO_ID, EVIDENCE_TO_ID, INTENT_TO_ID
from causal_diagsafe.models.cssb import CSSBHeads
from causal_diagsafe.models.egcr import EGCRSignals, MonotonicEGCR
from causal_diagsafe.models.losses import (
    categorical_invariance_loss,
    counterfactual_risk_calibration_loss,
    decision_flip_loss,
    evidence_flip_loss,
    evidence_present_log_odds,
    non_target_invariance_loss,
    stage_state_invariance_loss,
    supervised_cssb_loss,
    target_flip_loss,
)


def test_cssb_heads_shapes() -> None:
    model = CSSBHeads(8, head_kind="mlp", mlp_hidden_size=12, policy_id_count=4)
    out = model(torch.randn(2, 5, 8), torch.randn(2, 8))
    assert out.r_logits.shape == (2, 5)
    assert out.e_logits.shape == (2, 5, 3)
    assert out.decision_logits.shape == (2, 4)
    assert out.probabilities()["p_policy_id"].shape == (2, 4)


def test_masked_auxiliary_labels_are_not_evaluated() -> None:
    model = CSSBHeads(8)
    output = model(torch.randn(1, 5, 8), torch.randn(1, 8))
    losses = supervised_cssb_loss(
        output,
        {
            "policy_action": torch.tensor([-100]),
            "benign": torch.tensor([float("nan")]),
        },
        {
            "policy_action": torch.zeros(1),
            "benign": torch.zeros(1),
        },
    )
    assert losses["policy_action"].item() == 0.0
    assert losses["benign"].item() == 0.0
    assert torch.isfinite(losses["total"])


def test_egcr_outputs_logits_and_monotonic_weights_nonnegative() -> None:
    router = MonotonicEGCR()
    signals = _signals(batch=3)
    logits = router(signals)
    assert logits.shape == (3, 4)
    assert torch.all(torch.nn.functional.softplus(router.refuse_raw) >= 0)


def test_egcr_gates_risk_by_requirement_and_evidence() -> None:
    router = MonotonicEGCR()
    signals = _signals(batch=1)
    signals.response_present = None
    signals.pR.zero_()
    signals.pF.fill_(1.0)
    components = router.components(signals)
    assert components["stage_risk"].item() == 0.0
    assert components["exec_risk"].item() == 0.0
    assert components["evidence_missing"].item() == 0.0

    signals.pR[:, 4] = 1.0
    components = router.components(signals)
    assert components["stage_risk"].item() == 0.0
    assert components["exec_risk"].item() == 0.0

    signals.response_present = torch.ones(1)
    components = router.components(signals)
    assert components["stage_risk"].item() == 0.0
    assert components["exec_risk"].item() > 0.79

    signals.pR[:, 4] = 0.0
    signals.pR[:, 0] = 1.0
    signals.pE[:, 0].zero_()
    signals.pE[:, 0, EVIDENCE_TO_ID["absent"]] = 1.0
    components = router.components(signals)
    assert components["stage_risk"].item() == 0.0
    assert components["evidence_missing"].item() > 0.99


def test_target_flip_loss_rewards_correct_direction() -> None:
    a = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]])
    b = torch.tensor([[-2.0, 0.0, 0.0, 0.0, 0.0]])
    good = target_flip_loss(a, b, torch.tensor([0]), torch.tensor([1]), margin=1.0)
    bad = target_flip_loss(b, a, torch.tensor([0]), torch.tensor([1]), margin=1.0)
    assert good < bad


def test_pair_flip_losses_ignore_unsupervised_invalid_stage() -> None:
    f_logits = torch.zeros(2, 5)
    loss = target_flip_loss(
        f_logits,
        f_logits,
        torch.tensor([0, -1]),
        torch.tensor([1, 0]),
        supervision_mask=torch.tensor([1, 0]),
    )
    assert torch.isfinite(loss)


def test_evidence_flip_uses_translation_invariant_log_odds() -> None:
    a = torch.zeros(1, 5, 3)
    b = torch.zeros(1, 5, 3)
    a[:, 0, EVIDENCE_TO_ID["present"]] = 3.0
    b[:, 0, EVIDENCE_TO_ID["present"]] = -3.0
    target = torch.tensor([0])
    direction = torch.tensor([1])
    original = evidence_flip_loss(a, b, target, direction)
    shifted = evidence_flip_loss(a + 20.0, b - 17.0, target, direction)
    assert torch.allclose(original, shifted)
    assert torch.allclose(
        evidence_present_log_odds(a),
        evidence_present_log_odds(a + 20.0),
    )


def test_non_target_invariance_respects_mutable_mask() -> None:
    a = torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, -10.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([0])
    stable_mask = torch.tensor([[1, 0, 1, 1, 1]], dtype=torch.float)
    loss = non_target_invariance_loss(a, b, target, stable_mask)
    assert loss.item() == 0.0


def test_stage_state_invariance_covers_requirement_evidence_and_risk() -> None:
    r_a = torch.zeros(1, 5)
    r_b = r_a.clone()
    e_a = torch.zeros(1, 5, 3)
    e_b = e_a.clone()
    f_a = torch.zeros(1, 5)
    f_b = f_a.clone()
    r_b[:, 1] = 2.0
    e_b[:, 2, EVIDENCE_TO_ID["present"]] = 2.0
    f_b[:, 3] = 2.0
    loss = stage_state_invariance_loss(
        r_a,
        e_a,
        f_a,
        r_b,
        e_b,
        f_b,
        torch.tensor([0]),
        torch.tensor([[0, 1, 1, 1, 1]], dtype=torch.float),
    )
    assert loss.item() > 0.0


def test_categorical_invariance_respects_mask() -> None:
    a = torch.tensor([[2.0, 0.0], [5.0, -5.0]])
    b = torch.tensor([[2.0, 0.0], [-5.0, 5.0]])
    masked = categorical_invariance_loss(
        a,
        b,
        supervision_mask=torch.tensor([1.0, 0.0]),
    )
    unmasked = categorical_invariance_loss(a, b)
    assert masked.item() < 1e-7
    assert unmasked.item() > 0.0


def test_decision_flip_requires_each_side_to_have_margin() -> None:
    gold_a = torch.tensor([0])
    gold_b = torch.tensor([3])
    good_a = torch.tensor([[4.0, 0.0, 0.0, 0.0]])
    good_b = torch.tensor([[0.0, 0.0, 0.0, 4.0]])
    wrong_b = torch.tensor([[8.0, 0.0, 0.0, 0.0]])
    balanced = decision_flip_loss(good_a, good_b, gold_a, gold_b)
    one_side_wrong = decision_flip_loss(
        torch.tensor([[20.0, 0.0, 0.0, 0.0]]),
        wrong_b,
        gold_a,
        gold_b,
    )
    assert balanced < one_side_wrong


def test_counterfactual_calibration_uses_gated_probability_not_raw_evidence_logit() -> None:
    r_a = torch.full((1, 5), 3.0)
    r_b = torch.full((1, 5), -3.0)
    f_a = torch.full((1, 5), 3.0)
    f_b = torch.full((1, 5), -3.0)
    e_a = torch.zeros(1, 5, 3)
    e_b = torch.zeros(1, 5, 3)
    e_a[..., EVIDENCE_TO_ID["present"]] = 3.0
    e_b[..., EVIDENCE_TO_ID["present"]] = -3.0
    args = (
        torch.tensor([2.0]),
        torch.tensor([-2.0]),
        r_a,
        r_b,
        f_a,
        f_b,
        e_a,
        e_b,
        torch.tensor([0]),
        torch.ones(5),
    )
    original = counterfactual_risk_calibration_loss(*args)
    shifted_args = (*args[:6], e_a + 20.0, e_b - 17.0, *args[8:])
    shifted = counterfactual_risk_calibration_loss(*shifted_args)
    assert torch.isfinite(original)
    assert torch.allclose(original, shifted)


def _signals(batch: int) -> EGCRSignals:
    pR = torch.full((batch, 5), 0.8)
    pE = torch.zeros(batch, 5, 3)
    pE[:, :, EVIDENCE_TO_ID["present"]] = 0.8
    pE[:, :, EVIDENCE_TO_ID["absent"]] = 0.1
    pE[:, :, EVIDENCE_TO_ID["uncertain"]] = 0.1
    pF = torch.full((batch, 5), 0.2)
    pF[:, 4] = 0.7
    p_intent = torch.zeros(batch, len(INTENT_TO_ID))
    p_intent[:, INTENT_TO_ID["malicious"]] = 0.7
    p_intent[:, INTENT_TO_ID["ambiguous"]] = 0.1
    p_policy_action = torch.zeros(batch, 4)
    p_policy_action[:, DECISION_TO_ID["refuse"]] = 0.8
    p_policy_action[:, DECISION_TO_ID["warn"]] = 0.1
    p_policy_action[:, DECISION_TO_ID["review"]] = 0.1
    p_benign = torch.full((batch,), 0.2)
    return EGCRSignals(
        pR=pR,
        pE=pE,
        pF=pF,
        p_intent=p_intent,
        p_policy_action=p_policy_action,
        p_benign=p_benign,
    )
