from __future__ import annotations

import torch

from causal_diagsafe.constants import DECISION_TO_ID, EVIDENCE_TO_ID, INTENT_TO_ID
from causal_diagsafe.models.cssb import CSSBHeads
from causal_diagsafe.models.egcr import EGCRSignals, MonotonicEGCR
from causal_diagsafe.models.losses import non_target_invariance_loss, target_flip_loss


def test_cssb_heads_shapes() -> None:
    model = CSSBHeads(8, head_kind="mlp", mlp_hidden_size=12)
    out = model(torch.randn(2, 5, 8), torch.randn(2, 8))
    assert out.r_logits.shape == (2, 5)
    assert out.e_logits.shape == (2, 5, 3)
    assert out.decision_logits.shape == (2, 4)


def test_egcr_outputs_logits_and_monotonic_weights_nonnegative() -> None:
    router = MonotonicEGCR()
    signals = _signals(batch=3)
    logits = router(signals)
    assert logits.shape == (3, 4)
    assert torch.all(torch.nn.functional.softplus(router.refuse_raw) >= 0)


def test_target_flip_loss_rewards_correct_direction() -> None:
    a = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0]])
    b = torch.tensor([[-2.0, 0.0, 0.0, 0.0, 0.0]])
    good = target_flip_loss(a, b, torch.tensor([0]), torch.tensor([1]), margin=1.0)
    bad = target_flip_loss(b, a, torch.tensor([0]), torch.tensor([1]), margin=1.0)
    assert good < bad


def test_non_target_invariance_respects_mutable_mask() -> None:
    a = torch.tensor([[0.0, 10.0, 0.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, -10.0, 0.0, 0.0, 0.0]])
    target = torch.tensor([0])
    stable_mask = torch.tensor([[1, 0, 1, 1, 1]], dtype=torch.float)
    loss = non_target_invariance_loss(a, b, target, stable_mask)
    assert loss.item() == 0.0


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

