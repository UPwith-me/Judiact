from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from ..constants import DECISION_TO_ID
from .cssb import CSSBOutput


def supervised_cssb_loss(
    output: CSSBOutput,
    labels: dict[str, torch.Tensor],
    weights: dict[str, torch.Tensor] | None = None,
) -> dict[str, torch.Tensor]:
    weights = weights or {}
    losses: dict[str, torch.Tensor] = {}
    if "R" in labels:
        losses["R"] = _weighted_bce(output.r_logits, labels["R"].float(), weights.get("R"))
    if "E" in labels:
        losses["E"] = _weighted_ce_flat(output.e_logits, labels["E"].long(), weights.get("E"))
    if "F" in labels:
        losses["F"] = _weighted_bce(output.f_logits, labels["F"].float(), weights.get("F"))
    if "intent" in labels:
        losses["intent"] = F.cross_entropy(output.intent_logits, labels["intent"].long())
    if "policy_action" in labels:
        losses["policy_action"] = F.cross_entropy(
            output.policy_action_logits, labels["policy_action"].long()
        )
    if "benign" in labels:
        losses["benign"] = F.binary_cross_entropy_with_logits(
            output.benign_logits.squeeze(-1), labels["benign"].float()
        )
    if "decision" in labels:
        losses["decision"] = F.cross_entropy(output.decision_logits, labels["decision"].long())
    if output.policy_id_logits is not None and "policy_id" in labels:
        losses["policy_id"] = F.cross_entropy(output.policy_id_logits, labels["policy_id"].long())
    losses["total"] = sum(losses.values()) if losses else output.r_logits.sum() * 0.0
    return losses


def router_loss(router_logits: torch.Tensor, decision_labels: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(router_logits, decision_labels.long())


def target_flip_loss(
    f_logits_a: torch.Tensor,
    f_logits_b: torch.Tensor,
    target_stage: torch.Tensor,
    direction: torch.Tensor,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    a = f_logits_a.gather(1, target_stage.long().view(-1, 1)).squeeze(1)
    b = f_logits_b.gather(1, target_stage.long().view(-1, 1)).squeeze(1)
    signed_gap = direction.float() * (a - b)
    return F.softplus(margin - signed_gap).mean()


def evidence_flip_loss(
    e_logits_a: torch.Tensor,
    e_logits_b: torch.Tensor,
    target_stage: torch.Tensor,
    direction: torch.Tensor,
    *,
    present_index: int = 1,
    margin: float = 1.0,
) -> torch.Tensor:
    present_a = e_logits_a[:, :, present_index].gather(1, target_stage.long().view(-1, 1)).squeeze(1)
    present_b = e_logits_b[:, :, present_index].gather(1, target_stage.long().view(-1, 1)).squeeze(1)
    signed_gap = direction.float() * (present_a - present_b)
    return F.softplus(margin - signed_gap).mean()


def non_target_invariance_loss(
    f_logits_a: torch.Tensor,
    f_logits_b: torch.Tensor,
    target_stage: torch.Tensor,
    mutable_non_target_mask: torch.Tensor,
) -> torch.Tensor:
    if mutable_non_target_mask.shape != f_logits_a.shape:
        raise ValueError("mutable_non_target_mask must have shape [B,5]")
    batch, stages = f_logits_a.shape
    target_mask = F.one_hot(target_stage.long(), num_classes=stages).bool()
    stable_mask = mutable_non_target_mask.float() * (~target_mask).float()
    squared = (f_logits_a - f_logits_b).pow(2) * stable_mask
    return squared.sum() / stable_mask.sum().clamp_min(1.0)


def decision_flip_loss(
    decision_logits_a: torch.Tensor,
    decision_logits_b: torch.Tensor,
    gold_a: torch.Tensor,
    gold_b: torch.Tensor,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    alt_a = _best_non_gold(decision_logits_a, gold_a)
    alt_b = _best_non_gold(decision_logits_b, gold_b)
    gold_score_a = decision_logits_a.gather(1, gold_a.long().view(-1, 1)).squeeze(1)
    gold_score_b = decision_logits_b.gather(1, gold_b.long().view(-1, 1)).squeeze(1)
    alt_score_a = decision_logits_a.gather(1, alt_a.view(-1, 1)).squeeze(1)
    alt_score_b = decision_logits_b.gather(1, alt_b.view(-1, 1)).squeeze(1)
    return F.softplus(margin - ((gold_score_a - alt_score_a) + (gold_score_b - alt_score_b))).mean()


def decision_invariance_loss(decision_logits_a: torch.Tensor, decision_logits_b: torch.Tensor) -> torch.Tensor:
    logp_a = F.log_softmax(decision_logits_a, dim=-1)
    p_b = F.softmax(decision_logits_b, dim=-1)
    return F.kl_div(logp_a, p_b, reduction="batchmean")


def counterfactual_risk_calibration_loss(
    refuse_logits_a: torch.Tensor,
    refuse_logits_b: torch.Tensor,
    f_logits_a: torch.Tensor,
    f_logits_b: torch.Tensor,
    e_present_logits_a: torch.Tensor,
    e_present_logits_b: torch.Tensor,
    target_stage: torch.Tensor,
    stage_alpha_raw: torch.Tensor,
    stage_beta_raw: torch.Tensor,
) -> torch.Tensor:
    delta_refuse = refuse_logits_a - refuse_logits_b
    delta_f = (f_logits_a - f_logits_b).gather(1, target_stage.long().view(-1, 1)).squeeze(1)
    delta_e = (e_present_logits_a - e_present_logits_b).gather(
        1, target_stage.long().view(-1, 1)
    ).squeeze(1)
    alpha = F.softplus(stage_alpha_raw).gather(0, target_stage.long())
    beta = F.softplus(stage_beta_raw).gather(0, target_stage.long())
    explained = alpha * delta_f + beta * delta_e
    return (delta_refuse - explained).pow(2).mean()


def benign_guard_loss(
    router_logits: torch.Tensor,
    benign_labels: torch.Tensor,
    *,
    refuse_threshold: float = 0.25,
    margin: float = 0.5,
) -> torch.Tensor:
    probs = torch.softmax(router_logits, dim=-1)
    refuse = probs[:, DECISION_TO_ID["refuse"]]
    allow_warn = torch.stack(
        [router_logits[:, DECISION_TO_ID["allow"]], router_logits[:, DECISION_TO_ID["warn"]]], dim=-1
    ).max(dim=-1).values
    refuse_logits = router_logits[:, DECISION_TO_ID["refuse"]]
    mask = benign_labels.float()
    threshold_loss = F.relu(refuse - refuse_threshold).pow(2) * mask
    margin_loss = F.softplus(refuse_logits - allow_warn + margin) * mask
    return (threshold_loss + margin_loss).sum() / mask.sum().clamp_min(1.0)


def brier_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    target = F.one_hot(labels.long(), num_classes=logits.shape[-1]).float()
    return (probs - target).pow(2).sum(dim=-1).mean()


def total_weighted_loss(losses: dict[str, torch.Tensor], lambdas: dict[str, float]) -> torch.Tensor:
    total = None
    for key, value in losses.items():
        if key == "total":
            continue
        weighted = float(lambdas.get(key, 1.0)) * value
        total = weighted if total is None else total + weighted
    if total is None:
        first = next(iter(losses.values()))
        return first * 0.0
    return total


def _weighted_bce(logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    if weight is not None:
        loss = loss * weight.float()
        return loss.sum() / weight.float().sum().clamp_min(1.0)
    return loss.mean()


def _weighted_ce_flat(logits: torch.Tensor, labels: torch.Tensor, weight: torch.Tensor | None) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    loss = F.cross_entropy(flat_logits, flat_labels, reduction="none")
    if weight is not None:
        flat_weight = weight.reshape(-1).float()
        return (loss * flat_weight).sum() / flat_weight.sum().clamp_min(1.0)
    return loss.mean()


def _best_non_gold(logits: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    masked = logits.clone()
    masked.scatter_(1, gold.long().view(-1, 1), -1e9)
    return masked.argmax(dim=-1)

