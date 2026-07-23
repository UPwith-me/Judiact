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
        losses["intent"] = _weighted_ce(
            output.intent_logits,
            labels["intent"].long(),
            weights.get("intent"),
        )
    if "policy_action" in labels:
        losses["policy_action"] = _weighted_ce(
            output.policy_action_logits,
            labels["policy_action"].long(),
            weights.get("policy_action"),
        )
    if "benign" in labels:
        losses["benign"] = _weighted_bce(
            output.benign_logits.squeeze(-1),
            labels["benign"].float(),
            weights.get("benign"),
        )
    if "decision" in labels:
        losses["decision"] = _weighted_ce(
            output.decision_logits,
            labels["decision"].long(),
            weights.get("decision"),
        )
    if output.policy_id_logits is not None and "policy_id" in labels:
        losses["policy_id"] = _weighted_ce(
            output.policy_id_logits,
            labels["policy_id"].long(),
            weights.get("policy_id"),
        )
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
    supervision_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if direction.shape != target_stage.shape:
        raise ValueError("target direction must have shape [B]")
    active = _pair_supervision_mask(
        target_stage,
        supervision_mask,
        stage_count=f_logits_a.shape[1],
    )
    if not torch.any(active):
        return f_logits_a.sum() * 0.0
    active_stage = target_stage[active].long()
    active_direction = direction[active].float()
    if not torch.all((active_direction == 1) | (active_direction == -1)):
        raise ValueError("active target directions must be -1 or +1")
    a = f_logits_a[active].gather(1, active_stage.view(-1, 1)).squeeze(1)
    b = f_logits_b[active].gather(1, active_stage.view(-1, 1)).squeeze(1)
    signed_gap = active_direction * (a - b)
    return F.softplus(margin - signed_gap).mean()


def evidence_flip_loss(
    e_logits_a: torch.Tensor,
    e_logits_b: torch.Tensor,
    target_stage: torch.Tensor,
    direction: torch.Tensor,
    *,
    present_index: int = 1,
    margin: float = 1.0,
    supervision_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if direction.shape != target_stage.shape:
        raise ValueError("evidence direction must have shape [B]")
    active = _pair_supervision_mask(
        target_stage,
        supervision_mask,
        stage_count=e_logits_a.shape[1],
    )
    if not torch.any(active):
        return e_logits_a.sum() * 0.0
    active_stage = target_stage[active].long()
    active_direction = direction[active].float()
    if not torch.all((active_direction == 1) | (active_direction == -1)):
        raise ValueError("active evidence directions must be -1 or +1")
    present_a = evidence_present_log_odds(
        e_logits_a[active],
        present_index=present_index,
    ).gather(
        1,
        active_stage.view(-1, 1),
    ).squeeze(1)
    present_b = evidence_present_log_odds(
        e_logits_b[active],
        present_index=present_index,
    ).gather(
        1,
        active_stage.view(-1, 1),
    ).squeeze(1)
    signed_gap = active_direction * (present_a - present_b)
    return F.softplus(margin - signed_gap).mean()


def evidence_present_log_odds(
    e_logits: torch.Tensor,
    *,
    present_index: int = 1,
) -> torch.Tensor:
    """Return translation-invariant present-vs-non-present evidence logits."""

    if e_logits.ndim != 3:
        raise ValueError("e_logits must have shape [B,5,C]")
    classes = e_logits.shape[-1]
    if present_index < 0 or present_index >= classes:
        raise ValueError("present_index is outside the evidence class dimension")
    other_indices = [index for index in range(classes) if index != present_index]
    if not other_indices:
        raise ValueError("evidence logits must contain at least two classes")
    present = e_logits[..., present_index]
    non_present = torch.logsumexp(e_logits[..., other_indices], dim=-1)
    return present - non_present


def non_target_invariance_loss(
    f_logits_a: torch.Tensor,
    f_logits_b: torch.Tensor,
    target_stage: torch.Tensor,
    stable_non_target_mask: torch.Tensor,
) -> torch.Tensor:
    """Backward-compatible F-only non-target invariance loss."""

    stable_mask = _stable_mask(
        f_logits_a,
        target_stage,
        stable_non_target_mask,
    )
    squared = (f_logits_a - f_logits_b).pow(2) * stable_mask
    return squared.sum() / stable_mask.sum().clamp_min(1.0)


def stage_state_invariance_loss(
    r_logits_a: torch.Tensor,
    e_logits_a: torch.Tensor,
    f_logits_a: torch.Tensor,
    r_logits_b: torch.Tensor,
    e_logits_b: torch.Tensor,
    f_logits_b: torch.Tensor,
    target_stage: torch.Tensor,
    stable_non_target_mask: torch.Tensor,
    *,
    r_weight: float = 1.0,
    e_weight: float = 1.0,
    f_weight: float = 1.0,
) -> torch.Tensor:
    """Constrain the complete R/E/F state on stable non-target stages."""

    if r_logits_a.shape != f_logits_a.shape or r_logits_b.shape != f_logits_b.shape:
        raise ValueError("R and F logits must have matching [B,5] shapes")
    if r_logits_a.shape != r_logits_b.shape or f_logits_a.shape != f_logits_b.shape:
        raise ValueError("paired R/F logits must have matching shapes")
    if e_logits_a.shape != e_logits_b.shape or e_logits_a.shape[:2] != r_logits_a.shape:
        raise ValueError("paired E logits must have shape [B,5,C]")
    stable_mask = _stable_mask(r_logits_a, target_stage, stable_non_target_mask)
    denominator = stable_mask.sum().clamp_min(1.0)
    r_loss = ((r_logits_a - r_logits_b).pow(2) * stable_mask).sum() / denominator
    f_loss = ((f_logits_a - f_logits_b).pow(2) * stable_mask).sum() / denominator
    e_js = _categorical_js_divergence(e_logits_a, e_logits_b)
    e_loss = (e_js * stable_mask).sum() / denominator
    return float(r_weight) * r_loss + float(e_weight) * e_loss + float(f_weight) * f_loss


def _stable_mask(
    reference_logits: torch.Tensor,
    target_stage: torch.Tensor,
    stable_non_target_mask: torch.Tensor,
) -> torch.Tensor:
    if stable_non_target_mask.shape != reference_logits.shape:
        raise ValueError("stable_non_target_mask must have shape [B,5]")
    _, stages = reference_logits.shape
    target_mask = F.one_hot(target_stage.long(), num_classes=stages).bool()
    return stable_non_target_mask.float() * (~target_mask).float()


def decision_flip_loss(
    decision_logits_a: torch.Tensor,
    decision_logits_b: torch.Tensor,
    gold_a: torch.Tensor,
    gold_b: torch.Tensor,
    *,
    margin: float = 1.0,
) -> torch.Tensor:
    if gold_a.numel() == 0:
        return decision_logits_a.sum() * 0.0
    alt_a = _best_non_gold(decision_logits_a, gold_a)
    alt_b = _best_non_gold(decision_logits_b, gold_b)
    gold_score_a = decision_logits_a.gather(1, gold_a.long().view(-1, 1)).squeeze(1)
    gold_score_b = decision_logits_b.gather(1, gold_b.long().view(-1, 1)).squeeze(1)
    alt_score_a = decision_logits_a.gather(1, alt_a.view(-1, 1)).squeeze(1)
    alt_score_b = decision_logits_b.gather(1, alt_b.view(-1, 1)).squeeze(1)
    margin_a = gold_score_a - alt_score_a
    margin_b = gold_score_b - alt_score_b
    return (F.softplus(margin - margin_a) + F.softplus(margin - margin_b)).mean()


def decision_invariance_loss(
    decision_logits_a: torch.Tensor,
    decision_logits_b: torch.Tensor,
) -> torch.Tensor:
    if decision_logits_a.numel() == 0:
        return decision_logits_a.sum() * 0.0
    return _categorical_js_divergence(decision_logits_a, decision_logits_b).mean()


def categorical_invariance_loss(
    logits_a: torch.Tensor,
    logits_b: torch.Tensor,
    supervision_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Jensen-Shannon invariance for intent, policy, or other categorical heads."""

    if logits_a.shape != logits_b.shape:
        raise ValueError("paired categorical logits must have matching shapes")
    if logits_a.ndim != 2:
        raise ValueError("categorical logits must have shape [B,C]")
    per_row = _categorical_js_divergence(logits_a, logits_b)
    if supervision_mask is None:
        return per_row.mean() if per_row.numel() else logits_a.sum() * 0.0
    if supervision_mask.shape != per_row.shape:
        raise ValueError("categorical invariance mask must have shape [B]")
    active = supervision_mask > 0
    if not torch.any(active):
        return logits_a.sum() * 0.0
    weights = supervision_mask[active].float()
    return (per_row[active] * weights).sum() / weights.sum()


def counterfactual_risk_calibration_loss(
    refuse_logits_a: torch.Tensor,
    refuse_logits_b: torch.Tensor,
    r_logits_a: torch.Tensor,
    r_logits_b: torch.Tensor,
    f_logits_a: torch.Tensor,
    f_logits_b: torch.Tensor,
    e_logits_a: torch.Tensor,
    e_logits_b: torch.Tensor,
    target_stage: torch.Tensor,
    stage_alpha_raw: torch.Tensor,
    *,
    present_index: int = 1,
    eps: float = 1e-5,
    supervision_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Align router refusal changes with evidence-gated target-stage risk changes."""

    if refuse_logits_a.ndim != 1 or refuse_logits_b.shape != refuse_logits_a.shape:
        raise ValueError("paired refuse logits must have shape [B]")
    if stage_alpha_raw.ndim != 1 or stage_alpha_raw.shape[0] != r_logits_a.shape[1]:
        raise ValueError("stage_alpha_raw must contain one value per stage")
    active = _pair_supervision_mask(
        target_stage,
        supervision_mask,
        stage_count=r_logits_a.shape[1],
    )
    if not torch.any(active):
        return refuse_logits_a.sum() * 0.0
    delta_refuse = refuse_logits_a[active] - refuse_logits_b[active]
    confident_a = _confident_stage_risk_logit(
        r_logits_a[active],
        e_logits_a[active],
        f_logits_a[active],
        present_index=present_index,
        eps=eps,
    )
    confident_b = _confident_stage_risk_logit(
        r_logits_b[active],
        e_logits_b[active],
        f_logits_b[active],
        present_index=present_index,
        eps=eps,
    )
    delta_risk = (confident_a - confident_b).gather(
        1,
        target_stage[active].long().view(-1, 1),
    ).squeeze(1)
    alpha = F.softplus(stage_alpha_raw).gather(0, target_stage[active].long())
    explained = alpha * delta_risk
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
        [
            router_logits[:, DECISION_TO_ID["allow"]],
            router_logits[:, DECISION_TO_ID["warn"]],
        ],
        dim=-1,
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


def total_weighted_loss(
    losses: dict[str, torch.Tensor],
    lambdas: dict[str, float],
) -> torch.Tensor:
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


def _weighted_bce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    if weight is None:
        return F.binary_cross_entropy_with_logits(logits, labels)
    active = weight > 0
    if not torch.any(active):
        return logits.sum() * 0.0
    active_weight = weight[active].float()
    loss = F.binary_cross_entropy_with_logits(
        logits[active],
        labels[active],
        reduction="none",
    )
    return (loss * active_weight).sum() / active_weight.sum()


def _weighted_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    if weight is None:
        return F.cross_entropy(logits, labels)
    active = weight > 0
    if not torch.any(active):
        return logits.sum() * 0.0
    active_weight = weight[active].float()
    loss = F.cross_entropy(logits[active], labels[active], reduction="none")
    return (loss * active_weight).sum() / active_weight.sum()


def _weighted_ce_flat(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor | None,
) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1)
    if weight is None:
        return F.cross_entropy(flat_logits, flat_labels)
    flat_weight = weight.reshape(-1)
    active = flat_weight > 0
    if not torch.any(active):
        return flat_logits.sum() * 0.0
    active_weight = flat_weight[active].float()
    loss = F.cross_entropy(
        flat_logits[active],
        flat_labels[active],
        reduction="none",
    )
    return (loss * active_weight).sum() / active_weight.sum()


def _best_non_gold(logits: torch.Tensor, gold: torch.Tensor) -> torch.Tensor:
    masked = logits.clone()
    masked.scatter_(1, gold.long().view(-1, 1), -1e9)
    return masked.argmax(dim=-1)


def _pair_supervision_mask(
    target_stage: torch.Tensor,
    supervision_mask: torch.Tensor | None,
    *,
    stage_count: int,
) -> torch.Tensor:
    if target_stage.ndim != 1:
        raise ValueError("target_stage must have shape [B]")
    if supervision_mask is not None and supervision_mask.shape != target_stage.shape:
        raise ValueError("pair supervision mask must have shape [B]")
    active = (
        supervision_mask.bool()
        if supervision_mask is not None
        else torch.ones_like(target_stage, dtype=torch.bool)
    )
    valid_stage = (target_stage >= 0) & (target_stage < stage_count)
    if torch.any(active & ~valid_stage):
        raise ValueError("active target_stage values are outside the stage dimension")
    return active & valid_stage


def _categorical_js_divergence(logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
    logp_a = F.log_softmax(logits_a, dim=-1)
    logp_b = F.log_softmax(logits_b, dim=-1)
    p_a = logp_a.exp()
    p_b = logp_b.exp()
    midpoint = 0.5 * (p_a + p_b)
    log_midpoint = midpoint.clamp_min(torch.finfo(midpoint.dtype).tiny).log()
    kl_a = (p_a * (logp_a - log_midpoint)).sum(dim=-1)
    kl_b = (p_b * (logp_b - log_midpoint)).sum(dim=-1)
    return 0.5 * (kl_a + kl_b)


def _confident_stage_risk_logit(
    r_logits: torch.Tensor,
    e_logits: torch.Tensor,
    f_logits: torch.Tensor,
    *,
    present_index: int,
    eps: float,
) -> torch.Tensor:
    p_required = torch.sigmoid(r_logits)
    p_present = torch.softmax(e_logits, dim=-1)[..., present_index]
    p_risk = torch.sigmoid(f_logits)
    confident_risk = (p_required * p_present * p_risk).clamp(eps, 1.0 - eps)
    return torch.log(confident_risk) - torch.log1p(-confident_risk)
