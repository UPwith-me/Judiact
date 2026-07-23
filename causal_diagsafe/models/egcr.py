from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from ..constants import DECISION_TO_ID, EVIDENCE_TO_ID, INTENT_TO_ID


@dataclass
class EGCRConfig:
    eps: float = 1e-5
    benign_mask_threshold: float = 0.8
    policy_high_threshold: float = 0.9
    intent_high_threshold: float = 0.85
    stage_risk_high_threshold: float = 0.75
    visual_evidence_threshold: float = 0.35
    exec_high_threshold: float = 0.85
    hard_masks: bool = False


@dataclass
class EGCRSignals:
    pR: torch.Tensor
    pE: torch.Tensor
    pF: torch.Tensor
    p_intent: torch.Tensor
    p_policy_action: torch.Tensor
    p_benign: torch.Tensor
    response_present: torch.Tensor | None = None
    text_violation_prob: torch.Tensor | None = None


class MonotonicEGCR(nn.Module):
    """Evidence-Gated Causal Router with non-negative risk coefficients."""

    def __init__(self, config: EGCRConfig | None = None) -> None:
        super().__init__()
        self.config = config or EGCRConfig()
        self.bias = nn.Parameter(torch.zeros(4))
        self.refuse_raw = nn.Parameter(torch.ones(5))
        self.review_raw = nn.Parameter(torch.ones(3))
        self.warn_raw = nn.Parameter(torch.ones(3))
        self.allow_raw = nn.Parameter(torch.ones(4))
        self.cf_alpha_raw = nn.Parameter(torch.ones(5))

    def forward(self, signals: EGCRSignals, *, apply_masks: bool | None = None) -> torch.Tensor:
        self._validate_signals(signals)
        comps = self.components(signals)
        refuse_w = F.softplus(self.refuse_raw)
        review_w = F.softplus(self.review_raw)
        warn_w = F.softplus(self.warn_raw)
        allow_w = F.softplus(self.allow_raw)

        z_refuse = self.bias[DECISION_TO_ID["refuse"]] + (
            refuse_w[0] * _logit(comps["policy_refuse"], self.config.eps)
            + refuse_w[1] * _logit(comps["intent_risk"], self.config.eps)
            + refuse_w[2] * _logit(1.0 - comps["benign"], self.config.eps)
            + refuse_w[3] * _logit(comps["exec_risk"], self.config.eps)
            + refuse_w[4] * _logit(comps["stage_risk"], self.config.eps)
        )
        z_review = self.bias[DECISION_TO_ID["review"]] + (
            review_w[0] * _logit(comps["policy_uncertain"], self.config.eps)
            + review_w[1] * _logit(comps["evidence_missing"], self.config.eps)
            + review_w[2] * _logit(comps["intent_ambiguous"], self.config.eps)
        )
        z_warn = self.bias[DECISION_TO_ID["warn"]] + (
            warn_w[0] * _logit(comps["policy_warn"], self.config.eps)
            + warn_w[1] * _logit(comps["benign"], self.config.eps)
            - warn_w[2] * _logit(comps["exec_risk"], self.config.eps)
        )
        z_allow = self.bias[DECISION_TO_ID["allow"]] + (
            allow_w[0] * _logit(comps["benign"], self.config.eps)
            + allow_w[1] * _logit(1.0 - comps["policy_refuse"], self.config.eps)
            + allow_w[2] * _logit(1.0 - comps["exec_risk"], self.config.eps)
            + allow_w[3] * _logit(1.0 - comps["stage_risk"], self.config.eps)
        )
        logits = torch.stack([z_allow, z_warn, z_review, z_refuse], dim=-1)
        use_masks = self.config.hard_masks if apply_masks is None else apply_masks
        return self.apply_hard_masks(logits, comps) if use_masks else logits

    def components(self, signals: EGCRSignals) -> dict[str, torch.Tensor]:
        batch = signals.pR.shape[0]
        device = signals.pR.device
        dtype = signals.pR.dtype
        zeros = torch.zeros(batch, device=device, dtype=dtype)
        response_present = (
            signals.response_present.to(device=device, dtype=dtype).clamp(0.0, 1.0)
            if signals.response_present is not None
            else zeros
        )
        effective_requirement = torch.cat(
            [
                signals.pR[:, :4],
                signals.pR[:, 4:5] * response_present.unsqueeze(-1),
            ],
            dim=-1,
        )
        e_present = signals.pE[..., EVIDENCE_TO_ID["present"]]
        g_stage = (1.0 - effective_requirement) + effective_requirement * e_present
        g_evidence = torch.exp(torch.log(g_stage.clamp(self.config.eps, 1.0)).sum(dim=-1))
        evidence_missing = 1.0 - g_evidence
        stage_confident_risk = effective_requirement * e_present * signals.pF
        stage_risk = 1.0 - torch.prod(
            (1.0 - stage_confident_risk[:, :4]).clamp(self.config.eps, 1.0),
            dim=-1,
        )
        intent_risk = signals.p_intent[:, INTENT_TO_ID["malicious"]] + signals.p_intent[
            :, INTENT_TO_ID["jailbreak_or_evasion"]
        ]
        intent_risk = intent_risk.clamp(0.0, 1.0)
        intent_ambiguous = signals.p_intent[:, INTENT_TO_ID["ambiguous"]]
        policy_refuse = signals.p_policy_action[:, DECISION_TO_ID["refuse"]]
        policy_warn = signals.p_policy_action[:, DECISION_TO_ID["warn"]]
        policy_uncertain = signals.p_policy_action[:, DECISION_TO_ID["review"]]
        exec_risk = stage_confident_risk[:, 4]
        return {
            "effective_requirement": effective_requirement,
            "response_present": response_present,
            "g_stage": g_stage,
            "g_evidence": g_evidence,
            "evidence_missing": evidence_missing,
            "stage_confident_risk": stage_confident_risk,
            "stage_risk": stage_risk,
            "intent_risk": intent_risk,
            "intent_ambiguous": intent_ambiguous,
            "policy_refuse": policy_refuse,
            "policy_warn": policy_warn,
            "policy_uncertain": policy_uncertain,
            "exec_risk": exec_risk,
            "benign": signals.p_benign,
            "text_violation_prob": (
                signals.text_violation_prob if signals.text_violation_prob is not None else zeros
            ),
        }

    def apply_hard_masks(self, logits: torch.Tensor, comps: dict[str, torch.Tensor]) -> torch.Tensor:
        masked = logits.clone()
        refuse_idx = DECISION_TO_ID["refuse"]
        allow_idx = DECISION_TO_ID["allow"]
        review_idx = DECISION_TO_ID["review"]

        benign_guard = (comps["benign"] > self.config.benign_mask_threshold) & ~(
            (comps["policy_refuse"] > self.config.policy_high_threshold)
            & (comps["intent_risk"] > self.config.intent_high_threshold)
            & (comps["stage_risk"] > self.config.stage_risk_high_threshold)
        )
        masked[benign_guard, refuse_idx] = -1e4

        required_visual_missing = (
            comps["g_stage"][:, :2].min(dim=-1).values
            < self.config.visual_evidence_threshold
        ) & (comps["text_violation_prob"] < self.config.policy_high_threshold)
        masked[required_visual_missing, refuse_idx] = -1e4
        masked[required_visual_missing, review_idx] = (
            masked[required_visual_missing, review_idx] + 2.0
        )

        unsafe_response = (comps["exec_risk"] > self.config.exec_high_threshold) & (
            comps["policy_refuse"] > self.config.policy_high_threshold
        )
        masked[unsafe_response, allow_idx] = -1e4
        return masked

    @staticmethod
    def _validate_signals(signals: EGCRSignals) -> None:
        if signals.pR.ndim != 2 or signals.pR.shape[1] != 5:
            raise ValueError("pR must have shape [B,5]")
        if signals.pE.ndim != 3 or signals.pE.shape[:2] != signals.pR.shape:
            raise ValueError("pE must have shape [B,5,3]")
        if signals.pE.shape[2] != len(EVIDENCE_TO_ID):
            raise ValueError("pE must contain absent/present/uncertain probabilities")
        if signals.pF.shape != signals.pR.shape:
            raise ValueError("pF must have shape [B,5]")
        batch = signals.pR.shape[0]
        if signals.p_intent.shape != (batch, len(INTENT_TO_ID)):
            raise ValueError("p_intent has an incompatible shape")
        if signals.p_policy_action.shape != (batch, len(DECISION_TO_ID)):
            raise ValueError("p_policy_action has an incompatible shape")
        if signals.p_benign.shape != (batch,):
            raise ValueError("p_benign must have shape [B]")
        if signals.response_present is not None and signals.response_present.shape != (batch,):
            raise ValueError("response_present must have shape [B]")
        if (
            signals.text_violation_prob is not None
            and signals.text_violation_prob.shape != (batch,)
        ):
            raise ValueError("text_violation_prob must have shape [B]")


def build_signals_from_cssb(
    cssb_probs: dict[str, torch.Tensor],
    *,
    response_present: torch.Tensor | None = None,
    text_violation_prob: torch.Tensor | None = None,
) -> EGCRSignals:
    return EGCRSignals(
        pR=cssb_probs["pR"],
        pE=cssb_probs["pE"],
        pF=cssb_probs["pF"],
        p_intent=cssb_probs["p_intent"],
        p_policy_action=cssb_probs["p_policy_action"],
        p_benign=cssb_probs["p_benign"],
        response_present=response_present,
        text_violation_prob=text_violation_prob,
    )


def _logit(value: torch.Tensor, eps: float) -> torch.Tensor:
    clipped = value.clamp(eps, 1.0 - eps)
    return torch.log(clipped) - torch.log1p(-clipped)
