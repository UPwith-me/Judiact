from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..constants import DECISIONS, EVIDENCE_STATES, INTENT_CLASSES


@dataclass
class CSSBOutput:
    r_logits: torch.Tensor
    e_logits: torch.Tensor
    f_logits: torch.Tensor
    intent_logits: torch.Tensor
    policy_action_logits: torch.Tensor
    benign_logits: torch.Tensor
    decision_logits: torch.Tensor
    policy_id_logits: torch.Tensor | None = None

    def probabilities(self) -> dict[str, torch.Tensor]:
        probabilities = {
            "pR": torch.sigmoid(self.r_logits),
            "pE": torch.softmax(self.e_logits, dim=-1),
            "pF": torch.sigmoid(self.f_logits),
            "p_intent": torch.softmax(self.intent_logits, dim=-1),
            "p_policy_action": torch.softmax(self.policy_action_logits, dim=-1),
            "p_benign": torch.sigmoid(self.benign_logits).squeeze(-1),
            "p_decision": torch.softmax(self.decision_logits, dim=-1),
        }
        if self.policy_id_logits is not None:
            probabilities["p_policy_id"] = torch.softmax(self.policy_id_logits, dim=-1)
        return probabilities


class CSSBHeads(nn.Module):
    """Causal stage-state heads over sentinel hidden states.

    Input:
        stage_hidden: FloatTensor[B, 5, H]
        decision_hidden: FloatTensor[B, H]

    Output:
        R/E/F heads plus intent, policy action, benign and decision heads.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        head_kind: str = "linear",
        mlp_hidden_size: int | None = None,
        policy_id_count: int = 0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if head_kind not in {"linear", "mlp"}:
            raise ValueError("head_kind must be 'linear' or 'mlp'")
        self.hidden_size = hidden_size
        self.head_kind = head_kind
        self.policy_id_count = policy_id_count
        self.r_head = _make_head(hidden_size, 1, head_kind, mlp_hidden_size, dropout)
        self.e_head = _make_head(hidden_size, len(EVIDENCE_STATES), head_kind, mlp_hidden_size, dropout)
        self.f_head = _make_head(hidden_size, 1, head_kind, mlp_hidden_size, dropout)
        self.intent_head = _make_head(
            hidden_size, len(INTENT_CLASSES), head_kind, mlp_hidden_size, dropout
        )
        self.policy_action_head = _make_head(
            hidden_size, len(DECISIONS), head_kind, mlp_hidden_size, dropout
        )
        self.benign_head = _make_head(hidden_size, 1, head_kind, mlp_hidden_size, dropout)
        self.decision_head = _make_head(hidden_size, len(DECISIONS), head_kind, mlp_hidden_size, dropout)
        self.policy_id_head = (
            _make_head(hidden_size, policy_id_count, head_kind, mlp_hidden_size, dropout)
            if policy_id_count > 0
            else None
        )

    def forward(self, stage_hidden: torch.Tensor, decision_hidden: torch.Tensor) -> CSSBOutput:
        if stage_hidden.ndim != 3 or stage_hidden.shape[1] != 5:
            raise ValueError(f"stage_hidden must have shape [B,5,H], got {tuple(stage_hidden.shape)}")
        if decision_hidden.ndim != 2:
            raise ValueError("decision_hidden must have shape [B,H]")
        if stage_hidden.shape[0] != decision_hidden.shape[0]:
            raise ValueError("stage_hidden and decision_hidden must have the same batch size")
        if stage_hidden.shape[2] != self.hidden_size or decision_hidden.shape[1] != self.hidden_size:
            raise ValueError(f"hidden states must use hidden_size={self.hidden_size}")
        h3 = stage_hidden[:, 2, :]
        h4 = stage_hidden[:, 3, :]
        r_logits = self.r_head(stage_hidden).squeeze(-1)
        e_logits = self.e_head(stage_hidden)
        f_logits = self.f_head(stage_hidden).squeeze(-1)
        policy_id_logits = self.policy_id_head(h4) if self.policy_id_head is not None else None
        return CSSBOutput(
            r_logits=r_logits,
            e_logits=e_logits,
            f_logits=f_logits,
            intent_logits=self.intent_head(h3),
            policy_action_logits=self.policy_action_head(h4),
            benign_logits=self.benign_head(decision_hidden),
            decision_logits=self.decision_head(decision_hidden),
            policy_id_logits=policy_id_logits,
        )


def _make_head(
    hidden_size: int,
    out_size: int,
    head_kind: str,
    mlp_hidden_size: int | None,
    dropout: float,
) -> nn.Module:
    if head_kind == "linear":
        return nn.Linear(hidden_size, out_size)
    mid = mlp_hidden_size or hidden_size
    return nn.Sequential(
        nn.LayerNorm(hidden_size),
        nn.Linear(hidden_size, mid),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mid, out_size),
    )
