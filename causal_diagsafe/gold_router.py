from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset

from .constants import DECISION_TO_ID, EVIDENCE_TO_ID, INTENT_TO_ID, STAGES
from .metrics import decision_metrics
from .models.egcr import EGCRSignals, MonotonicEGCR
from .models.losses import brier_loss, router_loss
from .parser import ParsedPrediction
from .schema import DiagRecord, normalize_record


def build_gold_router_tensors(records: list[DiagRecord]) -> dict[str, torch.Tensor]:
    pR = torch.zeros(len(records), 5)
    pE = torch.zeros(len(records), 5, 3)
    pF = torch.zeros(len(records), 5)
    p_intent = torch.zeros(len(records), len(INTENT_TO_ID))
    p_policy_action = torch.zeros(len(records), len(DECISION_TO_ID))
    p_benign = torch.zeros(len(records))
    decision = torch.zeros(len(records), dtype=torch.long)

    for i, record in enumerate(records):
        for j, stage in enumerate(STAGES):
            state = record.stage_states.normalized().states[stage]
            pR[i, j] = float(state.R if state.R is not None else int(record.target_stage == stage))
            evidence = state.E or "uncertain"
            pE[i, j, EVIDENCE_TO_ID[evidence]] = 1.0
            pF[i, j] = float(state.F if state.F is not None else int(record.final_decision == "refuse" and record.target_stage == stage))
        intent_key = _intent_key(record)
        p_intent[i, INTENT_TO_ID[intent_key]] = 1.0
        action_key = _policy_action_key(record)
        p_policy_action[i, DECISION_TO_ID[action_key]] = 1.0
        p_benign[i] = float(_benign_label(record))
        decision[i] = DECISION_TO_ID[record.final_decision]
    return {
        "pR": pR,
        "pE": pE,
        "pF": pF,
        "p_intent": p_intent,
        "p_policy_action": p_policy_action,
        "p_benign": p_benign,
        "decision": decision,
    }


def train_gold_router(
    train_rows: list[dict[str, Any]],
    eval_rows: list[dict[str, Any]],
    *,
    epochs: int = 100,
    lr: float = 5e-2,
    batch_size: int = 128,
    device: str = "cpu",
) -> tuple[MonotonicEGCR, dict[str, Any]]:
    train_records = [normalize_record(row) for row in train_rows]
    eval_records = [normalize_record(row) for row in eval_rows]
    train_tensors = build_gold_router_tensors(train_records)
    router = MonotonicEGCR().to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr)
    dataset = TensorDataset(
        train_tensors["pR"],
        train_tensors["pE"],
        train_tensors["pF"],
        train_tensors["p_intent"],
        train_tensors["p_policy_action"],
        train_tensors["p_benign"],
        train_tensors["decision"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    history = []
    for epoch in range(1, epochs + 1):
        losses = []
        router.train()
        for batch in loader:
            pR, pE, pF, p_intent, p_policy_action, p_benign, decision = [
                item.to(device) for item in batch
            ]
            logits = router(EGCRSignals(pR, pE, pF, p_intent, p_policy_action, p_benign))
            loss = router_loss(logits, decision) + 0.1 * brier_loss(logits, decision)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            metrics = evaluate_gold_router(router, eval_records, device=device)
            history.append({"epoch": epoch, "train_loss": sum(losses) / max(len(losses), 1), **metrics["decision_metrics"]})
    return router, {"history": history, **evaluate_gold_router(router, eval_records, device=device)}


@torch.no_grad()
def evaluate_gold_router(
    router: MonotonicEGCR,
    eval_records: list[DiagRecord],
    *,
    device: str = "cpu",
) -> dict[str, Any]:
    tensors = build_gold_router_tensors(eval_records)
    signals = EGCRSignals(
        tensors["pR"].to(device),
        tensors["pE"].to(device),
        tensors["pF"].to(device),
        tensors["p_intent"].to(device),
        tensors["p_policy_action"].to(device),
        tensors["p_benign"].to(device),
    )
    router.eval()
    logits = router(signals, apply_masks=True)
    pred_ids = logits.argmax(dim=-1).detach().cpu().tolist()
    id_to_decision = {idx: name for name, idx in DECISION_TO_ID.items()}
    predictions = [
        ParsedPrediction(
            sample_id=record.sample_id,
            valid_json=True,
            final_decision=id_to_decision[int(pred_id)],
            policy_id="gold_router",
            stage_states=record.stage_states,
            legacy_stage_labels=False,
            raw_json=None,
            parse_error=None,
        )
        for record, pred_id in zip(eval_records, pred_ids)
    ]
    return {"decision_metrics": decision_metrics(eval_records, predictions)}


def _intent_key(record: DiagRecord) -> str:
    value = str(record.raw.get("intent_label") or record.raw.get("intent") or "").strip().lower()
    if value in INTENT_TO_ID:
        return value
    if record.final_decision == "refuse":
        return "malicious"
    if record.final_decision == "review":
        return "ambiguous"
    return "benign_educational"


def _policy_action_key(record: DiagRecord) -> str:
    value = str(record.raw.get("policy_action") or "").strip().lower()
    return value if value in DECISION_TO_ID else record.final_decision


def _benign_label(record: DiagRecord) -> int:
    if "benign" in record.raw:
        return int(bool(record.raw["benign"]))
    return int(record.final_decision in ("allow", "warn"))

