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


def build_gold_router_tensors(
    records: list[DiagRecord],
    *,
    ablations: set[str] | None = None,
) -> dict[str, torch.Tensor]:
    ablations = ablations or set()
    pR = torch.zeros(len(records), 5)
    pE = torch.zeros(len(records), 5, 3)
    pF = torch.zeros(len(records), 5)
    p_intent = torch.zeros(len(records), len(INTENT_TO_ID))
    p_policy_action = torch.full(
        (len(records), len(DECISION_TO_ID)),
        1.0 / len(DECISION_TO_ID),
    )
    policy_action_mask = torch.zeros(len(records))
    p_benign = torch.zeros(len(records))
    response_present = torch.zeros(len(records))
    decision = torch.zeros(len(records), dtype=torch.long)

    for i, record in enumerate(records):
        for j, stage in enumerate(STAGES):
            state = record.stage_states.normalized().states[stage]
            pR[i, j] = float(state.R if state.R is not None else 0)
            evidence = state.E or "uncertain"
            pE[i, j, EVIDENCE_TO_ID[evidence]] = 1.0
            pF[i, j] = float(state.F if state.F is not None else 0)
        intent_key = _intent_key(record)
        p_intent[i, INTENT_TO_ID[intent_key]] = 1.0
        action_key = _policy_action_key(record)
        if action_key is not None:
            p_policy_action[i].zero_()
            p_policy_action[i, DECISION_TO_ID[action_key]] = 1.0
            policy_action_mask[i] = 1.0
        p_benign[i] = float(_benign_label(record))
        response_present[i] = float(bool(record.response_candidate))
        decision[i] = DECISION_TO_ID[record.final_decision]
    if "requirement" in ablations:
        pR.fill_(0.0)
    if "evidence" in ablations:
        pE.zero_()
        pE[:, :, EVIDENCE_TO_ID["uncertain"]] = 1.0
    if "risk" in ablations:
        pF.fill_(0.0)
    if "intent" in ablations:
        p_intent.zero_()
        p_intent[:, INTENT_TO_ID["unknown"]] = 1.0
    if "policy_action" in ablations:
        p_policy_action.fill_(1.0 / len(DECISION_TO_ID))
        policy_action_mask.zero_()
    if "benign" in ablations:
        p_benign.fill_(0.5)
    return {
        "pR": pR,
        "pE": pE,
        "pF": pF,
        "p_intent": p_intent,
        "p_policy_action": p_policy_action,
        "policy_action_mask": policy_action_mask,
        "p_benign": p_benign,
        "response_present": response_present,
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
    ablations: set[str] | None = None,
    shuffle_train_stage: bool = False,
    seed: int = 0,
) -> tuple[MonotonicEGCR, dict[str, Any]]:
    train_records = [normalize_record(row) for row in train_rows]
    eval_records = [normalize_record(row) for row in eval_rows]
    train_tensors = build_gold_router_tensors(train_records, ablations=ablations)
    if shuffle_train_stage and len(train_records) > 1:
        generator = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(train_records), generator=generator)
        for key in ("pR", "pE", "pF"):
            train_tensors[key] = train_tensors[key][order]
    router = MonotonicEGCR().to(device)
    optimizer = torch.optim.AdamW(router.parameters(), lr=lr)
    dataset = TensorDataset(
        train_tensors["pR"],
        train_tensors["pE"],
        train_tensors["pF"],
        train_tensors["p_intent"],
        train_tensors["p_policy_action"],
        train_tensors["p_benign"],
        train_tensors["response_present"],
        train_tensors["decision"],
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    history = []
    for epoch in range(1, epochs + 1):
        losses = []
        router.train()
        for batch in loader:
            (
                pR,
                pE,
                pF,
                p_intent,
                p_policy_action,
                p_benign,
                response_present,
                decision,
            ) = [item.to(device) for item in batch]
            logits = router(
                EGCRSignals(
                    pR,
                    pE,
                    pF,
                    p_intent,
                    p_policy_action,
                    p_benign,
                    response_present=response_present,
                )
            )
            loss = router_loss(logits, decision) + 0.1 * brier_loss(logits, decision)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            metrics = evaluate_gold_router(
                router, eval_records, device=device, ablations=ablations
            )
            history.append({"epoch": epoch, "train_loss": sum(losses) / max(len(losses), 1), **metrics["decision_metrics"]})
    return router, {
        "history": history,
        "ablations": sorted(ablations or set()),
        "shuffle_train_stage": shuffle_train_stage,
        "seed": seed,
        **evaluate_gold_router(router, eval_records, device=device, ablations=ablations),
    }


@torch.no_grad()
def evaluate_gold_router(
    router: MonotonicEGCR,
    eval_records: list[DiagRecord],
    *,
    device: str = "cpu",
    ablations: set[str] | None = None,
) -> dict[str, Any]:
    tensors = build_gold_router_tensors(eval_records, ablations=ablations)
    signals = EGCRSignals(
        tensors["pR"].to(device),
        tensors["pE"].to(device),
        tensors["pF"].to(device),
        tensors["p_intent"].to(device),
        tensors["p_policy_action"].to(device),
        tensors["p_benign"].to(device),
        response_present=tensors["response_present"].to(device),
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
    return {
        "decision_metrics": decision_metrics(eval_records, predictions),
        "prediction_rows": [
            {
                "id": prediction.sample_id,
                "final_decision": prediction.final_decision,
                "policy_id": prediction.policy_id,
            }
            for prediction in predictions
        ],
    }


def _record_targets(record: DiagRecord) -> dict[str, Any]:
    targets = record.raw.get("targets")
    return targets if isinstance(targets, dict) else {}


def _intent_key(record: DiagRecord) -> str:
    targets = _record_targets(record)
    value = str(
        targets.get("intent_label")
        or record.raw.get("intent_label")
        or record.raw.get("intent")
        or ""
    ).strip().lower()
    if value in INTENT_TO_ID:
        return value
    return "unknown"


def _policy_action_key(record: DiagRecord) -> str | None:
    targets = _record_targets(record)
    raw_mask = (
        targets.get("policy_action_mask")
        if "policy_action_mask" in targets
        else record.raw.get("policy_action_mask")
    )
    if isinstance(raw_mask, str):
        raw_mask = raw_mask.strip().lower()
    if raw_mask not in (True, 1, 1.0, "1", "true", "yes"):
        return None
    value = (
        targets.get("policy_action")
        if "policy_action" in targets
        else record.raw.get("policy_action")
    )
    if isinstance(value, str):
        normalized = value.strip().lower()
        return normalized if normalized in DECISION_TO_ID else None
    if isinstance(value, (int, float)):
        id_to_decision = {index: name for name, index in DECISION_TO_ID.items()}
        return id_to_decision.get(int(value))
    return None


def _benign_label(record: DiagRecord) -> float:
    targets = _record_targets(record)
    if (
        int(targets.get("benign_mask", 0) or 0) == 1
        and targets.get("benign_source") == "explicit_manual"
    ):
        return float(int(bool(targets.get("benign"))))
    return 0.5
