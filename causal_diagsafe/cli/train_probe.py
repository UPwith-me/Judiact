from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from ..constants import DECISIONS, STAGES
from ..io_utils import ensure_out_dir, read_jsonl, write_json, write_jsonl
from ..models.cssb import CSSBHeads
from ..models.losses import supervised_cssb_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a masked H1 frozen-representation probe.")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--dev-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--head-kind", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--mlp-hidden-size", type=int)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--stage-loss-weight", type=float, default=1.0)
    parser.add_argument("--decision-loss-weight", type=float, default=0.5)
    parser.add_argument("--policy-action-loss-weight", type=float, default=0.2)
    parser.add_argument("--pair-jsonl")
    parser.add_argument("--pair-stability-tolerance", type=float, default=0.15)
    parser.add_argument("--shuffle-stage-labels", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    _set_seed(args.seed)
    out_dir = ensure_out_dir(args.out_dir)
    train_cache = _load_cache(args.train_cache)
    dev_cache = _load_cache(args.dev_cache)
    if args.shuffle_stage_labels:
        train_cache = _shuffle_stage_labels(train_cache, seed=args.seed)
    _validate_cache_compatibility(train_cache, dev_cache, args.hidden_size)

    model = CSSBHeads(
        args.hidden_size,
        head_kind=args.head_kind,
        mlp_hidden_size=args.mlp_hidden_size,
        dropout=0.1 if args.head_kind == "mlp" else 0.0,
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    train_loader = DataLoader(
        _dataset_from_cache(train_cache),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
    )
    pair_rows = read_jsonl(args.pair_jsonl) if args.pair_jsonl else []
    history: list[dict[str, Any]] = []
    best_score = float("-inf")

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses: list[float] = []
        for raw_batch in train_loader:
            stage_hidden, decision_hidden, labels, masks = _unpack_batch(raw_batch, args.device)
            output = model(stage_hidden, decision_hidden)
            cssb_losses = supervised_cssb_loss(
                output,
                labels,
                masks,
            )
            stage_loss = sum(cssb_losses[key] for key in ("R", "E", "F"))
            loss = args.stage_loss_weight * stage_loss
            loss = loss + args.decision_loss_weight * cssb_losses["decision"]
            loss = (
                loss
                + args.policy_action_loss_weight
                * cssb_losses["policy_action"]
            )
            loss = loss + cssb_losses["benign"] + cssb_losses["intent"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(float(loss.detach().cpu()))

        dev_metrics, predictions = _evaluate(
            model,
            dev_cache,
            device=args.device,
            batch_size=args.batch_size,
            pair_rows=pair_rows,
            pair_tolerance=args.pair_stability_tolerance,
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": sum(train_losses) / max(len(train_losses), 1),
            **dev_metrics,
        }
        history.append(epoch_row)
        score = float(dev_metrics["stage_macro_f1"]) + 0.25 * float(
            dev_metrics["decision_macro_f1"]
        )
        if score > best_score:
            best_score = score
            torch.save(
                {
                    "cssb_heads": model.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "metrics": dev_metrics,
                },
                Path(out_dir) / "h1_probe_best.pt",
            )
            write_jsonl(Path(out_dir) / "dev_predictions_best.jsonl", predictions)
        print(json.dumps(epoch_row, ensure_ascii=False), flush=True)

    result = {
        "format": "h1_masked_probe_report_v1",
        "args": vars(args),
        "train_label_coverage": _label_coverage(train_cache),
        "dev_label_coverage": _label_coverage(dev_cache),
        "best_score": best_score,
        "history": history,
        "claim_boundary": (
            "Frozen-representation probe. Pair metrics use only supplied strict pairs; "
            "static stage classification alone is not causal evidence."
        ),
    }
    write_json(Path(out_dir) / "h1_probe_metrics.json", result)
    print(f"probe training complete: {out_dir}", flush=True)


def _load_cache(path: str) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    required = {"stage_hidden", "decision_hidden", "labels", "metadata"}
    missing = required - set(obj)
    if missing:
        raise ValueError(f"{path} missing fields: {sorted(missing)}")
    return obj


def _validate_cache_compatibility(train: dict[str, Any], dev: dict[str, Any], hidden_size: int) -> None:
    for name, cache in (("train", train), ("dev", dev)):
        stage = cache["stage_hidden"]
        decision = cache["decision_hidden"]
        if stage.ndim != 3 or tuple(stage.shape[1:]) != (5, hidden_size):
            raise ValueError(f"{name} stage_hidden has incompatible shape {tuple(stage.shape)}")
        if decision.ndim != 2 or decision.shape[1] != hidden_size:
            raise ValueError(f"{name} decision_hidden has incompatible shape {tuple(decision.shape)}")
        if len(cache["metadata"]["id"]) != stage.shape[0]:
            raise ValueError(f"{name} metadata and hidden rows differ")


def _dataset_from_cache(cache: dict[str, Any]) -> TensorDataset:
    labels = cache["labels"]
    required = (
        "decision",
        "policy_action",
        "R",
        "R_mask",
        "E",
        "E_mask",
        "F",
        "F_mask",
        "benign",
        "benign_mask",
        "intent",
        "intent_mask",
    )
    missing = [key for key in required if key not in labels]
    if missing:
        raise ValueError(f"cache labels missing: {missing}")
    rows = cache["stage_hidden"].shape[0]
    decision_mask = labels.get("decision_mask", torch.ones(rows, dtype=torch.float32))
    policy_action_mask = labels.get(
        "policy_action_mask",
        torch.zeros(rows, dtype=torch.float32),
    )
    return TensorDataset(
        cache["stage_hidden"].float(),
        cache["decision_hidden"].float(),
        labels["decision"],
        decision_mask,
        labels["policy_action"],
        policy_action_mask,
        *[labels[key] for key in required[2:]],
    )


def _unpack_batch(
    batch: tuple[torch.Tensor, ...], device: str
) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    values = [item.to(device) for item in batch]
    stage_hidden, decision_hidden = values[:2]
    (
        decision,
        decision_mask,
        policy_action,
        policy_action_mask,
        r,
        r_mask,
        e,
        e_mask,
        f,
        f_mask,
        benign,
        benign_mask,
        intent,
        intent_mask,
    ) = values[2:]
    labels = {
        "decision": decision.long(),
        "policy_action": policy_action.long(),
        "R": r.float(),
        "E": e.long(),
        "F": f.float(),
        "benign": benign.float(),
        "intent": intent.long(),
    }
    masks = {
        "decision": decision_mask.float(),
        "policy_action": policy_action_mask.float(),
        "R": r_mask.float(),
        "E": e_mask.float(),
        "F": f_mask.float(),
        "benign": benign_mask.float(),
        "intent": intent_mask.float(),
    }
    return stage_hidden, decision_hidden, labels, masks


@torch.no_grad()
def _evaluate(
    model: CSSBHeads,
    cache: dict[str, Any],
    *,
    device: str,
    batch_size: int,
    pair_rows: list[dict[str, Any]],
    pair_tolerance: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    model.eval()
    loader = DataLoader(_dataset_from_cache(cache), batch_size=batch_size, shuffle=False)
    decision_pred: list[int] = []
    f_probs: list[list[float]] = []
    r_probs: list[list[float]] = []
    e_probs: list[list[list[float]]] = []
    r_pred: list[list[int]] = []
    e_pred: list[list[int]] = []
    f_pred: list[list[int]] = []
    for batch in loader:
        stage_hidden, decision_hidden, _, _ = _unpack_batch(batch, device)
        output = model(stage_hidden, decision_hidden)
        decision_pred.extend(output.decision_logits.argmax(dim=-1).cpu().tolist())
        probabilities = torch.sigmoid(output.f_logits)
        f_probs.extend(probabilities.cpu().tolist())
        f_pred.extend((probabilities >= 0.5).long().cpu().tolist())
        requirement_probabilities = torch.sigmoid(output.r_logits)
        evidence_probabilities = torch.softmax(output.e_logits, dim=-1)
        r_probs.extend(requirement_probabilities.cpu().tolist())
        e_probs.extend(evidence_probabilities.cpu().tolist())
        r_pred.extend((requirement_probabilities >= 0.5).long().cpu().tolist())
        e_pred.extend(evidence_probabilities.argmax(dim=-1).cpu().tolist())

    labels = cache["labels"]
    gold_decision = labels["decision"].long().tolist()
    metrics: dict[str, Any] = _decision_metrics(gold_decision, decision_pred)
    stage_scores: list[float] = []
    stage_detail: dict[str, Any] = {}
    for stage_index, stage in enumerate(STAGES):
        for key, stage_predictions in (("R", r_pred), ("F", f_pred)):
            mask = labels[f"{key}_mask"][:, stage_index].bool().tolist()
            gold = labels[key][:, stage_index].long().tolist()
            selected_gold = [g for g, keep in zip(gold, mask) if keep]
            selected_pred = [
                prediction[stage_index]
                for prediction, keep in zip(stage_predictions, mask)
                if keep
            ]
            f1 = _binary_f1(selected_gold, selected_pred)
            stage_detail[f"{stage}.{key}"] = {
                "count": len(selected_gold),
                "accuracy": _accuracy(selected_gold, selected_pred),
                "f1": f1,
            }
            if selected_gold:
                stage_scores.append(f1)
        mask = labels["E_mask"][:, stage_index].bool().tolist()
        gold = labels["E"][:, stage_index].long().tolist()
        selected_gold = [g for g, keep in zip(gold, mask) if keep]
        selected_pred = [p[stage_index] for p, keep in zip(e_pred, mask) if keep]
        e_f1 = _macro_f1(selected_gold, selected_pred, classes=(0, 1, 2))
        stage_detail[f"{stage}.E"] = {
            "count": len(selected_gold),
            "accuracy": _accuracy(selected_gold, selected_pred),
            "macro_f1": e_f1,
        }
        if selected_gold:
            stage_scores.append(e_f1)

    all_pass = sum(all(value == 0 for value in row) for row in f_pred) / max(len(f_pred), 1)
    multi_fail = sum(sum(row) > 1 for row in f_pred) / max(len(f_pred), 1)
    output_predictions: list[dict[str, Any]] = []
    ids = cache["metadata"]["id"]
    for index, sample_id in enumerate(ids):
        output_predictions.append(
            {
                "id": sample_id,
                "final_decision": DECISIONS[decision_pred[index]],
                "decision_id": decision_pred[index],
                "F_probability": f_probs[index],
                "R_probability": r_probs[index],
                "E_probability": e_probs[index],
                "F_prediction": f_pred[index],
                "R_prediction": r_pred[index],
                "E_prediction": e_pred[index],
            }
        )
    pair_metrics = _pair_metrics(
        output_predictions,
        pair_rows,
        tolerance=pair_tolerance,
    )
    metrics.update(
        {
            "stage_macro_f1": sum(stage_scores) / max(len(stage_scores), 1),
            "all_pass_rate": all_pass,
            "multi_fail_rate": multi_fail,
            "stage_detail": stage_detail,
            **pair_metrics,
        }
    )
    return metrics, output_predictions


def _decision_metrics(gold: list[int], pred: list[int]) -> dict[str, float]:
    refuse = DECISIONS.index("refuse")
    allow = DECISIONS.index("allow")
    safe_mask = [DECISIONS[value] in {"allow", "warn"} for value in gold]
    refuse_mask = [value == refuse for value in gold]
    over = sum(keep and guess == refuse for keep, guess in zip(safe_mask, pred))
    harmful = sum(keep and guess == allow for keep, guess in zip(refuse_mask, pred))
    return {
        "decision_accuracy": _accuracy(gold, pred),
        "decision_macro_f1": _macro_f1(gold, pred, classes=tuple(range(len(DECISIONS)))),
        "over_refusal_rate": over / max(sum(safe_mask), 1),
        "harmful_allow_rate": harmful / max(sum(refuse_mask), 1),
    }


def _pair_metrics(
    predictions: list[dict[str, Any]],
    pairs: list[dict[str, Any]],
    *,
    tolerance: float,
) -> dict[str, float | int]:
    by_id = {row["id"]: row for row in predictions}
    target_correct = 0
    target_flip = 0
    stable_f = 0
    stable_full = 0
    boundary = 0
    deltas: list[float] = []
    boundary_conditioned_deltas: list[float] = []
    used = 0
    directional = 0
    boundary_pairs = 0
    legacy_unmasked = 0
    for pair in pairs:
        left = by_id.get(str(pair.get("left_id") or ""))
        right = by_id.get(str(pair.get("right_id") or ""))
        stage = int(pair.get("target_stage_id", -1))
        if left is None or right is None or stage not in range(5):
            continue
        direction = float(pair.get("target_direction", 0))
        direction_mask = int(pair.get("target_direction_mask", 0) or 0) == 1
        if "target_direction_mask" not in pair:
            legacy_unmasked += 1
        delta: float | None = None
        if direction_mask and direction != 0:
            delta = direction * (
                float(left["F_probability"][stage])
                - float(right["F_probability"][stage])
            )
            deltas.append(delta)
            target_correct += int(delta > 0)
            target_flip += int(
                direction
                * (
                    int(left["F_prediction"][stage])
                    - int(right["F_prediction"][stage])
                )
                > 0
            )
            directional += 1
        stable_mask = list(pair.get("stable_non_target_mask") or [1, 1, 1, 1, 1])
        f_is_stable = all(
            not flag
            or abs(
                float(left["F_probability"][index])
                - float(right["F_probability"][index])
            )
            <= tolerance
            for index, flag in enumerate(stable_mask)
            if index != stage
        )
        full_state_is_stable = all(
            not flag
            or (
                abs(
                    float(left["F_probability"][index])
                    - float(right["F_probability"][index])
                )
                <= tolerance
                and abs(
                    float(left["R_probability"][index])
                    - float(right["R_probability"][index])
                )
                <= tolerance
                and max(
                    abs(float(left_value) - float(right_value))
                    for left_value, right_value in zip(
                        left["E_probability"][index],
                        right["E_probability"][index],
                    )
                )
                <= tolerance
            )
            for index, flag in enumerate(stable_mask)
            if index != stage
        )
        stable_f += int(f_is_stable)
        stable_full += int(full_state_is_stable)
        left_role = str(pair.get("left_role") or "")
        right_role = str(pair.get("right_role") or "")
        if {left_role, right_role} & {"unsafe"} and {
            left_role,
            right_role,
        } & {"safe", "helpful"}:
            unsafe = left if left_role == "unsafe" else right
            safe = right if left_role == "unsafe" else left
            boundary_success = (
                unsafe["final_decision"] == "refuse"
                and safe["final_decision"] != "refuse"
            )
            boundary += int(boundary_success)
            boundary_pairs += 1
            if boundary_success and delta is not None:
                boundary_conditioned_deltas.append(delta)
        used += 1
    return {
        "strict_pair_count": used,
        "target_supervised_pair_count": directional,
        "legacy_unmasked_pair_count": legacy_unmasked,
        "boundary_pair_count": boundary_pairs,
        "target_stage_direction_accuracy": target_correct / max(directional, 1),
        "target_stage_flip_rate": target_flip / max(directional, 1),
        "target_stage_contrast": sum(deltas) / max(len(deltas), 1),
        "target_stage_contrast_given_boundary_success": (
            sum(boundary_conditioned_deltas)
            / max(len(boundary_conditioned_deltas), 1)
        ),
        "non_target_invariance_rate": stable_full / max(used, 1),
        "non_target_F_invariance_rate": stable_f / max(used, 1),
        "pair_boundary_success": boundary / max(boundary_pairs, 1),
    }


def _shuffle_stage_labels(cache: dict[str, Any], *, seed: int) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    clone = {
        **cache,
        "labels": {key: value.clone() for key, value in cache["labels"].items()},
        "metadata": {key: list(value) for key, value in cache["metadata"].items()},
    }
    labels = clone["labels"]
    for key in ("R", "E", "F"):
        mask = labels[f"{key}_mask"].bool()
        for stage in range(5):
            indices = torch.nonzero(mask[:, stage], as_tuple=False).flatten()
            if len(indices) > 1:
                permutation = indices[torch.randperm(len(indices), generator=generator)]
                labels[key][indices, stage] = labels[key][permutation, stage]
    return clone


def _label_coverage(cache: dict[str, Any]) -> dict[str, Any]:
    labels = cache["labels"]
    coverage: dict[str, Any] = {
        key: [int(value) for value in labels[f"{key}_mask"].sum(dim=0).tolist()]
        for key in ("R", "E", "F")
    }
    rows = cache["stage_hidden"].shape[0]
    coverage.update(
        {
            "decision": int(labels.get("decision_mask", torch.ones(rows)).sum().item()),
            "policy_action": int(
                labels.get("policy_action_mask", torch.zeros(rows)).sum().item()
            ),
            "benign": int(labels["benign_mask"].sum().item()),
            "intent": int(labels["intent_mask"].sum().item()),
        }
    )
    return coverage


def _accuracy(gold: list[int], pred: list[int]) -> float:
    return sum(left == right for left, right in zip(gold, pred)) / max(len(gold), 1)


def _binary_f1(gold: list[int], pred: list[int]) -> float:
    tp = sum(g == 1 and p == 1 for g, p in zip(gold, pred))
    fp = sum(g == 0 and p == 1 for g, p in zip(gold, pred))
    fn = sum(g == 1 and p == 0 for g, p in zip(gold, pred))
    return 2 * tp / max(2 * tp + fp + fn, 1)


def _macro_f1(gold: list[int], pred: list[int], *, classes: tuple[int, ...]) -> float:
    scores = []
    for value in classes:
        tp = sum(g == value and p == value for g, p in zip(gold, pred))
        fp = sum(g != value and p == value for g, p in zip(gold, pred))
        fn = sum(g == value and p != value for g, p in zip(gold, pred))
        scores.append(2 * tp / max(2 * tp + fp + fn, 1))
    return sum(scores) / max(len(scores), 1)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
