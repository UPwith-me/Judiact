from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from ..constants import DECISIONS
from ..io_utils import ensure_out_dir, write_json
from ..models.cssb import CSSBHeads
from ..models.egcr import MonotonicEGCR, build_signals_from_cssb
from ..models.losses import brier_loss, router_loss, supervised_cssb_loss


def main() -> None:
    parser = argparse.ArgumentParser(description="Train H1 cached hidden-state CSSB/EGCR probe.")
    parser.add_argument("--train-cache", required=True)
    parser.add_argument("--dev-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--hidden-size", type=int, required=True)
    parser.add_argument("--head-kind", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--router-loss-weight", type=float, default=1.0)
    parser.add_argument("--cal-loss-weight", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    out_dir = ensure_out_dir(args.out_dir)
    train_cache = _load_cache(args.train_cache)
    dev_cache = _load_cache(args.dev_cache)
    model = CSSBHeads(args.hidden_size, head_kind=args.head_kind).to(args.device)
    router = MonotonicEGCR().to(args.device)
    optimizer = torch.optim.AdamW([*model.parameters(), *router.parameters()], lr=args.lr)

    train_loader = DataLoader(
        _dataset_from_cache(train_cache), batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        router.train()
        losses = []
        for batch in train_loader:
            stage_hidden, decision_hidden, labels = _unpack_batch(batch, args.device)
            output = model(stage_hidden, decision_hidden)
            cssb_losses = supervised_cssb_loss(output, labels)
            signals = build_signals_from_cssb(output.probabilities())
            router_logits = router(signals)
            loss = (
                cssb_losses["total"]
                + args.router_loss_weight * router_loss(router_logits, labels["decision"])
                + args.cal_loss_weight * brier_loss(router_logits, labels["decision"])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_([*model.parameters(), *router.parameters()], 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        dev_metrics = _evaluate(model, router, dev_cache, args.device, args.batch_size)
        epoch_row = {
            "epoch": float(epoch),
            "train_loss": sum(losses) / max(len(losses), 1),
            **dev_metrics,
        }
        history.append(epoch_row)
        print(
            "epoch={epoch} train_loss={train_loss:.4f} dev_acc={dev_router_acc:.4f} "
            "dev_over_refusal={dev_over_refusal:.4f} dev_harmful_allow={dev_harmful_allow:.4f}".format(
                **epoch_row
            ),
            flush=True,
        )

    torch.save(
        {
            "cssb_heads": model.state_dict(),
            "egcr": router.state_dict(),
            "args": vars(args),
            "history": history,
        },
        Path(out_dir) / "h1_probe.pt",
    )
    write_json(Path(out_dir) / "h1_probe_metrics.json", {"history": history})
    print(f"probe training complete: {out_dir}")


def _load_cache(path: str) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu")
    required = {"stage_hidden", "decision_hidden", "labels"}
    missing = required - set(obj)
    if missing:
        raise ValueError(f"{path} missing fields: {sorted(missing)}")
    return obj


def _dataset_from_cache(cache: dict[str, Any]) -> TensorDataset:
    labels = cache["labels"]
    required_labels = ("decision", "R", "E", "F", "intent", "policy_action", "benign")
    missing = [key for key in required_labels if key not in labels]
    if missing:
        raise ValueError(f"cache labels missing: {missing}")
    return TensorDataset(
        cache["stage_hidden"].float(),
        cache["decision_hidden"].float(),
        labels["decision"].long(),
        labels["R"].float(),
        labels["E"].long(),
        labels["F"].float(),
        labels["intent"].long(),
        labels["policy_action"].long(),
        labels["benign"].float(),
    )


def _unpack_batch(batch: tuple[torch.Tensor, ...], device: str) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    stage_hidden, decision_hidden, decision, r, e, f, intent, policy_action, benign = [
        item.to(device) for item in batch
    ]
    labels = {
        "decision": decision,
        "R": r,
        "E": e,
        "F": f,
        "intent": intent,
        "policy_action": policy_action,
        "benign": benign,
    }
    return stage_hidden, decision_hidden, labels


@torch.no_grad()
def _evaluate(
    model: CSSBHeads,
    router: MonotonicEGCR,
    cache: dict[str, Any],
    device: str,
    batch_size: int,
) -> dict[str, float]:
    model.eval()
    router.eval()
    loader = DataLoader(_dataset_from_cache(cache), batch_size=batch_size, shuffle=False)
    golds: list[int] = []
    preds: list[int] = []
    for batch in loader:
        stage_hidden, decision_hidden, labels = _unpack_batch(batch, device)
        output = model(stage_hidden, decision_hidden)
        logits = router(build_signals_from_cssb(output.probabilities()), apply_masks=True)
        pred = logits.argmax(dim=-1).detach().cpu().tolist()
        gold = labels["decision"].detach().cpu().tolist()
        preds.extend(int(x) for x in pred)
        golds.extend(int(x) for x in gold)
    correct = sum(1 for g, p in zip(golds, preds) if g == p)
    refuse = DECISIONS.index("refuse")
    allow = DECISIONS.index("allow")
    safe_total = sum(1 for g in golds if DECISIONS[g] in ("allow", "warn"))
    refuse_total = sum(1 for g in golds if g == refuse)
    over_refusal = sum(1 for g, p in zip(golds, preds) if DECISIONS[g] in ("allow", "warn") and p == refuse)
    harmful_allow = sum(1 for g, p in zip(golds, preds) if g == refuse and p == allow)
    return {
        "dev_router_acc": correct / max(len(golds), 1),
        "dev_over_refusal": over_refusal / max(safe_total, 1),
        "dev_harmful_allow": harmful_allow / max(refuse_total, 1),
    }


if __name__ == "__main__":
    main()

