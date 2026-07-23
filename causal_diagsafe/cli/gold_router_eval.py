from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..gold_router import train_gold_router
from ..io_utils import ensure_out_dir, read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate H1c gold-state oracle EGCR router.")
    parser.add_argument("--train", required=True, help="Gold train JSONL.")
    parser.add_argument("--eval", required=True, help="Gold eval JSONL.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--ablate",
        default="",
        help="Comma-separated signals: requirement,evidence,risk,intent,policy_action,benign.",
    )
    parser.add_argument("--shuffle-train-stage", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    ablations = {item.strip() for item in args.ablate.split(",") if item.strip()}

    router, summary = train_gold_router(
        read_jsonl(args.train),
        read_jsonl(args.eval),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
        ablations=ablations,
        shuffle_train_stage=args.shuffle_train_stage,
        seed=args.seed,
    )
    out_dir = ensure_out_dir(args.out_dir)
    prediction_rows = summary.pop("prediction_rows")
    write_json(Path(out_dir) / "gold_router_summary.json", summary)
    write_jsonl(Path(out_dir) / "gold_router_predictions.jsonl", prediction_rows)
    torch.save({"egcr": router.state_dict(), "args": vars(args)}, Path(out_dir) / "gold_router.pt")
    print(f"gold-router evaluation complete: {out_dir}")


if __name__ == "__main__":
    main()
