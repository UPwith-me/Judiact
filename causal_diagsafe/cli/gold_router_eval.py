from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ..gold_router import train_gold_router
from ..io_utils import ensure_out_dir, read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Train/evaluate H1c gold-state oracle EGCR router.")
    parser.add_argument("--train", required=True, help="Gold train JSONL.")
    parser.add_argument("--eval", required=True, help="Gold eval JSONL.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    router, summary = train_gold_router(
        read_jsonl(args.train),
        read_jsonl(args.eval),
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        device=args.device,
    )
    out_dir = ensure_out_dir(args.out_dir)
    write_json(Path(out_dir) / "gold_router_summary.json", summary)
    torch.save({"egcr": router.state_dict(), "args": vars(args)}, Path(out_dir) / "gold_router.pt")
    print(f"gold-router evaluation complete: {out_dir}")


if __name__ == "__main__":
    main()

