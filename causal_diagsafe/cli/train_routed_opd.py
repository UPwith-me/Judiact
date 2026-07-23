from __future__ import annotations

import argparse
import json

from ..training.routed_opd import RoutedOPDConfig, train_routed_opd


def main() -> None:
    parser = argparse.ArgumentParser(description="Train clean decision-routed multimodal OPD.")
    parser.add_argument("--train-jsonl", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--student-init-adapter", required=True)
    parser.add_argument("--safety-teacher-adapter", required=True)
    parser.add_argument("--helpful-teacher-adapter", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["text_only", "multimodal"], default="multimodal")
    parser.add_argument("--max-steps", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--rollout-tokens", type=int, default=96)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--opd-weight", type=float, default=1.0)
    parser.add_argument("--anchor-sft-weight", type=float, default=0.35)
    parser.add_argument("--repair-sft-weight", type=float, default=0.70)
    parser.add_argument("--margin-weight", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=127)
    parser.add_argument("--log-every", type=int, default=2)
    args = parser.parse_args()

    config = RoutedOPDConfig(**vars(args))
    summary = train_routed_opd(config)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
