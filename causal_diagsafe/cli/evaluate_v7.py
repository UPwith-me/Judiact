"""Evaluate the fixed natural dev using autonomous V7 inference."""

import argparse
import json

import torch
from accelerate import PartialState
from transformers import set_seed

from ..evaluation_v7 import evaluate_natural
from ..models.diagsafe_v3 import load_model_v3, load_processor_v3


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "release", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    state = PartialState()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = load_processor_v3(args.model)
    model = load_model_v3(args.model, args.adapter, training=False).to(state.device).eval()
    metrics = evaluate_natural(model, processor, args.release, args.output, state, args.max_new_tokens)
    if state.is_main_process:
        print(json.dumps(metrics, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
