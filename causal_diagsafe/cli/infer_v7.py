"""Run autonomous V7 audits from visible inputs, without annotation context."""

import argparse
import json
from pathlib import Path

import torch
from transformers import set_seed

from ..inference_v7 import AdjudicationPredictor
from ..io_utils import read_jsonl
from ..models.diagsafe_v3 import load_model_v3, load_processor_v3


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "inputs", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--class-batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    set_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    processor = load_processor_v3(args.model)
    model = load_model_v3(args.model, args.adapter, training=False).to(args.device).eval()
    predictor = AdjudicationPredictor(model, processor, args.max_new_tokens,
                                      class_batch_size=args.class_batch_size)
    rows = read_jsonl(args.inputs)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start:start + args.batch_size]
            traces = predictor.predict_batch([r.get("model_input", r) for r in batch],
                                             [r.get("task_type") for r in batch])
            for index, (row, trace) in enumerate(zip(batch, traces), start):
                handle.write(json.dumps({"sample_id": row.get("sample_id", str(index)),
                                         "trace": trace}, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
