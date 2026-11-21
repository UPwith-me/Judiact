"""Frozen full-train confusion from the selected autonomous short-review model."""

import argparse
from datetime import timedelta
import json
from pathlib import Path
import time

import torch
from accelerate import PartialState

from ..short_review import PROTOCOL_VERSION
from ..checkpoint import read_checkpoint
from ..short_predictor import ShortReviewPredictor
from ..io_utils import write_json
from ..training.action_cost import action_population, make_cost_snapshot
from ..training.shared_data import SharedAuditData


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "adapter", "release", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bank", type=Path, help="Train-only reference bank (defaults to the bundle's bank)")
    parser.add_argument("--max-batch-tokens", type=int, default=65536)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--short-batch-size", type=int, default=64)
    parser.add_argument("--short-max-batch-tokens", type=int, default=131072)
    args = parser.parse_args()

    from ..models.backbone import load_model, load_processor

    state = PartialState(timeout=timedelta(seconds=7200))
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = read_checkpoint(args.adapter)
    if config.get("phase") != "sft":
        raise ValueError("Compute costs with the selected SFT checkpoint")
    bank = args.bank or args.release / "process_references/bank.rank0.jsonl"
    data = SharedAuditData(args.release, bank, "natural", True, False)
    population = action_population(data)
    processor = load_processor(args.model)
    model = load_model(args.model, args.adapter, training=False,
                          attn_implementation="flash_attention_2").to(state.device).eval()
    predictor = ShortReviewPredictor(model, processor, short_max_new_tokens=config["short_max_new_tokens"],
        class_batch_size=64, text_batch_size=args.batch_size, max_batch_tokens=args.max_batch_tokens,
        short_batch_size=args.short_batch_size, short_max_batch_tokens=args.short_max_batch_tokens,
        action_objective=config["action_objective"]["objective"])
    predictor.encoder.vision.maximum = 128
    owners, loads = [[] for _ in range(state.num_processes)], [0] * state.num_processes
    costs = [(sum(len(str(v)) for k, v in data.cases[sid]["model_input"].items() if k != "images") +
              6000 * len(data.cases[sid]["model_input"]["images"]), sid) for sid in population]
    for cost, sid in sorted(costs, reverse=True):
        rank = min(range(state.num_processes), key=loads.__getitem__)
        owners[rank].append(sid)
        loads[rank] += cost
    args.output.mkdir(parents=True, exist_ok=True)
    assigned = owners[state.process_index]
    started = time.perf_counter()
    with (args.output / f"predictions.rank{state.process_index}.jsonl").open("w", encoding="utf-8") as stream:
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for start in range(0, len(assigned), args.batch_size):
                ids = assigned[start:start + args.batch_size]
                cases = [data.cases[sid] for sid in ids]
                traces = predictor.predict_batch([c["model_input"] for c in cases],
                    [c["task_type"] for c in cases], include_evidence=False)
                for sid, trace in zip(ids, traces):
                    stream.write(json.dumps({"sample_id": sid,
                        "log_probabilities": trace["nodes"]["action"]["log_probabilities"],
                        "short_review_errors": [o["error"] for o in trace["short_review"].values()]}) + "\n")
                stream.flush()
                print(json.dumps({"phase": "frozen_short_train_readout", "rank": state.process_index,
                    "members": start + len(ids), "rank_total": len(assigned),
                    "seconds": time.perf_counter() - started}), flush=True)
    state.wait_for_everyone()
    if state.is_main_process:
        predictions = {}
        for rank in range(state.num_processes):
            for line in (args.output / f"predictions.rank{rank}.jsonl").read_text().splitlines():
                row = json.loads(line)
                assert row["sample_id"] not in predictions
                predictions[row["sample_id"]] = row["log_probabilities"]
        snapshot = make_cost_snapshot(population, predictions, args.adapter, args.release)
        snapshot["readout"] = {"protocol": PROTOCOL_VERSION, "world_size": state.num_processes,
            "elapsed_seconds": time.perf_counter() - started, "dropout": False,
            "short_batch_size": args.short_batch_size,
            "short_max_batch_tokens": args.short_max_batch_tokens,
            "population": "complete train, single fixed selected checkpoint, autonomous greedy short context",
            "hidden_test_read": False}
        write_json(args.output / "action_costs.json", snapshot)
        print(json.dumps({"phase": "fixed_short_costs_complete", "members": len(predictions)}), flush=True)
    state.wait_for_everyone()


if __name__ == "__main__":
    main()
