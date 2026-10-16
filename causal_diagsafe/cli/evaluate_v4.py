"""Full dev and mechanism inference, four replicas and measured adaptive batches."""

import argparse
import json
import time
from pathlib import Path

from accelerate import PartialState

from ..data_contract_v1 import load_relations_v1, load_s2_quartets_v1
from ..evaluation_v4 import summarize
from ..inference_v4 import ContextualAuditPredictor
from ..io_utils import read_jsonl, write_json, write_jsonl
from ..training.v1_data import root_assignments


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "adapter", "release", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--initial-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--predictions-only", action="store_true")
    parser.add_argument("--protocol", choices=("v4", "v5", "v6"), default="v4")
    args = parser.parse_args()
    state = PartialState()
    release, output = Path(args.release), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(release / "data/core_request/dev.jsonl") + read_jsonl(release / "data/s5_response/dev.jsonl")
    if not args.predictions_only:
        predictor_class = ContextualAuditPredictor
        if args.protocol == "v5":
            from ..inference_v5 import CoherentAuditPredictor
            predictor_class = CoherentAuditPredictor
        elif args.protocol == "v6":
            from ..inference_v6 import GroundedAuditPredictor
            predictor_class = GroundedAuditPredictor
        predictor = predictor_class(args.model, args.adapter, device=state.device)
        started = time.perf_counter()
        with (output / f"predictions.rank{state.process_index}.jsonl").open("w", encoding="utf-8") as handle:
            for count, prediction in enumerate(predictor.iter_predictions(rows[state.process_index::state.num_processes],
                    args.initial_batch_size, args.max_batch_size), 1):
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                handle.flush()
                if count % 25 == 0:
                    print(json.dumps({"rank": state.process_index, "dev_processed": count,
                                      "elapsed_seconds": time.perf_counter() - started}), flush=True)
        write_json(output / f"batching.rank{state.process_index}.json", predictor.eval_batcher.summary())
        del predictor
    state.wait_for_everyone()
    if not state.is_main_process:
        return
    samples = {row["sample_id"]: row for row in rows}
    predictions = {}
    for rank in range(state.num_processes):
        for row in read_jsonl(output / f"predictions.rank{rank}.jsonl"):
            assert row["sample_id"] not in predictions
            predictions[row["sample_id"]] = row
    assert predictions.keys() == samples.keys()
    natural = {row["sample_id"] for task in ("natural_request", "natural_response")
               for row in read_jsonl(release / f"data/{task}/dev.jsonl")}
    edges = list(load_relations_v1(str(release / "data/causal/verified_dev.jsonl"), samples))
    quartets = list(load_s2_quartets_v1(str(release / "data/causal/verified_s2_quartets_dev.jsonl"), samples))
    report, groups = summarize(samples, predictions, natural, root_assignments(rows, release / "reviews"), edges, quartets)
    if args.protocol == "v5":
        from ..audit_v5 import PROTOCOL_VERSION
        report["method"] = PROTOCOL_VERSION
    elif args.protocol == "v6":
        from ..audit_v6 import PROTOCOL_VERSION
        report["method"] = PROTOCOL_VERSION
    report["settings"] = {"release": args.release, "adapter": args.adapter, "split": "dev",
        "hidden_tests_loaded": False, "do_sample": False, "complete_audit": True, "protocol": args.protocol,
        "initial_batch_size": args.initial_batch_size, "max_batch_size": args.max_batch_size,
        "batching": "four independent replicas; shape buckets; measured throughput and memory; no input truncation"}
    write_json(output / "summary.json", report)
    write_jsonl(output / "predictions.jsonl", (predictions[row["sample_id"]] for row in rows))
    write_jsonl(output / "mechanism_groups.jsonl", groups)
    write_jsonl(output / "decision_errors.jsonl", ({"sample_id": row["sample_id"], "model_input": row["model_input"],
        "gold": row["gold"][key], "predicted": predictions[row["sample_id"]][key]}
        for row in rows for key in ("response_publishability" if row["task_type"] == "response_audit" else "request_action",)
        if row["gold"][key] != predictions[row["sample_id"]][key]))
    print(json.dumps({"dev_members": len(rows), "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
