"""Full natural and mechanism dev evaluation with visible-only inference."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from accelerate import PartialState

from ..data_contract_v1 import load_relations_v1, load_s2_quartets_v1
from ..evaluation_v1 import summarize
from ..inference_v1 import DiagnosticPredictor
from ..io_utils import read_jsonl, write_json, write_jsonl
from ..training.v1_data import root_assignments


def main(predictor_class=DiagnosticPredictor, summarizer=summarize):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--release", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--predictions-only", action="store_true", help="Recompute metrics from existing predictions.")
    args = parser.parse_args()
    state = PartialState()
    release, directory = Path(args.release), Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(release / "data/core_request/dev.jsonl") + read_jsonl(release / "data/s5_response/dev.jsonl")
    samples = {row["sample_id"]: row for row in rows}
    assert len(samples) == len(rows) and all(row["split"] == "dev" for row in rows)
    shard = directory / f"predictions.rank{state.process_index}.jsonl"
    if not args.predictions_only:
        predictor = predictor_class(args.model, args.adapter, device=state.device)
        started = time.perf_counter()
        with shard.open("w", encoding="utf-8") as handle:
            for index in range(state.process_index, len(rows), state.num_processes):
                row = rows[index]
                before = time.perf_counter()
                prediction = predictor.predict(row["model_input"], row["sample_id"])
                prediction["timing"]["wall_seconds_with_serialization"] = time.perf_counter() - before
                handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
                if (index // state.num_processes + 1) % 100 == 0:
                    handle.flush()
                    print(json.dumps({"rank": state.process_index, "processed": index // state.num_processes + 1,
                                      "elapsed_seconds": time.perf_counter() - started}), flush=True)
        del predictor
    state.wait_for_everyone()
    if not state.is_main_process:
        return
    predictions = {}
    for rank in range(state.num_processes):
        for row in read_jsonl(directory / f"predictions.rank{rank}.jsonl"):
            assert row["sample_id"] not in predictions
            predictions[row["sample_id"]] = row
    assert predictions.keys() == samples.keys()
    write_jsonl(directory / "predictions.jsonl", (predictions[row["sample_id"]] for row in rows))
    natural_ids = {row["sample_id"] for task in ("natural_request", "natural_response")
                   for row in read_jsonl(release / "data" / task / "dev.jsonl")}
    assert natural_ids <= samples.keys()
    roots = root_assignments(rows, args.reviews)
    edges = list(load_relations_v1(str(release / "data/causal/verified_dev.jsonl"), samples))
    quartets = list(load_s2_quartets_v1(str(release / "data/causal/verified_s2_quartets_dev.jsonl"), samples))
    raw_predictions = {key: {**value, "action_probabilities": value["uncalibrated_action_probabilities"]}
                       for key, value in predictions.items()}
    report, group_rows = summarizer(samples, raw_predictions, natural_ids, roots, edges, quartets)
    report["settings"] = {"adapter": args.adapter, "release": args.release,
                          "split": "dev", "hidden_tests_loaded": False,
                          "actions": "argmax; primary metrics uncalibrated for ABC comparability"}
    if "joint_method" in report:
        report["settings"]["actions"] = "joint action marginal argmax; no separate action temperature scaling"
    if (Path(args.adapter) / "temperatures.json").exists():
        from ..evaluation_v1 import task_metrics
        report["natural_calibrated"] = {key: task_metrics(selected, predictions, roots, key)
            for key in ("request_action", "response_publishability")
            if (selected := [row for row in rows if row["sample_id"] in natural_ids and row["gold"][key] is not None])}
    report["latency"] = {"note": "batch1 dev forward, includes processor for wall time; no answer generation",
                         "by_task": {}}
    for task in ("request_decision", "response_audit"):
        members = [item for item in predictions.values() if item["task"] == task]
        report["latency"]["by_task"][task] = {
            key: {"denominator": len(members), "mean": float(np.mean(values)),
                  "P50": float(np.quantile(values, .5)), "P95": float(np.quantile(values, .95))}
            for key in ("gpu_forward_seconds", "input_and_forward_seconds", "wall_seconds_with_serialization")
            if (values := [item["timing"][key] for item in members])}
    write_json(directory / "summary.json", report)
    write_jsonl(directory / "mechanism_groups.jsonl", group_rows)
    errors = []
    for row in rows:
        key = "response_publishability" if row["task_type"] == "response_audit" else "request_action"
        prediction = predictions[row["sample_id"]]
        if prediction[key] != row["gold"][key]:
            errors.append({"sample_id": row["sample_id"], "group_id": row["group_id"],
                           "task": key, "gold": row["gold"][key], "predicted": prediction[key],
                           "probabilities": prediction["action_probabilities"], "model_input": row["model_input"]})
    write_jsonl(directory / "decision_errors.jsonl", errors)
    print(json.dumps({"output": str(directory), "natural": {key: {
        name: report["subsets"]["natural"][key][name] for name in ("denominator", "accuracy", "macro_F1", "NLL")}
        for key in ("request_action", "response_publishability")}}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
