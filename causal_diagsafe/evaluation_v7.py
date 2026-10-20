"""Autonomous natural-dev selection, retaining invalid outputs in denominators."""

from collections import Counter, defaultdict
import math
from pathlib import Path
import time

import torch

from .audit_v4 import audit_slots
from .audit_v7 import PROTOCOL_VERSION
from .data_contract_v1 import DECISIONS_V1
from .inference_v7 import AdjudicationPredictor
from .io_utils import read_jsonl, write_json, write_jsonl


def summarize_natural(rows, predictions):
    by_task, fields, invalid = {}, defaultdict(lambda: [0, 0]), []
    process_counts = Counter()
    for task, name in (("request_decision", "request_action"), ("response_audit", "response_publishability")):
        selected = [row for row in rows if (row["task_type"] == "response_audit") == (task == "response_audit")]
        counts = [[0] * 5 for _ in DECISIONS_V1]
        class_losses = defaultdict(list)
        for row in selected:
            trace = predictions[row["sample_id"]]["trace"]
            process_counts[task + "/records"] += 1
            process_counts[task + "/empty_facts"] += not trace.get("witness", {}).get("facts")
            process_counts[task + "/empty_rule_links"] += not trace.get("links")
            for evidence in trace.get("evidence", []):
                process_counts[task + "/evidence_blocks"] += 1
                process_counts[task + "/evidence_error_blocks"] += bool(evidence["errors"])
                process_counts[task + "/evidence_nonempty_blocks"] += bool(
                    evidence["spans"] or evidence["regions"])
            gold = row["gold"][name]
            output = trace["nodes"].get("action")
            value = trace["values"].get("action") if trace["valid"] else None
            counts[DECISIONS_V1.index(gold)][DECISIONS_V1.index(value) if value in DECISIONS_V1 else 4] += 1
            probability = 0.
            if trace["valid"] and output is not None:
                if output["forced_by_contract"] is not None:
                    probability = float(gold == output["forced_by_contract"])
                else:
                    probability = math.exp(output["log_probabilities"][DECISIONS_V1.index(gold)])
            # Same numerical floor as V6. Missing output is zero probability,
            # not substituted review or removal from the evaluation denominator.
            class_losses[gold].append(-math.log(max(probability, 1e-300)))
            if not trace["valid"]:
                invalid.append(row["sample_id"])
            for slot in audit_slots(row["model_input"], row["task_type"]):
                if slot.kind != "classify":
                    continue
                if slot.name.startswith(("R", "E", "F")) and len(slot.name) == 2:
                    target = row["gold"]["stage_states"]["S" + slot.name[1]][slot.name[0]]
                else:
                    key = {"action": name, "intent": "intent_class", "behavior": "safe_response_behavior"}.get(
                        slot.name, slot.name)
                    target = row["gold"][key]
                fields[task + "/" + slot.name][0] += trace["values"].get(slot.name) == target
                fields[task + "/" + slot.name][1] += 1
        supports = [sum(line) for line in counts]
        selected_counts = [sum(line[i] for line in counts) for i in range(4)]
        scores = [2 * counts[i][i] / (supports[i] + selected_counts[i])
                  if supports[i] + selected_counts[i] else 0. for i in range(4)]
        n = len(selected)
        by_task[task] = {
            "denominator": n, "accuracy": sum(counts[i][i] for i in range(4)) / n if n else None,
            "macro_F1": sum(scores) / 4 if n else None,
            "confusion_matrix": counts, "gold_order": list(DECISIONS_V1),
            "prediction_order": list(DECISIONS_V1) + ["invalid"],
            "NLL": sum(sum(v) for v in class_losses.values()) / n if n else None,
            "balanced_NLL": sum(sum(v) / len(v) for v in class_losses.values()) / len(class_losses)
                if class_losses else None,
            "class_denominators": {k: len(v) for k, v in class_losses.items()},
            "invalid": sum(line[4] for line in counts),
        }
    return {"tasks": by_task, "invalid_sample_ids": invalid,
            "process_counts": dict(process_counts),
            "process_count_note": "empty facts/links and empty evidence are reported, not repaired or excluded",
            "field_accuracy": {name: {"correct": a, "denominator": b, "value": a / b}
                               for name, (a, b) in fields.items()},
            "protocol": PROTOCOL_VERSION,
            "criterion": "mean task, then mean observed action class, then mean conditional NLL",
            "invalid_output_handling": "kept; action probability zero; existing 1e-300 log floor"}


def evaluate_natural(model, processor, release, output, state, max_new_tokens=2048):
    output, release = Path(output), Path(release)
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for task in ("natural_request", "natural_response")
            for row in read_jsonl(release / f"data/{task}/dev.jsonl")]
    predictor = AdjudicationPredictor(model, processor, max_new_tokens, class_batch_size=16)
    started = time.perf_counter()
    predictions = []
    local = rows[state.process_index::state.num_processes]
    for start in range(0, len(local), 8):
        batch = local[start:start + 8]
        traces = predictor.predict_batch([r["model_input"] for r in batch], [r["task_type"] for r in batch])
        predictions.extend({"sample_id": row["sample_id"], "trace": trace}
                           for row, trace in zip(batch, traces))
    write_jsonl(output / f"predictions.rank{state.process_index}.jsonl", predictions)
    state.wait_for_everyone()
    metrics = None
    if state.is_main_process:
        predictions = {row["sample_id"]: row for rank in range(state.num_processes)
                       for row in read_jsonl(output / f"predictions.rank{rank}.jsonl")}
        assert set(predictions) == {row["sample_id"] for row in rows}
        report = summarize_natural(rows, predictions)
        tasks = report["tasks"]
        metrics = {"eval_loss": sum(v["balanced_NLL"] for v in tasks.values()) / 2,
                   "eval_population_NLL": sum(v["NLL"] for v in tasks.values()) / 2,
                   "eval_invalid": len(report["invalid_sample_ids"]),
                   "eval_runtime": time.perf_counter() - started}
        write_json(output / "summary.json", report)
        write_json(output / "selection_metrics.json", metrics)
        write_jsonl(output / "predictions.jsonl", [predictions[row["sample_id"]] for row in rows])
    if state.num_processes > 1:
        shared = [metrics]
        torch.distributed.broadcast_object_list(shared, src=0)
        metrics = shared[0]
    return metrics
