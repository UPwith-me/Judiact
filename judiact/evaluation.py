"""Autonomous natural-split evaluation, retaining invalid outputs in denominators."""

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import time

import torch

from .audit_schema import audit_slots
from .reference_audit import DEFAULT_MAX_NEW_TOKENS, PROTOCOL_VERSION, canonical, source_identity, source_node
from .data_contract import DECISIONS, audit_target_withheld
from .reference_predictor import AdjudicationPredictor
from .io_utils import read_jsonl, write_json, write_jsonl
from .judibench_metrics import diagnostic_scores, mechanism_scores
from .review_files import review_file


def summarize_natural(rows, predictions, supervision=None):
    supervision = supervision or {}
    by_task, fields, invalid = {}, defaultdict(lambda: [0, 0]), []
    process_counts = Counter()
    witness_lengths = defaultdict(list)
    for task, name in (("request_decision", "request_action"), ("response_audit", "response_publishability")):
        selected = [row for row in rows if (row["task_type"] == "response_audit") == (task == "response_audit")]
        counts = [[0] * 5 for _ in DECISIONS]
        class_losses = defaultdict(list)
        withheld_actions = []
        for row in selected:
            metadata = supervision.get(row["sample_id"], {})
            withheld = metadata.get("withheld_fields", [])
            mask = metadata.get("target_mask", {})
            trace = predictions[row["sample_id"]]["trace"]
            process_counts[task + "/records"] += 1
            if "audit_complete" in trace:
                process_counts[task + "/audit_complete_denominator"] += 1
                process_counts[task + "/audit_incomplete"] += not trace["audit_complete"]
            witness_output = trace.get("nodes", {}).get("witness", {})
            if "tokens" in witness_output:
                witness_lengths[task].append(witness_output["tokens"])
            if "terminated" in witness_output:
                process_counts[task + "/witness_unterminated"] += not witness_output["terminated"]
            facts = trace.get("witness", {}).get("facts", [])
            signatures = [canonical({k: v for k, v in fact.items() if k != "id"}) for fact in facts]
            repeats = len(signatures) - len(set(signatures))
            if "witness" in trace.get("nodes", {}):
                process_counts[task + "/witness_repeated_fact_records"] += bool(repeats)
                process_counts[task + "/witness_repeated_facts"] += repeats
                process_counts[task + "/empty_facts"] += not trace.get("witness", {}).get("facts")
            if "links" in trace.get("nodes", {}):
                process_counts[task + "/empty_rule_links"] += not trace.get("links")
            basis = trace.get("nodes", {}).get("basis")
            if basis is not None:
                process_counts[task + "/basis_outputs"] += 1
                process_counts[task + "/basis_unterminated"] += not basis.get("terminated", True)
                process_counts[task + "/basis_error_outputs"] += bool(basis.get("error"))
            for evidence in trace.get("evidence", []):
                process_counts[task + "/evidence_blocks"] += 1
                process_counts[task + "/evidence_error_blocks"] += bool(evidence["errors"])
                process_counts[task + "/evidence_nonempty_blocks"] += bool(
                    evidence["spans"] or evidence["regions"])
            if mask.get("action") is False or audit_target_withheld("action", row["task_type"], withheld):
                withheld_actions.append(row["sample_id"])
            else:
                gold = row["gold"][name]
                output = trace["nodes"].get("action")
                value = trace["values"].get("action") if trace["valid"] else None
                counts[DECISIONS.index(gold)][DECISIONS.index(value) if value in DECISIONS else 4] += 1
                probability = 0.
                if trace["valid"] and output is not None:
                    if output["forced_by_contract"] is not None:
                        probability = float(gold == output["forced_by_contract"])
                    else:
                        probability = math.exp(output["log_probabilities"][DECISIONS.index(gold)])
                # Invalid predictions remain in every determined-label denominator.
                class_losses[gold].append(-math.log(max(probability, 1e-300)))
            if not trace["valid"]:
                invalid.append(row["sample_id"])
            for slot in audit_slots(row["model_input"], row["task_type"]):
                if slot.kind != "classify":
                    continue
                if trace.get("action_only") and slot.name != "action":
                    continue
                if mask.get(slot.name) is False or audit_target_withheld(slot.name, row["task_type"], withheld):
                    continue
                if slot.name.startswith(("R", "E", "F")) and len(slot.name) == 2:
                    target = row["gold"]["stage_states"]["S" + slot.name[1]][slot.name[0]]
                else:
                    key = {"action": name, "intent": "intent_class", "behavior": "safe_response_behavior"}.get(
                        slot.name, slot.name)
                    target = row["gold"][key]
                if target not in slot.labels:
                    continue
                fields[task + "/" + slot.name][0] += trace["valid"] and trace["values"].get(slot.name) == target
                fields[task + "/" + slot.name][1] += 1
        supports = [sum(line) for line in counts]
        selected_counts = [sum(line[i] for line in counts) for i in range(4)]
        scores = [2 * counts[i][i] / (supports[i] + selected_counts[i])
                  if supports[i] + selected_counts[i] else 0. for i in range(4)]
        n = sum(supports)
        by_task[task] = {
            "denominator": n, "accuracy": sum(counts[i][i] for i in range(4)) / n if n else None,
            "population_denominator": len(selected),
            "withheld_action_denominator": len(withheld_actions),
            "withheld_action_sample_ids": withheld_actions,
            "macro_F1": sum(scores) / 4 if n else None,
            "confusion_matrix": counts, "gold_order": list(DECISIONS),
            "prediction_order": list(DECISIONS) + ["invalid"],
            "NLL": sum(sum(v) for v in class_losses.values()) / n if n else None,
            "balanced_NLL": sum(sum(v) / len(v) for v in class_losses.values()) / len(class_losses)
                if class_losses else None,
            "class_denominators": {k: len(v) for k, v in class_losses.items()},
            "per_class": {label: {"precision": counts[i][i] / selected_counts[i] if selected_counts[i] else 0.,
                "recall": counts[i][i] / supports[i] if supports[i] else 0., "F1": scores[i],
                "support": supports[i], "predicted": selected_counts[i]}
                for i, label in enumerate(DECISIONS)},
            "invalid": sum(line[4] for line in counts),
        }
    return {"tasks": by_task, "invalid_sample_ids": invalid,
            "process_counts": dict(process_counts),
            "witness_lengths": {task: {"denominator": len(values), "median": sorted(values)[len(values) // 2],
                "p95": sorted(values)[int(.95 * (len(values) - 1))], "max": max(values)}
                for task, values in witness_lengths.items() if values},
            "repetition_note": "exact fact duplicates ignore assigned IDs; not a semantic truth or completeness check",
            "process_count_note": "empty facts/links and empty evidence are reported, not repaired or excluded",
            "field_accuracy": {name: {"correct": a, "denominator": b, "value": a / b}
                               for name, (a, b) in fields.items()},
            "protocol": PROTOCOL_VERSION,
            "criterion": "mean task, then mean observed action class, then mean conditional NLL",
            "annotation_uncertainty_handling": "explicit field masks only; determined-label score and full population/withheld denominators reported separately; not selected by prediction correctness",
            "invalid_output_handling": "kept; action probability zero; existing 1e-300 log floor"}


def evaluate_natural(model, processor, release, output, state, max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                     pool_size=32, class_batch_size=64, text_batch_size=32,
                     max_batch_tokens=65536, max_decode_tokens=327680, witness_frequency_penalty=.1,
                     compact_witness_json=True, predictor_type=AdjudicationPredictor,
                     protocol=PROTOCOL_VERSION, resume=False, split="dev", include_mechanism=False):
    output, release = Path(output), Path(release)
    output.mkdir(parents=True, exist_ok=True)
    natural_rows = [row for task in ("natural_request", "natural_response")
                    for row in read_jsonl(release / f"data/{task}/{split}.jsonl")]
    rows = ([row for task in ("core_request", "s5_response")
             for row in read_jsonl(release / f"data/{task}/{split}.jsonl")]
            if include_mechanism else natural_rows)
    natural_ids = {row["sample_id"] for row in natural_rows}
    metadata_path = review_file(release, "supervision.jsonl")
    supervision = {row["sample_id"]: row for row in read_jsonl(metadata_path)
                   if row["split"] == split} if metadata_path.exists() else {}
    saved = [[] for _ in range(state.num_processes)]
    if resume:
        for rank in range(state.num_processes):
            path = output / f"predictions.rank{rank}.jsonl"
            if path.exists():
                saved[rank] = read_jsonl(path)
    existing = {record["sample_id"]: record for worker in saved for record in worker}
    assert len(existing) == sum(map(len, saved)), "duplicate completed predictions"
    assert existing.keys() <= {row["sample_id"] for row in rows}, "completed predictions outside this split"
    predictor = predictor_type(model, processor, max_new_tokens,
        class_batch_size=class_batch_size, text_batch_size=text_batch_size,
        max_batch_tokens=max_batch_tokens, max_decode_tokens=max_decode_tokens, source_cache={},
        witness_frequency_penalty=witness_frequency_penalty, compact_witness_json=compact_witness_json)
    for row in rows:
        record = existing.get(row["sample_id"])
        if record is not None and "witness" in record["trace"]["nodes"]:
            trace = record["trace"]
            predictor.source_cache[trace["task_type"], source_identity(row["model_input"])] = trace.get(
                "draft_witness_output", trace["nodes"]["witness"])
    started = time.perf_counter()
    predictions = list(saved[state.process_index])
    costs = []
    for index in range(state.process_index, len(rows), state.num_processes):
        row = rows[index]
        node = source_node(row["model_input"], row["task_type"])
        length = predictor.encoder.prepare(node)["input_ids"].numel()
        cost = length * (2 + len(audit_slots(row["model_input"], row["task_type"])))
        costs.append((cost, index, length))
    if state.num_processes > 1:
        gathered = [None] * state.num_processes
        torch.distributed.all_gather_object(gathered, costs)
        costs = [row for worker in gathered for row in worker]
    bins, loads = [[] for _ in range(state.num_processes)], [0] * state.num_processes
    for cost, index, length in sorted(costs, reverse=True):
        owner = min(range(state.num_processes), key=loads.__getitem__)
        bins[owner].append(index)
        loads[owner] += cost
    lengths = {index: length for _, index, length in costs}
    pools = [[] for _ in bins]
    pending = 0
    for owner, indices in enumerate(bins):
        ordered = sorted(indices, key=lambda index: (
            rows[index]["task_type"], bool(rows[index]["model_input"]["images"]), -lengths[index]))
        for start in range(0, len(ordered), pool_size):
            batch = [rows[index] for index in ordered[start:start + pool_size]
                     if rows[index]["sample_id"] not in existing]
            if batch:
                # Resume assigns whole unfinished pools, retaining their batch shapes and order.
                pools[pending % state.num_processes if resume else owner].append(batch)
                pending += 1
    denominators = [len(saved[rank]) + sum(map(len, worker)) for rank, worker in enumerate(pools)]
    rank_key = "dev_rank" if split == "dev" else split + "_rank"
    print(json.dumps({rank_key: state.process_index, "resumed": len(predictions),
                      "pending": sum(map(len, pools[state.process_index]))}), flush=True)
    with (output / f"predictions.rank{state.process_index}.jsonl").open(
            "a" if resume else "w", encoding="utf-8") as handle:
        for batch in pools[state.process_index]:
            traces = predictor.predict_batch([r["model_input"] for r in batch], [r["task_type"] for r in batch])
            produced = [{"sample_id": row["sample_id"], "trace": trace} for row, trace in zip(batch, traces)]
            predictions.extend(produced)
            for record in produced:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({rank_key: state.process_index, "completed": len(predictions),
                              "denominator": denominators[state.process_index],
                              "elapsed_seconds": time.perf_counter() - started}), flush=True)
    state.wait_for_everyone()
    metrics = None
    if state.is_main_process:
        predictions = {row["sample_id"]: row for rank in range(state.num_processes)
                       for row in read_jsonl(output / f"predictions.rank{rank}.jsonl")}
        assert set(predictions) == {row["sample_id"] for row in rows}
        report = summarize_natural(natural_rows, predictions, supervision)
        report["diagnostic_macro_F1"] = diagnostic_scores(natural_rows, predictions, supervision)
        report["predicted_members"] = len(rows)
        if include_mechanism:
            extra_rows = [row for row in rows if row["sample_id"] not in natural_ids]
            report["non_natural_members"] = summarize_natural(extra_rows, predictions, supervision)
            report["non_natural_members"]["scope"] = "Additional members, not a count or score of causal groups"
            report["contextual_evaluation"] = mechanism_scores(release, split, rows, predictions, supervision)
        report["split"] = split
        report["supervision_metadata"] = str(metadata_path) if metadata_path.exists() else None
        report["protocol"] = protocol
        report["execution"] = {"pool_size": pool_size, "class_batch_size": class_batch_size,
            "text_batch_size": text_batch_size, "max_batch_tokens": max_batch_tokens,
            "max_decode_tokens": max_decode_tokens, "max_new_tokens": max_new_tokens,
            "greedy_free_text_probability": "not_recorded; categorical probabilities unchanged",
            "batching": "node type, modality and visible prefix length; no gold inputs",
            "source_reuse": "identical visible source and task only; never sampled paths"}
        report["execution"]["witness_frequency_penalty"] = witness_frequency_penalty
        report["execution"]["witness_repair"] = predictor.witness_repair
        report["execution"]["compact_witness_json"] = compact_witness_json
        report["execution"]["rank_denominators"] = denominators
        report["execution"]["sharding"] = "visible prefill length times declared node count; no gold or observed errors"
        if resume:
            report["execution"].update(resumed_records=[len(worker) for worker in saved],
                resume_assignment="whole unfinished pools distributed across ranks; completed outputs retained",
                runtime_scope="current invocation only; excludes work retained from the previous invocation")
        report["execution"]["accelerate_forward_autocast"] = hasattr(model, "_original_forward")
        if hasattr(predictor, "short_batch_size"):
            report["execution"]["short_review_generation"] = {
                "max_sequences": predictor.short_batch_size or text_batch_size,
                "max_prefill_and_decode_tokens": predictor.short_max_batch_tokens or max_batch_tokens,
                "max_new_tokens": predictor.encoder.short_max_new_tokens,
                "finished_row_compaction_after": predictor.encoder.short_compact_after,
                "candidate_head_projection": predictor.encoder.candidate_head_projection,
                "condition_text_frequency_penalty": .1,
                "compact_json": True}
        if hasattr(predictor, "action_objective"):
            report["criterion"] = "equal-task mean four-action Macro-F1; full per-class and field metrics retained"
            report["execution"].update(action_training_objective=predictor.action_objective,
                action_readout="raw four-action scores; no inference prior correction; "
                    "raw softmax is not claimed as a calibrated natural posterior",
                basis_output=predictor.include_basis)
        tasks = report["tasks"]
        metrics = {"eval_loss": sum(v["balanced_NLL"] for v in tasks.values()) / 2,
                   "eval_macro_F1": sum(v["macro_F1"] for v in tasks.values()) / 2,
                   "eval_request_macro_F1": tasks["request_decision"]["macro_F1"],
                   "eval_response_macro_F1": tasks["response_audit"]["macro_F1"],
                   "eval_population_NLL": sum(v["NLL"] for v in tasks.values()) / 2,
                   "eval_invalid": len(report["invalid_sample_ids"]),
                   "eval_runtime": time.perf_counter() - started}
        write_json(output / "summary.json", report)
        write_json(output / ("selection_metrics.json" if split == "dev" else "evaluation_metrics.json"), metrics)
        write_jsonl(output / "predictions.jsonl", [predictions[row["sample_id"]] for row in rows])
    if state.num_processes > 1:
        shared = [metrics]
        torch.distributed.broadcast_object_list(shared, src=0)
        metrics = shared[0]
    return metrics
