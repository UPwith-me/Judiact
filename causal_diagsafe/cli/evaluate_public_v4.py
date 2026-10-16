"""The same locked public tasks/metrics as external guards, without retraining them."""

import argparse
import json
from pathlib import Path

from ..eval_batching import EvalBatcher, kv_bytes_per_token
from ..external_eval_metrics import summarize_benchmark
from ..inference_v4 import ContextualAuditPredictor
from ..io_utils import read_jsonl, write_json, write_jsonl


class ContextualGuard:
    def __init__(self, model, adapter, device, initial_batch_size, max_batch_size, protocol="v4"):
        predictor_class = ContextualAuditPredictor
        if protocol == "v5":
            from ..inference_v5 import CoherentAuditPredictor
            predictor_class = CoherentAuditPredictor
        elif protocol == "v6":
            from ..inference_v6 import GroundedAuditPredictor
            predictor_class = GroundedAuditPredictor
        self.predictor = predictor_class(model, adapter, device=device)
        self.model, self.processor = self.predictor.model, self.predictor.processor
        self.family = {"v4": "contextual_v4", "v5": "coherent_v5", "v6": "grounded_v6"}[protocol]
        self.protocol, self.device = "shared_deck", device
        self.eval_batcher = EvalBatcher(devices=[device], initial_batch_size=initial_batch_size,
            max_batch_size=max_batch_size, generation_tokens=2048, kv_bytes_per_token=kv_bytes_per_token(self.model),
            image_max_pixels=self.processor.image_processor.size["longest_edge"])

    def predict_batch(self, visibles, tasks=None):
        values = self.predictor.predict_batch(visibles)
        result = []
        for visible, value in zip(visibles, values):
            response = visible["response_candidate"] is not None
            action = value["response_publishability" if response else "request_action"]
            probabilities = value["action_probabilities"]
            timing = value["timing"]
            result.append({"status": "ok", "harmful": action != "allow" if response else action in ("review", "refuse"),
                "score": 1 - probabilities["allow"] if response else probabilities["review"] + probabilities["refuse"],
                "score_kind": "policy_hold_not_native_harm", "action": action, "action_probabilities": probabilities,
                "input_tokens": value["input_tokens"], "output_tokens": value["output_tokens"],
                "timing": {"wall_seconds": timing["wall_seconds_with_serialization"],
                           "batch_wall_seconds": timing["batch_wall_seconds"], "batch_size": timing["batch_size"],
                           "semantics": "amortized complete-audit batch service time, not isolated latency"},
                "complete_audit": value})
        return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "adapter", "config", "data", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--initial-batch-size", type=int, default=4)
    parser.add_argument("--max-batch-size", type=int, default=64)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--protocol", choices=("v4", "v5", "v6"), default="v4")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tasks = args.tasks or list(dict.fromkeys(config["primary_tasks"] + config["extended_tasks"]))
    tasks = [task for task in tasks if task not in config.get("deferred_tasks", {})]
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if args.merge_only:
        from ..external_lmms import native_result
        summary = {}
        for task in tasks:
            directory = output / task
            docs = read_jsonl(Path(args.data) / (task + ".jsonl"))
            predictions = {}
            for rank in range(args.world_size):
                for row in read_jsonl(directory / f"outcomes.rank{rank}.jsonl"):
                    assert row["sample_id"] not in predictions
                    predictions[row["sample_id"]] = row
            assert predictions.keys() == {doc["sample_id"] for doc in docs}, (task, "public denominator mismatch")
            rows = [native_result(doc, [json.dumps(predictions[doc["sample_id"]])])["native_safety"] for doc in docs]
            write_jsonl(directory / "outputs.jsonl", rows)
            report = {"status": "evaluated", "partial_probe": False, "protocol": "shared_deck",
                "metrics": summarize_benchmark(rows),
                "interpretation": "Same fixed policy-hold proxy as v3/shared-deck baselines; not native harm or system ASR."}
            write_json(directory / "summary.json", report)
            summary[task] = report
        write_json(output / "summary.json", summary)
        return
    from lmms_eval import evaluator
    from ..external_lmms import NativeGuardLM, NativeSafetyTask
    guard = ContextualGuard(args.model, args.adapter, args.device, args.initial_batch_size, args.max_batch_size, args.protocol)
    write_json(output / f"settings.rank{args.rank}.json", {"adapter": args.adapter, "model": args.model,
        "framework": "existing lmms-eval native task transport and external metrics", "protocol": "shared_deck",
        "complete_audit": True, "do_sample": False, "rank": args.rank, "world_size": args.world_size,
        "audit_protocol": args.protocol,
        "native_gold_note": "Native benchmark labels; no fabricated REF or four-action gold; warn permits request assistance but never unchanged response publication."})
    for task in tasks:
        directory = output / task
        directory.mkdir(parents=True, exist_ok=True)
        docs = read_jsonl(Path(args.data) / (task + ".jsonl"))[args.rank::args.world_size]
        lm = NativeGuardLM(guard, directory / f"predictions.rank{args.rank}.jsonl")
        benchmark = NativeSafetyTask(task, docs)
        lm.task_dict = {task: benchmark.dataset}
        evaluator.evaluate(lm=lm, task_dict={task: benchmark}, limit=None, log_samples=False,
                           bootstrap_iters=0, process_docs_parallel=1)
        # Keep observed missing/error outcomes in the denominator, outside the resumable success cache.
        write_jsonl(directory / f"outcomes.rank{args.rank}.jsonl",
                    (lm.current[doc["sample_id"]] for doc in docs))
        print(json.dumps({"public_task": task, "rank": args.rank, "members": len(docs)}), flush=True)


if __name__ == "__main__":
    main()
