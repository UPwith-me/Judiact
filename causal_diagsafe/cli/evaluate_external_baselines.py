"""Full native public splits through lmms-eval; do not train any baseline."""

import argparse
import importlib.metadata
import json
import traceback
from pathlib import Path

from ..external_benchmark_data import source_metadata
from ..external_eval_io import record_settings
from ..external_eval_metrics import summarize_benchmark
from ..external_guard_models import NativeGuard, model_directory
from ..io_utils import read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", help="Selected current-v3 adapter, for the matched shared-deck comparison only.")
    parser.add_argument("--protocol", choices=("native", "shared_deck"), default="native")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--predictions-only", action="store_true", help="Rebuild metrics from saved outputs without loading a model.")
    parser.add_argument("--limit", type=int, help="Integration probe only; never labeled a full benchmark result.")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    spec = next(item for item in config["models"] + config.get("method_models", []) if item["name"] == args.model)
    if args.protocol not in spec["protocols"]:
        raise ValueError(f"No released {args.protocol} interface for {args.model}")
    if spec["family"] == "diagsafe_v3" and not args.adapter:
        raise ValueError("A trained-v3 comparison requires its actual selected adapter, not the base model")
    tasks = args.tasks or list(dict.fromkeys(config["primary_tasks"] + config["extended_tasks"]))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    settings = {"protocol_version": config["protocol_version"], "model": spec, "protocol": args.protocol,
                "model_source": source_metadata(model_directory(args.root, "qwen3vl4b" if args.adapter else args.model)),
                "adapter": args.adapter,
                "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "lmms-eval")},
                "do_sample": False, "limit": args.limit, "device": args.device,
                "input_preprocessing": "Released processor for native; v3 processor for matched Qwen shared-deck.",
                "native_gold_note": "Benchmark native labels, never fabricated R/E/F or four-action gold."}
    if not args.predictions_only:
        record_settings(output, settings)
    predictor = None if args.predictions_only else NativeGuard(args.root, spec, args.protocol, args.device, adapter=args.adapter)
    from lmms_eval import evaluator
    from ..external_lmms import NativeGuardLM, NativeSafetyTask, native_result

    summary_file = output / "summary.json"
    summaries = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    if not args.predictions_only:
        tasks = list(dict.fromkeys([name for name, report in summaries.items()
                                   if report["status"] == "failed"] + tasks))
    tasks = [task for task in tasks if task not in config.get("deferred_tasks", {})]
    inventory_file = Path(args.data) / "inventory.json"
    inventory = json.loads(inventory_file.read_text()) if inventory_file.exists() else {}
    for task in tasks:
        directory = output / task
        directory.mkdir(parents=True, exist_ok=True)
        if not (Path(args.data) / (task + ".jsonl")).exists():
            summaries[task] = {"status": "unavailable_data"}
            write_json(output / "summary.json", summaries)
            continue
        try:
            docs = read_jsonl(Path(args.data) / (task + ".jsonl"))
            if args.limit is not None:
                docs = docs[:args.limit]
            lm = NativeGuardLM(predictor, directory / "predictions.jsonl")
            benchmark = NativeSafetyTask(task, docs)
            lm.task_dict = {task: benchmark.dataset}
            evaluator.evaluate(lm=lm, task_dict={task: benchmark},
                               limit=None, log_samples=False, bootstrap_iters=0, process_docs_parallel=1)
            rows = [native_result(doc, [json.dumps(lm.current[doc["sample_id"]])])["native_safety"] for doc in docs]
            write_jsonl(directory / "outputs.jsonl", rows)
            report = {"status": "evaluated", "partial_probe": args.limit is not None,
                      "partial_source": inventory.get(task, {}).get("partial_source", False),
                      "data_inventory": inventory.get(task, {}),
                      "protocol": args.protocol, "metrics": summarize_benchmark(rows)}
            if args.protocol == "shared_deck":
                report["interpretation"] = "Policy automatic hold versus native harm is a proxy comparison, not native harmfulness or system ASR."
            write_json(directory / "summary.json", report)
            summaries[task] = report
        except Exception as exc:
            # Keep an actual traceback, finish independent benchmarks, then report the failed branch.
            traceback.print_exc()
            summaries[task] = {"status": "failed", "error_type": type(exc).__name__, "error": str(exc)[:2000]}
        write_json(output / "summary.json", summaries)
    if any(report["status"] == "failed" for report in summaries.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
