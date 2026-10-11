"""Cache the same frozen-Qwen answer for all pre/post guard comparisons."""

import argparse
import json
from pathlib import Path

from ..external_answering import FixedAnswerer
from ..external_benchmark_data import source_metadata
from ..external_eval_io import record_settings
from ..external_guard_models import model_directory
from ..io_utils import read_jsonl, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--predictions-only", action="store_true", help="Rebuild completion counts from cached answers without loading a model.")
    parser.add_argument("--limit", type=int, help="Short interface probe only, not a full system evaluation.")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output) / f"rank{args.rank}"
    output.mkdir(parents=True, exist_ok=True)
    if not args.predictions_only:
        record_settings(output, {"protocol_version": config["protocol_version"], "answerer": config["answerer"],
            "model_source": source_metadata(model_directory(args.root, "qwen3vl4b")), "rank": args.rank,
            "world_size": args.world_size, "sampling": "one greedy answer per original visible request"})
        record_settings(output / "probe_scope", {"limit": args.limit})
    predictor = None if args.predictions_only else FixedAnswerer(args.root, config["answerer"])
    from lmms_eval import evaluator
    from ..external_lmms import NativeGuardLM, NativeSafetyTask
    summary_file = output / "summary.json"
    reports = json.loads(summary_file.read_text()) if summary_file.exists() else {}
    tasks = [task for task in args.tasks or config["system_tasks"]
             if task not in config.get("deferred_tasks", {})]
    for task in tasks:
        file = Path(args.data) / (task + ".jsonl")
        if not file.exists():
            reports[task] = {"status": "unavailable_data"}
        else:
            docs = read_jsonl(file)[args.rank::args.world_size]
            if args.limit is not None:
                docs = docs[:args.limit]
            lm = NativeGuardLM(predictor, output / (task + ".jsonl"))
            try:
                benchmark = NativeSafetyTask(task, docs)
                lm.task_dict = {task: benchmark.dataset}
                evaluator.evaluate(lm=lm, task_dict={task: benchmark},
                    limit=None, log_samples=False, bootstrap_iters=0, process_docs_parallel=1)
                reports[task] = {"status": "generated", "members": len(docs),
                                 "successful_answers": sum(row["status"] == "ok" for row in lm.current.values())}
            except Exception as exc:
                import traceback
                traceback.print_exc()
                reports[task] = {"status": "failed", "error": str(exc)[:1500]}
        write_json(output / "summary.json", reports)
    if any(row["status"] == "failed" for row in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
