"""Run complete locked public classification sets through lmms-eval."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from accelerate import PartialState

from ..io_utils import read_jsonl, write_json, write_jsonl
from ..public_benchmarks_v1 import PUBLIC_TASKS, load_public_documents, public_result, summarize_public


def main(predictor_class=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--policy-deck", required=True, help="Explicit frozen shared-policy input.")
    parser.add_argument("--prepared", required=True)
    parser.add_argument("--output")
    parser.add_argument("--model")
    parser.add_argument("--adapter")
    parser.add_argument("--backend", choices=("diagnostic", "qwen_prompt"), default="diagnostic")
    parser.add_argument("--processor", help="Same saved processor used for the trained comparisons.")
    parser.add_argument("--tasks", nargs="+", choices=PUBLIC_TASKS, default=list(PUBLIC_TASKS))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--diagnostic-limit", type=int, help="Only for interface diagnosis, never final metrics.")
    args = parser.parse_args()
    state = PartialState()
    prepared = Path(args.prepared)
    prepared.mkdir(parents=True, exist_ok=True)
    deck = json.loads(Path(args.policy_deck).read_text(encoding="utf-8"))
    if state.is_main_process:
        inventory_file = prepared / "inventory.json"
        inventory = json.loads(inventory_file.read_text()) if inventory_file.exists() else {}
        for name in args.tasks:
            file = prepared / f"{name}.jsonl"
            if not file.exists():
                write_jsonl(file, load_public_documents(name, args.raw, deck))
            docs = read_jsonl(file)
            assert len({d["sample_id"] for d in docs}) == len(docs), name
            assert all(json.loads(d["visible_json"])["policy_deck"] == deck for d in docs), name
            inventory[name] = {"records": len(docs), "labeled": sum(d["native_harm"] is not None for d in docs),
                "image_records": sum(bool(json.loads(d["visible_json"])["images"]) for d in docs),
                "subsets": dict(Counter(d["subset"] for d in docs))}
        write_json(prepared / "inventory.json", inventory)
        write_json(prepared / "policy_deck.json", deck)
        print(json.dumps(inventory), flush=True)
    state.wait_for_everyone()
    if args.prepare_only:
        return
    from lmms_eval import evaluator
    from ..lmms_v1 import DiagnosticLM, PublicSafetyTask, QwenPromptLM

    directory = Path(args.output)
    directory.mkdir(parents=True, exist_ok=True)
    shard = directory / f"predictions.rank{state.process_index}.jsonl"
    predictor_options = {"predictor_class": predictor_class} if predictor_class is not None else {}
    model = DiagnosticLM(args.model, args.adapter, state.device, state, shard, **predictor_options) \
        if args.backend == "diagnostic" else QwenPromptLM(
            args.model, args.processor, state.device, state, shard)
    tasks = {name: PublicSafetyTask(name, read_jsonl(prepared / f"{name}.jsonl")) for name in args.tasks}
    model.task_dict = {name: task.dataset for name, task in tasks.items()}
    predictor_name = type(model.predictor).__name__
    result = evaluator.evaluate(lm=model, task_dict=tasks, limit=args.diagnostic_limit,
        bootstrap_iters=0, log_samples=True, distributed_executor_backend="accelerate", process_docs_parallel=1)
    if state.is_main_process:
        write_json(directory / "lmms_results.json", json.loads(json.dumps(
            {k: v for k, v in result.items() if k != "samples"}, default=str)))
        reports = {}
        for name, rows in result["samples"].items():
            write_jsonl(directory / f"{name}_samples.jsonl", rows)
            items = [public_result(r["doc"], r["filtered_resps"])["public_safety"] for r in rows]
            summary = summarize_public(items)
            summary["by_subset"] = {key: summarize_public([r for r in items if r["subset"] == key])
                                    for key in sorted({r["subset"] for r in items})}
            summary["by_category"] = {key: summarize_public([r for r in items if r["category"] == key])
                                      for key in sorted({r["category"] for r in items})}
            reports[name] = summary
        write_json(directory / "summary.json", {"settings": {
            "adapter": args.adapter, "model": args.model, "backend": args.backend, "policy_deck": deck,
            "framework": "lmms-eval 0.7.3 caa8d98b5c109ff1b0dd2f65381dc4674622be18",
            "protocol": "shared explicit deck, native benchmark binary labels",
            "diagnostic_limit": args.diagnostic_limit, "generation": False,
            "predictor": predictor_name,
            "native_head_context": "same visible shared-deck input; reported separately from policy decisions",
            "note": "Classification only. No ASR, harmful completion, or helpfulness claim; policy alignment is not human-certified."
        }, "benchmarks": reports})
        print(json.dumps({name: r["denominator"] for name, r in reports.items()}), flush=True)


if __name__ == "__main__":
    main()
