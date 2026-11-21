"""Evaluate JudiAct on prepared public benchmark inputs and the source judging rubrics."""

import argparse
import json
from collections import Counter
from importlib.resources import files
from pathlib import Path

from ..benchmark_response_eval import (
    assemble, delivered, expanded, indexed, judgment_documents, post_documents,
    score_systems, translation_documents,
)
from ..external_eval_io import record_settings
from ..io_utils import read_jsonl, write_json, write_jsonl


def prepare(args):
    defaults = json.loads(files("judiact").joinpath("public_protocol.json").read_text(encoding="utf-8"))
    tasks = args.tasks or [p.stem for p in sorted(args.inputs.glob("*.jsonl"))
                          if p.stem in defaults["generation_tasks"] + defaults["native_tasks"]]
    if not tasks:
        raise ValueError("No supported benchmark manifests found in --inputs")
    for key in ("generation_tasks", "native_tasks"):
        defaults[key] = [task for task in defaults[key] if task in tasks]
    if set(tasks) != set(defaults["generation_tasks"] + defaults["native_tasks"]):
        raise ValueError("Unsupported public benchmark task")
    args.output.mkdir(parents=True, exist_ok=True)
    record_settings(args.output, {"input_directory": str(args.inputs.resolve()), "protocol": defaults})
    for task in tasks:
        source = args.inputs / f"{task}.jsonl"
        docs = list(indexed(read_jsonl(source)).values())
        for row in docs:
            visible = json.loads(row["visible_json"])
            visible["images"] = [{**im, "uri": str((source.parent / im["uri"]).resolve())}
                                 for im in visible["images"]]
            row["visible_json"] = json.dumps(visible, ensure_ascii=False)
        write_jsonl(args.output / "inputs" / f"{task}.jsonl", docs)
    write_json(args.output / "protocol.json", defaults)


def run_guard(args, config):
    from ..benchmark_response_models import transport
    from ..guard import JudiActGuard

    if not args.adapter or not args.model:
        raise ValueError("Guard inference needs --model and --adapter")
    name = args.name or "judiact"
    if not name.replace("-", "").replace("_", "").isalnum():
        raise ValueError("--name must be a simple system name")
    protocol = "native_harm" if args.native_harm else "four_action"
    target = args.output / "guards" / name / protocol
    record_settings(target / f"rank{args.rank}", {
        "model": str(args.model.resolve()), "adapter": str(args.adapter.resolve()), "native_harm": args.native_harm,
        "rank": args.rank, "world_size": args.world_size, "phases": args.phases,
        "attention_backend": args.attention_backend})
    predictor = JudiActGuard(args.model, args.adapter, native_harm=args.native_harm, include_evidence=False,
                             attention_backend=args.attention_backend)
    for phase in args.phases:
        tasks = config["native_tasks"] if phase == "native" else config["generation_tasks"]
        for task in tasks:
            if args.tasks and task not in args.tasks:
                continue
            docs = (post_documents(args.output, task, config) if phase == "post" else
                    read_jsonl(args.output / "inputs" / f"{task}.jsonl"))
            results = transport(predictor, task, docs[args.rank::args.world_size],
                                target / phase / f"{task}.rank{args.rank}.jsonl")
            print(json.dumps({"task": task, "phase": phase, "rank": args.rank,
                              "status_counts": dict(Counter(r["status"] for r in results.values()))}), flush=True)


def score_native(args, config):
    from ..binary_projection import four_action_harm_prediction
    from ..external_eval_metrics import summarize_benchmark

    for task in args.tasks or config["native_tasks"]:
        docs = read_jsonl(args.output / "inputs" / f"{task}.jsonl")
        for system in sorted((args.output / "guards").glob("*/*")):
            paths = sorted((system / "native").glob(f"{task}.rank*.outcomes.jsonl"))
            if not paths:
                continue
            outcomes = indexed(r for path in paths for r in read_jsonl(path))
            rows = []
            for doc in docs:
                outcome = outcomes.get(doc["sample_id"], {"status": "missing_prediction"})
                row = {**doc, **outcome}
                if "action_probabilities" in outcome and outcome["status"] == "ok":
                    row["score"], row["harmful"] = four_action_harm_prediction(outcome["action_probabilities"])
                    row["score_kind"] = "four_action_hold_probability"
                rows.append(row)
            report = summarize_benchmark(rows)
            c = report["counts"]
            errors = c["fp"] + c["fn"]
            def f1(correct, unknown):
                total = 2 * correct + errors + unknown
                return 2 * correct / total if total else 0.
            harmful = f1(c["tp"], c["unknown_positive"])
            safe = f1(c["tn"], c["unknown_negative"])
            report["full_denominator_classification"] = {
                "harmful_F1": harmful, "macro_F1": (harmful + safe) / 2}
            report["readout"] = system.name
            write_json(system / "native" / f"{task}.summary.json", report)


def run_inference(args, config):
    from transformers import set_seed
    from ..benchmark_response_models import BenchmarkGenerator, transport

    judge = args.command in ("translate", "judge")
    set_seed(config["seed"] + args.rank)
    path = args.judge_model if judge else args.model
    record_settings(args.output / "execution" / args.command / f"rank{args.rank}", {
        "model": str(path.resolve()) if path else None, "backend": args.backend,
        "harmbench_classifier": str(args.harmbench_classifier) if args.harmbench_classifier else None,
        "references": str(args.references) if args.references else None,
        "rank": args.rank, "world_size": args.world_size, "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len, "gpu_memory_utilization": args.gpu_memory_utilization})
    tasks = args.tasks or (["sorrybench"] if args.command == "translate" else config["generation_tasks"])
    generator = None
    for task in tasks:
        name = f"{task}.rank{args.rank}.jsonl"
        if args.command == "generate":
            docs = expanded(read_jsonl(args.output / "inputs" / f"{task}.jsonl"), task, config)
            target = args.output / "answers/base" / name
        elif args.command == "controlled":
            docs = read_jsonl(args.output / "delivery" / f"{task}.generation.jsonl")
            target = args.output / "answers/controlled" / name
        else:
            answers = read_jsonl(args.output / "delivery" / f"{task}.actual.jsonl")[args.rank::args.world_size]
            if args.command == "translate":
                if task != "sorrybench":
                    raise ValueError("Translation is only used by SORRY-Bench")
                docs, target = translation_documents(answers), args.output / "translations" / name
            elif task == "harmbench":
                from ..harmbench import HarmBenchTransport

                if not (args.harmbench_classifier and args.references):
                    raise ValueError("HarmBench requires --harmbench-classifier and --references")
                transport(HarmBenchTransport(args.harmbench_classifier, args.references), task,
                          HarmBenchTransport.documents(answers), args.output / "judgments" / name)
                continue
            else:
                if not args.references:
                    raise ValueError("Source-rubric judging requires --references")
                translations = (indexed(r for p in sorted((args.output / "translations").glob(
                    "sorrybench.rank*.outcomes.jsonl")) for r in read_jsonl(p)) if task == "sorrybench" else {})
                docs = judgment_documents(answers, task, args.references, translations)
                target = args.output / "judgments" / name
        if not judge:
            docs = docs[args.rank::args.world_size]
        if not docs:
            write_jsonl(target.with_suffix(".outcomes.jsonl"), [])
            continue
        if generator is None:
            if path is None:
                raise ValueError("Supply --judge-model for judging or --model for answer generation")
            if args.backend == "vllm":
                from ..benchmark_vllm import VLLMBenchmarkJudge

                if not judge:
                    raise ValueError("Sampled or beam-search answers use the Transformers backend")
                generator = VLLMBenchmarkJudge(path, config, tensor_parallel_size=args.tensor_parallel_size,
                    max_model_len=args.max_model_len, memory_utilization=args.gpu_memory_utilization,
                    max_num_seqs=128, max_num_batched_tokens=8192, disable_custom_all_reduce=True)
            else:
                generator = BenchmarkGenerator(path, config, judge=judge)
        generator.eval_batcher.generation_tokens = (
            max(json.loads(d["visible_json"])["judge_generation"]["max_new_tokens"] for d in docs) if judge else
            config["generation"][task]["max_new_tokens"] * config["generation"][task].get("num_beams", 1))
        outcomes = transport(generator, task, docs, target)
        print(json.dumps({"task": task, "command": args.command, "rank": args.rank, "members": len(outcomes)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "generate", "guard", "assemble", "controlled", "materialize",
                                             "translate", "judge", "score", "score-native"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="Directory of prepared benchmark JSONL manifests")
    parser.add_argument("--model", type=Path, help="Local Qwen3-VL-4B-Instruct snapshot")
    parser.add_argument("--adapter", type=Path, help="Selected JudiAct SFT or RL adapter")
    parser.add_argument("--judge-model", type=Path, help="Local Qwen3-VL-32B-Instruct snapshot")
    parser.add_argument("--harmbench-classifier", type=Path, help="Local official HarmBench classifier")
    parser.add_argument("--references", type=Path, help="Directory of trusted upstream rubric checkouts")
    parser.add_argument("--name", help="Distinct system name for each checkpoint")
    parser.add_argument("--tasks", nargs="+")
    parser.add_argument("--phases", nargs="+", choices=("pre", "post", "native"), default=["pre", "post"])
    parser.add_argument("--native-harm", action="store_true", help="Use trained native binary interface rather than four actions")
    parser.add_argument("--attention-backend", choices=("flash_attention_2", "sdpa"), default="flash_attention_2")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--backend", choices=("transformers", "vllm"), default="transformers")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--gpu-memory-utilization", type=float, default=.90)
    args = parser.parse_args()
    if not 0 <= args.rank < args.world_size:
        parser.error("--rank must be in [0, world-size)")
    if args.command == "prepare":
        if args.inputs is None:
            parser.error("prepare requires --inputs")
        prepare(args)
        return
    config = json.loads((args.output / "protocol.json").read_text(encoding="utf-8"))
    if args.command == "guard":
        run_guard(args, config)
    elif args.command == "assemble":
        print(json.dumps(assemble(args.output, config)))
    elif args.command == "materialize":
        for task in args.tasks or config["generation_tasks"]:
            delivered(args.output, task)
    elif args.command == "score":
        for task in args.tasks or config["generation_tasks"]:
            score_systems(args.output, task, config)
    elif args.command == "score-native":
        score_native(args, config)
    else:
        run_inference(args, config)


if __name__ == "__main__":
    main()
