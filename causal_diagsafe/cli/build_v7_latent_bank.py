"""Train-only weak bootstrap records from the frozen 32B teacher, continuously batched."""

import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import time

from ..audit_v7 import (
    PROTOCOL_VERSION, canonical, diagnostic_node, materialize_links,
    materialize_witness, source_identity, source_node,
)
from ..io_utils import read_jsonl, write_json
from ..models.diagsafe_v3 import PROCESSOR_SIZE
from ..training.v3_views import open_rgb
from ..training.v5_data import coherent_case
from ..training.v7_latent_data import candidate_views


class Teacher:
    def __init__(self, args):
        from transformers import AutoProcessor
        from vllm import LLM
        self.processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
        self.engine = LLM(
            model=args.model, tokenizer=args.model, dtype="bfloat16", kv_cache_dtype="auto",
            tensor_parallel_size=args.tensor_parallel_size, gpu_memory_utilization=args.memory,
            max_model_len=32768, max_num_seqs=32, max_num_batched_tokens=8192,
            enable_prefix_caching=True, enable_chunked_prefill=True, seed=args.seed,
            limit_mm_per_prompt={"image": 8, "video": 0},
            mm_processor_kwargs={"size": dict(PROCESSOR_SIZE)}, generation_config="vllm")

    def generate(self, nodes, budget):
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
        requests, settings = [], []
        for node in nodes:
            images = []
            for uri, box in node.media:
                image = open_rgb(uri)
                if box is not None:
                    import math
                    w, h = image.size
                    image = image.crop((math.floor(box[0] * w / 1000), math.floor(box[1] * h / 1000),
                                        math.ceil(box[2] * w / 1000), math.ceil(box[3] * h / 1000)))
                images.append(image)
            content = [{"type": "image", "image": im} for im in images]
            content.append({"type": "text", "text": node.text})
            prompt = self.processor.apply_chat_template(
                [{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)
            requests.append({"prompt": prompt, **({"multi_modal_data": {"image": images}} if images else {})})
            settings.append(SamplingParams(
                temperature=0, top_p=1, top_k=-1, max_tokens=budget,
                structured_outputs=StructuredOutputsParams(json=json.loads(canonical(node.schema)))))
        result = []
        for output in self.engine.generate(requests, settings, use_tqdm=False):
            completion = output.outputs[0]
            raw = completion.text
            error = "teacher_token_budget" if completion.finish_reason == "length" else None
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                value, error = None, "teacher_invalid_json"
            result.append({"raw": raw, "value": value, "error": error,
                           "terminated": completion.finish_reason != "length",
                           "tokens": len(completion.token_ids)})
        return result


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "release", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--memory", type=float, default=.86)
    parser.add_argument("--window", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--ids-file")
    args = parser.parse_args()
    release, output = Path(args.release), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [r for task in ("core_request", "s5_response")
            for r in read_jsonl(release / f"data/{task}/train.jsonl")]
    if args.ids_file:
        wanted = set(Path(args.ids_file).read_text().splitlines())
        rows = [r for r in rows if r["sample_id"] in wanted]
        assert {r["sample_id"] for r in rows} == wanted
    if args.limit:
        # Spread the small implementation trial over task, modality and action.
        buckets = {}
        for row in rows:
            task = row["task_type"]
            key = (task, bool(row["model_input"]["images"]),
                   row["gold"]["response_publishability" if task == "response_audit" else "request_action"])
            buckets.setdefault(key, []).append(row)
        rows = [bucket[i] for i in range(args.limit) for bucket in buckets.values()
                if i < len(bucket)][:args.limit]
    cases = [coherent_case(r) for r in rows]
    sources = {}
    for case in cases:
        key = (case["task_type"], source_identity(case["model_input"]))
        sources.setdefault(key, []).append(case)
    selected = [c for i, group in enumerate(sources.values())
                if i % args.world_size == args.rank for c in group]
    path = output / f"bank.rank{args.rank}.jsonl"
    done = {r["sample_id"] for r in read_jsonl(path)} if path.exists() else set()
    selected = [c for c in selected if c["sample_id"] not in done]
    if not selected:
        return
    teacher = Teacher(args)
    cache, counts = {}, Counter()
    started = time.perf_counter()
    with path.open("a", encoding="utf-8") as handle:
        for start in range(0, len(selected), args.window):
            batch = selected[start:start + args.window]
            missing = {}
            for c in batch:
                key = (c["task_type"], source_identity(c["model_input"]))
                if key not in cache:
                    missing[key] = source_node(c["model_input"], c["task_type"])
            if missing:
                generated = teacher.generate(list(missing.values()), args.max_new_tokens)
                cache.update(zip(missing, generated))
                repair_keys, repair_nodes = [], []
                for key, node in missing.items():
                    raw = cache[key]
                    error = raw["error"]
                    if error is None:
                        try:
                            # The key stores the exact original visible source.
                            case = next(c for c in batch if
                                        (c["task_type"], source_identity(c["model_input"])) == key)
                            materialize_witness(case["model_input"], raw["value"])
                        except (ValueError, TypeError, KeyError) as exception:
                            error = str(exception)
                    if error is not None:
                        repaired = copy.deepcopy(node)
                        repaired.instruction += (
                            "\nThe previous attempt had a structural error: " + error +
                            ". Produce a fresh concise record from the original source. "
                            "No disposition label or semantic answer is supplied.")
                        repair_keys.append(key)
                        repair_nodes.append(repaired)
                if repair_nodes:
                    retries = teacher.generate(repair_nodes, args.max_new_tokens * 2)
                    for key, retry in zip(repair_keys, retries):
                        retry["previous_attempt"] = cache[key]
                        cache[key] = retry
            candidates, pending, nodes = {}, [], []
            for c in batch:
                sid, visible = c["sample_id"], c["model_input"]
                raw = cache[c["task_type"], source_identity(visible)]
                candidate = {"witness": copy.deepcopy(raw), "links": None}
                candidates[sid] = candidate
                if raw["error"]:
                    continue
                try:
                    witness = materialize_witness(visible, raw["value"])
                except (ValueError, TypeError, KeyError) as error:
                    candidate["witness"]["error"] = str(error)
                    continue
                pending.append(sid)
                nodes.append(diagnostic_node(visible, c["task_type"], "links", witness))
            if nodes:
                for sid, result in zip(pending, teacher.generate(nodes, args.max_new_tokens)):
                    candidates[sid]["links"] = result
            for c in batch:
                sid = c["sample_id"]
                candidate = candidates[sid]
                _, _, error = candidate_views(c, candidate)
                counts[error or "supported"] += 1
                handle.write(json.dumps({
                    "sample_id": sid, "candidate": candidate, "error": error,
                    "protocol": PROTOCOL_VERSION,
                    "source_identity": source_identity(c["model_input"]),
                    "policy_identity": canonical(c["model_input"]["policy_deck"]),
                    "provenance": {"teacher": args.model, "semantic_status": "machine_weak_supervision",
                        "human_final": False, "teacher_fact_prompt_has_gold": False,
                        "teacher_prompt_has_gold_locations": False,
                        "bank_proposal": "fixed label-blind teacher record, summed as a separate atom",
                        "precision": "BF16, unquantized weights and KV", "teacher_temperature": 0},
                }, ensure_ascii=False) + "\n")
            handle.flush()
            print(json.dumps({"completed": sum(counts.values()), "planned": len(selected),
                              "counts": dict(counts), "seconds": time.perf_counter() - started}), flush=True)
    write_json(output / f"bank_summary.rank{args.rank}.json",
               {"arguments": vars(args), "counts": dict(counts), "seconds": time.perf_counter() - started})


if __name__ == "__main__":
    main()
