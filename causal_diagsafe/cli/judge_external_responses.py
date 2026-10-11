"""Cache response judgments independently of guard decisions and benchmark tuning."""

import argparse
import json
import time
import traceback
from collections import Counter
from pathlib import Path

from ..external_answering import HarmBenchJudge, OpenMultimodalJudge
from ..external_benchmark_data import source_metadata
from ..external_eval_io import append_prediction, read_predictions, record_settings
from ..io_utils import read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--kind", choices=("multimodal", "harmbench"), default="multimodal")
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--merge-only", action="store_true")
    args = parser.parse_args()
    base_output = Path(args.output)
    if args.merge_only:
        reports = {}
        for task in args.tasks:
            source = Path(args.data) / (task + ".jsonl")
            if not source.exists():
                reports[task] = {"status": "unavailable_data"}
                continue
            docs = read_jsonl(source)
            if args.limit is not None:
                docs = docs[:args.limit]
            joined = {}
            for rank in range(args.world_size):
                path = base_output / f"rank{rank}" / task / "outputs.jsonl"
                if path.exists():
                    for row in read_jsonl(path):
                        if row["sample_id"] in joined:
                            raise ValueError("Judge rank shards overlap on a candidate ID")
                        joined[row["sample_id"]] = row
            rows = [joined.get(doc["sample_id"], {"sample_id": doc["sample_id"],
                    "status": "missing_prediction", "unsafe": None}) for doc in docs]
            write_jsonl(base_output / task / "outputs.jsonl", rows)
            reports[task] = {"status": "judged", "members": len(docs),
                "statuses": dict(Counter(row["status"] for row in rows)), "world_size": args.world_size}
        write_json(base_output / "summary.json", reports)
        return
    output = base_output / f"rank{args.rank}" if args.world_size > 1 else base_output
    output.mkdir(parents=True, exist_ok=True)
    model_name = "multimodal_judge32b" if args.kind == "multimodal" else "harmbench_judge13b"
    record_settings(output, {"judge": model_name, "source": source_metadata(Path(args.root) / "raw/external_models" / model_name),
        "protocol": "open_binary_questions_20261009_v1" if args.kind == "multimodal" else "official_harmbench_text",
        "limit": args.limit, "rank": args.rank, "world_size": args.world_size,
        "selection": "All selected generated candidates; no guard-conditioned filtering."})
    predictor = OpenMultimodalJudge(args.root) if args.kind == "multimodal" else HarmBenchJudge(args.root, args.references)
    import torch
    from PIL import UnidentifiedImageError
    reports = {}
    for task in args.tasks:
        source = Path(args.data) / (task + ".jsonl")
        if not source.exists():
            reports[task] = {"status": "unavailable_data"}
            continue
        directory = output / task
        directory.mkdir(parents=True, exist_ok=True)
        docs = read_jsonl(source)
        if args.limit is not None:
            docs = docs[:args.limit]
        docs = docs[args.rank::args.world_size]
        cache = read_predictions(directory / "predictions.jsonl")
        rows = []
        try:
            with (directory / "predictions.jsonl").open("a", encoding="utf-8") as handle:
                entries, positions = [], {}
                for index, doc in enumerate(docs):
                    identifier = doc["sample_id"]
                    visible, provenance = json.loads(doc["visible_json"]), json.loads(doc["provenance_json"])
                    if identifier in cache:
                        value = cache[identifier]
                        if value["visible_json"] != doc["visible_json"] or value["provenance_json"] != doc["provenance_json"]:
                            raise ValueError(f"Changed response/criterion for cached judgment {identifier}")
                    elif visible["response_candidate"] is None:
                        value = {"sample_id": identifier, "status": "missing_candidate", "unsafe": None}
                    else:
                        positions[len(entries)] = index
                        entries.append({"visible": visible, "provenance": provenance, "identifier": identifier,
                            "visible_json": doc["visible_json"], "provenance_json": doc["provenance_json"]})
                        value = None
                    rows.append(value)
                from ..eval_batching import EvalBatcher
                batcher = EvalBatcher(devices=[predictor.device], max_batch_size=32 if args.kind == "multimodal" else 128,
                    image_max_pixels=1003520 if args.kind == "multimodal" else None, include_policy=False)
                def forward(batch):
                    arguments = ([entry["visible"] for entry in batch], [entry["provenance"] for entry in batch])
                    return predictor.judge_batch(*arguments) if args.kind == "multimodal" else \
                        predictor.judge_batch(*arguments, [entry["identifier"] for entry in batch])
                def failure(entry, exc):
                    return {"status": "failed", "unsafe": None, "error": f"{type(exc).__name__}: {exc}"[:1500]}
                for index, value in batcher.run(entries, forward, on_error=failure):
                    entry = entries[index]
                    value.update(sample_id=entry["identifier"], visible_json=entry["visible_json"], provenance_json=entry["provenance_json"])
                    append_prediction(handle, value)
                    cache[entry["identifier"]] = value
                    rows[positions[index]] = value
                write_json(directory / "batching.json", batcher.summary())
            write_jsonl(directory / "outputs.jsonl", rows)
            reports[task] = {"status": "judged", "members": len(docs),
                             "statuses": dict(Counter(row["status"] for row in rows)), "partial_probe": args.limit is not None}
        except Exception as exc:
            traceback.print_exc()
            reports[task] = {"status": "failed", "error": str(exc)[:2000]}
        write_json(output / "summary.json", reports)
    if any(row["status"] == "failed" for row in reports.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
