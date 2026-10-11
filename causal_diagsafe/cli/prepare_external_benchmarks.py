"""Materialize full selected public splits once, independently of GPU experiments."""

import argparse
import json
from pathlib import Path

from ..external_benchmark_data import inventory, load_external_documents
from ..io_utils import write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--deck", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="*")
    args = parser.parse_args()
    config, deck = json.loads(Path(args.config).read_text()), json.loads(Path(args.deck).read_text())
    tasks = args.tasks or list(dict.fromkeys(config["primary_tasks"] + config["extended_tasks"]))
    output = Path(args.output)
    report = {}
    new_sources = {"siuo": "siuo", "beavertails_v_response": "beavertails_v", "mm_safetybench": "mm_safetybench",
                   "jailbreakv": "jailbreakv", "sorrybench": "sorrybench", "spa_vl_eval_prompt": "guardreasoner_test",
                   "spa_vl_eval_response": "guardreasoner_test"}
    for task in tasks:
        try:
            docs = list(load_external_documents(task, args.root, deck, args.references))
            if not docs or len({doc["sample_id"] for doc in docs}) != len(docs):
                raise ValueError(f"Empty split or repeated sample IDs: {task}")
            destination = output / (task + ".jsonl")
            temporary = destination.with_suffix(".jsonl.tmp")
            write_jsonl(temporary, docs)
            temporary.replace(destination)
            report[task] = {"status": "prepared", "file": str(destination), **inventory(docs)}
            if task in new_sources:
                complete = (Path(args.root) / "raw/external_baselines" / new_sources[task] / "download_source.json").exists()
                report[task]["source_snapshot_completed"] = complete
                report[task]["partial_source"] = not complete
            if task == "jailbreakv":
                report[task]["full_media_available"] = report[task]["missing_media_members"] == 0
            print(task, json.dumps(report[task]), flush=True)
        except Exception as exc:
            report[task] = {"status": "unavailable", "error_type": type(exc).__name__, "error": str(exc)[:1500]}
            print(task, json.dumps(report[task]), flush=True)
        write_json(output / "inventory.json", report)


if __name__ == "__main__":
    main()
