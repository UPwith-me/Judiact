"""Join answer shards by ID, preserving failed-generation denominators."""

import argparse
import json
from collections import Counter
from pathlib import Path

from ..external_eval_io import read_predictions
from ..io_utils import read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--answers", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    report = {}
    for task in args.tasks:
        original = Path(args.data) / (task + ".jsonl")
        if not original.exists():
            report[task] = {"status": "unavailable_data"}
            continue
        answers = {}
        for file in sorted(Path(args.answers).glob(f"rank*/{task}.jsonl")):
            for identifier, row in read_predictions(file).items():
                if identifier in answers:
                    raise ValueError(f"Repeated answer shard ID: {identifier}")
                answers[identifier] = row
        docs = []
        for original_doc in read_jsonl(original):
            identifier = original_doc["sample_id"]
            answer = answers.get(identifier, {})
            if answer and answer["visible_json"] != original_doc["visible_json"]:
                raise ValueError(f"Changed request for generated answer: {identifier}")
            visible = json.loads(original_doc["visible_json"])
            if visible["query"] is None:
                answer = {"status": "missing_query"}
            visible["response_candidate"] = answer.get("response_candidate") if answer.get("status") == "ok" else None
            docs.append({**original_doc, "native_harm": None, "audit_target": "response",
                "visible_json": json.dumps(visible, ensure_ascii=False),
                "input_error": None if answer.get("status") == "ok" else "No successfully generated candidate.",
                "generation_status": answer.get("status", "not_completed"),
                "request_native_harm": original_doc["native_harm"],
                "answer_provenance": "Frozen Qwen3-VL-4B-Instruct; one shared greedy cached answer; not native response gold."})
        write_jsonl(output / (task + ".jsonl"), docs)
        report[task] = {"members": len(docs), "generation_status": dict(Counter(doc["generation_status"] for doc in docs))}
        write_json(output / "inventory.json", report)


if __name__ == "__main__":
    main()
