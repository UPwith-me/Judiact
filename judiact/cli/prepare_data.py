"""Materialize the evaluation layout from a local JudiBench snapshot."""

import argparse
from pathlib import Path

from ..io_utils import read_jsonl, write_jsonl


def prepare_benchmark(snapshot, output):
    records = read_jsonl(snapshot / "data/test.jsonl")
    by_task = {"request_decision": ("core_request", "natural_request"),
               "response_audit": ("s5_response", "natural_response")}
    if len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("Duplicate JudiBench member IDs")
    for row in records:
        visible = row["model_input"]
        visible["images"] = [{**im, "uri": str((snapshot / im["uri"]).resolve())} for im in visible["images"]]
        if row["split"] != "test_hidden" or row["task_type"] not in by_task:
            raise ValueError(f"Unexpected benchmark split or task for {row['sample_id']}")
    for task, (full, natural) in by_task.items():
        selected = [row for row in records if row["task_type"] == task]
        write_jsonl(output / f"data/{full}/test_hidden.jsonl", selected)
        write_jsonl(output / f"data/{natural}/test_hidden.jsonl", (r for r in selected if r["natural"]))
    write_jsonl(output / "reviews/supervision.jsonl", (
        {"sample_id": row["sample_id"], "split": row["split"], "withheld_fields": row["supervision"]["withheld_fields"],
         "target_mask": row["supervision"]["target_mask"]} for row in records))
    write_jsonl(output / "data/causal/verified_test_hidden.jsonl",
                read_jsonl(snapshot / "relations/test_verified.jsonl"))
    write_jsonl(output / "data/causal/verified_s2_quartets_test_hidden.jsonl",
                read_jsonl(snapshot / "quartets/test.jsonl"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True, help="Local JudiBench files and referenced media")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare_benchmark(args.snapshot, args.output)


if __name__ == "__main__":
    main()
