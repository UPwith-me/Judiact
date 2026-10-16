"""Use the existing common-answer system protocol for one contextual method run."""

import argparse
import json
from pathlib import Path

from ..external_system_metrics import system_summary
from ..io_utils import read_jsonl, write_json, write_jsonl
from .summarize_external_systems import index_file


def main():
    parser = argparse.ArgumentParser()
    for name in ("config", "data", "generated", "pre", "post", "judges", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output, reports = Path(args.output), {}
    for task in config["system_tasks"]:
        if task in config.get("deferred_tasks", {}):
            continue
        generated = index_file(Path(args.generated) / (task + ".jsonl"))
        pre = index_file(Path(args.pre) / task / "outputs.jsonl")
        post = index_file(Path(args.post) / task / "outputs.jsonl")
        kind = "harmbench" if task == "harmbench_test" else "multimodal"
        judges = index_file(Path(args.judges) / kind / task / "outputs.jsonl")
        rows = []
        for doc in read_jsonl(Path(args.data) / (task + ".jsonl")):
            identifier = doc["sample_id"]
            candidate, judged = generated.get(identifier, {}), judges.get(identifier, {})
            request_pred, response_pred = pre.get(identifier, {}), post.get(identifier, {})
            if request_pred and request_pred["visible_json"] != doc["visible_json"]:
                raise ValueError(f"Wrong cached pre-guard context: {identifier}")
            if response_pred and response_pred["visible_json"] != candidate["visible_json"]:
                raise ValueError(f"Wrong cached post-guard candidate: {identifier}")
            rows.append({"sample_id": identifier, "root_id": doc["root_id"], "category": doc["category"],
                "request_native_harm": doc["native_harm"],
                "risk_context": True if task == "siuo" else doc["native_harm"],
                "candidate_available": candidate.get("generation_status") == "ok",
                "pre_hold": request_pred.get("harmful"), "post_hold": response_pred.get("harmful"),
                "judge_unsafe": judged.get("unsafe"), "judge_useful": judged.get("useful"),
                "judge_refusal": judged.get("refusal"), "judge_protocol": judged.get("protocol", "missing"),
                "pre_status": request_pred.get("status", "missing"),
                "post_status": response_pred.get("status", "missing")})
        directory = output / task
        write_jsonl(directory / "system_rows.jsonl", rows)
        reports[task] = system_summary(rows)
        reports[task]["method"] = "contextual_v4/shared_deck; complete generated audit"
        reports[task]["warning_execution"] = (
            "Cached-answer filtering only; does not simulate regenerating a bounded answer after warn.")
        write_json(directory / "summary.json", reports[task])
    write_json(output / "summary.json", reports)


if __name__ == "__main__":
    main()
