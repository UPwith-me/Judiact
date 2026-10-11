"""Join common-answer judgment and pre/post decisions with full expected denominators."""

import argparse
import json
from pathlib import Path

from ..external_system_metrics import system_summary
from ..io_utils import read_jsonl, write_json, write_jsonl


def index_file(path):
    if not path.exists():
        return {}
    rows = read_jsonl(path)
    if len({row["sample_id"] for row in rows}) != len(rows):
        raise ValueError(f"Repeated output IDs: {path}")
    return {row["sample_id"]: row for row in rows}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--generated", required=True)
    parser.add_argument("--native", required=True)
    parser.add_argument("--post", required=True)
    parser.add_argument("--judges", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tasks", nargs="+", required=True)
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    output = Path(args.output)
    previous = output / "summary.json"
    reports = json.loads(previous.read_text()) if previous.exists() else {}
    for model in config["models"] + config.get("method_models", []):
        for protocol in model["protocols"]:
            key = f"{model['name']}/{protocol}"
            reports.setdefault(key, {})
            for task in args.tasks:
                file = Path(args.data) / (task + ".jsonl")
                if not file.exists():
                    reports[key][task] = {"status": "unavailable_data"}
                    continue
                generated = index_file(Path(args.generated) / (task + ".jsonl"))
                pre = index_file(Path(args.native) / key / task / "outputs.jsonl")
                post = index_file(Path(args.post) / key / task / "outputs.jsonl")
                kind = "harmbench" if task == "harmbench_test" else "multimodal"
                judges = index_file(Path(args.judges) / kind / task / "outputs.jsonl")
                rows = []
                for doc in read_jsonl(file):
                    identifier = doc["sample_id"]
                    candidate = generated.get(identifier, {})
                    judged = judges.get(identifier, {})
                    request_pred, response_pred = pre.get(identifier, {}), post.get(identifier, {})
                    if request_pred and request_pred["visible_json"] != doc["visible_json"]:
                        raise ValueError(f"Wrong cached pre-guard context: {identifier}")
                    if response_pred and response_pred["visible_json"] != candidate["visible_json"]:
                        raise ValueError(f"Wrong cached post-guard candidate: {identifier}")
                    available = candidate.get("generation_status") == "ok"
                    rows.append({"sample_id": identifier, "root_id": doc["root_id"], "category": doc["category"],
                        "request_native_harm": doc["native_harm"], "risk_context": True if task == "siuo" else doc["native_harm"],
                        "candidate_available": available, "pre_hold": request_pred.get("harmful"),
                        "post_hold": response_pred.get("harmful"), "judge_unsafe": judged.get("unsafe"),
                        "judge_useful": judged.get("useful"), "judge_refusal": judged.get("refusal"),
                        "judge_protocol": judged.get("protocol", "missing"),
                        "pre_status": request_pred.get("status", "missing"), "post_status": response_pred.get("status", "missing")})
                directory = output / key / task
                write_jsonl(directory / "system_rows.jsonl", rows)
                reports[key][task] = system_summary(rows)
                write_json(directory / "summary.json", reports[key][task])
            write_json(output / "summary.json", reports)


if __name__ == "__main__":
    main()
