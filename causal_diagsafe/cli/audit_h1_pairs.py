from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from ..io_utils import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit strict H1 pairs and cached representations.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument(
        "--cache",
        action="append",
        default=[],
        help="Named cache in NAME=PATH form. Repeat as needed.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    manifest = read_jsonl(args.manifest)
    pairs = read_jsonl(args.pairs)
    by_id = {str(row.get("id") or ""): row for row in manifest}
    caches = _load_named_caches(args.cache)
    file_hashes: dict[str, str] = {}
    pair_reports: list[dict[str, Any]] = []

    for pair in pairs:
        left_id = str(pair.get("left_id") or "")
        right_id = str(pair.get("right_id") or "")
        left = by_id.get(left_id)
        right = by_id.get(right_id)
        if left is None or right is None:
            pair_reports.append(
                {"pair_id": pair.get("pair_id"), "valid": False, "issue": "missing_manifest_id"}
            )
            continue
        stage = int(pair.get("target_stage_id", -1))
        left_input = left.get("model_input") or {}
        right_input = right.get("model_input") or {}
        left_images = [str(value) for value in left_input.get("images") or []]
        right_images = [str(value) for value in right_input.get("images") or []]
        left_hashes = [_sha256(path, file_hashes) for path in left_images]
        right_hashes = [_sha256(path, file_hashes) for path in right_images]
        query_equal = str(left_input.get("query") or "") == str(
            right_input.get("query") or ""
        )
        image_hashes_equal = left_hashes == right_hashes
        response_equal = _input_value(left, "response_candidate") == _input_value(
            right, "response_candidate"
        )
        policy_text_equal = _input_value(left, "policy_text") == _input_value(
            right, "policy_text"
        )
        target_f_left = _stage_value(left, "F", stage)
        target_f_right = _stage_value(right, "F", stage)
        intervention_observable = _observable_intervention(
            str(pair.get("changed_factor") or ""),
            query_equal=query_equal,
            image_equal=image_hashes_equal,
            response_equal=response_equal,
            policy_equal=policy_text_equal,
        )
        direction_supervised = int(pair.get("target_direction_mask", 0) or 0) == 1
        target_f_flip = (
            target_f_left is not None
            and target_f_right is not None
            and target_f_left != target_f_right
        )
        report: dict[str, Any] = {
            "pair_id": str(pair.get("pair_id") or ""),
            "group_id": str(pair.get("group_id") or ""),
            "left_id": left_id,
            "right_id": right_id,
            "target_stage": pair.get("target_stage"),
            "target_stage_id": stage,
            "changed_factor": pair.get("changed_factor"),
            "valid": stage in range(5),
            "query_equal": query_equal,
            "image_paths_equal": left_images == right_images,
            "image_hashes_equal": image_hashes_equal,
            "response_candidate_equal": response_equal,
            "policy_text_equal": policy_text_equal,
            "intervention_observable": intervention_observable,
            "left_images": left_images,
            "right_images": right_images,
            "target_F_left": target_f_left,
            "target_F_right": target_f_right,
            "target_direction_supervised": direction_supervised,
            "causal_pair_valid": intervention_observable
            and direction_supervised
            and target_f_flip,
        }
        for name, cache in caches.items():
            left_index = cache["index"].get(left_id)
            right_index = cache["index"].get(right_id)
            if left_index is None or right_index is None or stage not in range(5):
                report[f"{name}_cache_present"] = False
                continue
            left_stage = cache["stage_hidden"][left_index, stage].float()
            right_stage = cache["stage_hidden"][right_index, stage].float()
            left_decision = cache["decision_hidden"][left_index].float()
            right_decision = cache["decision_hidden"][right_index].float()
            report[f"{name}_cache_present"] = True
            report[f"{name}_target_l2"] = float(torch.linalg.vector_norm(left_stage - right_stage))
            report[f"{name}_target_mean_abs"] = float((left_stage - right_stage).abs().mean())
            report[f"{name}_target_exact_equal"] = bool(torch.equal(left_stage, right_stage))
            report[f"{name}_decision_l2"] = float(
                torch.linalg.vector_norm(left_decision - right_decision)
            )
            report[f"{name}_decision_exact_equal"] = bool(
                torch.equal(left_decision, right_decision)
            )
        pair_reports.append(report)

    valid_rows = [row for row in pair_reports if row.get("valid")]
    summary: dict[str, Any] = {
        "format": "h1_pair_input_representation_audit_v1",
        "pair_count": len(pair_reports),
        "valid_pair_count": len(valid_rows),
        "target_stage_distribution": dict(Counter(row.get("target_stage") for row in valid_rows)),
        "changed_factor_distribution": dict(
            Counter(row.get("changed_factor") for row in valid_rows)
        ),
        "query_equal_count": sum(bool(row.get("query_equal")) for row in valid_rows),
        "image_paths_equal_count": sum(
            bool(row.get("image_paths_equal")) for row in valid_rows
        ),
        "image_hashes_equal_count": sum(
            bool(row.get("image_hashes_equal")) for row in valid_rows
        ),
        "target_F_flip_count": sum(
            row.get("target_F_left") is not None
            and row.get("target_F_right") is not None
            and row.get("target_F_left") != row.get("target_F_right")
            for row in valid_rows
        ),
        "target_direction_supervised_count": sum(
            bool(row.get("target_direction_supervised")) for row in valid_rows
        ),
        "observable_intervention_count": sum(
            bool(row.get("intervention_observable")) for row in valid_rows
        ),
        "causal_pair_valid_count": sum(bool(row.get("causal_pair_valid")) for row in valid_rows),
        "caches": {},
    }
    summary["valid_for_causal_evaluation"] = (
        summary["valid_pair_count"] > 0
        and summary["causal_pair_valid_count"] == summary["valid_pair_count"]
    )
    for name in caches:
        l2_values = [
            float(row[f"{name}_target_l2"])
            for row in valid_rows
            if f"{name}_target_l2" in row
        ]
        summary["caches"][name] = {
            "pair_count": len(l2_values),
            "target_hidden_exact_equal_count": sum(
                bool(row.get(f"{name}_target_exact_equal")) for row in valid_rows
            ),
            "decision_hidden_exact_equal_count": sum(
                bool(row.get(f"{name}_decision_exact_equal")) for row in valid_rows
            ),
            "target_l2_mean": statistics.fmean(l2_values) if l2_values else 0.0,
            "target_l2_median": statistics.median(l2_values) if l2_values else 0.0,
            "target_l2_max": max(l2_values, default=0.0),
        }
    write_json(args.output_json, summary)
    write_jsonl(args.output_jsonl, pair_reports)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


def _load_named_caches(specs: list[str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"cache must be NAME=PATH: {spec}")
        name, path = spec.split("=", 1)
        obj = torch.load(path, map_location="cpu", weights_only=False)
        ids = [str(value) for value in obj["metadata"]["id"]]
        result[name] = {
            "stage_hidden": obj["stage_hidden"],
            "decision_hidden": obj["decision_hidden"],
            "index": {sample_id: index for index, sample_id in enumerate(ids)},
        }
    return result


def _stage_value(row: dict[str, Any], key: str, stage: int) -> int | None:
    targets = row.get("targets") or {}
    stage_targets = targets.get("stage") or {}
    values = stage_targets.get(key) or []
    masks = stage_targets.get(f"{key}_mask") or []
    if (
        stage not in range(len(values))
        or stage not in range(len(masks))
        or not int(masks[stage])
    ):
        return None
    return int(values[stage])


def _input_value(row: dict[str, Any], key: str) -> Any:
    model_input: dict[str, Any] = (
        row["model_input"] if isinstance(row.get("model_input"), dict) else {}
    )
    return model_input.get(key) if key in model_input else row.get(key)


def _observable_intervention(
    changed_factor: str,
    *,
    query_equal: bool,
    image_equal: bool,
    response_equal: bool,
    policy_equal: bool,
) -> bool:
    if changed_factor in {"visual_evidence", "ocr_text"}:
        return query_equal and not image_equal and response_equal and policy_equal
    if changed_factor == "image_text_binding":
        return (query_equal != image_equal) and response_equal and policy_equal
    if changed_factor == "intent":
        return not query_equal and image_equal and response_equal and policy_equal
    if changed_factor == "policy_rule":
        return query_equal and image_equal and response_equal and not policy_equal
    if changed_factor == "response_execution":
        return query_equal and image_equal and not response_equal and policy_equal
    return False


def _sha256(path: str, cache: dict[str, str]) -> str:
    if path in cache:
        return cache[path]
    source = Path(path)
    if not source.is_file():
        value = "missing"
    else:
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        value = digest.hexdigest()
    cache[path] = value
    return value


if __name__ == "__main__":
    main()
