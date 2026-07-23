from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from ..io_utils import read_jsonl, write_json, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit within-split identical-input label conflicts.")
    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Named JSONL in NAME=PATH form. Repeat as needed.",
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-jsonl", required=True)
    args = parser.parse_args()

    image_hash_cache: dict[str, str] = {}
    reports: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    for spec in args.input:
        name, path = _split_spec(spec)
        report, conflicts = audit_file(name, path, image_hash_cache)
        reports.append(report)
        conflict_rows.extend(conflicts)
    result = {
        "format": "identical_input_label_conflict_audit_v1",
        "valid_for_supervised_training": all(
            report["decision_conflict_group_count"] == 0 for report in reports
        ),
        "reports": reports,
    }
    write_json(args.output_json, result)
    write_jsonl(args.output_jsonl, conflict_rows)
    print(json.dumps(result, ensure_ascii=False), flush=True)


def audit_file(
    name: str,
    path: str,
    image_hash_cache: dict[str, str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = read_jsonl(path)
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_counts: Counter[str] = Counter()
    for row in rows:
        (
            query,
            images,
            response_candidate,
            policy_text,
            policy_text_supplied,
        ) = _model_input(row)
        image_hashes = tuple(_sha256(value, image_hash_cache) for value in images)
        key_payload = json.dumps(
            {
                "query": _normalize_text(query),
                "image_hashes": image_hashes,
                "response_candidate": _normalize_text(response_candidate),
                "policy_text": _normalize_text(policy_text),
                "policy_text_supplied": policy_text_supplied,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        key = hashlib.sha256(key_payload.encode("utf-8")).hexdigest()
        groups[key].append(row)
        source_counts[str(_metadata_value(row, "source_id") or "unknown")] += 1

    duplicate_groups = [values for values in groups.values() if len(values) > 1]
    decision_conflicts = [
        values
        for values in duplicate_groups
        if len({_target_value(row, "final_decision") for row in values}) > 1
    ]
    policy_conflicts = [
        values
        for values in duplicate_groups
        if len({_target_value(row, "policy_id") for row in values}) > 1
    ]
    role_conflicts = [
        values
        for values in duplicate_groups
        if len({_target_value(row, "role") for row in values}) > 1
    ]
    decision_correct_ceiling = sum(
        max(Counter(_target_value(row, "final_decision") for row in values).values())
        for values in groups.values()
    )
    policy_correct_ceiling = sum(
        max(Counter(_target_value(row, "policy_id") for row in values).values())
        for values in groups.values()
    )
    conflicts: list[dict[str, Any]] = []
    for values in decision_conflicts:
        (
            query,
            images,
            response_candidate,
            policy_text,
            policy_text_supplied,
        ) = _model_input(values[0])
        conflicts.append(
            {
                "dataset": name,
                "query": query,
                "images": images,
                "response_candidate": response_candidate,
                "policy_text": policy_text,
                "policy_text_supplied": policy_text_supplied,
                "row_count": len(values),
                "ids": [str(row.get("id") or row.get("candidate_id") or "") for row in values],
                "source_ids": sorted(
                    {str(_metadata_value(row, "source_id") or "") for row in values}
                ),
                "decisions": [_target_value(row, "final_decision") for row in values],
                "policies": [_target_value(row, "policy_id") for row in values],
                "roles": [_target_value(row, "role") for row in values],
                "target_stages": [_target_value(row, "target_stage") for row in values],
                "changed_factors": [
                    _target_value(row, "changed_factor") for row in values
                ],
            }
        )

    return (
        {
            "name": name,
            "path": path,
            "row_count": len(rows),
            "unique_input_count": len(groups),
            "duplicate_group_count": len(duplicate_groups),
            "duplicate_row_count": sum(len(values) for values in duplicate_groups),
            "decision_conflict_group_count": len(decision_conflicts),
            "decision_conflict_row_count": sum(len(values) for values in decision_conflicts),
            "policy_conflict_group_count": len(policy_conflicts),
            "role_conflict_group_count": len(role_conflicts),
            "deterministic_decision_accuracy_upper_bound": decision_correct_ceiling
            / max(len(rows), 1),
            "minimum_unavoidable_decision_errors": len(rows) - decision_correct_ceiling,
            "deterministic_policy_accuracy_upper_bound": policy_correct_ceiling
            / max(len(rows), 1),
            "minimum_unavoidable_policy_errors": len(rows) - policy_correct_ceiling,
            "source_id_count": len(source_counts),
            "rows_per_source_distribution": dict(Counter(source_counts.values())),
            "decision_distribution": dict(
                Counter(_target_value(row, "final_decision") for row in rows)
            ),
            "role_distribution": dict(Counter(_target_value(row, "role") for row in rows)),
        },
        conflicts,
    )


def _split_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"input must be NAME=PATH: {spec}")
    return tuple(spec.split("=", 1))  # type: ignore[return-value]


def _model_input(row: dict[str, Any]) -> tuple[str, list[str], str, str, bool]:
    model_input: dict[str, Any] = (
        row["model_input"] if isinstance(row.get("model_input"), dict) else {}
    )
    query = str(
        model_input.get("query")
        or row.get("compact_user_query")
        or row.get("query")
        or ""
    )
    images = model_input.get("images") or row.get("images") or []
    response_candidate = str(
        model_input.get("response_candidate")
        or row.get("response_candidate")
        or row.get("candidate_response")
        or ""
    )
    policy_text_supplied = (
        "policy_text" in model_input
        or "policy_deck" in model_input
        or bool(row.get("policy_text_supplied"))
        or "policy_text" in row
        or "policy_deck" in row
    )
    policy_text = str(
        model_input.get("policy_text")
        if "policy_text" in model_input
        else model_input.get("policy_deck")
        if "policy_deck" in model_input
        else row.get("policy_text")
        if "policy_text" in row
        else row.get("policy_deck")
        or ""
    )
    return (
        query,
        [str(value) for value in images],
        response_candidate,
        policy_text,
        policy_text_supplied,
    )


def _target_value(row: dict[str, Any], key: str) -> str:
    targets: dict[str, Any] = (
        row["targets"] if isinstance(row.get("targets"), dict) else {}
    )
    return str(targets.get(key) or row.get(key) or "unknown")


def _metadata_value(row: dict[str, Any], key: str) -> Any:
    metadata: dict[str, Any] = (
        row["metadata"] if isinstance(row.get("metadata"), dict) else {}
    )
    return metadata.get(key) or row.get(key)


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).casefold()


def _sha256(path: str, cache: dict[str, str]) -> str:
    if path in cache:
        return cache[path]
    source = Path(path)
    if not source.is_file():
        digest_value = f"missing:{path}"
    else:
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        digest_value = digest.hexdigest()
    cache[path] = digest_value
    return digest_value


if __name__ == "__main__":
    main()
