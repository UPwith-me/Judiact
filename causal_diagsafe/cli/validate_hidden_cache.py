from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from ..io_utils import read_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate exported H1 hidden-state caches.")
    parser.add_argument("--cache", action="append", required=True)
    parser.add_argument("--manifest", action="append", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()
    if len(args.cache) != len(args.manifest):
        raise ValueError("--cache and --manifest must have equal counts")

    reports = [
        validate_cache(cache_path, manifest_path)
        for cache_path, manifest_path in zip(args.cache, args.manifest)
    ]
    result = {
        "format": "hidden_cache_validation_v2",
        "valid": all(report["valid"] for report in reports),
        "cache_count": len(reports),
        "reports": reports,
    }
    write_json(args.output_json, result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    if not result["valid"]:
        raise SystemExit(2)


def validate_cache(cache_path: str, manifest_path: str) -> dict[str, Any]:
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    manifest = read_jsonl(manifest_path)
    issues: list[str] = []
    stage_hidden = cache.get("stage_hidden")
    decision_hidden = cache.get("decision_hidden")
    labels = cache.get("labels") or {}
    metadata = cache.get("metadata") or {}
    export = cache.get("export") if isinstance(cache.get("export"), dict) else {}
    max_samples = int(export.get("max_samples", 0) or 0)
    if max_samples > 0:
        manifest = manifest[:max_samples]

    if cache.get("format") != "causal_diagsafe_hidden_cache_v2":
        issues.append("cache format must be causal_diagsafe_hidden_cache_v2")
    if export.get("prompt_contract_version") != "model_visible_policy_response_v2":
        issues.append("cache prompt contract is stale or missing")
    if export.get("input_sha256") != _sha256_file(manifest_path):
        issues.append("cache input hash does not match the current manifest")
    if not isinstance(export.get("sentinel_block_token_count"), int):
        issues.append("cache is missing sentinel block token count")
    terminal_indices = export.get("sentinel_terminal_token_indices")
    if not isinstance(terminal_indices, list) or len(terminal_indices) != 6:
        issues.append("cache must record six sentinel terminal token indices")

    if not isinstance(stage_hidden, torch.Tensor) or stage_hidden.ndim != 3:
        issues.append("stage_hidden must be a rank-3 tensor")
        row_count = 0
        hidden_size = 0
    else:
        row_count = int(stage_hidden.shape[0])
        hidden_size = int(stage_hidden.shape[-1])
        if tuple(stage_hidden.shape[1:2]) != (5,):
            issues.append(f"stage_hidden stage axis must be 5: {tuple(stage_hidden.shape)}")
        if not torch.isfinite(stage_hidden).all().item():
            issues.append("stage_hidden contains non-finite values")

    if not isinstance(decision_hidden, torch.Tensor) or decision_hidden.ndim != 2:
        issues.append("decision_hidden must be a rank-2 tensor")
    else:
        if int(decision_hidden.shape[0]) != row_count:
            issues.append("decision_hidden row count mismatch")
        if hidden_size and int(decision_hidden.shape[1]) != hidden_size:
            issues.append("decision_hidden hidden size mismatch")
        if not torch.isfinite(decision_hidden).all().item():
            issues.append("decision_hidden contains non-finite values")

    if row_count != len(manifest):
        issues.append(f"cache/manifest row mismatch: {row_count} != {len(manifest)}")
    expected_ids = [str(row.get("id") or "") for row in manifest]
    cache_ids = [str(value) for value in metadata.get("id", [])]
    if cache_ids != expected_ids:
        issues.append("cache IDs do not exactly match manifest order")
    if len(cache_ids) != len(set(cache_ids)):
        issues.append("cache IDs are not unique")
    required_labels = {
        "decision",
        "decision_mask",
        "policy_action",
        "policy_action_mask",
        "benign",
        "benign_mask",
        "intent",
        "intent_mask",
        "R",
        "R_mask",
        "E",
        "E_mask",
        "F",
        "F_mask",
        "target_stage",
    }
    missing_labels = sorted(required_labels - set(labels))
    if missing_labels:
        issues.append(f"cache labels missing: {missing_labels}")
    required_metadata = {
        "id",
        "group_id",
        "source_dataset",
        "modality_scope",
        "role",
        "changed_factor",
        "policy_condition",
        "has_response_candidate",
        "policy_text_supplied",
    }
    missing_metadata = sorted(required_metadata - set(metadata))
    if missing_metadata:
        issues.append(f"cache metadata missing: {missing_metadata}")
    for key, value in labels.items():
        if not isinstance(value, torch.Tensor):
            issues.append(f"label {key} is not a tensor")
        elif int(value.shape[0]) != row_count:
            issues.append(f"label {key} row count mismatch")
    stage_shapes = {
        key: (row_count, 5)
        for key in ("R", "R_mask", "E", "E_mask", "F", "F_mask")
    }
    vector_shapes = {
        key: (row_count,)
        for key in (
            "decision",
            "decision_mask",
            "policy_action",
            "policy_action_mask",
            "benign",
            "benign_mask",
            "intent",
            "intent_mask",
            "target_stage",
        )
    }
    for key, expected_shape in {**stage_shapes, **vector_shapes}.items():
        value = labels.get(key)
        if isinstance(value, torch.Tensor) and tuple(value.shape) != expected_shape:
            issues.append(
                f"label {key} shape mismatch: {tuple(value.shape)} != {expected_shape}"
            )
    for key in (
        "decision_mask",
        "policy_action_mask",
        "benign_mask",
        "intent_mask",
        "R_mask",
        "E_mask",
        "F_mask",
    ):
        value = labels.get(key)
        if isinstance(value, torch.Tensor) and not torch.all(
            (value == 0) | (value == 1)
        ).item():
            issues.append(f"label {key} must be binary")
    for key, value in metadata.items():
        if len(value) != row_count:
            issues.append(f"metadata {key} row count mismatch")

    return {
        "cache": str(Path(cache_path)),
        "manifest": str(Path(manifest_path)),
        "valid": not issues,
        "row_count": row_count,
        "hidden_size": hidden_size,
        "stage_shape": list(stage_hidden.shape) if isinstance(stage_hidden, torch.Tensor) else None,
        "decision_shape": (
            list(decision_hidden.shape) if isinstance(decision_hidden, torch.Tensor) else None
        ),
        "issues": issues,
    }


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
