from __future__ import annotations

import torch

from causal_diagsafe.cli.validate_hidden_cache import _sha256_file, validate_cache
from causal_diagsafe.io_utils import write_jsonl


def test_hidden_cache_validation_binds_cache_to_manifest(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    write_jsonl(manifest, [{"id": "sample-1"}])
    labels = {
        "decision": torch.zeros(1, dtype=torch.long),
        "decision_mask": torch.ones(1),
        "policy_action": torch.zeros(1, dtype=torch.long),
        "policy_action_mask": torch.zeros(1),
        "benign": torch.zeros(1),
        "benign_mask": torch.zeros(1),
        "intent": torch.zeros(1, dtype=torch.long),
        "intent_mask": torch.zeros(1),
        "R": torch.zeros(1, 5),
        "R_mask": torch.zeros(1, 5),
        "E": torch.zeros(1, 5, dtype=torch.long),
        "E_mask": torch.zeros(1, 5),
        "F": torch.zeros(1, 5),
        "F_mask": torch.zeros(1, 5),
        "target_stage": torch.full((1,), -1, dtype=torch.long),
    }
    metadata = {
        "id": ["sample-1"],
        "group_id": [""],
        "source_dataset": ["manual"],
        "modality_scope": ["text_only"],
        "role": ["safe"],
        "changed_factor": ["none"],
        "policy_condition": ["unknown"],
        "has_response_candidate": [False],
        "policy_text_supplied": [False],
    }
    cache = {
        "format": "causal_diagsafe_hidden_cache_v2",
        "stage_hidden": torch.zeros(1, 5, 8),
        "decision_hidden": torch.zeros(1, 8),
        "labels": labels,
        "metadata": metadata,
        "export": {
            "prompt_contract_version": "model_visible_policy_response_v2",
            "input_sha256": _sha256_file(manifest),
            "sentinel_block_token_count": 12,
            "sentinel_terminal_token_indices": [1, 3, 5, 7, 9, 11],
        },
    }
    cache_path = tmp_path / "cache.pt"
    torch.save(cache, cache_path)
    assert validate_cache(str(cache_path), str(manifest))["valid"]

    write_jsonl(manifest, [{"id": "sample-1", "changed": True}])
    report = validate_cache(str(cache_path), str(manifest))
    assert not report["valid"]
    assert "cache input hash does not match the current manifest" in report["issues"]
