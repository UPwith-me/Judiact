from __future__ import annotations

from pathlib import Path

from causal_diagsafe.io_utils import write_jsonl
from causal_diagsafe.release import inspect_release
from causal_diagsafe.schema import normalize_record


def test_unified_nested_record_is_normalized() -> None:
    row = {
        "id": "nested-1",
        "modality_scope": "cross_modal",
        "model_input": {
            "query": "inspect this",
            "images": ["/images/a.jpg"],
            "response_candidate": "candidate",
            "policy_text": "Policy deck unavailable.",
        },
        "targets": {
            "final_decision": "warn",
            "target_stage": "S2",
            "changed_factor": "image_text_binding",
            "role": "safe",
            "policy_id": "P_MISINFO",
        },
        "metadata": {"source_dataset": "manual"},
    }
    record = normalize_record(row)
    assert record.query == "inspect this"
    assert record.images == ("/images/a.jpg",)
    assert record.response_candidate == "candidate"
    assert record.policy_text == "Policy deck unavailable."
    assert record.policy_text_supplied is True
    assert record.final_decision == "warn"
    assert record.target_stage == "S2"
    assert record.changed_factor == "image_text_binding"
    assert record.role == "safe"
    assert record.policy_id == "P_MISINFO"
    assert record.source_dataset == "manual"
    assert record.stage_states.states["S1"].E is None


def test_release_inspector_checks_nested_input_and_audit(tmp_path: Path) -> None:
    image_path = tmp_path / "a.jpg"
    image_path.write_bytes(b"test")
    split_specs = {}
    registry_rows = []
    for split in ("train", "dev", "test"):
        sample_id = f"sample-{split}"
        row = {
            "id": sample_id,
            "modality_scope": "cross_modal",
            "model_input": {"query": f"query-{split}", "images": [str(image_path)]},
            "targets": {
                "final_decision": "allow",
                "target_stage": "S2",
                "changed_factor": "image_text_binding",
                "role": "helpful",
                "policy_id": "P_BENIGN",
            },
            "metadata": {"source_dataset": "manual"},
        }
        audit = {
            "id": sample_id,
            "query": f"query-{split}",
            "images": [str(image_path)],
            "final_decision": "allow",
            "target_stage": "S2",
            "changed_factor": "image_text_binding",
            "role": "helpful",
            "policy_id": "P_BENIGN",
            "manual_binding_evidence": "The query depends on the image context.",
        }
        input_path = tmp_path / f"{split}.jsonl"
        audit_path = tmp_path / f"{split}_audit.jsonl"
        write_jsonl(input_path, [row])
        write_jsonl(audit_path, [audit])
        split_specs[split] = {"input": str(input_path), "audit": str(audit_path)}
        registry_rows.append({"group_id": f"g-{split}", "candidate_ids": [sample_id]})

    registry_path = tmp_path / "groups.jsonl"
    write_jsonl(registry_path, registry_rows)
    config = {
        "schema_version": "causal_diagsafe_unified_v1",
        "releases": {
            "combined": {
                "input_format": "nested_model_input",
                "group_registry": str(registry_path),
                "splits": split_specs,
            }
        },
    }
    summary = inspect_release(config)
    assert summary["blocking_issue_count"] == 0
    assert summary["releases"]["combined"]["splits"]["train"]["row_count"] == 1
    assert summary["releases"]["combined"]["splits"]["train"]["missing_image_count"] == 0
