from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from causal_diagsafe.algorithm_data import build_algorithm_examples
from causal_diagsafe.clean_reproduction import (
    POLICY_DECK,
    build_clean_reproduction_bundle,
    validate_clean_rows,
)
from causal_diagsafe.io_utils import write_jsonl
from causal_diagsafe.training.multimodal_sft import validate_training_manifest


def _row(sample_id: str, role: str, decision: str, *, image: str = "") -> dict:
    return {
        "id": sample_id,
        "modality_scope": "cross_modal" if image else "text_only",
        "model_input": {"query": f"query for {sample_id}", "images": [image] if image else []},
        "targets": {
            "final_decision": decision,
            "policy_id": "P_BENIGN" if decision != "refuse" else "P_VIOLENCE",
            "fine_policy_id": "P_BENIGN" if decision != "refuse" else "P_VIOLENCE",
            "role": role,
            "target_stage": "S2" if image else "S3",
            "changed_factor": "image_text_binding" if image else "intent",
            "gold_response_policy": f"safe policy for {sample_id}",
        },
        "metadata": {"source_dataset": "manual_source"},
    }


def test_clean_bundle_keeps_images_and_hides_training_controls() -> None:
    rows = [
        _row("unsafe", "unsafe", "refuse", image="/images/a.jpg"),
        _row("safe", "safe", "warn"),
        _row("helpful", "helpful", "allow"),
    ]
    examples, issues = build_algorithm_examples(
        rows,
        split="train",
        group_by_id={row["id"]: f"group-{row['id']}" for row in rows},
    )
    assert not issues
    bundle = build_clean_reproduction_bundle(examples, max_calibration_groups=1, seed=7)
    assert len(bundle.joint_sft) == 3
    assert len(bundle.calibration_groups) == 1
    assert bundle.calibration_groups[0]["causal_group"] is False
    unsafe = next(row for row in bundle.joint_sft if row["id"] == "unsafe")
    assert unsafe["images"] == ["/images/a.jpg"]
    assert unsafe["training_target"]["safe_response_policy"] == "safe policy for unsafe"
    visible_prompt = " ".join(message["content"] for message in unsafe["messages"][:2])
    assert "manual_source" not in visible_prompt
    assert "group-unsafe" not in visible_prompt
    assert not validate_clean_rows(bundle.joint_sft)


def test_clean_sft_preflight_checks_every_image(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    image_path = tmp_path / "image.jpg"
    image_path.write_bytes(b"placeholder")
    rows = [_row("helpful", "helpful", "allow", image=str(image_path))]
    examples, issues = build_algorithm_examples(
        rows,
        split="train",
        group_by_id={"helpful": "g1"},
    )
    assert not issues
    bundle = build_clean_reproduction_bundle(examples, max_calibration_groups=0)
    manifest = tmp_path / "train.jsonl"
    write_jsonl(manifest, bundle.joint_sft)
    report = validate_training_manifest(manifest, model_path=model_dir)
    assert report["row_count"] == 1
    assert report["image_count"] == 1
    assert report["missing_image_count"] == 0
    assert report["issue_count"] == 0


def test_response_and_policy_interventions_are_model_visible() -> None:
    row = _row("s5", "unsafe", "refuse")
    row["model_input"]["response_candidate"] = "candidate response under audit"
    row["model_input"]["policy_text"] = "Policy deck unavailable."
    examples, issues = build_algorithm_examples(
        [row],
        split="train",
        group_by_id={"s5": "g1"},
    )
    assert not issues
    bundle = build_clean_reproduction_bundle(examples, max_calibration_groups=0)
    exported = bundle.joint_sft[0]
    assert exported["model_input"]["response_candidate"] == "candidate response under audit"
    assert exported["model_input"]["policy_text"] == "Policy deck unavailable."
    assert "candidate response under audit" in exported["messages"][1]["content"]
    assert "Policy deck unavailable." in exported["messages"][0]["content"]
    assert POLICY_DECK not in exported["messages"][0]["content"]


def test_clean_validation_rejects_policy_id_missing_from_fixed_deck() -> None:
    rows = [_row("safe", "safe", "warn")]
    examples, issues = build_algorithm_examples(
        rows,
        split="train",
        group_by_id={"safe": "g1"},
    )
    assert not issues
    bundle = build_clean_reproduction_bundle(examples, max_calibration_groups=0)
    invalid_row = deepcopy(bundle.joint_sft[0])
    invalid_row["training_target"]["policy_id"] = "P_NOT_IN_DECK"
    assert validate_clean_rows([invalid_row]) == ["safe:policy_id_not_in_deck:P_NOT_IN_DECK"]
