from __future__ import annotations

import json

from causal_diagsafe.cli.audit_h1_pairs import _observable_intervention
from causal_diagsafe.cli.audit_input_conflicts import audit_file


def test_observable_intervention_contracts() -> None:
    assert _observable_intervention(
        "intent",
        query_equal=False,
        image_equal=True,
        response_equal=True,
        policy_equal=True,
    )
    assert _observable_intervention(
        "image_text_binding",
        query_equal=True,
        image_equal=False,
        response_equal=True,
        policy_equal=True,
    )
    assert not _observable_intervention(
        "response_execution",
        query_equal=True,
        image_equal=True,
        response_equal=True,
        policy_equal=True,
    )
    assert not _observable_intervention(
        "policy_rule",
        query_equal=True,
        image_equal=True,
        response_equal=False,
        policy_equal=False,
    )
    assert not _observable_intervention(
        "response_execution",
        query_equal=True,
        image_equal=True,
        response_equal=False,
        policy_equal=False,
    )


def test_identical_input_conflicting_decisions_are_reported(tmp_path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"same-image")
    source = tmp_path / "rows.jsonl"
    rows = [
        {
            "id": "left",
            "model_input": {"query": "same query", "images": [str(image)]},
            "targets": {"final_decision": "refuse", "policy_id": "P_A"},
        },
        {
            "id": "right",
            "model_input": {"query": "same query", "images": [str(image)]},
            "targets": {"final_decision": "warn", "policy_id": "P_A"},
        },
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    report, conflicts = audit_file("test", str(source), {})

    assert report["unique_input_count"] == 1
    assert report["decision_conflict_group_count"] == 1
    assert report["decision_conflict_row_count"] == 2
    assert conflicts[0]["decisions"] == ["refuse", "warn"]


def test_missing_and_explicitly_unavailable_policy_are_distinct_inputs(
    tmp_path,
) -> None:
    source = tmp_path / "rows.jsonl"
    rows = [
        {
            "id": "fixed-deck",
            "model_input": {"query": "same query", "images": []},
            "targets": {"final_decision": "review"},
        },
        {
            "id": "missing-deck",
            "model_input": {
                "query": "same query",
                "images": [],
                "policy_text": "",
            },
            "targets": {"final_decision": "review"},
        },
    ]
    source.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    report, _ = audit_file("test", str(source), {})
    assert report["unique_input_count"] == 2
