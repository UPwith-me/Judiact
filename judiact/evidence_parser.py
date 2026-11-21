"""Coherent audit records; unestablished localization is distinct from absence."""

import json

from . import audit_schema
from .audit_schema import audit_slots, visible_only


PROTOCOL_VERSION = "coherent_audit"


def audit_prompt(visible, task_type=None):
    prompt = audit_schema.audit_prompt(visible, task_type)
    if audit_slots(visible, task_type)[0].name.startswith("native_"):
        return prompt
    return prompt + (
        "\nEach evidence block selects observed supporting facts; it is not an exhaustive inventory. "
        "Describe a region's visible object or wording before its media_index and bbox. "
        "If localization cannot be established, emit JSON null for that evidence block. "
        "null means unestablished localization, never absent evidence or an inactive risk. "
        "Empty quotes/regions arrays record that no support was located; judge E separately."
    )


def evidence_schema(visible, stage):
    schema = audit_schema.evidence_schema(visible, stage)
    return {"anyOf": [schema, {"type": "null"}]}


def parse_evidence(answer, visible, stage):
    try:
        value = json.loads(answer)
    except json.JSONDecodeError:
        value = ...
    if value is None:
        return {"stage": stage, "raw": answer, "spans": [], "regions": [], "errors": [],
                "localization_status": "unestablished"}
    result = audit_schema.parse_evidence(answer, visible, stage)
    result["localization_status"] = "selected_support"
    return result
