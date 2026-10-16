"""Evidence remains context; categorical predictions are readouts, not new facts."""

from .audit_v5 import audit_slots, evidence_schema, parse_evidence, visible_only
from .audit_v5 import audit_prompt as coherent_prompt


PROTOCOL_VERSION = "grounded_audit_v6_20261016"


def audit_prompt(visible, task_type=None):
    prompt = coherent_prompt(visible, task_type)
    if audit_slots(visible, task_type)[0].name.startswith("native_"):
        return prompt
    return prompt + (
        "\nCategorical answers are recorded separately. Their positions in the continuing "
        "context contain a fixed question mark, not a selected answer or an uncertain label. "
        "Judge each slot from the original input and the factual evidence. No earlier "
        "categorical answer is supplied, and none should be inferred from a question mark."
    )
