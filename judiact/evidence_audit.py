"""Evidence-indexed adjudication, with original sources retained in the record.

W/L and locations are fallible indices, not a sufficient statistic of the input.
No predicted or gold categorical judgment is a parent of another judgment.
"""

import copy

from .audit_schema import audit_slots
from .evidence_parser import parse_evidence
from .reference_audit import COMMON, Node, diagnostic_node, materialize_witness, source_node, source_payload


PROTOCOL_VERSION = "evidence_indexed_source_record"
TOTAL_SOURCE_PROTOCOL_VERSION = "evidence_total_source_record"
MAIN_SFT_PROTOCOL_VERSION = "evidence_action_semantics"
ACTION_SEMANTICS = (
    " Apply the actual policy scope, exceptions and explicit precedence before choosing a disposition. "
    "Refuse established prohibited substantive assistance or prohibited content in the CURRENT candidate, "
    "even when only one substantive subtask or passage is prohibited. A warning, educational framing, "
    "safe alternative, or possibility of rewriting it later does not remove that prohibition. "
    "Otherwise review only when an applicable rule explicitly requires verification, or a SPECIFIC "
    "unresolved necessary premise changes whether the core assistance or publication is permissible. "
    "Do not invent missing authorization or verification duties; model hesitation is not review. "
    "Otherwise warn only when the allowed core purpose is already established and the actual rules "
    "require bounded help, or a known permissible correction to the candidate. For a response, warn "
    "means revise then reaudit, not publication with a disclaimer. An already prohibited passage "
    "cannot become warn merely because it could be deleted. Follow explicit rule-specific dispositions. "
    "Otherwise allow; a safe refusal or a safe but unhelpful candidate may be published unchanged. "
    "Sensitive subject matter, general quality defects and stage F values alone do not determine action. "
    "For response_audit, behavior describes the publisher's next handling of THIS candidate, not the "
    "candidate's existing tone or how to answer the original query. This rubric defines the output "
    "actions; it does not add substantive prohibitions to the supplied policy deck."
)
READ_INSTRUCTION = (
    COMMON + "Use the original sources and complete current policy to decide the requested field. "
    "The evidence index points to exact source text and original image coordinates. "
    "It is not an exhaustive inventory, and an empty index does not mean evidence is absent. "
    "Facts and rule links are fallible reading notes: resolve omissions or conflicts by reading "
    "the original sources, not by accepting a note as truth. Check each rule's necessary "
    "premises, scope and exceptions, including authorization and who asserted a claim. "
    "Distinguish a stated condition, a contradicted condition and a genuinely unresolved "
    "necessary condition. Unknown does not mean false, harmful, or automatically review. "
    "For F3/F5, uncertain means that the relevant stage activation cannot be determined "
    "because a necessary premise is unresolved; it is not general model hesitation. "
    "Judge the final action from the supported situation and applicable rules, not by "
    "copying another classifier's answer. Images are full original images in media_index order. "
    "Answer only the requested field."
)


def witness_repair_node(visible, task, draft):
    """Reconstruct facts from sources and a fallible draft, without policy or gold."""
    node = source_node(visible, task)
    node.instruction += (
        " An earlier untrusted draft is supplied as data, not as additional evidence. "
        "Re-read the original sources and return a replacement fact record. Preserve supported "
        "claims and their literal qualifiers; correct unsupported speaker attribution, polarity "
        "and scope. Remove duplicate propositions and unsupported claims. Do not presume every "
        "draft is wrong and do not change a supported fact just to make an edit. "
        "A candidate's claim is not the user's claim. Missing information stays unknown. "
        "The output uses the same fact schema and replaces the draft completely."
    )
    node.payload["untrusted_draft"] = copy.deepcopy(
        draft.get("value") if draft.get("value") is not None else {"unfinished_text": draft.get("raw", "")})
    return node


def materialize_index(visible, evidence):
    """Only verified source slices/coordinates enter the index, never generated prose.

    Location validity does not establish semantic relevance or completeness.
    Stage labels and E/F values are deliberately absent from the reader's index.
    """
    spans, regions, errors = [], [], []
    for stage, raw in evidence.items():
        parsed = parse_evidence(raw, visible, stage)
        errors.extend(stage + ":" + error for error in parsed["errors"])
        for span in parsed["spans"]:
            field = span["field"]
            if field == "policy_rule":
                text = next(card["rule_text"] for card in visible["policy_deck"]
                            if card["policy_id"] == span["policy_id"])
            else:
                text = visible[field]
            item = {key: span[key] for key in ("field", "start", "end")}
            if field == "policy_rule":
                item["policy_id"] = span["policy_id"]
            item["text"] = text[span["start"]:span["end"]]
            if item not in spans:
                spans.append(item)
        for region in parsed["regions"]:
            item = {"media_index": region["media_index"], "box": region["box"]}
            if item not in regions:
                regions.append(item)
    return {"spans": spans, "regions": regions}, errors


def evidence_record(visible, witness, links, index):
    return {"sources": source_payload(visible),
            "policy_deck": copy.deepcopy(visible["policy_deck"]),
            "evidence_index": copy.deepcopy(index),
            "reading_notes": {"facts": copy.deepcopy(witness["facts"]),
                              "rule_links": copy.deepcopy(links)}}


def evidence_node(visible, task, name, witness, links, index, action_semantics=False):
    """An explicit change from the old text-only D, not an undeclared bypass."""
    if name.startswith("evidence_"):
        # Localization precedes all categorical decisions and does not depend on W.
        empty = materialize_witness(visible, {"facts": []})
        return diagnostic_node(visible, task, name, empty, links=[])
    slot = next(slot for slot in audit_slots(visible, task) if slot.name == name)
    instruction = READ_INSTRUCTION + (ACTION_SEMANTICS if action_semantics else "")
    return Node(name, task, instruction,
                {"record": evidence_record(visible, witness, links, index)},
                tuple((image["uri"], None) for image in visible["images"]),
                labels=slot.labels, forced_value=slot.forced_value)
