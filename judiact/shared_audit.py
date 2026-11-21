"""Source-visible audit with fixed categorical inputs and shared hidden states."""

from dataclasses import dataclass

from .audit_schema import audit_slots, choice_codebook
from .reference_audit import Node, canonical
from .source_audit import READ_INSTRUCTION, source_judgment_node, source_record


PROTOCOL_VERSION = "shared_source_audit"
SHARED_INSTRUCTION = READ_INSTRUCTION.replace(
    "Answer only the requested field.",
    "Read each audit field in order. The fixed question marks in the assistant "
    "audit scaffold are placeholders, not observed classifications. Each field "
    "must be judged from the actual sources and rules; a preceding classifier "
    "answer is never supplied. Integrate the applicable evidence and conditions "
    "for the final action. Source reads remain fallible selections, not new facts.")


@dataclass
class SharedAuditNode(Node):
    slots: tuple = ()

    @property
    def text(self):
        return (self.instruction + "\nTask: " + self.task + "\n" + canonical(self.payload)
                + "\n" + choice_codebook(self.slots))


def shared_slots(visible, task):
    slots = tuple(slot for slot in audit_slots(visible, task) if slot.kind == "classify")
    # The action is read last; neither behavior nor policy conclusions are fed back.
    return tuple(slot for slot in slots if slot.name != "action") + tuple(
        slot for slot in slots if slot.name == "action")


def shared_judgment_node(visible, task, name, index=None):
    if name == "basis" or name.startswith("evidence_"):
        return source_judgment_node(visible, task, name, index)
    slots = shared_slots(visible, task)
    slot = next(slot for slot in slots if slot.name == name)
    record, media = source_record(visible, index)
    return SharedAuditNode(name, task, SHARED_INSTRUCTION, {"record": record}, media,
                           labels=slot.labels, forced_value=slot.forced_value, slots=slots)
