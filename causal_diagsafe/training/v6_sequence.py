"""Candidate V6: compatible token decoding and explicit record-likelihood scoring."""

from dataclasses import replace

import torch
import torch.nn.functional as F

from .v4_sequence import AuditTrajectory
from .v5_sequence import CoherentAuditEncoder, EncodedRecord


def trajectory_facts(trajectory):
    return {entry["name"]: {"raw": entry["raw"], "token_ids": trajectory.input_ids[
        0, entry["answer_start"]:entry["answer_end"]].tolist()}
        for entry in trajectory.entries if entry["kind"] == "evidence"}


class GroundedAuditEncoder(CoherentAuditEncoder):
    def __init__(self, *args, record_objective="joint", label_context="autoregressive", **kwargs):
        super().__init__(*args, **kwargs)
        self.record_objective = record_objective
        self.grammar = None
        self.label_context = label_context
        if label_context == "blind":
            from .. import audit_v6
            self.protocol = audit_v6
            tokens = self.token_ids(" ?")
            assert len(tokens) == 1, "categorical context replacement must preserve positions"
            self.placeholder_id = tokens[0]

    def context_tokens(self, slot, emitted):
        if self.label_context == "blind" and slot.kind == "classify":
            return torch.full_like(emitted, self.placeholder_id)
        return emitted

    def encode_case(self, case):
        record = super().encode_case(case)
        if self.label_context == "blind":
            for entry in record.trajectory.entries:
                if entry["kind"] == "classify":
                    assert entry["answer_end"] - entry["answer_start"] == 1
                    record.trajectory.input_ids[0, entry["answer_start"]] = self.placeholder_id
        return record

    def evidence_constraint(self, visible, stage):
        from ..audit_constraints import AuditGrammar
        if self.grammar is None:
            self.grammar = AuditGrammar(self.processor.tokenizer)
        return self.grammar.constraint(self.protocol.evidence_schema(visible, stage))

    def encode_factual_context(self, case, facts):
        """Exact generated evidence tokens; gold classes are targets only, never context."""
        visible, task = case["model_input"], case["task_type"]
        base = self.prepare(visible, task)
        pieces = [base["input_ids"].flatten()]
        length = prompt_length = pieces[0].numel()
        entries, values = [], {}
        for slot in self.protocol.audit_slots(visible, task):
            header = torch.tensor(self.token_ids(slot.header), dtype=torch.long)
            pieces.append(header)
            length += len(header)
            start = length
            if slot.kind == "classify":
                candidates = self.candidate_ids(slot)
                body = [self.placeholder_id]
                values[slot.name] = case["targets"][slot.name]
            else:
                candidates, body = None, facts[slot.name]["token_ids"]
            pieces.append(torch.tensor(body, dtype=torch.long))
            length += len(body)
            entry = {"name": slot.name, "kind": slot.kind, "answer_start": start, "answer_end": length,
                     "candidate_token_ids": candidates, "forced_by_contract": slot.forced_value}
            if slot.kind == "classify":
                entry.update(labels=list(slot.labels), value=values[slot.name])
            entries.append(entry)
        trajectory = AuditTrajectory(visible, task, base, torch.cat(pieces).reshape(1, -1), prompt_length,
                                     entries, values, {}, [], {}, {"input_tokens": prompt_length})
        return EncodedRecord(case["sample_id"], trajectory, {}, {})

    def evidence_loss(self, logits, targets):
        return F.cross_entropy(logits.float(), targets,
                               reduction="mean" if self.record_objective == "legacy" else "sum")

    def cross_record(self, input_record, candidate_record):
        """The same candidate prefix is read under every input, with no row-specific gold."""
        source = candidate_record.trajectory
        base = input_record.trajectory.base_inputs
        length = base["input_ids"].numel()
        delta = length - source.prompt_length
        ids = torch.cat((base["input_ids"].flatten(), source.input_ids[0, source.prompt_length:])).reshape(1, -1)
        entries = [{**entry, "answer_start": entry["answer_start"] + delta,
                    "answer_end": entry["answer_end"] + delta} for entry in source.entries]
        trajectory = replace(source, visible=input_record.trajectory.visible, base_inputs=base,
                             input_ids=ids, prompt_length=length, entries=entries)
        return replace(candidate_record, sample_id=input_record.sample_id + "::" + candidate_record.sample_id,
                       trajectory=trajectory,
                       token_targets={name: (positions + delta, targets) for name, (positions, targets)
                                      in candidate_record.token_targets.items()})
