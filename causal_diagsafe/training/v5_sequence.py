"""Teacher-forced compatible records, read in one causal LM forward per batch."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .. import audit_v5
from ..audit_v5 import audit_slots, visible_only
from ..inference_v1 import move_tensors
from .v4_losses import MemberLosses
from .v4_sequence import AuditEncoder, AuditTrajectory


@dataclass
class EncodedRecord:
    sample_id: str
    trajectory: AuditTrajectory
    # Each field stores its actual prediction positions and targets, not header tokens.
    token_targets: dict
    localization_states: dict


class CoherentAuditEncoder(AuditEncoder):
    def __init__(self, processor, max_evidence_tokens=512, use_cache=True, optimize=False):
        super().__init__(processor, max_evidence_tokens, use_cache, optimize, audit_protocol=audit_v5)

    def evidence_loss(self, logits, targets):
        return F.cross_entropy(logits.float(), targets)

    def encode_case(self, case):
        visible, task = visible_only(case["model_input"]), case["task_type"]
        base = self.prepare(visible, task)
        pieces = [base["input_ids"].flatten()]
        length = prompt_length = pieces[0].numel()
        entries, values, token_targets = [], {}, {}
        for slot in audit_slots(visible, task):
            header = torch.tensor(self.token_ids(slot.header), dtype=torch.long)
            pieces.append(header)
            length += header.numel()
            start = length
            observed = slot.name in case["targets"]
            candidates = None
            if slot.kind == "classify":
                assert observed, (case["sample_id"], slot.name, "missing categorical target")
                target = case["targets"][slot.name]
                assert slot.forced_value is None or target == slot.forced_value
                candidates = self.candidate_ids(slot)
                ids = [candidates[slot.labels.index(target)]]
                values[slot.name] = target
            else:
                # This marker states that localization is unestablished. It is not
                # an empty evidence target, and none of its tokens receive loss.
                ids = self.token_ids(case["targets"].get(slot.name, "null"))
                if observed:
                    assert len(ids) + 1 <= self.max_evidence_tokens, (case["sample_id"], slot.name, len(ids))
                    positions = torch.arange(start - 1, start + len(ids), dtype=torch.long)
                    token_targets[slot.name] = (positions, torch.tensor(
                        ids + [self.processor.tokenizer.eos_token_id], dtype=torch.long))
            pieces.append(torch.tensor(ids, dtype=torch.long))
            length += len(ids)
            entry = {"name": slot.name, "kind": slot.kind, "answer_start": start,
                     "answer_end": length, "candidate_token_ids": candidates,
                     "forced_by_contract": slot.forced_value}
            if slot.kind == "classify":
                entry.update(labels=list(slot.labels), value=target)
            entries.append(entry)
        ids = torch.cat(pieces).reshape(1, -1)
        trajectory = AuditTrajectory(visible, task, base, ids, prompt_length, entries,
                                     values, {}, [], {}, {"input_tokens": prompt_length})
        return EncodedRecord(case["sample_id"], trajectory, token_targets,
                             case.get("localization_states", {}))

    def read_records(self, model, records, absolute_ids=()):
        """All readouts use known preceding fields; each absolute member is counted once."""
        supervised = set(absolute_ids)
        trajectories = [item.trajectory for item in records]
        length = max(t.input_ids.numel() for t in trajectories)
        maps = [t.classification_readouts() for t in trajectories]
        row_positions = []
        for record, mapping in zip(records, maps):
            offset = length - record.trajectory.input_ids.numel()
            positions = [entry["position"] + offset for entry in mapping.values()]
            if record.sample_id in supervised:
                positions.extend(int(pos) + offset for local, _ in record.token_targets.values() for pos in local)
            row_positions.append(positions)
        selected = sorted({pos for positions in row_positions for pos in positions})
        columns = {position: index for index, position in enumerate(selected)}
        device = next(model.parameters()).device
        inputs = move_tensors(self.batch_inputs([t.base_inputs for t in trajectories],
                                                [t.input_ids.flatten() for t in trajectories]), device)
        with self.visual_context(model, [t.visible for t in trajectories]):
            logits = model(**inputs, use_cache=False,
                           logits_to_keep=torch.tensor(selected, device=device)).logits
        readouts, members = {}, {}
        for row, (record, mapping) in enumerate(zip(records, maps)):
            trajectory = record.trajectory
            offset = length - trajectory.input_ids.numel()
            scores = {name: logits[row, columns[entry["position"] + offset], entry["candidate_token_ids"]]
                      .float().log_softmax(-1) for name, entry in mapping.items()}
            readouts[record.sample_id] = scores
            if record.sample_id not in supervised:
                continue
            losses = {name: -scores[name][entry["labels"].index(trajectory.values[name])]
                      for name, entry in ((item["name"], item) for item in trajectory.entries)
                      if entry["kind"] == "classify"}
            if trajectory.task_type.startswith("native_"):
                members[record.sample_id] = MemberLosses(None, {}, losses[trajectory.task_type])
                continue
            action = losses.pop("action")
            for name, (positions, targets) in record.token_targets.items():
                indices = torch.tensor([columns[int(p) + offset] for p in positions], device=device)
                losses[name] = self.evidence_loss(logits[row, indices], targets.to(device))
            members[record.sample_id] = MemberLosses(action, losses, None)
        return readouts, members
