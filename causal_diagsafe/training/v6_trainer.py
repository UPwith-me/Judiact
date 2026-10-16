"""Focused V6 objective comparisons on the unchanged coherent-audit contract."""

from collections import defaultdict

import torch

from .v4_losses import MemberLosses, absolute_loss, absolute_weights
from .v5_matching import conditional_matching_nll
from .v5_trainer import CoherentTrainer
from .v6_matching import candidate_log_odds, compatible_prefix_codebooks, crossed_matching_nll
from .v6_sequence import GroundedAuditEncoder
from .v6_supervision import balanced_members, risk_action_balance


def record_normalizer(data, tokenizer):
    """One fixed train-population constant; no target-length-dependent member weight."""
    weights = absolute_weights(data.populations)
    total, blocks, categorical = 0., [], 0.
    field_mass, field_tokens = defaultdict(float), defaultdict(float)
    for sid in data.records:
        targets = data.cases[sid]["targets"]
        count = sum(not name.startswith("evidence_") for name in targets)
        categorical += weights[sid] * count
        for name, target in targets.items():
            if name.startswith("evidence_"):
                blocks.append((weights[sid], name, target))
    total = categorical
    for start in range(0, len(blocks), 512):
        part = blocks[start:start + 512]
        ids = tokenizer([text for _, _, text in part], add_special_tokens=False)["input_ids"]
        for (weight, name, _), tokens in zip(part, ids):
            total += weight * (len(tokens) + 1)
            field_tokens[name] += weight * (len(tokens) + 1)
            field_mass[name] += weight
    return {"constant": total, "categorical_tokens": categorical,
            "evidence_tokens_including_stop": total - categorical,
            "evidence_field_mean_tokens": {name: value / field_mass[name] for name, value in field_tokens.items()},
            "scope": "full frozen training population; request/response budgets and root weights retained"}


class GroundedTrainer(CoherentTrainer):
    def __init__(self, *args, record_objective="joint", matching_mode="crossed",
                 normalization=1., label_context="autoregressive", evidence_normalization=None, fact_bank=None,
                 class_balance="none", **kwargs):
        super().__init__(*args, **kwargs)
        self.record_objective, self.matching_mode = record_objective, matching_mode
        self.normalization = normalization
        self.evidence_normalization = evidence_normalization
        self.fact_bank = fact_bank
        self.class_balance = class_balance
        self.balance = risk_action_balance(self.data, determinate_only=class_balance == "determinate_risk_action") \
            if class_balance != "none" else None
        if self.balance is not None:
            assert record_objective == "field_fixed", "class masses refer to the actual field-fixed objective"
        self.encoder = GroundedAuditEncoder(self.processing_class, self.encoder.max_evidence_tokens,
                                           optimize=True, record_objective=record_objective,
                                           label_context=label_context)

    def packet_jobs(self, packet):
        jobs = super().packet_jobs(packet)
        if self.fact_bank is not None:
            for job in jobs:
                job["gold_evidence_records"] = [r for r in job["records"]
                    if r.sample_id in job["absolute_ids"] and not r.trajectory.task_type.startswith("native_")]
                job["records"] = [r if r.trajectory.task_type.startswith("native_") else
                    self.encoder.encode_factual_context(self.data.cases[r.sample_id], self.fact_bank[r.sample_id])
                    for r in job["records"]]
        if self.matching_mode.startswith("crossed"):
            for job in jobs:
                terms = []
                for field, component, weight in job["matching_terms"]:
                    cases = [self.data.cases[sid] for sid in component.sample_ids]
                    if compatible_prefix_codebooks(cases, field):
                        terms.append((field, component, weight))
                    else:
                        self.work_counts["incompatible_prefix_components"] += 1
                job["matching_terms"] = tuple(terms)
        return jobs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        records = inputs["records"]
        crossed = self.matching_mode.startswith("crossed") and bool(inputs["matching_terms"])
        if crossed:
            scores, members = {}, {}
            # Keep microbatches small while retaining differentiable matrix entries.
            for left in records:
                row = [left if left.sample_id == right.sample_id else self.encoder.cross_record(left, right)
                       for right in records]
                row_scores, row_members = self.encoder.read_records(model, row, inputs["absolute_ids"])
                scores.update(row_scores)
                members.update(row_members)
            self.work_counts["extra_crossed_forwards"] += len(records) * (len(records) - 1)
        else:
            scores, members = self.encoder.read_records(model, records, inputs["absolute_ids"])
        gold_evidence = inputs.get("gold_evidence_records", ())
        if gold_evidence:
            _, gold_members = self.encoder.read_records(model, gold_evidence, [r.sample_id for r in gold_evidence])
            for sid, gold in gold_members.items():
                member = members[sid]
                members[sid] = MemberLosses(member.action, {**member.diagnostic,
                    **{name: value for name, value in gold.diagnostic.items() if name.startswith("evidence_")}}, None)
            self.work_counts["extra_gold_evidence_forwards"] += len(gold_evidence)
        if self.balance is not None:
            members = balanced_members(members, self.data.cases, self.balance)
        exemplar = next(iter(next(iter(scores.values())).values()))
        terms = {name: exemplar.sum() * 0 for name in ("L_A", "L_D", "L_N", "L_match")}
        if members and self.record_objective == "field_fixed":
            scaled = {sid: MemberLosses(member.action, {name: value / self.evidence_normalization[name]
                if name.startswith("evidence_") else value for name, value in member.diagnostic.items()},
                member.native) for sid, member in members.items()}
            terms.update(absolute_loss(scaled, self.data.absolute_weights))
        elif members and self.record_objective == "legacy":
            terms.update(absolute_loss(members, self.data.absolute_weights))
        elif members:
            for sid, member in members.items():
                weight = self.data.absolute_weights[sid]
                if member.native is not None:
                    terms["L_N"] = terms["L_N"] + weight * member.native
                else:
                    terms["L_A"] = terms["L_A"] + weight * member.action / self.normalization
                    if member.diagnostic:
                        terms["L_D"] = terms["L_D"] + weight * torch.stack(
                            list(member.diagnostic.values())).sum() / self.normalization
        if self.matching_mode != "none":
            for field, component, weight in inputs["matching_terms"]:
                if crossed:
                    ids = component.sample_ids
                    def cell(left, right, j):
                        values = scores[left if left == right else left + "::" + right][field]
                        index = component.gold_indices[j]
                        return candidate_log_odds(values, index) if self.matching_mode == "crossed_odds" else values[index]
                    matrix = torch.stack([torch.stack([
                        cell(left, right, j)
                        for j, right in enumerate(ids)]) for left in ids])
                    value = crossed_matching_nll(matrix, component.gold_indices)
                else:
                    value = conditional_matching_nll([scores[sid][field] for sid in component.sample_ids],
                                                     component.gold_indices)
                terms["L_match"] = terms["L_match"] + weight * value
        loss = self.update_scale * (terms["L_A"] + terms["L_D"] +
            self.native_coefficient * terms["L_N"] + self.matching_coefficient * terms["L_match"])
        for name, value in terms.items():
            observed = value.detach() * self.update_scale
            if name in self.observed_terms:
                self.observed_terms[name].add_(observed)
            else:
                self.observed_terms[name] = observed.clone()
        return (loss, {"loss": loss}) if return_outputs else loss
