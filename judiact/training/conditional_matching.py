"""Oriented conditional allocation likelihood on audit's explicitly reviewed scope.

This module does not replace absolute supervision. In particular, conditioning on
class counts removes shared class biases and supplies no class-prior supervision.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

import torch

from ..audit_schema import audit_slots
from .relation_events import build_relation_events


def conditional_matching_nll(log_probabilities, gold_indices):
    """Scalar -log P(actual allocation | observed class counts).

    Rows are members, columns share a semantic codebook. The count-DP includes
    ALL distinct allocations, not just assignments allowed by the old event.
    Each transition assigns one class once: there are no factorial multiplicities.
    Cost is O(K * product(n_k + 1)) states/transitions for the observed classes.
    Float64 is retained; low-precision readouts are accumulated in float32.
    """
    scores = log_probabilities if isinstance(log_probabilities, torch.Tensor) \
        else torch.stack(tuple(log_probabilities))
    if scores.dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()
    gold = torch.as_tensor(gold_indices, dtype=torch.long, device=scores.device)
    assert scores.ndim == 2 and gold.shape == (scores.shape[0],) and len(gold)
    indices = gold.tolist()
    assert min(indices) >= 0 and max(indices) < scores.shape[1]
    counts = Counter(indices)
    if len(counts) == 1:
        return scores.sum() * 0

    classes = tuple(sorted(counts))
    target = tuple(counts[k] for k in classes)
    # Subtract the actual numerator row by row; no probability-space products.
    relative = scores - scores.gather(1, gold[:, None])
    states = {(0,) * len(classes): scores.new_zeros(())}
    for row in relative:
        following = {}
        for used, value in states.items():
            for j, k in enumerate(classes):
                if used[j] == target[j]:
                    continue
                updated = used[:j] + (used[j] + 1,) + used[j + 1:]
                candidate = value + row[k]
                following[updated] = torch.logaddexp(following[updated], candidate) \
                    if updated in following else candidate
        states = following
    return states[target]


@dataclass(frozen=True)
class MatchingComponent:
    sample_ids: tuple
    gold_indices: tuple[int, ...]
    labels: tuple  # Full common semantic codebook, including absent classes.

    @property
    def is_constant(self):
        return len(set(self.gold_indices)) == 1


@dataclass(frozen=True)
class MatchingField:
    slot: str
    components: tuple[MatchingComponent, ...]

    @property
    def is_directional(self):
        return any(not component.is_constant for component in self.components)

    @property
    def sample_ids(self):
        return tuple(sorted({sid for c in self.components for sid in c.sample_ids}))


@dataclass(frozen=True)
class MatchingExperiment:
    key: tuple
    cluster: tuple
    group_id: str
    stage: str
    kind: str  # Original intervention grouping, not a claim of directionality.
    fields: tuple[MatchingField, ...]
    source_edge_ids: tuple = ()

    @property
    def sample_ids(self):
        return tuple(sorted({sid for field in self.fields for sid in field.sample_ids}))


def build_matching_experiments(data):
    """Orient reviewed components only; equal labels never create new relations.

    Determined S2 quartets (including zero controls) and known partial edges of
    uncertain quartets inherit audit's scope. Old assignments only check compatibility
    of the actual targets; neither Delta nor symmetric event NLL is optimized.
    Constant components remain visible and are reported as zero, not directional.
    Dynamic policy IDs are omitted even if a pairwise ID alignment was reviewed;
    their absolute supervision is untouched.
    """
    reviewed, omissions = build_relation_events(data)
    omissions = list(omissions)
    slots = {sid: {slot.name: slot for slot in audit_slots(case["model_input"], case["task_type"])}
             for sid, case in data.cases.items() if sid in data.records}
    experiments = []
    for experiment in reviewed:
        fields = []

        def report(slot, ids, reason):
            omissions.append({"key": experiment.key, "group_id": experiment.group_id,
                              "field": slot, "sample_ids": ids, "reason": reason})

        for event in experiment.fields:
            if event.slot == "policy_id":
                report(event.slot, event.sample_ids,
                       "dynamic_policy_id_matching_skipped_absolute_supervision_retained")
                continue
            assert event.slot in {"F1", "F2", "F3", "F4", "F5", "action", "policy_action"}
            components = []
            for reviewed_component in event.components:
                ids = reviewed_component.sample_ids
                if any(event.slot not in data.cases[sid]["targets"] for sid in ids):
                    report(event.slot, ids, "matching_target_unavailable")
                    continue
                labels = slots[ids[0]][event.slot].labels
                assert all(slots[sid][event.slot].labels == labels for sid in ids), \
                    (experiment.key, event.slot, "matching requires a common semantic codebook")
                gold = tuple(labels.index(data.cases[sid]["targets"][event.slot]) for sid in ids)
                if gold not in reviewed_component.assignments:
                    report(event.slot, ids, "matching_targets_conflict_with_reviewed_component")
                    continue
                component = MatchingComponent(ids, gold, labels)
                components.append(component)
                if component.is_constant:
                    report(event.slot, ids, "constant_label_component_zero_matching_loss")
            if components:
                fields.append(MatchingField(event.slot, tuple(components)))
        if fields:
            kind = experiment.kind
            # Three legacy S5 edges name the factor "none", but explicitly record
            # replacement of the response only. Do not report them as invariance.
            if experiment.stage == "S5" and kind == "invariance" and experiment.source_edge_ids:
                interventions = [data.raw_edges[eid]["intervention"] for eid in experiment.source_edge_ids]
                if all(item.get("operation") == "response_replacement" and
                       item.get("changed_fields") == ["response_candidate"] for item in interventions):
                    kind = "execution"
            experiments.append(MatchingExperiment(
                experiment.key, experiment.cluster, experiment.group_id, experiment.stage,
                kind, tuple(fields), experiment.source_edge_ids))
    return experiments, omissions


def matching_weights(experiments):
    """Average active root/contexts, then active experiments, then active fields.

    Constant-only fields have explicit zero weights and consume no directional
    mass at any level. Compute on the full population, not each minibatch.
    """
    weights = {(experiment.key, field.slot): 0.
               for experiment in experiments for field in experiment.fields}
    by_cluster = defaultdict(list)
    for experiment in experiments:
        active = tuple(field for field in experiment.fields if field.is_directional)
        if active:
            by_cluster[experiment.cluster].append((experiment, active))
    for cluster in by_cluster.values():
        for experiment, fields in cluster:
            for field in fields:
                weights[experiment.key, field.slot] = 1 / (len(by_cluster) * len(cluster) * len(fields))
    return weights


def field_matching_nll(readouts, field):
    """SUM disconnected component NLLs, i.e. a product of conditional likelihoods.

    There is no member/component averaging. Adding zero controls cannot dilute
    directional components; field/root scaling is solely in matching_weights.
    """
    return torch.stack([
        conditional_matching_nll([readouts[sid][field.slot] for sid in component.sample_ids],
                                 component.gold_indices)
        for component in field.components
    ]).sum()
