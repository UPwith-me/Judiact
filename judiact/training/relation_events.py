"""Exact likelihood of explicitly reviewed audit-label events."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from itertools import product

import torch

from ..audit_schema import audit_slots


@dataclass(frozen=True)
class LabelConstraint:
    sample_ids: tuple
    assignments: tuple

    def canonical(self):
        order = sorted(range(len(self.sample_ids)), key=self.sample_ids.__getitem__)
        return LabelConstraint(tuple(self.sample_ids[i] for i in order),
            tuple(sorted({tuple(row[i] for i in order) for row in self.assignments})))


@dataclass(frozen=True)
class FieldEvent:
    slot: str
    components: tuple[LabelConstraint, ...]

    @property
    def sample_ids(self):
        return tuple(sorted({sid for component in self.components for sid in component.sample_ids}))


@dataclass(frozen=True)
class RelationExperiment:
    key: tuple
    cluster: tuple
    group_id: str
    stage: str
    kind: str
    fields: tuple[FieldEvent, ...]
    source_edge_ids: tuple

    @property
    def sample_ids(self):
        return tuple(sorted({sid for event in self.fields for sid in event.sample_ids}))


def binary_pair(ids, relation):
    assert relation in ("stable", "flip")
    pairs = ((0, 0), (1, 1)) if relation == "stable" else ((0, 1), (1, 0))
    return LabelConstraint(tuple(ids), pairs)


def categorical_pair(ids, count, relation):
    assert relation in ("same", "different")
    return LabelConstraint(tuple(ids), tuple((a, b) for a, b in product(range(count), repeat=2)
                                             if (a == b) == (relation == "same")))


def s2_quartet(ids, delta):
    assert delta in (-2, -1, 0, 1, 2)
    return LabelConstraint(tuple(ids), tuple(row for row in product((0, 1), repeat=4)
                                            if row[3] - row[2] - row[1] + row[0] == delta))


def event_nll(log_probabilities, assignments):
    """The original categorical denominator includes uncertainty; no binary renormalization."""
    rows = torch.as_tensor(assignments, dtype=torch.long, device=log_probabilities[0].device)
    assert rows.ndim == 2 and rows.shape[1] == len(log_probabilities) and len(rows)
    joint = sum(scores[rows[:, i]] for i, scores in enumerate(log_probabilities))
    return -torch.logsumexp(joint, dim=0)


def compile_field(slot, constraints):
    """Conjoin reviewed constraints once, factoring only disconnected member sets."""
    constraints = sorted(set(c.canonical() for c in constraints),
                         key=lambda c: (c.sample_ids, c.assignments))
    remaining = set(range(len(constraints)))
    components = []
    while remaining:
        selected = {min(remaining)}
        ids = set(constraints[min(remaining)].sample_ids)
        while True:
            added = {i for i in remaining - selected if ids.intersection(constraints[i].sample_ids)}
            if not added:
                break
            selected.update(added)
            ids.update(sid for i in added for sid in constraints[i].sample_ids)
        remaining.difference_update(selected)
        items = [constraints[i] for i in sorted(selected)]
        ordered_ids = tuple(sorted(ids))
        domains, touching = {}, defaultdict(list)
        for constraint in items:
            for i, sid in enumerate(constraint.sample_ids):
                values = {row[i] for row in constraint.assignments}
                domains[sid] = domains[sid] & values if sid in domains else values
                touching[sid].append(constraint)
        visit_order = sorted(ids, key=lambda sid: (-len(touching[sid]), len(domains[sid]), sid))
        chosen, solutions = {}, []

        def visit(depth):
            if depth == len(visit_order):
                solutions.append(tuple(chosen[sid] for sid in ordered_ids))
                return
            sid = visit_order[depth]
            for value in sorted(domains[sid]):
                chosen[sid] = value
                if all(any(all(chosen[node] == row[i] for i, node in enumerate(c.sample_ids)
                               if node in chosen) for row in c.assignments) for c in touching[sid]):
                    visit(depth + 1)
            chosen.pop(sid, None)

        visit(0)
        assert solutions, (slot, ordered_ids, "reviewed constraints have no satisfying assignment")
        components.append(LabelConstraint(ordered_ids, tuple(sorted(solutions))))
    return FieldEvent(slot, tuple(components))


def field_event_nll(readouts, event):
    return sum(event_nll([readouts[sid][event.slot] for sid in component.sample_ids],
                         component.assignments) for component in event.components)


def _policy_pairs(edge, records):
    counts = [len(records[edge[key]]["model_input"]["policy_deck"]) for key in ("left_id", "right_id")]
    return tuple(sorted({tuple(index // 4 if index < 4 * count else count + index - 4 * count
                               for index, count in zip(pair, counts))
                         for pair in edge["policy_alignment"]}))


def build_relation_events(data):
    """Preserve judgment object, intervention type and actual field-review scope."""
    experiments, omissions = [], []
    slots = {sid: {slot.name: slot for slot in audit_slots(case["model_input"], case["task_type"])}
             for sid, case in data.cases.items() if sid in data.records}

    def supervised(ids, name):
        return all("sft_targets" not in data.cases[sid] or name in data.cases[sid]["sft_targets"]
                   for sid in ids)

    for object_key, group in data.relation_groups.items():
        buckets = defaultdict(lambda: {"constraints": defaultdict(list), "edge_ids": set()})
        quartet = group["quartet"]
        determined = quartet is not None and quartet["use_quartet_event"] and supervised(quartet["sample_ids"], "F2")
        quartet_kind = "quartet_zero" if determined and quartet["Delta_gold"] == 0 else "quartet_nonzero"
        if determined:
            buckets[("S2", quartet_kind)]["constraints"]["F2"].append(
                s2_quartet(quartet["sample_ids"], quartet["Delta_gold"]))

        def omit(edge, field, reason):
            omissions.append({"edge_id": edge["edge_id"], "group_id": group["group_id"],
                              "field": field, "reason": reason})

        for edge in group["edges"]:
            raw = data.raw_edges[edge["edge_id"]]
            if not edge["use_causal_loss"] or raw["causal_status"] != "single_factor_verified":
                continue
            stage, factor = edge["target_stage"], raw["intervention"]["changed_factor"]
            ids = (edge["left_id"], edge["right_id"])
            inside_quartet = determined and stage == "S2" and set(ids) <= set(quartet["sample_ids"])
            kind = quartet_kind if inside_quartet else (
                "invariance" if factor == "none" else {"S4": "rule_boundary", "S5": "execution"}.get(stage, "pair"))
            bucket = buckets[(stage, kind)]
            bucket["edge_ids"].add(edge["edge_id"])
            left, right = [data.records[sid] for sid in ids]
            gold = [row["gold"] for row in (left, right)]
            relation = raw["relation"]

            def add_binary(which_stage, claim):
                name = "F" + which_stage[1:]
                if not supervised(ids, name):
                    omit(edge, name, "member_field_withheld_from_supervision")
                    return
                states = [g["stage_states"][which_stage] for g in gold]
                values = [state["F"] if state is not None else None for state in states]
                if not all(value in (0, 1) for value in values):
                    omit(edge, name, "F_relation_not_binary_determined")
                elif (values[0] == values[1]) != (claim == "stable"):
                    omit(edge, name, "F_review_conflicts_with_member_labels")
                else:
                    bucket["constraints"][name].append(binary_pair(ids, claim))

            if inside_quartet:
                omit(edge, "F2", "pair_replaced_by_complete_quartet_event")
            elif stage == "S4" and factor == "none" and relation["target_F"] == "flip":
                omit(edge, "F4", "policy_invariance_conflicts_with_F_flip")
            elif relation["target_F"] in ("flip", "stable"):
                add_binary(stage, relation["target_F"])
            for other in relation["stable_non_target_stages"]:
                assert other != stage, (edge["edge_id"], "non-target review includes target stage")
                add_binary(other, "stable")

            action_relation = relation["decision"]
            if not supervised(ids, "action"):
                omit(edge, "action", "member_field_withheld_from_supervision")
            elif stage == "S4" and factor == "none" and action_relation == "different":
                omit(edge, "action", "policy_invariance_conflicts_with_action_change")
            elif action_relation in ("same", "different"):
                name = "response_publishability" if left["task_type"] == "response_audit" else "request_action"
                values = [g[name] for g in gold]
                if (values[0] == values[1]) == (action_relation == "same"):
                    bucket["constraints"]["action"].append(categorical_pair(ids, 4, action_relation))
                else:
                    omit(edge, "action", "action_review_conflicts_with_member_labels")

            if edge["policy_alignment"]:
                pairs = _policy_pairs(edge, data.records)
                targets = tuple(slots[sid]["policy_id"].labels.index(g["policy_id"]) for sid, g in zip(ids, gold))
                if not supervised(ids, "policy_id"):
                    omit(edge, "policy_id", "member_field_withheld_from_supervision")
                elif targets in pairs:
                    bucket["constraints"]["policy_id"].append(LabelConstraint(ids, pairs))
                else:
                    omit(edge, "policy_id", "aligned_policy_labels_disagree")
                if not supervised(ids, "policy_action"):
                    omit(edge, "policy_action", "member_field_withheld_from_supervision")
                elif gold[0]["policy_action"] == gold[1]["policy_action"]:
                    count = len(slots[ids[0]]["policy_action"].labels)
                    bucket["constraints"]["policy_action"].append(categorical_pair(ids, count, "same"))
                else:
                    omit(edge, "policy_action", "policy_invariance_conflicts_with_policy_action")

        for (stage, kind), bucket in sorted(buckets.items()):
            fields = tuple(compile_field(name, constraints)
                           for name, constraints in sorted(bucket["constraints"].items()) if constraints)
            if fields:
                event = RelationExperiment(object_key + (stage, kind), group["cluster"], group["group_id"],
                                           stage, kind, fields, tuple(sorted(bucket["edge_ids"])))
                for sid in event.sample_ids:
                    cluster = ("response", object_key[2]) if object_key[0] == "response" \
                        else ("request", data.root_ids[sid])
                    assert cluster == event.cluster, (event.key, sid, "relation spans independent roots")
                experiments.append(event)
    return experiments, omissions


def relation_weights(experiments):
    by_cluster = defaultdict(list)
    for experiment in experiments:
        by_cluster[experiment.cluster].append(experiment)
    return {(experiment.key, event.slot): 1 / (len(by_cluster) * len(cluster) * len(experiment.fields))
            for cluster in by_cluster.values() for experiment in cluster for event in experiment.fields}


def relation_loss(readouts, experiments, weights):
    """Weights are from the full relation population, never renormalized per minibatch."""
    terms = {(experiment.key, event.slot): field_event_nll(readouts, event)
             for experiment in experiments for event in experiment.fields}
    zero = next(iter(next(iter(readouts.values())).values())).sum() * 0
    return sum((weights[key] * value for key, value in terms.items()), zero), terms
