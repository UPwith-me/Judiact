"""Reuse reviewed local supervision for short review, without new semantic gold."""

from collections import Counter
from copy import deepcopy
import random

from ..short_review import short_review_node, short_judgment_node
from .sequence import NodeTarget
from .shared_data import SharedAuditData


KIND_GROUP = {"evidence": "fact", "condition": "boundary", "rule": "rule"}


def short_reference_groups(examples):
    """Equal existing groups, then equal valid items inside each group."""
    groups = {kind: [] for kind in KIND_GROUP}
    for example in examples:
        target, name = example.target, example.node.name
        if name.startswith("local_fact_"):
            if target["facts"]:
                groups["evidence"].append(deepcopy(target["facts"]))
        elif name.startswith("local_rule_"):
            items = [{"policy_id": target["policy_id"], "relation": role, "fact": deepcopy(fact)}
                     for role in ("premises", "exceptions", "unresolved") for fact in target[role]]
            groups["rule"].append(items or [{"policy_id": target["policy_id"],
                                            "relation": "none", "fact": None}])
        elif name.startswith("local_boundary_"):
            groups["condition"].append([{"kind": name.removeprefix("local_boundary_"),
                                        "text": target["condition"]}])
    return groups


class ShortReviewData(SharedAuditData):
    def __init__(self, *args, reference_seed=42, **kwargs):
        super().__init__(*args, **kwargs)
        assert not self.student_source_reads and self.local_boundary_conditions
        self.reference_seed = reference_seed
        self.short_groups = {sid: short_reference_groups(examples)
                             for sid, examples in self.local_targets.items()}
        counts = Counter()
        for sid, groups in self.short_groups.items():
            for kind, pool in groups.items():
                task = self.cases[sid]["task_type"]
                counts[task + "/" + kind + "/members"] += bool(pool)
                counts[task + "/" + kind + "/valid_targets"] += sum(map(len, pool))
        self.inventory["short_review"] = {
            "coverage": dict(counts), "reference_seed": reference_seed,
            "references": "all existing complete local references retained; one supported short item "
                "sampled per available semantic group per member/epoch; no action-conditioned construction",
            "budgets": "within each available fact/rule/boundary block, half its old budget keeps full local "
                "supervision and half teaches the short form; action, stages, evidence, native and relation "
                "budgets unchanged; this equal two-view split is a declared design choice, not an optimum",
            "missing": "unannotated condition has no short-condition loss, never null/absent gold",
            "context": "first-visit current-student three items per epoch; original X/P retained; "
                "same autonomous context for categorical supervision and relations; no reference items in prefix",
        }
        actual_mass = Counter()
        for sid, examples in self.local_targets.items():
            task = self.cases[sid]["task_type"]
            for example in examples:
                group = self.semantic_group(example.node.name)
                actual_mass[task + "/" + group] += self.member_sft_coefficient(sid, example.node.name, 0, 0)
            for kind, groups in self.short_groups[sid].items():
                if groups:
                    actual_mass[task + "/short_" + kind] += self.member_sft_coefficient(
                        sid, "short_review_" + kind, 0, 0)
        previous = self.inventory["shared_audit_sft"]
        previous["original_local_coefficient_mass"] = previous["local_coefficient_mass"]
        previous.update(local_coefficient_mass=dict(actual_mass), budgets=self.inventory["short_review"]["budgets"],
                        read_context=self.inventory["short_review"]["context"],
                        relation_context="same actual student short reviews as categorical supervision")

    def member_sft_coefficient(self, sid, name, prior_count, diagnostic_count):
        if name.startswith("short_review_"):
            kind = name.removeprefix("short_review_")
            if not self.short_groups[sid][kind]:
                return 0.
            return .5 * self.absolute_weights[sid] / self.base_diagnostic_counts[sid]
        coefficient = super().member_sft_coefficient(sid, name, prior_count, diagnostic_count)
        # The parent computes its inventory before short_groups exists.
        if hasattr(self, "short_groups"):
            group = self.semantic_group(name)
            kind = next((k for k, g in KIND_GROUP.items() if g == group), None)
            if kind and self.short_groups[sid][kind]:
                coefficient *= .5
        return coefficient

    def short_views(self, sid, review, epoch):
        case = self.cases[sid]
        visible, task = case["model_input"], case["task_type"]
        targets = case.get("sft_targets", case["targets"])
        examples = [NodeTarget(sid, short_judgment_node(visible, task, name, review), value,
                               "own_short_review" if name != "basis" and not name.startswith("evidence_")
                               else "original_source") for name, value in targets.items()]
        examples.extend(self.local_targets[sid])
        rng = random.Random(f"{self.reference_seed}:{epoch}:{sid}:short_reference")
        for kind, groups in self.short_groups[sid].items():
            if groups:
                target = deepcopy(rng.choice(rng.choice(groups)))
                examples.append(NodeTarget(sid, short_review_node(visible, task, kind), target,
                                           "reviewed_short_item"))
        return [([], examples)]
