"""Frozen original field budgets plus root-normalized local semantic targets."""

from collections import Counter
import math

from ..shared_audit import shared_judgment_node
from .sequence import NodeTarget
from .source_data import SourceAuditData
from .shared_local import local_boundary_targets, local_projection


class SharedAuditData(SourceAuditData):
    def __init__(self, release, bank, action_objective="natural", local_boundary_conditions=False,
                 student_source_reads=True, action_budget=None, stratified_exposure=False, action_costs=None):
        self.stratified_exposure = stratified_exposure
        self.exposure_counts = {}
        super().__init__(release, bank)
        self.action_objective = action_objective
        self.local_boundary_conditions = local_boundary_conditions
        self.student_source_reads = student_source_reads
        self.action_budget = action_budget
        self.action_costs = action_costs
        assert (action_objective == "macro_f1_cost") == (action_costs is not None)
        if action_objective == "target_budget":
            assert action_budget is not None and set(action_budget) == {"allow", "warn", "review", "refuse"}
            assert all(math.isfinite(v) and v > 0 for v in action_budget.values())
            assert math.isclose(sum(action_budget.values()), 1., rel_tol=0., abs_tol=1e-9)
        else:
            assert action_budget is None, "A class budget is used only by the target_budget action objective"
        self.action_priors = {}
        for task in ("request_decision", "response_audit"):
            masses = Counter({label: 0. for label in ("allow", "warn", "review", "refuse")})
            for sid in self.records:
                case = self.cases[sid]
                targets = case.get("sft_targets", case["targets"])
                if case["task_type"] == task and "action" in targets:
                    masses[targets["action"]] += self.absolute_weights[sid]
            total = sum(masses.values())
            assert total > 0 and all(m > 0 for m in masses.values()), (task, masses)
            self.action_priors[task] = {label: mass / total for label, mass in masses.items()}
        if action_costs is not None:
            from .action_cost import check_cost_population
            check_cost_population(action_costs, self)
        self.local_targets = {}
        self.base_diagnostic_counts = {}
        self.local_group_counts = {}
        counts, masses, omissions = Counter(), Counter(), Counter()
        for sid in self.records:
            case = self.cases[sid]
            targets = case.get("sft_targets", case["targets"])
            self.base_diagnostic_counts[sid] = len(targets.keys() - {"action"})
            if local_boundary_conditions:
                case["sft_targets"] = {name: value for name, value in targets.items() if name != "basis"}
            projected = local_projection(case, self.bank[sid])
            examples = list(projected.targets)
            if local_boundary_conditions:
                examples.extend(local_boundary_targets(case))
            omissions.update(projected.omissions.values())
            self.local_targets[sid] = examples
            self.local_group_counts[sid] = Counter(self.semantic_group(e.node.name) for e in examples)
            for example in examples:
                group = self.semantic_group(example.node.name)
                counts[case["task_type"] + "/" + group] += 1
                masses[case["task_type"] + "/" + group] += self.sft_coefficient(sid, example.node.name, 0, 0)
        self.inventory["shared_audit_sft"] = {
            "local_target_counts": dict(counts), "local_coefficient_mass": dict(masses),
            "local_projection_omissions": dict(omissions),
            "budgets": "original per-member action/classification/evidence/basis coefficients unchanged; "
                "each local fact/rule block receives one existing diagnostic-field unit per root/member, "
                "averaged over available local items; no additional class-frequency balance; native .1; relations 1",
            "semantic_budget_rationale": "bounded addition tied to the existing field unit, not the number or "
                "length of facts; a declared conservative design choice, not an empirically optimal coefficient",
            "categorical_context": "fixed placeholders, shared source prefill, original X/P always visible",
            "read_context": "first-visit current-student source locations each epoch; exact original text/pixels; "
                "each existing classification budget averaged across distinct direct/read contexts; "
                "invalid/empty reads collapse to the unchanged direct input, never drop a supervised member",
            "relation_context": "direct input only; no gold locations or semantic targets in the prefix",
        }
        self.inventory["shared_action_objective"] = {
            "objective": action_objective, "priors": self.action_priors,
            "prior_measure": "observed action targets; full train task/root/context/member weights; not dev or batch counts",
            "scope": "action only; all other field coefficients unchanged; matching uses raw scores",
            "inference": "argmax raw action logits; no additional prior correction",
            "basis_output": not local_boundary_conditions,
            "condition_supervision": "positive observed content only; original diagnostic denominator frozen; "
                "unknown/null is not a need-to-reason negative" if local_boundary_conditions else "legacy basis JSON",
        }
        if action_budget is not None:
            self.inventory["shared_action_objective"].update(
                target_budget=action_budget,
                loss_multiplier={task: {label: action_budget[label] / prior[label] for label in prior}
                                 for task, prior in self.action_priors.items()},
                normalization="fixed full-train prior; weighted action mass preserved per task; "
                    "no batch renormalization or class resampling; multiplier applied once in action CE")
        if action_costs is not None:
            self.inventory["shared_action_objective"].update(
                algorithm=action_costs["algorithm"], gain_weight=action_costs["gain_weight"],
                source_adapter=action_costs["source_adapter"], frozen_tasks=action_costs["tasks"],
                normalization="half balanced hard CE plus half globally normalized full-information gain CE; "
                    "frozen train-only costs; no further q/prior multiplier; each task retains its original mass")
        if local_boundary_conditions:
            self.inventory["shared_audit_sft"]["budgets"] = (
                "original action/diagnostic/evidence/fact/rule coefficients; original diagnostic denominator frozen; "
                "positive boundary content shares at most its old basis field unit; no empty/existence targets; "
                "action objective recorded separately; native .1; relations 1")
        self.inventory["shared_audit_sft"]["student_source_reads"] = student_source_reads
        if not student_source_reads:
            self.inventory["shared_audit_sft"]["read_context"] = (
                "original X/P only; each field retains its full existing coefficient on the direct input; "
                "reference localization remains a supervised output, never an action input; "
                "no student roll-in, generated context or autoregressive preparation during SFT")

    @staticmethod
    def semantic_group(name):
        if name.startswith("local_boundary_"):
            return "boundary"
        if name.startswith("local_fact_"):
            return "fact"
        if name.startswith("local_rule_"):
            return "rule"
        return None

    def schedule(self, world, seed=42, effective_roots=64, locality=True):
        dataset, sampling = super().schedule(world, seed, effective_roots, locality)
        if self.stratified_exposure:
            from .exposure import exposure_schedule
            dataset, sampling = exposure_schedule(self, dataset, sampling, seed)
            self.inventory["stratified_exposure"] = sampling["stratified_exposure"]
            self.inventory["shared_action_objective"].update(
                normalization="fixed full-train prior; repeated member coefficients divided by exposure count; "
                    "no batch renormalization; action class multiplier applied once",
                scope="action class budget changed; all other field epoch coefficient totals unchanged")
            if self.action_costs is not None:
                self.inventory["shared_action_objective"]["normalization"] = (
                    "frozen whole-train hard/gain coefficients; repeated member mass divided by exposure count; "
                    "no additional inverse-prior or batch normalization")
                sampling["stratified_exposure"]["coefficient"] = (
                    "every member field divided by exposure count; action uses frozen hard/gain coefficients once")
        return dataset, sampling

    def sft_coefficient(self, sid, name, prior_count, diagnostic_count):
        return self.member_sft_coefficient(sid, name, prior_count, diagnostic_count) / self.exposure_counts.get(sid, 1)

    def member_sft_coefficient(self, sid, name, prior_count, diagnostic_count):
        group = self.semantic_group(name)
        if group:
            return (self.absolute_weights[sid] / self.base_diagnostic_counts[sid]
                    / self.local_group_counts[sid][group])
        if self.local_boundary_conditions and name != "action":
            case = self.cases[sid]
            targets = case.get("sft_targets", case["targets"])
            mass = self.absolute_weights[sid]
            balance = self.sft_balance.get(case["task_type"] + "/" + name)
            if balance is not None:
                mass *= balance["loss_multiplier"][str(targets[name])]
            return mass / self.base_diagnostic_counts[sid]
        return super().sft_coefficient(sid, name, prior_count, diagnostic_count)

    def shared_views(self, sid, index=None):
        case = self.cases[sid]
        visible, task = case["model_input"], case["task_type"]
        targets = case.get("sft_targets", case["targets"])
        direct = [NodeTarget(sid, shared_judgment_node(visible, task, name), value, "shared_direct")
                  for name, value in targets.items()]
        direct.extend(self.local_targets[sid])
        views = [([], direct)]
        if self.student_source_reads and index and (index["spans"] or index["regions"]):
            views.append(([], [NodeTarget(sid, shared_judgment_node(visible, task, name, index),
                                         value, "student_source_read")
                              for name, value in targets.items()
                              if name != "basis" and not name.startswith("evidence_")]))
        return views
