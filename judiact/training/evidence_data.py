"""Existing labels on evidence-indexed and unindexed views of the SAME source."""

from collections import Counter
import json

from ..reference_audit import diagnostic_node, materialize_links, materialize_witness, source_node
from ..evidence_audit import evidence_node, materialize_index
from ..data_contract import audit_target_withheld
from ..review_files import review_file
from .supervision import risk_action_balance
from .latent_data import LatentAuditData, generated_target
from .sequence import NodeTarget


def supervised_targets(case, withheld_fields):
    """Keep historical values in the record, not in the supervised field set."""
    return {name: value for name, value in case["targets"].items()
            if not audit_target_withheld(name, case["task_type"], withheld_fields)}


def known_evidence_index_targets(case, withheld_fields):
    result = {}
    for name, raw in case["targets"].items():
        if not name.startswith("evidence_"):
            continue
        value = json.loads(raw)
        if value is None:
            continue
        for field, component in (("evidence_spans", "quotes"), ("evidence_regions", "regions")):
            masks = [path for path in withheld_fields if path.split(".")[0] == field]
            if audit_target_withheld(name, case["task_type"], masks):
                value[component] = []
        if value["quotes"] or value["regions"]:
            result[name] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return result


def evidence_views(case, candidate, indexed=True, action_semantics=False):
    sid, task, visible = case["sample_id"], case["task_type"], case["model_input"]
    witness = materialize_witness(visible, candidate["witness"]["value"])
    links = materialize_links(visible["policy_deck"], witness, candidate["links"]["value"])
    prior = [generated_target(sid, source_node(visible, task), candidate["witness"])]
    if candidate["links"].get("supervised", True):
        prior.append(generated_target(sid, diagnostic_node(visible, task, "links", witness), candidate["links"]))
    targets = case.get("sft_targets", case["targets"])
    index_targets = case.get("sft_index_targets", targets)
    evidence = {name.removeprefix("evidence_"): target for name, target in index_targets.items()
                if name.startswith("evidence_")}
    index, errors = materialize_index(visible, evidence)
    if errors:
        raise ValueError((sid, "reference evidence does not identify its source", errors))
    empty = materialize_witness(visible, {"facts": []})
    reference, unindexed = [], []
    for name, target in targets.items():
        if name.startswith("evidence_"):
            reference.append(NodeTarget(sid, evidence_node(visible, task, name, empty, [], {}),
                                        target, "source_localization"))
            continue
        blank = evidence_node(visible, task, name, empty, [], {"spans": [], "regions": []}, action_semantics)
        reference.append(NodeTarget(sid,
            evidence_node(visible, task, name, witness, links, index, action_semantics) if indexed else blank,
            target, "indexed_reference" if indexed else "source_control"))
        unindexed.append(NodeTarget(sid, blank, target, "unindexed_source"))
    return [(prior, reference), ([], unindexed)]


class EvidenceAuditData(LatentAuditData):
    def set_boundary_measure(self):
        # Only the two diagnosed necessary-condition fields change measure.
        self.sft_balance = risk_action_balance(self, determinate_only=True)
        complete = risk_action_balance(self, determinate_only=False)
        for name in self.sft_balance:
            if name.rsplit("/", 1)[-1] in ("F3", "F5"):
                self.sft_balance[name] = complete[name]
            if self.action_measure == "root" and name.endswith("/action"):
                self.sft_balance[name]["loss_multiplier"] = {
                    label: 1. for label in self.sft_balance[name]["loss_multiplier"]}
                self.sft_balance[name]["balanced_labels"] = []

    def __init__(self, release, bank, action_measure="class_balanced", action_semantics=False):
        self.action_measure = action_measure
        self.action_semantics = action_semantics
        super().__init__(release, bank)
        metadata = review_file(self.release, "supervision.jsonl")
        masked = Counter()
        if metadata.exists():
            with metadata.open(encoding="utf-8-sig") as handle:
                rows = [json.loads(line) for line in handle if line.strip()]
            by_id = {row["sample_id"]: row for row in rows if row["split"] == "train"}
            assert len(by_id) == sum(row["split"] == "train" for row in rows)
            assert by_id.keys() <= self.records.keys(), "supervision metadata is not this train split"
            for sid, row in by_id.items():
                case = self.cases[sid]
                case["withheld_fields"] = list(row["withheld_fields"])
                case["sft_targets"] = supervised_targets(case, row["withheld_fields"])
                case["sft_index_targets"] = known_evidence_index_targets(case, row["withheld_fields"])
                masked.update(case["targets"].keys() - case["sft_targets"].keys())
        self.inventory["evidence_sft_supervision"] = {
            "metadata": str(metadata) if metadata.exists() else None,
            "withheld_node_counts": dict(masked),
            "supervised_node_counts": dict(Counter(name for sid in self.records
                for name in self.cases[sid].get("sft_targets", self.cases[sid]["targets"]))),
            "historical_values_retained": True,
            "scope": "evidence-indexed SFT only; second-stage objective is not revised here",
        }
        self.set_boundary_measure()
