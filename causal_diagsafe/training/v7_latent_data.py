"""Existing observed labels plus unlabelled source/rule records, without new human targets."""

from collections import Counter, defaultdict
import json
from pathlib import Path

from ..audit_v7 import (
    INITIAL_REQUEST, RULE_FIELDS, canonical, diagnostic_node,
    materialize_links, materialize_witness, native_node, source_identity, source_node,
)
from ..io_utils import read_jsonl
from ..audit_v4 import quote_to_span
from .v4_losses import absolute_weights
from .v5_data import CoherentAuditData
from .v7_sequence import NodeTarget


def evidence_visible(case, name, witness):
    """Do not supervise a gold location discarded from the selected source view."""
    value = json.loads(case["targets"][name])
    if value is None:
        return True
    for quote in value["quotes"]:
        if quote["field"] == "policy_rule":
            if name not in ("evidence_S4", "evidence_S5"):
                return False
            continue
        span = quote_to_span(case["model_input"], name.removeprefix("evidence_"), quote)
        if not any(item["field"] == span["field"] and item["start"] <= span["start"]
                   and item["end"] >= span["end"] for item in witness["spans"]):
            return False
    for region in value["regions"]:
        a = region["bbox"]
        if not any(item["media_index"] == region["media_index"] and
                   item["bbox"][0] <= a[0] and item["bbox"][1] <= a[1] and
                   item["bbox"][2] >= a[2] and item["bbox"][3] >= a[3]
                   for item in witness["regions"]):
            return False
    return True


def generated_target(sid, node, output):
    return NodeTarget(sid, node, output.get("raw", canonical(output.get("value"))),
                      view="latent", continuation_ids=output.get("token_ids"),
                      terminated=output.get("terminated", True))


def bank_output(value, raw=None):
    return {"value": value, "raw": raw or canonical(value), "error": None, "terminated": True}


def candidate_views(case, candidate):
    """A normalized DAG: W -> L; observed diagnostics -> action -> behavior.

    L is proposed without gold diagnostics. Observed categorical parents are
    teacher-forced only while evaluating the joint density, never to propose z.
    Invalid/censored latent paths have zero observed likelihood, not fake labels.
    """
    sid, task, visible = case["sample_id"], case["task_type"], case["model_input"]
    w_output = candidate["witness"]
    prior = [generated_target(sid, source_node(visible, task), w_output)]
    if w_output.get("error"):
        return prior, [], w_output["error"]
    try:
        witness = materialize_witness(visible, w_output["value"])
    except (ValueError, TypeError, KeyError) as error:
        return prior, [], "witness:" + str(error)
    link_output = candidate.get("links")
    if link_output is None:
        return prior, [], "links_not_generated"
    prior.append(generated_target(sid, diagnostic_node(visible, task, "links", witness), link_output))
    if link_output.get("error"):
        return prior, [], link_output["error"]
    try:
        links = materialize_links(visible["policy_deck"], witness, link_output["value"])
    except (ValueError, TypeError, KeyError) as error:
        return prior, [], "links:" + str(error)
    targets = case["targets"]
    values = {"empty_request_deck": task != "response_audit" and not visible["policy_deck"]}
    observed = []
    initial = ("E5", "F5") if task == "response_audit" else INITIAL_REQUEST
    for name in initial:
        observed.append(NodeTarget(sid, diagnostic_node(visible, task, name, witness),
                                   targets[name], "observed"))
        values[name] = targets[name]
    if task != "response_audit":
        for name in RULE_FIELDS:
            observed.append(NodeTarget(sid, diagnostic_node(visible, task, name, witness, values, links),
                                       targets[name], "observed"))
            values[name] = targets[name]
    for name in ("action", "behavior"):
        observed.append(NodeTarget(sid, diagnostic_node(visible, task, name, witness, values, links),
                                   targets[name], "observed"))
        values[name] = targets[name]
    for name in targets:
        if name.startswith("evidence_"):
            if not evidence_visible(case, name, witness):
                return prior, observed, "observed_evidence_outside_witness"
            observed.append(NodeTarget(sid, diagnostic_node(visible, task, name, witness, values, links),
                                       targets[name], "observed"))
    return prior, observed, None


def identity(candidate, encoder, case):
    prior, _, _ = candidate_views(case, candidate)
    return tuple((e.node.name, tuple(encoder.target_ids(e)), e.terminated) for e in prior)


class LatentAuditData(CoherentAuditData):
    def __init__(self, release, bank):
        super().__init__(release, split="train")
        path = Path(bank)
        paths = sorted(path.glob("bank.rank*.jsonl")) if path.is_dir() else [path]
        rows = [row for file in paths for row in read_jsonl(file)]
        self.bank = {row["sample_id"]: row for row in rows}
        assert len(self.bank) == len(rows), "duplicate bank rows would change the proposal mass"
        assert self.bank.keys() <= self.records.keys(), "proposal bank must be train-only"
        for sid, row in self.bank.items():
            visible = self.cases[sid]["model_input"]
            assert row["source_identity"] == source_identity(visible), (sid, "wrong source")
            assert row["policy_identity"] == canonical(visible["policy_deck"]), (sid, "wrong deck")
        self.absolute_weights = absolute_weights(self.populations)
        self.action_balance = {}
        for task in ("request_decision", "response_audit"):
            masses = defaultdict(float)
            for sid, row in self.records.items():
                if (row["task_type"] == "response_audit") == (task == "response_audit"):
                    masses[self.cases[sid]["targets"]["action"]] += self.absolute_weights[sid]
            total = sum(masses.values())
            self.action_balance[task] = {label: total / (len(masses) * mass)
                                         for label, mass in masses.items()}
        states = Counter()
        for sid in self.records:
            row = self.bank.get(sid)
            if row is None or row.get("candidate") is None:
                states["no_bank_candidate"] += 1
            else:
                _, _, error = candidate_views(self.cases[sid], row["candidate"])
                states[error or "supported"] += 1
        self.inventory["v7_latent_bank"] = dict(states)
        self.inventory["bank_provenance"] = "machine proposals; no new human-final status"

    def root_coefficient(self, sid):
        case = self.cases[sid]
        if case["task_type"].startswith("native_"):
            return .1 * self.absolute_weights[sid]
        task = "response_audit" if case["task_type"] == "response_audit" else "request_decision"
        return self.absolute_weights[sid] * self.action_balance[task][case["targets"]["action"]]

    def native_example(self, sid):
        case = self.cases[sid]
        task = case["task_type"]
        return NodeTarget(sid, native_node(case["model_input"], task), case["targets"][task])

    def restrict_trial(self, native_roots=32):
        """An explicitly labelled train-only exercise; never rewrite the release."""
        ids = set(self.bank)
        self.records = {sid: row for sid, row in self.records.items() if sid in ids}
        for kind in ("request", "response"):
            self.populations[kind] = {root: [sid for sid in members if sid in ids]
                                      for root, members in self.populations[kind].items()
                                      if any(sid in ids for sid in members)}
        self.populations["auxiliary"] = dict(list(self.populations["auxiliary"].items())[:native_roots])
        native = {sid for members in self.populations["auxiliary"].values() for sid in members}
        self.native_records = {sid: row for sid, row in self.native_records.items() if sid in native}
        self.cases = {sid: case for sid, case in self.cases.items() if sid in ids | native}
        self.relation_groups = {key: group for key, group in self.relation_groups.items()
                                if set(group["sample_ids"]) <= ids}
        self.absolute_weights = absolute_weights(self.populations)
        self.inventory["implementation_trial"] = {
            "members": len(ids), "native_roots": len(self.populations["auxiliary"]),
            "complete_relation_objects": len(self.relation_groups), "split": "train"}
