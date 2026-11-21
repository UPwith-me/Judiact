"""Existing supervision on original and source-read contexts, without W/L parents."""

from collections import Counter

from ..source_audit import CONDITION_KINDS, source_judgment_node
from ..evidence_audit import materialize_index
from ..io_utils import read_jsonl
from ..review_files import review_file
from .evidence_data import EvidenceAuditData
from .sequence import NodeTarget
from .supervision import risk_action_balance


def source_views(case, read_views=False):
    sid, task, visible = case["sample_id"], case["task_type"], case["model_input"]
    targets = dict(case.get("sft_targets", case["targets"]))
    if case.get("basis_target") is not None:
        targets["basis"] = case["basis_target"]
    index_targets = case.get("sft_index_targets", targets)
    index, errors = materialize_index(visible, {
        name.removeprefix("evidence_"): value for name, value in index_targets.items()
        if name.startswith("evidence_")})
    if errors:
        raise ValueError((sid, "source evidence read", errors))
    direct = [NodeTarget(sid, source_judgment_node(visible, task, name), value, "source_direct")
              for name, value in targets.items()]
    focused = [NodeTarget(sid, source_judgment_node(visible, task, name, index), value,
                          "reference_source_read") for name, value in targets.items()
               if not name.startswith("evidence_") and name != "basis"] if read_views and (index["spans"] or index["regions"]) else []
    return [([], direct)] + ([([], focused)] if focused else [])


def adjudicated_basis(row, case, supervision=None):
    """Project individually observed, unmasked conditions, never action-derived text."""
    targets = case.get("sft_targets", case["targets"])
    action_field = "response_publishability" if case["task_type"] == "response_audit" else "request_action"
    review = row.get("review_record", {})
    if row["split"] != "train" or "action" not in targets:
        return None
    if action_field not in row.get("unique_hard_target_fields", []):
        return None
    if row.get("working_canonical_gold", {}).get(action_field) != targets["action"]:
        raise ValueError((case["sample_id"], "adjudication/action target mismatch"))
    masks = []
    for owner in (row, case, supervision or {}):
        for key in ("field_uncertainties", "withheld_fields", "ignored_fields"):
            for item in owner.get(key, []) or []:
                path = item if isinstance(item, str) else item["field"]
                masks.append(path.removeprefix("review_record.").removeprefix("basis."))
    result = {name: review[name] for name in CONDITION_KINDS if name in review
              and not any(mask in ("review_record", "basis", name) or mask.startswith(name + ".")
                          for mask in masks)}
    duty = result.get("explicit_verification_basis")
    if isinstance(duty, list):
        # The final release contains empty lists and singleton rule citations.
        assert len(duty) <= 1, (case["sample_id"], "multiple verification citations")
        duty = duty[0] if duty else None
        result["explicit_verification_basis"] = duty
    if isinstance(duty, dict):
        assert duty["policy_id"] in {card["policy_id"] for card in case["model_input"]["policy_deck"]}
        key = "text" if "text" in duty else "clause"
        assert set(duty) == {"policy_id", key} and isinstance(duty[key], str) and duty[key].strip()
        result["explicit_verification_basis"] = duty["policy_id"] + ": " + duty[key]
    assert all(value is None or isinstance(value, str) and value.strip() for value in result.values()), case["sample_id"]
    return result or None


class SourceAuditData(EvidenceAuditData):
    def set_boundary_measure(self):
        # Retain audit's uncertain mass; balance only determinate risk labels.
        self.sft_balance = risk_action_balance(self, determinate_only=True)
        for name, values in self.sft_balance.items():
            if name.endswith("/action"):
                values["loss_multiplier"] = {label: 1. for label in values["loss_multiplier"]}
                values["balanced_labels"] = []

    def __init__(self, release, bank):
        super().__init__(release, bank, action_measure="root", action_semantics=True)
        self.read_views = False
        counts, origins, nonnull, available = Counter(), Counter(), Counter(), Counter()
        history = review_file(self.release, "historical_conditions.jsonl")
        current = review_file(self.release, "conditions.jsonl")
        rows = {}
        for filename in (history, current):
            for row in read_jsonl(filename):
                if row["split"] == "train":
                    rows[row["sample_id"]] = (row, filename)
        metadata = {r["sample_id"]: r for r in read_jsonl(review_file(self.release, "projection.jsonl"))}
        supervision = {r["sample_id"]: r for r in read_jsonl(review_file(self.release, "supervision.jsonl"))
                       if r["split"] == "train"}
        for sid in self.records:
            case = self.cases[sid]
            cell = case["task_type"] + "/" + str(case["targets"]["action"])
            available[cell] += 1
            if sid not in rows:
                continue
            row, filename = rows[sid]
            if filename == history:
                # The release explicitly retained these fields. Never substitute an
                # old rationale for a newly adjudicated rule-card decision.
                assert not metadata[sid]["policy_targeted"], (sid, "missing current policy review")
            target = adjudicated_basis(row, case, supervision.get(sid))
            if target is not None:
                case["basis_target"] = target
                # One structured target joins the diagnostic budget; it does not
                # acquire the removed W/L budget or create extra action examples.
                case.setdefault("sft_targets", dict(case["targets"]))["basis"] = target
                counts[cell] += 1
                origins[filename.name] += 1
                nonnull.update(name for name, value in target.items() if value is not None)
        self.set_boundary_measure()
        self.inventory["source_first_sft"] = {
            "basis_targets": dict(counts), "basis_origins": dict(origins), "basis_is_action_parent": False,
            "action_members": dict(available), "basis_nonnull_fields": dict(nonnull),
            "basis_scope": "observed unmasked gaps, revisions and explicit verification duties; current review supersedes inherited fields; no action-derived rationales",
            "long_witness_and_link_targets": 0,
            "read_policy": "main SFT uses original X/P only; optional source-read inference is separately reported, not claimed as a trained policy",
            "budgets": "action 1; observed diagnostic fields (including evidence and boundary object) mean 1; removed W/L mass is not reassigned; native .1; verified relations 1",
            "review_history": "model/human provenance retained; not whole-dataset human certification"}

    def sft_coefficient(self, sid, name, prior_count, diagnostic_count):
        case = self.cases[sid]
        targets = case.get("sft_targets", case["targets"])
        mass = self.absolute_weights[sid]
        if name == "action":
            return mass
        balance = self.sft_balance.get(case["task_type"] + "/" + name)
        if balance is not None:
            mass *= balance["loss_multiplier"][str(targets[name])]
        return mass / len(targets.keys() - {"action"})
