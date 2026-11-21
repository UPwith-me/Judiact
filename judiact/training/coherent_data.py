"""Keep the fixed contract and distinguish observed, partial and missing targets."""

import json
from collections import Counter

from ..audit_schema import region_to_json, span_to_quote
from ..evidence_parser import audit_slots, visible_only
from .audit_data import AuditData, audit_targets


def coherent_case(record):
    targets = audit_targets(record)
    states = {}
    if not record["task_type"].startswith("native_"):
        visible, gold = record["model_input"], record["gold"]
        for slot in audit_slots(visible, record["task_type"]):
            if slot.kind != "evidence":
                continue
            state = gold["stage_states"][slot.stage]
            quotes = [span_to_quote(visible, span) for span in gold["evidence_spans"]
                      if span["stage"] == slot.stage]
            regions = []
            for region in gold["evidence_regions"]:
                if region["stage"] == slot.stage:
                    item = region_to_json(region)
                    regions.append({"description": item["description"],
                                    "media_index": item["media_index"], "bbox": item["bbox"]})
            if state is not None and state["E"] == "absent":
                if quotes or regions:
                    # Do not silently choose between a stage label and a legacy
                    # location. Keep the raw data and mask only this evidence block.
                    states[slot.name] = "annotation_conflict"
                    targets.pop(slot.name, None)
                    continue
                states[slot.name] = "known_absent"
            elif quotes or regions:
                states[slot.name] = "localized" if state is not None and state["E"] == "present" else "partial_localization"
            else:
                states[slot.name] = "unestablished"
                targets.pop(slot.name, None)
                continue
            targets[slot.name] = json.dumps({"quotes": quotes, "regions": regions},
                                            ensure_ascii=False, separators=(",", ":"))
    return {"sample_id": record["sample_id"], "task_type": record["task_type"],
            "model_input": visible_only(record["model_input"]), "targets": targets,
            "localization_states": states}


class CoherentAuditData(AuditData):
    def __init__(self, release, split="train", reviews=None):
        super().__init__(release, split, reviews)
        self.cases = {sid: coherent_case(row) for sid, row in (self.records | self.native_records).items()}
        self.inventory["supervised_slots"] = dict(Counter(name for c in self.cases.values() for name in c["targets"]))
        self.inventory["localization_states"] = dict(Counter(s for c in self.cases.values() for s in c["localization_states"].values()))
        self.inventory["record_targets"] = "known compatible fields; complete JSON and stop; unknown localization is masked null"

    def schedule(self, world, seed=42, effective_roots=64, locality=True):
        from .conditional_matching import build_matching_experiments, matching_weights
        return super().schedule(world, seed, effective_roots, locality,
                                build_matching_experiments, matching_weights)
