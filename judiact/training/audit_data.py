"""Frozen audit samples and reviewed indices, without the old contrast/SVD jobs."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from ..audit_schema import audit_slots, region_to_json, span_to_quote, visible_only
from ..data_contract import load_native_auxiliary, load_relations, load_s2_quartets
from ..io_utils import read_jsonl
from .populations import request_key, root_assignments


def audit_targets(record):
    """Targets are separate from rollout; missing localization is not empty gold."""
    visible = visible_only(record["model_input"])
    task = record["task_type"]
    slots = audit_slots(visible, task)
    if task.startswith("native_"):
        return {task: ("safe", "harmful")[record["native_target"]]}
    gold, targets = record["gold"], {}
    for slot in slots:
        name = slot.name
        if slot.kind == "evidence":
            state = gold["stage_states"][slot.stage]
            if state is None or state["E"] != "present":
                continue
            quotes = [span_to_quote(visible, span) for span in gold["evidence_spans"]
                      if span["stage"] == slot.stage]
            regions = [region_to_json(region) for region in gold["evidence_regions"]
                       if region["stage"] == slot.stage]
            if quotes or regions:
                targets[name] = json.dumps({"quotes": quotes, "regions": regions},
                                           ensure_ascii=False, separators=(",", ":"))
            continue
        if name[0] in "REF" and name[1:].isdigit():
            value = gold["stage_states"]["S" + name[1:]][name[0]]
        else:
            field = {"action": "response_publishability" if task == "response_audit" else "request_action",
                     "intent": "intent_class", "behavior": "safe_response_behavior"}.get(name, name)
            value = gold[field]
        if value is not None and value != "not_applicable":
            assert value in slot.labels, (record["sample_id"], name, value)
            targets[name] = value
    return targets


def audit_case(record):
    return {"sample_id": record["sample_id"], "task_type": record["task_type"],
            "model_input": visible_only(record["model_input"]), "targets": audit_targets(record)}


def relation_object_key(record, group_id):
    """A shared group ID does not merge request decisions with response publication."""
    response = record["task_type"] == "response_audit"
    return ("response" if response else "request", group_id,
            request_key(record["model_input"]) if response else None)


class AuditData:
    """Root/context membership and explicit review files; no hidden split is opened."""

    def __init__(self, release, split="train", reviews=None):
        assert split in ("train", "dev"), "hidden evaluation is a separate, unopened protocol"
        self.release, self.split = Path(release), split
        requests = read_jsonl(self.release / f"data/core_request/{split}.jsonl")
        responses = read_jsonl(self.release / f"data/s5_response/{split}.jsonl")
        self.records = {row["sample_id"]: row for row in requests + responses}
        assert len(self.records) == len(requests) + len(responses), "duplicate member IDs"
        assert all(row["split"] == split for row in self.records.values())
        self.root_ids = root_assignments(requests, reviews or self.release / "reviews")
        roots, contexts, native = defaultdict(list), defaultdict(list), defaultdict(list)
        for row in requests:
            roots[self.root_ids[row["sample_id"]]].append(row["sample_id"])
        for row in responses:
            contexts[request_key(row["model_input"])].append(row["sample_id"])
        self.native_records = {}
        if split == "train":
            for row in load_native_auxiliary(
                    str(self.release / "data/auxiliary/native_text_train.jsonl"),
                    str(self.release / "data/auxiliary/native_image_train.jsonl")):
                assert row["sample_id"] not in self.native_records
                self.native_records[row["sample_id"]] = row
                native[row["root_id"]].append(row["sample_id"])
        self.populations = {"request": dict(roots), "response": dict(contexts), "auxiliary": dict(native)}
        self.cases = {sample_id: audit_case(row) for sample_id, row in self.records.items()}
        assert not self.records.keys() & self.native_records.keys(), "native/Core member ID collision"
        self.cases.update({sample_id: audit_case(row) for sample_id, row in self.native_records.items()})
        raw_edges = read_jsonl(self.release / f"data/causal/verified_{split}.jsonl")
        self.raw_edges = {edge["edge_id"]: edge for edge in raw_edges}
        self.edges = list(load_relations(str(self.release / f"data/causal/verified_{split}.jsonl"),
                                          self.records))
        assert all(edge["use_causal_loss"] for edge in self.edges)
        self.quartets = [{key: item[key] for key in ("group_id", "split", "sample_ids", "observed_F2", "Delta_gold")}
                         | {"use_quartet_event": item["Delta_gold"] is not None}
                         for item in load_s2_quartets(
                             str(self.release / f"data/causal/verified_s2_quartets_{split}.jsonl"), self.records)]
        self.noncausal_edges = read_jsonl(self.release / f"data/causal/noncausal_{split}.jsonl")
        self.relation_groups = defaultdict(lambda: {"edges": [], "quartet": None})
        for edge in self.edges:
            left, right = (self.records[edge[name]] for name in ("left_id", "right_id"))
            key = relation_object_key(left, edge["group_id"])
            assert key == relation_object_key(right, edge["group_id"]), "relation changed the judgment context"
            self.relation_groups[key]["edges"].append(edge)
        for quartet in self.quartets:
            key = relation_object_key(self.records[quartet["sample_ids"][0]], quartet["group_id"])
            self.relation_groups[key]["quartet"] = quartet
        for (_, group_id, _), group in self.relation_groups.items():
            ids = sorted({edge[key] for edge in group["edges"] for key in ("left_id", "right_id")} |
                         set((group["quartet"] or {}).get("sample_ids", [])))
            group["sample_ids"] = ids
            first = self.records[ids[0]]
            group["cluster"] = ("response", request_key(first["model_input"])) \
                if first["task_type"] == "response_audit" else ("request", self.root_ids[ids[0]])
            group["group_id"] = group_id
        target_counts = Counter(name for case in self.cases.values() for name in case["targets"])
        self.inventory = {
            "split": split, "hidden_tests_loaded": False,
            "task_roots": {kind: len(units) for kind, units in self.populations.items()},
            "task_members": {kind: sum(map(len, units.values())) for kind, units in self.populations.items()},
            "image_request_members": sum(bool(row["model_input"]["images"]) for row in requests),
            "image_response_candidates": sum(bool(row["model_input"]["images"]) for row in responses),
            "supervised_slots": dict(target_counts),
            "verified_edges": len(self.edges), "noncausal_edges_excluded": len(self.noncausal_edges),
            "relation_group_ids": len({key[1] for key in self.relation_groups}),
            "relation_objects": len(self.relation_groups),
            "verified_s2_quartets": len(self.quartets),
            "s2_nonzero": sum(q["Delta_gold"] not in (None, 0) for q in self.quartets),
            "s2_zero_controls": sum(q["Delta_gold"] == 0 for q in self.quartets),
            "s2_uncertain": sum(q["Delta_gold"] is None for q in self.quartets),
            "s2_determined_event_indices": sum(q["use_quartet_event"] for q in self.quartets),
            "relation_note": "reviewed indices only; relation_events compiles field events separately; no scheduler here",
            "provenance_note": "existing review status retained; readability does not certify human finalization",
        }

    def schedule(self, world, seed=42, effective_roots=64, locality=False,
                 relation_builder=None, relation_weight_builder=None):
        """One full population pass; per-update scaling preserves the declared objective."""
        from .packets import ManifestDataset
        from .relation_events import build_relation_events, relation_weights
        from .relation_losses import absolute_weights

        experiments, omissions = (relation_builder or build_relation_events)(self)
        weights = absolute_weights(self.populations)
        event_weights = (relation_weight_builder or relation_weights)(experiments)
        total = sum(map(len, self.populations.values()))
        steps = math.ceil(total / effective_roots)
        rng = random.Random(seed)
        orders = {}
        for kind, roots in self.populations.items():
            orders[kind] = sorted(roots)
            rng.shuffle(orders[kind])
        by_cluster = defaultdict(list)
        for experiment in experiments:
            by_cluster[experiment.cluster].append(experiment)
        packets, uses = [], Counter()
        absolute_totals, relation_total = Counter(), 0.
        for step in range(steps):
            bins = [{"step": step, "absolute_ids": [], "native_ids": [], "experiments": []}
                    for _ in range(world)]
            costs = [0.] * world
            jobs = []
            clusters = defaultdict(lambda: {"absolute_ids": [], "experiments": []})
            for kind, order in orders.items():
                for root in order[step * len(order) // steps:(step + 1) * len(order) // steps]:
                    for sid in self.populations[kind][root]:
                        case = self.cases[sid]
                        visible = case["model_input"]
                        text = sum(len(visible.get(k) or "") for k in ("query", "response_candidate", "ocr_text"))
                        localized = sum(k.startswith("evidence_") for k in case["targets"])
                        cost = 1 + text / 1500 + 2 * len(visible["images"]) + 8 * localized
                        if locality and kind != "auxiliary":
                            clusters[kind, root]["absolute_ids"].append(sid)
                        else:
                            jobs.append((cost, "native" if kind == "auxiliary" else "absolute", sid))
                        absolute_totals[kind] += weights[sid]
                        uses[kind] += 1
                    for experiment in by_cluster.get((kind, root), []):
                        if locality:
                            clusters[kind, root]["experiments"].append(experiment)
                        else:
                            jobs.append((12 * len(experiment.sample_ids), "relation", experiment))
                        relation_total += sum(event_weights[experiment.key, field.slot] for field in experiment.fields)
            if locality:
                # Same global update/population weights; keep reusable audit prefixes on one rank.
                natives = sorted(jobs, key=lambda job: job[0], reverse=True)
                for rank, (cost, _, sid) in enumerate(natives[:world]):
                    bins[rank]["native_ids"].append(sid)
                    costs[rank] += cost
                jobs = natives[world:]
                for cluster in clusters.values():
                    ids = set(cluster["absolute_ids"]) | {sid for e in cluster["experiments"] for sid in e.sample_ids}
                    cost = 0.
                    # All workers must break cost ties from the same float sum.
                    for sid in sorted(ids):
                        case = self.cases[sid]
                        visible = case["model_input"]
                        text = sum(len(visible.get(k) or "") for k in ("query", "response_candidate", "ocr_text"))
                        # Every audit generates evidence even when its localization lacks a gold target.
                        blocks = sum(slot.kind == "evidence" for slot in audit_slots(visible, case["task_type"]))
                        cost += 1 + text / 1500 + 2 * len(visible["images"]) + 8 * blocks
                    jobs.append((cost, "cluster", cluster))
            for cost, kind, item in sorted(jobs, key=lambda job: job[0], reverse=True):
                rank = min(range(world), key=costs.__getitem__)
                costs[rank] += cost
                if kind == "cluster":
                    bins[rank]["absolute_ids"].extend(item["absolute_ids"])
                    bins[rank]["experiments"].extend(item["experiments"])
                    continue
                bins[rank]["experiments" if kind == "relation" else
                           "native_ids" if kind == "native" else "absolute_ids"].append(item)
            # Native forward/backward is the final real DDP synchronization on every rank.
            assert all(packet["native_ids"] for packet in bins)
            packets.extend(bins)
        report = {"steps_per_epoch": steps, "world_size": world, "effective_roots_nominal": effective_roots,
            "root_units_per_epoch": total, "root_repeats_for_padding": 0,
            "member_uses_per_epoch": dict(uses), "absolute_population_weight_totals": dict(absolute_totals),
            "relation_population_weight_total": relation_total, "relation_experiments": len(experiments),
            "relation_root_context_clusters": len(by_cluster), "omission_counts": dict(Counter(x["reason"] for x in omissions)),
            "update_scale": world * steps,
            "gradient_scaling": "world*steps times full-population contributions; DDP averages ranks; epoch-mean equals the declared frozen-parameter objective",
            "epoch_order": "fixed seed-shuffled full pass; no root oversampling or sharding padding",
            "gradient_accumulation_steps": 1, "packet_order": "sequential; world consecutive packets form one update",
            "weights": "request/response 0.5/0.5; roots/contexts then members; native roots then members; relation clusters/experiments/fields"}
        report["rank_assignment"] = "root/context locality; unchanged global update membership" if locality else "original member/experiment balancing"
        self.absolute_weights, self.event_weights = weights, event_weights
        return ManifestDataset(packets), report
