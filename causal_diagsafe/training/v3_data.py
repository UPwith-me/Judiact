"""A full-pass root manifest, with task-balanced updates and separate relation jobs."""

import math
import random
from collections import Counter, defaultdict
from pathlib import Path

from torch.utils.data import Dataset

from ..data_contract_v1 import load_native_auxiliary_v1, load_relations_v1, load_s2_quartets_v1
from ..io_utils import read_jsonl
from .v1_data import request_key, root_assignments
from .v3_views import classification_views, evidence_views, labels_for_view


def target_for(record, name):
    return dict(classification_views(record))[name]


def relation_jobs(records, edges, quartet, raw_edges):
    """Build only explicitly reviewed field differences; no inferred R/E stability."""
    ids = sorted({edge[key] for edge in edges for key in ("left_id", "right_id")} |
                 set((quartet or {}).get("sample_ids", [])))
    local = {sample_id: index for index, sample_id in enumerate(ids)}
    operators = defaultdict(list)
    policy_edges = []

    def pair(name, left, right):
        row = [0.] * len(ids)
        row[local[left]], row[local[right]] = -1., 1.
        operators[name].append(row)

    if quartet and quartet["Delta_gold"] is not None:
        row = [0.] * len(ids)
        for sample_id, sign in zip(quartet["sample_ids"], (1., -1., -1., 1.)):
            row[local[sample_id]] = sign
        operators["F2"].append(row)
    for edge in edges:
        a, b, stage = edge["left_id"], edge["right_id"], edge["target_stage"]
        original = raw_edges[edge["edge_id"]]
        if original["relation"]["target_F"] in ("flip", "stable"):
            pair("F" + stage[1:], a, b)
        for other in edge["stable_non_target_stages"]:
            ga, gb = (records[index]["gold"]["stage_states"][other] for index in (a, b))
            # Current review files certify non-target F, not independent R/E constancy.
            if ga and gb and ga["F"] in (0, 1) and ga["F"] == gb["F"]:
                pair("F" + other[1:], a, b)
        if original["relation"]["decision"] in ("same", "different"):
            pair("action", a, b)
        if edge["policy_alignment"]:
            mapping = []
            nleft = len(records[a]["model_input"]["policy_deck"])
            nright = len(records[b]["model_input"]["policy_deck"])
            for left, right in edge["policy_alignment"]:
                item = (left // 4 if left < 4 * nleft else nleft + left - 4 * nleft,
                        right // 4 if right < 4 * nright else nright + right - 4 * nright)
                if item not in mapping:
                    mapping.append(item)
            left_indices, right_indices = ([item[i] for item in mapping] for i in (0, 1))
            policy_edges.append((a, b, dict(mapping)))
            pair("policy_action", a, b)
    jobs = []
    for name, rows in sorted(operators.items()):
        used = [index for index in range(len(ids)) if any(row[index] for row in rows)]
        specs = []
        for index in used:
            record = records[ids[index]]
            choices = labels_for_view(record["model_input"], name)
            specs.append({"record": record, "name": name, "target": choices.index(target_for(record, name))})
        jobs.append({"kind": "contrast", "specs": specs,
                     "operators": [[row[index] for index in used] for row in rows], "field": name})
    # Compose the reviewed ID maps in each connected invariant-deck component.
    # Duplicated policy edges now share one row space, just like all other fields.
    remaining = list(policy_edges)
    while remaining:
        a, _, mapping = remaining[0]
        gathers = {a: sorted(mapping)}
        selected = []
        changed = True
        while changed:
            changed = False
            for edge in list(remaining):
                left, right, mapping = edge
                if left in gathers or right in gathers:
                    if left not in gathers:
                        inverse = {value: key for key, value in mapping.items()}
                        gathers[left] = [inverse[index] for index in gathers[right]]
                    expected = [mapping[index] for index in gathers[left]]
                    if right in gathers:
                        assert gathers[right] == expected, "inconsistent reviewed policy semantic maps"
                    gathers[right] = expected
                    selected.append(edge)
                    remaining.remove(edge)
                    changed = True
        members = sorted(gathers)
        specs, rows = [], []
        for index in members:
            gather = gathers[index]
            choices = labels_for_view(records[index]["model_input"], "policy_id")
            specs.append({"record": records[index], "name": "policy_id", "gather": gather,
                          "target": gather.index(choices.index(target_for(records[index], "policy_id")))})
        for left, right, _ in selected:
            row = [0.] * len(members)
            row[members.index(left)], row[members.index(right)] = -1., 1.
            rows.append(row)
        jobs.append({"kind": "contrast", "specs": specs, "operators": rows, "field": "policy_id"})
    return jobs


def semantic_jobs(record, weight):
    classified, located = classification_views(record), evidence_views(record)
    scale = weight / (len(classified) + len(located))
    jobs = [{"kind": "classify", "record": record, "name": name,
             "target": labels_for_view(record["model_input"], name).index(target), "weight": scale}
            for name, target in classified]
    jobs.extend({"kind": "evidence", "record": record, "stage": stage, "field": field, "answer": answer,
                 "weight": scale} for stage, field, answer in located)
    return jobs


class ManifestDataset(Dataset):
    def __init__(self, packets):
        self.packets = packets

    def __len__(self):
        return len(self.packets)

    def __getitem__(self, index):
        return self.packets[index]


def packet_collator(packets):
    assert len(packets) == 1
    return packets[0]


class V3TrainingData:
    def __init__(self, release, reviews):
        self.release, self.reviews = Path(release), Path(reviews)
        requests = read_jsonl(self.release / "data/core_request/train.jsonl")
        responses = read_jsonl(self.release / "data/s5_response/train.jsonl")
        self.records = {row["sample_id"]: row for row in requests + responses}
        self.root_ids = root_assignments(requests, reviews)
        roots, contexts, native = defaultdict(list), defaultdict(list), defaultdict(list)
        for row in requests:
            roots[self.root_ids[row["sample_id"]]].append(row)
        for row in responses:
            contexts[request_key(row["model_input"])].append(row)
        for row in load_native_auxiliary_v1(
                str(self.release / "data/auxiliary/native_text_train.jsonl"),
                str(self.release / "data/auxiliary/native_image_train.jsonl")):
            native[row["root_id"]].append(row)
        self.populations = {"request": roots, "response": contexts, "auxiliary": native}
        edge_path = self.release / "data/causal/verified_train.jsonl"
        self.raw_edges = {edge["edge_id"]: edge for edge in read_jsonl(edge_path)}
        grouped_edges = defaultdict(list)
        for edge in load_relations_v1(str(edge_path), self.records):
            if edge["use_causal_loss"]:
                grouped_edges[edge["group_id"]].append(edge)
        quartets = {item["group_id"]: item for item in load_s2_quartets_v1(
            str(self.release / "data/causal/verified_s2_quartets_train.jsonl"), self.records)}
        self.relation_groups = []
        for group in sorted(grouped_edges.keys() | quartets.keys()):
            edges, quartet = grouped_edges[group], quartets.get(group)
            jobs = relation_jobs(self.records, edges, quartet, self.raw_edges)
            if not jobs:
                continue
            first = jobs[0]["specs"][0]["record"]
            cluster = ("response", request_key(first["model_input"])) if first["task_type"] == "response_audit" \
                else ("request", self.root_ids[first["sample_id"]])
            self.relation_groups.append({"group_id": group, "cluster": cluster, "jobs": jobs})
        self.inventory = {
            "task_roots": {kind: len(units) for kind, units in self.populations.items()},
            "task_members": {kind: sum(map(len, units.values())) for kind, units in self.populations.items()},
            "relation_group_ids": len(self.relation_groups),
            "relation_root_context_clusters": len({group["cluster"] for group in self.relation_groups}),
            "verified_edges": len(self.raw_edges), "verified_s2_quartets": len(quartets),
            "s2_nonzero": sum(item["Delta_gold"] not in (None, 0) for item in quartets.values()),
            "s2_zero_controls": sum(item["Delta_gold"] == 0 for item in quartets.values()),
            "s2_uncertain": sum(item["Delta_gold"] is None for item in quartets.values()),
            "non_target_stability_fields": "F only: existing reviews do not separately certify R/E",
        }

    def schedule(self, world, seed, effective_roots=64):
        total = sum(map(len, self.populations.values()))
        steps = math.ceil(total / effective_roots)
        rng = random.Random(seed)
        orders = {}
        for kind, population in self.populations.items():
            orders[kind] = list(sorted(population))
            rng.shuffle(orders[kind])
        relations = list(self.relation_groups)
        rng.shuffle(relations)
        cluster_counts = Counter(group["cluster"] for group in relations)
        packets = []
        semantic_totals, group_total = Counter(), 0.
        view_counts, member_counts = Counter(), Counter()
        for step in range(steps):
            bins = [{"classify": [], "other": [], "root_ids": [], "step": step} for _ in range(world)]
            costs = [0] * world
            units = []
            for kind, order in orders.items():
                for key in order[step * len(order) // steps:(step + 1) * len(order) // steps]:
                    records = self.populations[kind][key]
                    weight = world * steps / (3 * len(order))
                    jobs = [job for record in records for job in semantic_jobs(record, weight / len(records))]
                    semantic_totals[kind] += weight / world / steps
                    units.append((sum(1 if job["kind"] == "classify" else 3 for job in jobs),
                                  kind, key, records, jobs))
            # Keep expensive roots intact while avoiding systematically overloading one rank.
            units.sort(key=lambda item: item[0], reverse=True)
            for cost, kind, key, records, jobs in units:
                rank = min(range(world), key=lambda index: costs[index])
                costs[rank] += cost
                bins[rank]["root_ids"].append((kind, key))
                for job in jobs:
                    bins[rank]["classify" if job["kind"] == "classify" else "other"].append(job)
                    view_counts[job["kind"]] += 1
                member_counts[kind] += len(records)
            for group in relations[step * len(relations) // steps:(step + 1) * len(relations) // steps]:
                rank = min(range(world), key=lambda index: costs[index])
                weight = world * steps / (len(cluster_counts) * cluster_counts[group["cluster"]])
                jobs = group["jobs"]
                for job in jobs:
                    bins[rank]["other"].append({**job, "group_id": group["group_id"], "weight": weight / len(jobs)})
                    view_counts["contrast_forward_views"] += len(job["specs"])
                costs[rank] += sum(len(job["specs"]) for job in jobs)
                group_total += weight / world / steps
            assert all(item["classify"] for item in bins), "each DDP rank needs a real final backward"
            packets.extend(bins)
        report = {"steps_per_epoch": steps, "world_size": world, "effective_roots_nominal": effective_roots,
                  "root_units_per_epoch": total, "root_repeats_for_padding": 0,
                  "semantic_weight_sum_per_epoch": dict(semantic_totals),
                  "relation_weight_sum_per_epoch": group_total,
                  "member_uses_per_epoch": dict(member_counts), "forward_views_per_epoch": dict(view_counts),
                  "epoch_order": "fixed seed-shuffled full-pass manifest; identical exposure for both variants",
                  "relation_absolute_weight": 0, "relation_population": "root/context clusters, groups then fields averaged",
                  "gradient_accumulation_steps": 1, "packet_order": "sequential; world consecutive packets form one update"}
        return ManifestDataset(packets), report
