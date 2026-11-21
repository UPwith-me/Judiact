"""Spread the existing root-weighted supervision over stratified updates."""

from collections import Counter, defaultdict
import heapq
import math
import random

from transformers import TrainerCallback


def stratified_exposures(cells, steps, seed):
    """Split large coefficient masses, not roots or independently counted data."""
    rng = random.Random(seed)
    updates = [[] for _ in range(steps)]
    repetitions, report = {}, {}
    for cell, members in sorted(cells.items()):
        total = math.fsum(mass for _, mass in members)
        pieces = []
        for sid, mass in sorted(members):
            # Avoid an extra exposure from roundoff at an integer boundary.
            count = max(1, math.ceil(steps * mass / total - 1e-10))
            repetitions[sid] = count
            pieces.extend((mass / total / count, rng.random(), sid) for _ in range(count))
        order = list(range(steps))
        rng.shuffle(order)
        heap = [(0., tie, step) for tie, step in enumerate(order)]
        heapq.heapify(heap)
        seen = [set() for _ in range(steps)]
        for mass, _, sid in sorted(pieces, reverse=True):
            skipped = []
            while True:
                current, tie, step = heapq.heappop(heap)
                if sid not in seen[step]:
                    break
                skipped.append((current, tie, step))
            seen[step].add(sid)
            updates[step].append(sid)
            heapq.heappush(heap, (current + mass, tie, step))
            for item in skipped:
                heapq.heappush(heap, item)
        masses = [mass for mass, _, _ in heap]
        report[cell] = {
            "members": len(members), "exposures": len(pieces),
            "repetitions": dict(sorted(Counter(repetitions[sid] for sid, _ in members).items())),
            "empty_updates": sum(not group for group in seen),
            "conditional_mass": math.fsum(masses),
            "update_mass_cv": math.sqrt(math.fsum((mass * steps - 1.) ** 2 for mass in masses) / steps),
        }
    return updates, repetitions, report


def exposure_schedule(data, dataset, sampling, seed):
    world, steps = sampling["world_size"], sampling["steps_per_epoch"]
    cells = defaultdict(list)
    clusters = {}
    for kind in ("request", "response"):
        for root, ids in data.populations[kind].items():
            for sid in ids:
                case = data.cases[sid]
                target = case.get("sft_targets", case["targets"])["action"]
                cells[case["task_type"] + "/" + target].append((sid, data.absolute_weights[sid]))
                clusters[sid] = kind, root
    updates, repetitions, report = stratified_exposures(cells, steps, seed)
    data.exposure_counts = repetitions
    positions = defaultdict(list)
    for step, ids in enumerate(updates):
        for sid in ids:
            positions[sid].append(step)

    experiments = [experiment for packet in dataset.packets for experiment in packet["experiments"]]
    relation_updates = [[] for _ in range(steps)]
    relation_load = [0] * steps
    for experiment in experiments:
        overlap = Counter(step for sid in experiment.sample_ids for step in positions[sid])
        step = min(overlap, key=lambda k: (-overlap[k], relation_load[k], k))
        relation_updates[step].append(experiment)
        relation_load[step] += len(experiment.sample_ids)

    costs = {}
    for sid in data.cases:
        case = data.cases[sid]
        visible = case["model_input"]
        text = sum(len(visible.get(name) or "") for name in ("query", "response_candidate", "ocr_text"))
        costs[sid] = 1 + text / 1500 + 2 * len(visible["images"]) + 8 * sum(
            name.startswith("evidence_") for name in case["targets"])
    packets = []
    for step, ids in enumerate(updates):
        bins = [{"step": step, "absolute_ids": [], "experiments": [],
                 "native_ids": list(dataset.packets[step * world + rank]["native_ids"])}
                for rank in range(world)]
        loads = [sum(costs[sid] for sid in packet["native_ids"]) for packet in bins]
        groups = defaultdict(lambda: {"absolute_ids": [], "experiments": []})
        for sid in ids:
            groups[clusters[sid]]["absolute_ids"].append(sid)
        for experiment in relation_updates[step]:
            groups[experiment.cluster]["experiments"].append(experiment)
        jobs = []
        for key, group in groups.items():
            required = set(group["absolute_ids"]) | {
                sid for experiment in group["experiments"] for sid in experiment.sample_ids}
            jobs.append((math.fsum(costs[sid] for sid in sorted(required)), key, group))
        for cost, _, group in sorted(jobs, key=lambda item: (-item[0], item[1])):
            rank = min(range(world), key=loads.__getitem__)
            bins[rank]["absolute_ids"].extend(group["absolute_ids"])
            bins[rank]["experiments"].extend(group["experiments"])
            loads[rank] += cost
        packets.extend(bins)
    dataset.packets = packets
    sampling = dict(sampling)
    sampling["member_uses_per_epoch"] = {
        **sampling["member_uses_per_epoch"],
        **{kind: sum(repetitions[sid] for ids in data.populations[kind].values() for sid in ids)
           for kind in ("request", "response")},
    }
    sampling["epoch_order"] = "rebuild with seed+epoch; task/action stratification; same member never twice in one update"
    sampling["stratified_exposure"] = {
        "cells": report, "formula": "m_i=max(1,ceil(steps*a_i/sum_class(a)))",
        "coefficient": "every existing member SFT field divided by m_i; action q/p applied once separately",
        "relations": "each original complete experiment once; placed near overlapping direct members; weights unchanged",
        "native": "original complete population, coefficients and per-rank DDP sync batches unchanged",
        "independent_members": len(repetitions), "member_exposures": sum(repetitions.values()),
        "repeated_member_max": max(repetitions.values()),
    }
    return dataset, sampling


class ExposureEpochs(TrainerCallback):
    """HF epoch callbacks also preserve the schedule on a mid-epoch resume."""

    def __init__(self, data, dataset, world, seed, effective_roots):
        self.data, self.dataset = data, dataset
        self.world, self.seed, self.effective_roots = world, seed, effective_roots
        self.epoch = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        epoch = math.floor(state.epoch or 0)
        if epoch != self.epoch:
            dataset, _ = self.data.schedule(self.world, self.seed + epoch, self.effective_roots, locality=True)
            self.dataset.packets = dataset.packets
            self.epoch = epoch
        return control
