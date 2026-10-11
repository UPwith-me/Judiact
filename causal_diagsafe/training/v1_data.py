"""Frozen-contract packets, root/group sampling, and official Qwen evidence alignment."""

from __future__ import annotations

import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset

from ..data_contract_v1 import (
    encode_sample_v1, load_native_auxiliary_v1, load_relations_v1,
    load_s2_quartets_v1, messages_v1,
)
from ..io_utils import read_jsonl
from ..models.diagsafe_v1 import FIELDS


SYSTEM_PROMPT = (
    "Assess the visible user request under the supplied policy deck. Use only the supplied "
    "images and text; do not infer missing evidence. Distinguish perceptual evidence, "
    "image-text binding, user intent, and policy applicability. If a candidate response is "
    "present, assess that response's safety and publishability separately from the request. "
    "Educational, protective, analytical, authorized, and fictional contexts may be safe."
)


def request_key(visible):
    return json.dumps({
        "query": visible["query"], "images": [image["sha256"] for image in visible["images"]],
        "ocr_text": visible["ocr_text"], "policy_deck": visible["policy_deck"],
    }, sort_keys=True, ensure_ascii=False)


def root_assignments(samples, review_dir):
    groups = defaultdict(list)
    for sample in samples:
        groups[sample["group_id"]].append(sample)
    parent = {group: group for group in groups}

    def find(group):
        while parent[group] != group:
            parent[group] = parent[parent[group]]
            group = parent[group]
        return group

    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)

    aliases = {row["right_sha256"]: row["left_sha256"] for row in
               read_jsonl(Path(review_dir) / "confirmed_core_image_aliases.jsonl")}
    families = {sha: row["family_id"] for row in
                read_jsonl(Path(review_dir) / "confirmed_image_root_families.jsonl")
                for sha in row["image_sha256s"]}
    owners = {}
    for group, members in sorted(groups.items()):
        keys = set()
        has_image = any(member["model_input"]["images"] for member in members)
        for sample in members:
            for image in sample["model_input"]["images"]:
                sha = image["sha256"]
                keys.add(("image", aliases.get(sha, sha)))
                if sha in families:
                    keys.add(("image_family", families[sha]))
            source = sample["source"]
            if source["query_dataset"] and source["query_record_id"]:
                keys.add(("source", source["query_dataset"], source["query_record_id"]))
            if not has_image and sample["task_type"] != "response_audit":
                keys.add(("text", " ".join(sample["model_input"]["query"].casefold().split()),
                          sample["model_input"]["ocr_text"]))
        for key in sorted(keys, key=str):
            if key in owners:
                union(group, owners[key])
            else:
                owners[key] = group
    return {sample["sample_id"]: find(sample["group_id"]) for sample in samples}


class PacketDataset(Dataset):
    def __init__(self, packets):
        self.packets = packets

    def __len__(self):
        return len(self.packets)

    def __getitem__(self, index):
        return self.packets[index]


def repeated_permutation(values, size, rng):
    result = []
    while len(result) < size:
        cycle = list(values)
        rng.shuffle(cycle)
        result.extend(cycle)
    return result[:size]


def separate_update_duplicates(order, block, key):
    for i in range(len(order)):
        start = i // block * block
        previous = {key(item) for item in order[start:i]}
        if key(order[i]) in previous:
            candidates = [j for j in range(i + 1, len(order)) if key(order[j]) not in previous]
            if not candidates:
                duplicate = key(order[i])
                candidates = [j for j in range(start) if key(order[j]) not in previous and duplicate not in
                              {key(item) for item in order[j // block * block:(j // block + 1) * block]}]
            j = candidates[0]
            order[i], order[j] = order[j], order[i]
    return order


class FrozenTrainingData:
    def __init__(self, release, review_dir):
        self.release = Path(release)
        requests = list(read_jsonl(self.release / "data/core_request/train.jsonl"))
        responses = list(read_jsonl(self.release / "data/s5_response/train.jsonl"))
        raw = {row["sample_id"]: row for row in requests + responses}
        assert len(raw) == len(requests) + len(responses)
        assert all(row["split"] == "train" for row in raw.values())
        self.records = {key: encode_sample_v1(row) for key, row in raw.items()}
        self.root_ids = root_assignments(raw.values(), review_dir)
        self.core_roots = defaultdict(list)
        self.core_bins = defaultdict(lambda: defaultdict(list))
        for row in requests:
            record = self.records[row["sample_id"]]
            root = self.root_ids[row["sample_id"]]
            self.core_roots[root].append(record)
            visible = record["model_input"]
            modality = "image" if visible["images"] else "ocr" if visible["ocr_text"] is not None else "text"
            self.core_bins[(row["gold"]["request_action"], modality)][root].append(record)
        self.native_roots = defaultdict(list)
        for row in load_native_auxiliary_v1(
            str(self.release / "data/auxiliary/native_text_train.jsonl"),
            str(self.release / "data/auxiliary/native_image_train.jsonl"),
        ):
            self.native_roots[row["root_id"]].append(row)
        edges = list(load_relations_v1(str(self.release / "data/causal/verified_train.jsonl"), raw))
        quartets = {row["group_id"]: row for row in load_s2_quartets_v1(
            str(self.release / "data/causal/verified_s2_quartets_train.jsonl"), raw)}
        group_requests, group_edges = defaultdict(list), defaultdict(list)
        for row in requests:
            group_requests[row["group_id"]].append(self.records[row["sample_id"]])
        self.stage_groups = defaultdict(list)
        for edge in edges:
            assert edge["use_causal_loss"]
            if edge["target_stage"] != "S5":
                group_edges[edge["group_id"]].append(edge)
        for group, members in sorted(group_requests.items()):
            if group not in group_edges:
                continue
            unit = {"kind": "request", "id": group, "records": members,
                    "edges": group_edges[group], "quartet": quartets.get(group)}
            for stage in sorted({edge["target_stage"] for edge in group_edges[group]}):
                self.stage_groups[stage].append(unit)
        self.response_contexts = defaultdict(list)
        for row in responses:
            self.response_contexts[request_key(row["model_input"])].append(self.records[row["sample_id"]])
        response_edges = defaultdict(list)
        for edge in edges:
            if edge["target_stage"] == "S5":
                left, right = (self.records[edge[key]] for key in ("left_id", "right_id"))
                key = request_key(left["model_input"])
                assert key == request_key(right["model_input"]), edge["edge_id"]
                response_edges[key].append(edge)
        self.response_units = [{"kind": "response", "id": records[0]["sample_id"],
                                "records": records, "edges": response_edges[key], "quartet": None}
                               for key, records in sorted(self.response_contexts.items())]
        self.inventory = {
            "core_roots": len(self.core_roots), "core_members": len(requests),
            "native_roots": len(self.native_roots),
            "native_items": sum(map(len, self.native_roots.values())),
            "verified_group_pools": {stage: len(groups) for stage, groups in self.stage_groups.items()},
            "verified_s2_quartets": len(quartets),
            "s2_nonzero_interactions": sum(row["use_interaction_loss"] for row in quartets.values()),
            "response_candidates": len(responses), "response_contexts": len(self.response_units),
            "image_response_candidates": sum(bool(row["model_input"]["images"]) for row in responses),
        }

    def draw_core(self, count, rng):
        bins = list(sorted(self.core_bins))
        chosen_roots, units = set(), []
        for _ in range(count):
            available = [(key, [root for root in self.core_bins[key] if root not in chosen_roots])
                         for key in bins]
            available = [(key, roots) for key, roots in available if roots]
            actions = sorted({key[0] for key, roots in available})
            action = rng.choice(actions)
            options = [(key, roots) for key, roots in available if key[0] == action]
            key, roots = rng.choices(options, weights=[{"image": .55, "text": .30, "ocr": .15}[key[1]]
                                                       for key, roots in options])[0]
            root = rng.choice(roots)
            chosen_roots.add(root)
            member = rng.choice(self.core_bins[key][root])
            units.append({"kind": "request", "id": root, "records": [member],
                          "edges": [], "quartet": None})
        return units

    def schedule(self, stage, world_size, seed):
        rng = random.Random(seed + ord(stage))
        accumulation = 4 if stage == "B" else 16
        block = world_size * accumulation
        packets = []

        def append_update(units):
            packets.extend({"units": [unit]} for unit in units)

        def replay():
            roots_per_packet = 4 if stage == "B" else 1
            units = self.draw_core(block * roots_per_packet, rng)
            packets.extend({"units": units[i:i + roots_per_packet]}
                           for i in range(0, len(units), roots_per_packet))

        if stage == "A":
            sources = defaultdict(list)
            for root in sorted(self.native_roots):
                sources[root.split(":", 1)[0]].append(root)
            largest = max(map(len, sources.values()))
            order = []
            for roots in sources.values():
                order.extend(repeated_permutation(roots, math.ceil(math.sqrt(largest * len(roots))), rng))
            rng.shuffle(order)
            updates = math.ceil(len(order) / block)
            # Keep repeated roots out of a single effective update without changing family counts.
            separate_update_duplicates(order, block, lambda root: root)
            for step in range(updates):
                roots = order[step * block:(step + 1) * block]
                if len(roots) < block:
                    roots += rng.sample([root for root in self.native_roots if root not in roots], block - len(roots))
                append_update([{"kind": "auxiliary", "id": root, "records": self.native_roots[root],
                                "edges": [], "quartet": None} for root in roots])
                if step % 2 == 1 or step == updates - 1:
                    replay()
        elif stage == "B":
            maximum = max(map(len, self.stage_groups.values()))
            order = []
            for groups in self.stage_groups.values():
                order.extend(repeated_permutation(groups, 2 * maximum, rng))
            rng.shuffle(order)
            order.extend(rng.choices(order, k=(-len(order)) % block))
            separate_update_duplicates(order, block, lambda unit: unit["id"])
            for i in range(0, len(order), block):
                append_update(order[i:i + block])
                replay()
        else:
            order = repeated_permutation(self.response_units, 2 * len(self.response_units), rng)
            order.extend(rng.choices(order, k=(-len(order)) % block))
            separate_update_duplicates(order, block, lambda unit: unit["id"])
            for i in range(0, len(order), block):
                append_update(order[i:i + block])
                replay()
        if stage == "A":
            replay_units = [unit for packet in packets for unit in packet["units"]
                            if unit["kind"] == "request"]
            uses = Counter(unit["id"] for unit in replay_units)
            missing = sorted(set(self.core_roots) - uses.keys())
            # Stratification can miss rare roots; replace repeated draws, not their labels.
            positions = iter(range(len(replay_units) - 1, -1, -1))
            for root in missing:
                index = next(index for index in positions if uses[replay_units[index]["id"]] > 1)
                unit = replay_units[index]
                uses[unit["id"]] -= 1
                member = rng.choice(self.core_roots[root])
                unit.update({"id": root, "records": [member]})
                uses[root] += 1
        draws = [unit for packet in packets for unit in packet["units"]]
        record_draws = [record for unit in draws for record in unit["records"]]
        report = {
            "stage": stage, "world_size": world_size, "gradient_accumulation_steps": accumulation,
            "optimizer_steps": len(packets) // block, "micro_packets_global": len(packets),
            "unit_draws": dict(Counter(unit["kind"] for unit in draws)),
            "distinct_units": {kind: len({unit["id"] for unit in draws if unit["kind"] == kind})
                               for kind in ("auxiliary", "request", "response")},
            "member_draws": len(record_draws),
            "sample_only_root_draws": sum(unit["kind"] == "request" and not unit["edges"] for unit in draws),
            "sample_only_unique_roots": len({unit["id"] for unit in draws
                                             if unit["kind"] == "request" and not unit["edges"]}),
            "causal_group_draws": sum(bool(unit["edges"]) for unit in draws),
            "unique_causal_groups": len({edge["group_id"] for unit in draws for edge in unit["edges"]}),
            "s2_interaction_draws": sum(bool(unit["quartet"] and unit["quartet"]["use_interaction_loss"])
                                        for unit in draws),
            "native_source_draws": dict(Counter(unit["id"].split(":", 1)[0] for unit in draws
                                                 if unit["kind"] == "auxiliary")),
            "core_action_draws": dict(Counter(record["targets"]["request_action"] for record in record_draws
                                               if record["task_type"] in ("request_decision", "perception_probe"))),
            "modality_draws": dict(Counter("image" if record["model_input"]["images"] else "ocr"
                                             if record["model_input"]["ocr_text"] is not None else "text"
                                             for record in record_draws)),
        }
        return PacketDataset(packets), report

    def natural_dev(self, include_responses):
        records = list(read_jsonl(self.release / "data/natural_request/dev.jsonl"))
        if include_responses:
            records += list(read_jsonl(self.release / "data/natural_response/dev.jsonl"))
        return PacketDataset([{"units": [{"kind": "response" if row["task_type"] == "response_audit" else "request",
                                           "id": row["sample_id"], "records": [encode_sample_v1(row)],
                                           "edges": [], "quartet": None}]} for row in records])


def expand_image_markers(text, token, counts):
    chunks = text.split(token)
    assert len(chunks) == len(counts) + 1, "image placeholders must match actual images"
    return chunks[0] + "".join(token * count + chunk for count, chunk in zip(counts, chunks[1:]))


def field_ranges(text, visible):
    ranges, rules = defaultdict(list), []
    cursor = text.index("User query:\n") + len("User query:\n")
    query = visible["query"]
    assert text[cursor:cursor + len(query)] == query
    ranges["query"].append((cursor, cursor + len(query)))
    cursor += len(query)
    if visible["ocr_text"] is not None:
        cursor = text.index("Visible OCR transcription:\n", cursor) + len("Visible OCR transcription:\n")
        ranges["ocr_text"].append((cursor, cursor + len(visible["ocr_text"])))
        cursor += len(visible["ocr_text"])
    cursor = text.index("Policy deck:\n", cursor) + len("Policy deck:\n")
    for card in visible["policy_deck"]:
        line = card["policy_id"] + ": " + card["rule_text"]
        assert text[cursor:cursor + len(line)] == line
        start = cursor + len(card["policy_id"]) + 2
        interval = (start, start + len(card["rule_text"]))
        ranges["policy_rule"].append(interval)
        rules.append(interval)
        cursor += len(line) + 1
    if visible["response_candidate"] is not None:
        cursor = text.index("Candidate response:\n", cursor) + len("Candidate response:\n")
        ranges["response_candidate"].append((cursor, cursor + len(visible["response_candidate"])))
    return ranges, rules


def overlapping_tokens(offsets, start, end):
    return ((offsets[:, 0] < end) & (offsets[:, 1] > start) &
            (offsets[:, 1] > offsets[:, 0])).nonzero().flatten()


def region_tokens(grid, image_positions, box, merge_size):
    temporal, h, w = map(int, grid)
    rows, cols = h // merge_size, w // merge_size
    x, y, width, height = box
    yy, xx = torch.meshgrid(torch.arange(rows), torch.arange(cols), indexing="ij")
    overlap = ((xx / cols < x + width) & ((xx + 1) / cols > x) &
               (yy / rows < y + height) & ((yy + 1) / rows > y))
    selected = overlap.flatten().repeat(temporal).nonzero().flatten()
    assert image_positions.numel() == temporal * rows * cols
    return image_positions[selected]


class EvidenceCollator:
    def __init__(self, processor, include_alignment_metadata=False):
        self.processor = processor
        self.include_alignment_metadata = include_alignment_metadata
        self.processor.tokenizer.padding_side = "right"

    def __call__(self, packets):
        assert len(packets) == 1, "one intact packet per DataLoader item"
        units = packets[0]["units"]
        records = [record for unit in units for record in unit["records"]]
        texts, images = [], []
        for record in records:
            visible = record["model_input"]
            assert not visible["images"] or visible["ocr_text"] is None
            texts.append(self.processor.apply_chat_template(messages_v1(visible, SYSTEM_PROMPT),
                                                            tokenize=False, add_generation_prompt=True))
            for image in visible["images"]:
                with Image.open(image["uri"]) as opened:
                    images.append(opened.convert("RGB"))
        batch = self.processor(text=texts, images=images or None, padding=True,
                               truncation=False, return_tensors="pt", return_offsets_mapping=True)
        offsets = batch.pop("offset_mapping")
        batch.pop("token_type_ids", None)
        grids = batch.get("image_grid_thw", torch.empty(0, 3, dtype=torch.long))
        merge = self.processor.image_processor.merge_size
        length = batch["input_ids"].shape[1]
        field_masks = torch.zeros(len(records), len(FIELDS), length, dtype=torch.bool)
        max_rules = max(len(record["model_input"]["policy_deck"]) for record in records)
        rule_masks = torch.zeros(len(records), max_rules, length, dtype=torch.bool)
        labels = {key: torch.full((len(records),), -100, dtype=torch.long) for key in (
            "request_action", "response_publishability", "intent", "policy_joint", "safe_response_behavior", "native_target")}
        labels.update({key: torch.full((len(records), 5), -100, dtype=torch.long) for key in ("R", "E", "F")})
        labels["native_tasks"] = [None] * len(records)
        evidence, metadata = [], []
        image_cursor = 0
        for row, record in enumerate(records):
            visible = record["model_input"]
            row_grids = grids[image_cursor:image_cursor + len(visible["images"])]
            counts = [int(grid.prod()) // merge**2 for grid in row_grids]
            expanded = expand_image_markers(texts[row], self.processor.image_token, counts)
            expected_ids = self.processor.tokenizer(expanded, add_special_tokens=False)["input_ids"]
            actual_ids = batch["input_ids"][row][batch["attention_mask"][row].bool()].tolist()
            assert expected_ids == actual_ids, record["sample_id"]
            ranges, rule_ranges = field_ranges(expanded, visible)
            for field, intervals in ranges.items():
                for start, end in intervals:
                    field_masks[row, FIELDS.index(field), overlapping_tokens(offsets[row], start, end)] = True
            for rule, (start, end) in enumerate(rule_ranges):
                rule_masks[row, rule, overlapping_tokens(offsets[row], start, end)] = True
            visual_positions = (batch["input_ids"][row] == self.processor.image_token_id).nonzero().flatten()
            field_masks[row, 0, visual_positions] = True
            per_image = list(visual_positions.split(counts)) if counts else []
            if record["task_type"].startswith("native_"):
                labels["native_tasks"][row] = record["task_type"]
                labels["native_target"][row] = record["native_target"]
            elif "targets" in record:
                targets = record["targets"]
                for variable in ("R", "E", "F"):
                    labels[variable][row] = torch.tensor(targets["stage"][variable])
                for key in ("request_action", "response_publishability", "intent", "safe_response_behavior"):
                    labels[key][row] = targets[key]
                candidate = targets["policy_joint"]
                own_rules = len(rule_ranges)
                labels["policy_joint"][row] = candidate if candidate < 4 * own_rules else \
                    4 * max_rules + candidate - 4 * own_rules
                if candidate == -100:
                    labels["policy_joint"][row] = -100
                for span in targets["evidence_spans"]:
                    field = span["field"]
                    if field == "policy_rule":
                        index = next(i for i, card in enumerate(visible["policy_deck"])
                                     if card["policy_id"] == span["policy_id"])
                        base = rule_ranges[index][0]
                        source = visible["policy_deck"][index]["rule_text"]
                    else:
                        base = ranges[field][0][0]
                        source = visible[field]
                    assert source[span["start"]:span["end"]] == span["text"], record["sample_id"]
                    positions = overlapping_tokens(offsets[row], base + span["start"], base + span["end"])
                    assert positions.numel() > 0, (record["sample_id"], span)
                    evidence.append({"row": row, "stage": int(span["stage"][1:]) - 1,
                                     "field": FIELDS.index(field), "positions": positions})
                for region in targets["evidence_regions"]:
                    index = region["media_index"]
                    positions = region_tokens(row_grids[index], per_image[index], region["box"], merge)
                    assert positions.numel() > 0, (record["sample_id"], region)
                    evidence.append({"row": row, "stage": int(region["stage"][1:]) - 1,
                                     "field": 0, "positions": positions})
            information = {"sample_id": record["sample_id"], "tokens": len(actual_ids),
                           "image_tokens": counts, "rule_tokens": rule_masks[row].sum(-1).tolist()}
            if self.include_alignment_metadata:
                information.update({"field_ranges": dict(ranges), "rule_ranges": rule_ranges,
                    "image_grids": row_grids.tolist(), "image_positions": [item.tolist() for item in per_image],
                    "offsets": offsets[row, :len(actual_ids)].tolist()})
            metadata.append(information)
            image_cursor += len(visible["images"])
        loss_units, cursor = [], 0
        for unit in units:
            rows = list(range(cursor, cursor + len(unit["records"])))
            local = {record["sample_id"]: row for row, record in zip(rows, unit["records"])}
            groups = defaultdict(lambda: {"edges": [], "quartet": None})
            for edge in unit["edges"]:
                left, right = local[edge["left_id"]], local[edge["right_id"]]
                alignment = []
                for a, b in edge["policy_alignment"]:
                    rebased = []
                    for index, endpoint in ((a, left), (b, right)):
                        own_rules = len(records[endpoint]["model_input"]["policy_deck"])
                        rebased.append(index if index < 4 * own_rules else 4 * max_rules + index - 4 * own_rules)
                    alignment.append(tuple(rebased))
                groups[edge["group_id"]]["edges"].append({**edge, "left": left, "right": right,
                    "stage": int(edge["target_stage"][1:]) - 1, "policy_alignment": alignment})
            quartet = unit["quartet"]
            if quartet:
                groups[quartet["group_id"]]["quartet"] = {
                    **quartet, "rows": [local[key] for key in quartet["sample_ids"]]}
            loss_units.append({"kind": unit["kind"], "rows": rows, "groups": list(groups.values())})
            cursor += len(rows)
        return {
            "encoded": dict(batch), "field_masks": field_masks, "rule_masks": rule_masks,
            "rule_counts": torch.tensor([len(record["model_input"]["policy_deck"]) for record in records]),
            "response_present": torch.tensor([record["model_input"]["response_candidate"] is not None for record in records]),
            "supervision": {"labels": labels, "evidence": evidence, "units": loss_units},
            "metadata": metadata,
        }
