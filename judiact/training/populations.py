"""Source-root and response-context grouping."""

import json
from collections import defaultdict
from pathlib import Path

from ..io_utils import read_jsonl


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
