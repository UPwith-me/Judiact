"""Copy train/dev exports, apply exact reviewed corrections, retain media/source links."""

import argparse
import copy
import os
import shutil
from pathlib import Path

from ..io_utils import read_jsonl, write_json, write_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--corrections", required=True)
    args = parser.parse_args()
    source, output = Path(args.release), Path(args.output)
    corrections = read_jsonl(args.corrections)
    correction_ids = {(item["split"], item["sample_id"]) for item in corrections}
    seen, relation_changes, media, counts = set(), [], {}, {}
    for split in ("train", "dev"):
        rows = [row for kind in ("core_request", "s5_response")
                for row in read_jsonl(source / "data" / kind / f"{split}.jsonl")]
        records = {row["sample_id"]: copy.deepcopy(row) for row in rows}
        counts[split] = {"request_members": sum(row["task_type"] != "response_audit" for row in rows),
                         "response_candidates": sum(row["task_type"] == "response_audit" for row in rows)}
        for row in rows:
            for image in row["model_input"]["images"]:
                media[image["uri"]] = image
        for item in corrections:
            if item["split"] != split:
                continue
            record = records[item["sample_id"]]
            target = record["gold"]
            for field in item["field_path"][:-1]:
                target = target[field]
            target[item["field_path"][-1]] = item["new_value"]
            seen.add((split, item["sample_id"]))
        for kind, task in (("core_request", False), ("s5_response", True)):
            write_jsonl(output / "data" / kind / f"{split}.jsonl",
                        (records[row["sample_id"]] for row in rows if (row["task_type"] == "response_audit") == task))
        for kind in ("natural_request", "natural_response"):
            write_jsonl(output / "data" / kind / f"{split}.jsonl",
                        (records[row["sample_id"]] for row in read_jsonl(source / "data" / kind / f"{split}.jsonl")))
        edges = read_jsonl(source / "data/causal" / f"verified_{split}.jsonl")
        for edge in edges:
            before = copy.deepcopy(edge["relation"])
            a, b = (records[edge[key]]["gold"]["stage_states"] for key in ("left_id", "right_id"))
            if (split, edge["left_id"]) in correction_ids or (split, edge["right_id"]) in correction_ids:
                fleft, fright = a[edge["target_stage"]]["F"], b[edge["target_stage"]]["F"]
                if before["target_F"] in ("stable", "flip"):
                    edge["relation"]["target_F"] = ("stable" if fleft == fright else "flip") \
                        if fleft in (0, 1) and fright in (0, 1) else "uncertain"
                edge["relation"]["stable_non_target_stages"] = [stage for stage in before["stable_non_target_stages"]
                    if a[stage] and b[stage] and a[stage]["F"] in (0, 1) and a[stage]["F"] == b[stage]["F"]]
            if before != edge["relation"]:
                relation_changes.append({"edge_id": edge["edge_id"], "before": before, "after": edge["relation"]})
        write_jsonl(output / "data/causal" / f"verified_{split}.jsonl", edges)
        quartets = read_jsonl(source / "data/causal" / f"verified_s2_quartets_{split}.jsonl")
        for quartet in quartets:
            states = [records[quartet["members"][corner]]["gold"]["stage_states"]["S2"]["F"]
                      for corner in ("00", "01", "10", "11")]
            quartet["Delta_gold"] = states[3] - states[2] - states[1] + states[0] \
                if all(value in (0, 1) for value in states) else None
        write_jsonl(output / "data/causal" / f"verified_s2_quartets_{split}.jsonl", quartets)
    assert seen == correction_ids, correction_ids - seen
    for name in ("native_text_train.jsonl", "native_image_train.jsonl"):
        target = output / "data/auxiliary" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        os.symlink((source / "data/auxiliary" / name).resolve(), target)
    shutil.copyfile(args.corrections, output / "corrections.jsonl")
    write_jsonl(output / "media_paths.jsonl", media.values())
    write_json(output / "counts.json", counts)
    write_json(output / "data_version.json", {"version": "v3_train_dev_20261008", "parent": str(source),
        "correction_records": len(corrections), "corrected_members": len(correction_ids),
        "relation_changes": relation_changes, "hidden_sample_files_read": False,
        "auxiliary_and_overlap_status": "same native sources and parent contamination coverage; outstanding parent gaps remain",
        "human_final_status": "not_certified", "media": "original server paths retained; no R2 changes"})
    print(str(output), flush=True)


if __name__ == "__main__":
    main()
