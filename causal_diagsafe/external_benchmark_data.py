"""Public test assets to visible-only inputs. No inferred preference or stage gold."""

import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

from .external_guard_formats import parse_verdict
from .io_utils import read_jsonl
from .public_benchmarks_v1 import PUBLIC_TASKS, csv_rows, document, load_public_documents


def source_metadata(directory):
    path = Path(directory) / "download_source.json"
    return json.loads(path.read_text()) if path.exists() else {"download_revision": "not_yet_recorded"}


def store_image(directory, identifier, value):
    if value is None:
        return None, None
    pixels = value.get("bytes") if isinstance(value, dict) else value
    suffix = ".jpg" if pixels.startswith(b"\xff\xd8") else ".png" if pixels.startswith(b"\x89PNG") else ".img"
    path = Path(directory) / (identifier + suffix)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(pixels)
    # Repeated candidate responses to the same picture must stay in one bootstrap cluster.
    return path, hashlib.sha256(pixels).hexdigest()


def decorate(doc):
    visible = json.loads(doc["visible_json"])
    doc["audit_target"] = "response" if visible["response_candidate"] is not None else "request"
    doc["modality"] = "image_text" if visible["images"] else "text"
    missing = [item["uri"] for item in visible["images"] if not Path(item["uri"]).is_file()]
    doc["input_error"] = "Missing media: " + ", ".join(missing) if missing else None
    return doc


def released_spa_label(text, response):
    value = parse_verdict("guardreasoner_vl", text, response)["harmful"]
    if value is not None:
        return value
    field = "Response" if response else "Request"
    normalized = re.sub(r"(?im)^\s*Answers:\s*", "", text.replace("**", ""))
    declarations = re.findall(
        rf"(?im)^\s*(?:-\s*)?{field}\s*:\s*"
        r"((?:harmful|unharmful)(?:\s*/\s*(?:harmful|unharmful))*)(?=\s*(?:$|\.|\())",
        normalized,
    )
    labels = {part.strip().lower() for declaration in declarations for part in declaration.split("/")}
    return next(iter(labels)) == "harmful" if len(labels) == 1 else None


def load_external_documents(task, root, deck, references):
    root, references = Path(root), Path(references)
    locked, new = root / "raw/external_lock", root / "raw/external_baselines"
    if task in PUBLIC_TASKS:
        yield from (decorate(doc) for doc in load_public_documents(task, locked, deck))
        return
    if task == "siuo":
        directory = new / "siuo"
        for row in json.loads((directory / "siuo_gen.json").read_text()):
            # SIUO supplies safe-response criteria, not binary request-harm annotations.
            yield decorate(document(task, row["question_id"], row["question"], deck,
                images=[directory / "images" / row["image"]], harmful=None, category=row["category"],
                root=row["image"], provenance={**source_metadata(directory),
                    "native_label_basis": "No binary request-harm gold is provided by SIUO.",
                    "system_judge_safety_criterion": row["safety_warning"]}))
    elif task.startswith("spa_vl_eval_"):
        directory = new / "guardreasoner_test"
        for index, row in enumerate(json.loads((directory / "test/SPA_VL_Eval.json").read_text())):
            text = row["messages"][0]["content"]
            body = text.split("Human user:\n", 1)[1]
            query, response = body.split("\n\nAI assistant:\n", 1)
            query = query.replace("<image>", "", 1).strip()
            response = response.strip()
            is_response = task.endswith("response")
            harmful = released_spa_label(row["messages"][1]["content"], is_response)
            images = []
            for path in row["images"]:
                image = directory / "test" / path.removeprefix("./")
                # The release stores numeric filenames with four-digit zero padding.
                if not image.is_file() and image.stem.isdecimal():
                    image = image.with_stem(f"{int(image.stem):04d}")
                images.append(image)
            root_id = hashlib.sha256(images[0].read_bytes()).hexdigest() if images[0].is_file() else str(images[0])
            yield decorate(document(task, index, query, deck, images=images,
                response=response if is_response else None, harmful=harmful, root=root_id,
                provenance={**source_metadata(directory), "record_index": index,
                    "released_image_paths": row["images"],
                    "resolved_image_paths": [str(image.relative_to(directory)) for image in images],
                    "native_label_status": "parsed" if harmful is not None else "ambiguous_or_unrecognized",
                    "native_label_basis": "Released GuardReasoner-VLTest request/response label; not chosen/rejected conversion."}))
    elif task == "beavertails_v_response":
        directory = new / "beavertails_v"
        for file in sorted((directory / "data").glob("*/evaluation.parquet")):
            offset = 0
            for batch in pq.ParquetFile(file).iter_batches(batch_size=16):
                for row in batch.to_pylist():
                    identifier = f"{file.parent.name}-{offset}"
                    uri, image_root = store_image(root / "eval_media/external/beavertails_v", identifier, row["image"])
                    label = row["is_response_safe"].strip().lower()
                    if label not in ("yes", "no"):
                        raise ValueError((identifier, label))
                    yield decorate(document(task, identifier, row["question"], deck, images=[uri],
                        response=row["response"], harmful=label == "no", category=row["category"], root=image_root,
                        subset="evaluation", provenance={**source_metadata(directory), "file": str(file),
                            "row": offset, "label_field": "is_response_safe", "label": label}))
                    offset += 1
    elif task == "mm_safetybench":
        directory = new / "mm_safetybench"
        files = sorted((directory / "data").glob("*/*.parquet"))
        if not files:
            raise FileNotFoundError(directory / "data/*/*.parquet")
        for file in files:
            category, mode = file.parent.name, file.stem
            offset = 0
            for batch in pq.ParquetFile(file).iter_batches(batch_size=16):
                for row in batch.to_pylist():
                    identifier = f"{category}:{mode}:{row['id']}"
                    uri, _ = store_image(root / "eval_media/external/mm_safetybench", identifier.replace(":", "-"), row.get("image"))
                    # Text_only is a distinct official reference, never an invented ablation query.
                    images = [] if mode == "Text_only" else [uri]
                    if mode != "Text_only" and uri is None:
                        raise ValueError(f"Missing image bytes in {identifier}")
                    yield decorate(document(task, identifier, row["question"], deck, images=images,
                        harmful=True, category=category, subset=mode, root=f"{category}:{row['id']}",
                        provenance={**source_metadata(directory), "row": offset, "variant": mode,
                            "image_source": "PKU-Alignment redistribution", "pixel_equivalence_to_original": "not_established",
                            "native_label_basis": "Benchmark safety-risk request membership, not four-action gold."}))
                    offset += 1
    elif task == "jailbreakv":
        directory = new / "jailbreakv"
        for row in csv_rows(directory / "JailBreakV_28K/JailBreakV_28K.csv"):
            # The released jailbreak_query already incorporates the target/persuaded request.
            uri = directory / "JailBreakV_28K" / row["image_path"]
            root_id = hashlib.sha256(row["redteam_query"].encode()).hexdigest()
            yield decorate(document(task, row["id"], row["jailbreak_query"], deck, images=[uri], harmful=True,
                category=row["policy"], subset=row["format"], root=root_id,
                provenance={**source_metadata(directory), "native_label_basis": "Released attack membership",
                    "redteam_behavior": row["redteam_query"], "released_image_path": row["image_path"],
                    "full_image_release": "Authors require full-data access; HF contains only some images."}))
    elif task == "sorrybench":
        directory = new / "sorrybench"
        for file in sorted(directory.glob("question*.jsonl")):
            for row in read_jsonl(file):
                if len(row["turns"]) != 1:
                    raise ValueError("SORRY single-turn adapter cannot drop a turn")
                yield decorate(document(task, f"{file.stem}:{row['question_id']}", row["turns"][0], deck,
                    harmful=True, category=row["category"], subset=row.get("prompt_style", file.stem),
                    root=row["question_id"], provenance={**source_metadata(directory), "file": file.name}))
    elif task == "harmbench_test":
        file = references / "HarmBench/data/behavior_datasets/harmbench_behaviors_text_test.csv"
        for row in csv_rows(file):
            query = (row["ContextString"] + "\n\n" if row["ContextString"] else "") + row["Behavior"]
            yield decorate(document(task, row["BehaviorID"], query, deck, harmful=True,
                category=row["SemanticCategory"], subset=row["FunctionalCategory"], root=row["BehaviorID"],
                provenance={"file": str(file), "tags": row["Tags"], "behavior": row["Behavior"],
                    "context": row["ContextString"], "protocol": "Official held-out text test; not all/train/val."}))
    else:
        raise ValueError(task)


def inventory(docs):
    return {"members": len(docs), "context_roots": len({row["root_id"] for row in docs}),
            "native_label_members": sum(row["native_harm"] is not None for row in docs),
            "native_labels": dict(Counter(str(row["native_harm"]) for row in docs)),
            "missing_media_members": sum(row["input_error"] is not None for row in docs),
            "subsets": dict(Counter(row["subset"] for row in docs)),
            "audit_targets": dict(Counter(row["audit_target"] for row in docs))}
