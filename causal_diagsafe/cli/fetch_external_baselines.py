"""Download public evaluation assets only; never train or edit annotation tasks."""

import argparse
import importlib.metadata
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, snapshot_download
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout

from ..io_utils import write_json


DATA_SOURCES = {
    "siuo": ("sinwang/SIUO", ["*.json", "*.jsonl", "*.csv", "*.md", "images/*"]),
    "guardreasoner_test": ("yueliu1999/GuardReasoner-VLTest", ["test/SPA_VL_Eval.json", "test/SPA_VL_Eval/*", "README.md"]),
    "beavertails_v": ("PKU-Alignment/BeaverTails-V", ["data/*/evaluation.parquet", "README.md"]),
    "mm_safetybench": ("PKU-Alignment/MM-SafetyBench", ["*.parquet", "*.json", "*.jpg", "*.png", "*.md"]),
    "jailbreakv": ("JailbreakV-28K/JailBreakV-28k", ["*.csv", "*.png", "*.jpg", "*.json", "*.md"]),
    "sorrybench": ("sorry-bench/sorry-bench-202503", ["question*.jsonl", "README.md", "LICENSE"]),
}


def download(repo_id, destination, patterns, repo_type, endpoint):
    destination = Path(destination)
    source_file = destination / "download_source.json"
    previous = json.loads(source_file.read_text()) if source_file.exists() else None
    api = HfApi(endpoint=endpoint)
    revision = previous["revision"] if previous else api.repo_info(repo_id, repo_type=repo_type).sha
    for attempt in range(3):
        try:
            snapshot_download(repo_id, repo_type=repo_type, revision=revision, endpoint=endpoint,
                              local_dir=destination, allow_patterns=patterns, max_workers=2)
            break
        except (ChunkedEncodingError, ConnectionError, ReadTimeout):
            if attempt == 2:
                raise
            print(f"Transport interrupted; resuming {repo_id} ({attempt + 2}/3).", flush=True)
            time.sleep(5)
    source = {"repo_id": repo_id, "revision": revision, "repo_type": repo_type,
              "endpoint": endpoint, "allow_patterns": patterns}
    write_json(source_file, source)
    return source


def download_modelscope(repo_id, destination, patterns, revision, canonical_repo_id):
    from modelscope import snapshot_download as modelscope_download

    destination = Path(destination)
    modelscope_download(model_id=repo_id, revision=revision, local_dir=str(destination),
                        allow_file_pattern=patterns, max_workers=2)
    for weight in destination.glob("*.safetensors"):
        with weight.open("rb") as handle:
            if handle.read(80).startswith(b"version https://git-lfs.github.com/spec/v1"):
                raise ValueError(f"ModelScope returned a Git LFS pointer, not model weights: {weight.name}")
    source = {"repo_id": repo_id, "canonical_repo_id": canonical_repo_id,
              "revision": revision, "repo_type": "model", "endpoint": "https://modelscope.cn",
              "allow_patterns": patterns,
              "packages": {name: importlib.metadata.version(name) for name in ("modelscope", "modelscope-hub")}}
    write_json(destination / "download_source.json", source)
    return source


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--endpoint", default="https://hf-mirror.com")
    parser.add_argument("--models", nargs="*")
    parser.add_argument("--data", action="store_true")
    parser.add_argument("--judges", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    config = json.loads(Path(args.config).read_text())
    reports = []
    jobs = []
    model_specs = {model["name"]: model for model in config["models"]}
    if args.data:
        jobs += [(name, repo, root / "raw/external_baselines" / name, patterns, "dataset")
                 for name, (repo, patterns) in DATA_SOURCES.items() if name != "jailbreakv"]
    if args.models is not None:
        names = set(args.models) if args.models else {m["name"] for m in config["models"]}
        for model in config["models"]:
            if model["name"] not in names:
                continue
            if model["name"] == "qwen3vl4b":
                destination = root / "raw/models/Qwen3-VL-4B-Instruct"
            else:
                destination = root / "raw/external_models" / model["name"]
            patterns = ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.py", "*.pt", "*.bin"]
            if model.get("modelscope_repo_id"):
                patterns += ["LICENSE*", "USE_POLICY*", "README*"]
            jobs.append((model["name"], model.get("modelscope_repo_id", model["repo_id"]), destination,
                         patterns, "model"))
    if args.judges:
        for name, repo in (("multimodal_judge32b", config["judge"]["repo_id"]),
                           ("harmbench_judge13b", config["judge"]["harmbench_repo_id"])):
            jobs.append((name, repo, root / "raw/external_models" / name,
                         ["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja"], "model"))
    for name, repo, destination, patterns, kind in jobs:
        print(f"FETCH {name} {repo}", flush=True)
        try:
            if name == "qwen3vl4b" and (destination / "download_source.json").exists():
                # This is the active training backbone: reuse it, never refresh it in place.
                source = json.loads((destination / "download_source.json").read_text())
            elif model_specs.get(name, {}).get("modelscope_repo_id"):
                spec = model_specs[name]
                source = download_modelscope(repo, destination, patterns, spec["modelscope_revision"], spec["repo_id"])
            else:
                source = download(repo, destination, patterns, kind, args.endpoint)
            reports.append({"name": name, "status": "downloaded", "path": str(destination), **source})
        except Exception as exc:
            # Asset permissions/network errors do not prevent independent baselines.
            reports.append({"name": name, "status": "unavailable", "repo_id": repo,
                            "error_type": type(exc).__name__, "error": str(exc)[:1200]})
            print(f"UNAVAILABLE {name}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)
        kind = "data" if args.data else "judges" if args.judges and args.models is None else "models"
        if args.models:
            kind += "_" + "_".join(sorted(args.models))
        write_json(root / "work/external_baselines_20261009" / ("fetch_" + kind + ".json"), reports)


if __name__ == "__main__":
    main()
