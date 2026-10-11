"""lmms-eval tasks and transport for native guard verdicts, with per-item resume."""

import json
import time
import traceback
from pathlib import Path

from datasets import Dataset, DatasetDict
from lmms_eval.api.model import lmms
from lmms_eval.api.task import ConfigurableTask
from tqdm import tqdm

from .external_eval_io import append_prediction, read_predictions
from .external_eval_metrics import summarize_native


def native_result(doc, results):
    prediction = json.loads(results[0])
    return {"native_safety": {**prediction, **{key: doc[key] for key in (
        "native_harm", "root_id", "pair_id", "category", "subset", "modality", "audit_target")}}}


class NativeSafetyTask(ConfigurableTask):
    def __init__(self, name, docs):
        self.docs = docs
        super().__init__(config={"task": name, "test_split": "test", "num_fewshot": 0,
            "doc_to_text": "visible_json", "doc_to_target": lambda doc: str(doc["native_harm"]),
            "doc_to_visual": lambda doc: [], "output_type": "generate_until",
            "generation_kwargs": {"do_sample": False, "max_new_tokens": 1},
            "process_results": native_result,
            # Model harmfulness confidence is not a per-item correctness score.
            "score_key": "_no_scalar_transport_metric",
            "metric_list": [{"metric": "native_safety", "aggregation": summarize_native,
                             "higher_is_better": False}],
            "metadata": {"version": "external_native_20261009_v1",
                         "note": "Generation interface transports serialized verdicts, not generated assistant answers."}})

    def download(self, dataset_kwargs=None):
        self.dataset = DatasetDict({"test": Dataset.from_list(self.docs)})
        self.dataset_no_image = self.dataset


class NativeGuardLM(lmms):
    def __init__(self, predictor, prediction_file):
        super().__init__()
        self.predictor = predictor
        self._rank, self._world_size = 0, 1
        self.prediction_file = Path(prediction_file)
        self.completed = read_predictions(self.prediction_file)
        self.current = {}

    def loglikelihood(self, requests):
        raise NotImplementedError("These tasks use each guard's native audit interface.")

    def generate_until_multi_round(self, requests):
        raise NotImplementedError("No conversation turns may be dropped by a single-turn adapter.")

    def generate_until(self, requests):
        import torch
        from PIL import UnidentifiedImageError

        if hasattr(self.predictor, "predict_batch"):
            return self._generate_batched(requests)

        outputs = []
        with self.prediction_file.open("a", encoding="utf-8") as handle, \
                self.prediction_file.with_suffix(".errors.jsonl").open("a", encoding="utf-8") as errors:
            for request in tqdm(requests, desc="Native safety audit"):
                ctx, _, _, doc_id, task, split = request.args
                doc = self.task_dict[task][split][doc_id]
                identifier = doc["sample_id"]
                visible = json.loads(ctx)
                if visible["query"] is None:
                    value = {"sample_id": identifier, "visible_json": ctx, "harmful": None, "score": None,
                             "status": "missing_query", "reason": "The source record supplies no request text."}
                    if identifier not in self.completed:
                        append_prediction(handle, value)
                        self.completed[identifier] = value
                elif identifier in self.completed:
                    value = self.completed[identifier]
                    if value["visible_json"] != ctx:
                        raise ValueError(f"Changed visible input under a cached sample ID: {identifier}")
                else:
                    started = time.perf_counter()
                    try:
                        missing = [image["uri"] for image in visible["images"] if not Path(image["uri"]).is_file()]
                        if doc["audit_target"] == "response" and visible["response_candidate"] is None:
                            value = {"harmful": None, "score": None, "status": "missing_candidate"}
                        elif missing:
                            value = {"harmful": None, "score": None, "status": "missing_media", "reason": str(missing)}
                        else:
                            if self.predictor is None:
                                raise ValueError(f"No saved prediction for {identifier}; cache-only evaluation does not run inference.")
                            value = self.predictor.predict(visible, task=task)
                        value.update(sample_id=identifier, visible_json=ctx)
                        if value["status"] not in ("missing_media", "missing_candidate", "context_exceeded"):
                            append_prediction(handle, value)
                            self.completed[identifier] = value
                    except (FileNotFoundError, UnidentifiedImageError, torch.cuda.OutOfMemoryError) as exc:
                        # These observed per-input failures do not turn into safe verdicts or abort other models.
                        if isinstance(exc, torch.cuda.OutOfMemoryError):
                            torch.cuda.empty_cache()
                        value = {"sample_id": identifier, "visible_json": ctx, "harmful": None, "score": None,
                                 "status": "out_of_memory" if isinstance(exc, torch.cuda.OutOfMemoryError) else "image_error",
                                 "reason": f"{type(exc).__name__}: {exc}"[:1500],
                                 "timing": {"wall_seconds": time.perf_counter() - started}}
                        append_prediction(errors, {**value, "traceback": traceback.format_exc()[-5000:]})
                self.current[identifier] = value
                outputs.append(json.dumps(value, ensure_ascii=False, allow_nan=False))
        return outputs

    def _generate_batched(self, requests):
        import torch
        from .eval_batching import EvalBatcher, kv_bytes_per_token
        from .io_utils import write_json

        outputs, entries, missing_queries = [None] * len(requests), [], []
        for index, request in enumerate(requests):
            ctx, _, _, doc_id, task, split = request.args
            doc = self.task_dict[task][split][doc_id]
            identifier = doc["sample_id"]
            visible = json.loads(ctx)
            value = self.completed.get(identifier)
            if value is not None and value["visible_json"] != ctx:
                raise ValueError(f"Changed visible input under a cached sample ID: {identifier}")
            if visible["query"] is None:
                value = {"status": "missing_query", "harmful": None, "score": None,
                         "reason": "The source record supplies no request text."}
                if identifier not in self.completed:
                    value.update(sample_id=identifier, visible_json=ctx)
                    missing_queries.append(value)
                    self.completed[identifier] = value
            if value is None:
                missing = [image["uri"] for image in visible["images"] if not Path(image["uri"]).is_file()]
                if doc["audit_target"] == "response" and visible["response_candidate"] is None:
                    value = {"status": "missing_candidate", "harmful": None, "score": None}
                elif missing:
                    value = {"status": "missing_media", "harmful": None, "score": None, "reason": str(missing)}
            if value is not None:
                value.update(sample_id=identifier, visible_json=ctx)
                self.current[identifier] = value
                outputs[index] = json.dumps(value, ensure_ascii=False, allow_nan=False)
            else:
                entries.append({"visible": visible, "visible_json": ctx, "identifier": identifier,
                    "task": task, "index": index, "batch_key": task})
        if not hasattr(self.predictor, "eval_batcher"):
            family = getattr(self.predictor, "family", "answerer")
            generative = family not in ("qwen", "shieldstral", "safeatlas", "diagsafe_v3")
            generator = self.predictor.model.thinker if family == "guardreasoner_omni" else self.predictor.model
            processor = getattr(self.predictor, "processor", None)
            size = getattr(getattr(processor, "image_processor", None), "size", {})
            pixel_limit = size.get("longest_edge") if isinstance(size, dict) else None
            self.predictor.eval_batcher = EvalBatcher(devices=[self.predictor.device],
                max_batch_size=64 if generative else 512,
                generation_tokens=getattr(self.predictor, "max_tokens", 0) if generative else 0,
                kv_bytes_per_token=kv_bytes_per_token(generator) if generative else 0,
                image_max_pixels=pixel_limit, include_policy=getattr(self.predictor, "protocol", "native") == "shared_deck")
        batcher = self.predictor.eval_batcher
        def failure(entry, exc):
            return {"status": "out_of_memory" if isinstance(exc, torch.cuda.OutOfMemoryError) else "image_error",
                    "harmful": None, "score": None, "reason": f"{type(exc).__name__}: {exc}"[:1500]}
        with self.prediction_file.open("a", encoding="utf-8") as handle, \
                self.prediction_file.with_suffix(".errors.jsonl").open("a", encoding="utf-8") as errors, \
                tqdm(total=len(entries), desc="Batched native safety audit") as progress:
            for value in missing_queries:
                append_prediction(handle, value)
            for index, value in batcher.run(entries, lambda batch: self.predictor.predict_batch(
                    [entry["visible"] for entry in batch], [entry["task"] for entry in batch]), on_error=failure):
                entry = entries[index]
                value.update(sample_id=entry["identifier"], visible_json=entry["visible_json"])
                if value["status"] in ("out_of_memory", "image_error"):
                    append_prediction(errors, value)
                elif value["status"] not in ("context_exceeded", "missing_media", "missing_candidate"):
                    append_prediction(handle, value)
                    self.completed[entry["identifier"]] = value
                self.current[entry["identifier"]] = value
                outputs[entry["index"]] = json.dumps(value, ensure_ascii=False, allow_nan=False)
                progress.update()
        write_json(self.prediction_file.with_suffix(".batching.json"), batcher.summary())
        return outputs
