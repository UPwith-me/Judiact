"""Actual answer and source-rubric generation with released Transformers models."""

import copy
import json
import time
from pathlib import Path

from .eval_batching import EvalBatcher, completion_tokens, kv_bytes_per_token
from .external_eval_io import append_prediction, read_predictions
from .io_utils import write_jsonl


def move_inputs(inputs, device):
    return {key: value.to(device) if hasattr(value, "to") else value for key, value in inputs.items()}


def open_images(visible):
    from PIL import Image

    images = []
    for item in visible["images"]:
        with Image.open(item["uri"]) as image:
            images.append(image.convert("RGB"))
    return images


def generation_request(processor, visible, task, config, judge):
    """Keep the chat template and generation policy identical across execution engines."""
    mode = visible.get("execution_mode", "base")
    content = [{"type": "image", "image": im["uri"]} for im in visible["images"]]
    content.append({"type": "text", "text": visible["query"]})
    messages = []
    if judge:
        if visible.get("judge_system"):
            messages.append({"role": "system", "content": visible["judge_system"]})
        settings = visible["judge_generation"]
    else:
        if mode == "warn":
            messages.append({"role": "system", "content": config["system_controller"]["warn"]})
        settings = config["generation"][task]
    messages.append({"role": "user", "content": content})
    if mode == "revise":
        if visible.get("response_candidate") is None:
            raise ValueError("Revision requires the actual original answer")
        messages += [{"role": "assistant", "content": visible["response_candidate"]},
                     {"role": "user", "content": config["system_controller"]["revise"]}]
    elif not judge and visible.get("response_candidate") is not None:
        raise ValueError("Ordinary answer generation cannot consume a reference answer")
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), settings


class BenchmarkGenerator:
    def __init__(self, path, config, *, judge=False, device="cuda:0"):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        path = str(path)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, local_files_only=True,
            dtype=torch.bfloat16, attn_implementation="sdpa", device_map=device).eval()
        self.processor = AutoProcessor.from_pretrained(path, local_files_only=True)
        self.processor.tokenizer.padding_side = "left"
        self.device, self.config, self.judge = device, config, judge
        self.family, self.protocol = "benchmark_generator", "native"
        self.max_tokens = 1024
        self.eval_batcher = EvalBatcher(devices=[device], initial_batch_size=2 if judge else 4,
            max_batch_size=32 if judge else 64, generation_tokens=self.max_tokens,
            kv_bytes_per_token=kv_bytes_per_token(self.model), include_policy=False,
            image_max_pixels=self.processor.image_processor.size.get("longest_edge"))

    def predict_batch(self, visibles, tasks=None):
        """Judge visibles contain only the source rubric and allowed judge images."""
        import torch

        if self.judge and any("source_defined_judgment" in v for v in visibles):
            results, pending = [], []
            for index, visible in enumerate(visibles):
                raw = visible.get("source_defined_judgment")
                results.append({"status": "ok", "harmful": None, "score": None, "response_candidate": raw,
                    "source_defined_rule": "SIUO actual-empty-response safety, not a missing generation"} if raw is not None else None)
                if raw is None:
                    pending.append(index)
            if pending:
                computed = self.predict_batch([visibles[i] for i in pending], [tasks[i] for i in pending])
                for index, result in zip(pending, computed):
                    results[index] = result
            return results
        started = time.perf_counter()
        texts, images, profiles = [], [], []
        for visible, task in zip(visibles, tasks):
            text, settings = generation_request(self.processor, visible, task, self.config, self.judge)
            texts.append(text)
            images.extend(open_images(visible))
            profiles.append(settings)
        if any(s != profiles[0] for s in profiles):
            raise ValueError("Different benchmark generation settings in the same batch")
        inputs = self.processor(text=texts, images=images or None, padding=True, truncation=False, return_tensors="pt")
        inputs.pop("token_type_ids", None)
        inputs = move_inputs(dict(inputs), self.device)
        profile = copy.deepcopy(self.model.generation_config)
        profile.update(**profiles[0])
        if not profile.do_sample:
            profile.temperature, profile.top_p, profile.top_k = 1.0, 1.0, 50
        width = inputs["input_ids"].shape[1]
        with torch.inference_mode():
            # Transformers otherwise backfills global-default values (e.g. False)
            # from Qwen's sampling defaults, silently turning greedy into sampling.
            generated = self.model.generate(**inputs, generation_config=profile, use_model_defaults=False,
                **profiles[0], use_cache=True, logits_to_keep=1)
        counts = inputs["attention_mask"].sum(-1).tolist()
        eos = profile.eos_token_id
        eos_set = {eos} if isinstance(eos, int) else set(eos or [])
        elapsed = time.perf_counter() - started
        results = []
        for index, emitted in enumerate(generated[:, width:]):
            tokens = completion_tokens(emitted, eos)
            response = self.processor.tokenizer.decode(tokens, skip_special_tokens=True)
            results.append({"status": "ok" if response.strip() else "empty_response", "harmful": None, "score": None,
                "response_candidate": response, "input_tokens": counts[index], "output_tokens": len(tokens),
                "generation_limit_reached": len(tokens) == profile.max_new_tokens and int(tokens[-1]) not in eos_set,
                "generation_settings": {key: getattr(profile, key) for key in (
                    "do_sample", "temperature", "top_p", "top_k", "num_beams", "max_new_tokens", "repetition_penalty")},
                "timing": {"batch_wall_seconds": elapsed, "batch_size": len(visibles),
                           "wall_seconds": elapsed / len(visibles)}})
        return results


def transport(predictor, task, docs, destination):
    """Run each visible input once and retain failed outcomes in the scoring population."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    completed = read_predictions(destination)
    current, pending = {}, []
    if len({row["sample_id"] for row in docs}) != len(docs):
        raise ValueError(f"Duplicate {task} member IDs")
    with destination.open("a", encoding="utf-8") as stream:
        for row in docs:
            sid, encoded = row["sample_id"], row["visible_json"]
            if sid in completed:
                if completed[sid]["visible_json"] != encoded:
                    raise ValueError(f"Changed visible input for cached member {sid}")
                current[sid] = completed[sid]
                continue
            visible = json.loads(encoded)
            status = None
            if visible["query"] is None:
                status = "missing_query"
            elif row["audit_target"] == "response" and visible.get("response_candidate") is None:
                status = "missing_candidate"
            elif any(not Path(im["uri"]).is_file() for im in visible["images"]):
                status = "missing_media"
            if status:
                current[sid] = {"sample_id": sid, "visible_json": encoded, "status": status,
                                "harmful": None, "score": None}
                continue
            pending.append({"sample_id": sid, "visible_json": encoded, "visible": visible, "batch_key": task})

        def infer(batch):
            return predictor.predict_batch([row["visible"] for row in batch], [task] * len(batch))

        def unresolved(row, error):
            return {"status": "inference_error", "reason": str(error), "harmful": None, "score": None}

        for index, result in predictor.eval_batcher.run(pending, infer, on_error=unresolved):
            row = pending[index]
            sid = row["sample_id"]
            current[sid] = {**result, "sample_id": sid, "visible_json": row["visible_json"]}
            if result["status"] in ("ok", "empty_response"):
                append_prediction(stream, current[sid])
    write_jsonl(destination.with_suffix(".outcomes.jsonl"), (current[row["sample_id"]] for row in docs))
    return current
