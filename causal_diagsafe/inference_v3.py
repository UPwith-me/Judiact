"""Visible-only semantic inference using the original LM vocabulary."""

import json
import time

import torch

from .data_contract_v1 import BEHAVIORS_V1, DECISIONS_V1, INTENTS_V1
from .inference_v1 import move_tensors
from .models.diagsafe_v3 import load_model_v3, load_processor_v3
from .training.v3_views import ViewEncoder, labels_for_view


def parse_evidence(answer, visible, stage, field):
    """Reject unmatched locations as errors, never snap them to a gold/source substring."""
    result = {"stage": stage, "field": field, "raw": answer, "spans": [], "regions": [], "errors": []}
    try:
        value = json.loads(answer)
    except json.JSONDecodeError:
        result["errors"].append("invalid_json")
        return result
    key = "regions" if field == "image" else "spans"
    if not isinstance(value, dict) or not isinstance(value.get(key), list):
        result["errors"].append("invalid_structure")
        return result
    if set(value) != {key}:
        result["errors"].append("extra_unrequested_field")
    cards = {card["policy_id"]: card["rule_text"] for card in visible["policy_deck"]}
    for span in value["spans"] if field != "image" else []:
        if not isinstance(span, dict) or not all(key in span for key in ("field", "policy_id", "start", "end", "text")):
            result["errors"].append("invalid_span_structure")
            continue
        if span["field"] != field:
            result["errors"].append("wrong_requested_field")
            continue
        source = cards.get(span["policy_id"]) if span["field"] == "policy_rule" else visible.get(span["field"]) \
            if span["field"] in ("query", "ocr_text", "response_candidate") and span["policy_id"] is None else None
        start, end = span["start"], span["end"]
        if not isinstance(source, str) or not isinstance(start, int) or not isinstance(end, int) \
                or not 0 <= start < end <= len(source) or source[start:end] != span["text"]:
            result["errors"].append("span_not_in_visible_source")
            continue
        result["spans"].append({"stage": stage, **span})
    for region in value["regions"] if field == "image" else []:
        if field != "image":
            result["errors"].append("wrong_requested_field")
            continue
        if not isinstance(region, dict) or not isinstance(region.get("bbox"), list) or len(region["bbox"]) != 4:
            result["errors"].append("invalid_region_structure")
            continue
        index, box = region.get("media_index"), region["bbox"]
        valid = isinstance(index, int) and 0 <= index < len(visible["images"]) \
            and all(isinstance(value, (int, float)) for value in box)
        if not valid or not (0 <= box[0] < box[2] <= 1000 and 0 <= box[1] < box[3] <= 1000):
            result["errors"].append("invalid_original_image_box")
            continue
        x1, y1, x2, y2 = box
        result["regions"].append({"stage": stage, "media_index": index,
            "box": [x1 / 1000, y1 / 1000, (x2 - x1) / 1000, (y2 - y1) / 1000]})
    return result


class SemanticPredictor:
    def __init__(self, model_path, adapter_path=None, device="cuda", view_batch_size=4):
        self.device = device
        self.model = load_model_v3(model_path, adapter_path).to(device).eval()
        self.processor = load_processor_v3(adapter_path or model_path)
        self.encoder = ViewEncoder(self.processor)
        self.view_batch_size = view_batch_size

    @classmethod
    def from_model(cls, model, processor, view_batch_size=4):
        predictor = cls.__new__(cls)
        predictor.model, predictor.processor = model, processor
        predictor.device = next(model.parameters()).device
        predictor.encoder = ViewEncoder(processor)
        predictor.view_batch_size = view_batch_size
        return predictor

    @torch.inference_mode()
    def predict_batch(self, model_inputs, sample_ids=None):
        """The identical action-only interface, batched across independent visible inputs."""
        started = time.perf_counter()
        inputs, choices = self.encoder.texts([({"model_input": visible}, "action") for visible in model_inputs])
        token_counts = inputs["attention_mask"].sum(-1).tolist()
        inputs = move_tensors(inputs, self.device)
        gpu = str(self.device).startswith("cuda")
        if gpu:
            before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            before.record()
        logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        values = [logits[index, ids].softmax(-1).tolist() for index, ids in enumerate(choices)]
        if gpu:
            after.record()
            after.synchronize()
        elapsed = time.perf_counter() - started
        gpu_seconds = before.elapsed_time(after) / 1000 if gpu else 0.
        results = []
        for index, (visible, probs) in enumerate(zip(model_inputs, values)):
            actions = dict(zip(DECISIONS_V1, probs))
            response = visible["response_candidate"] is not None
            results.append({"sample_id": sample_ids[index] if sample_ids else "input",
                "task": "response_audit" if response else "request_decision",
                "response_publishability" if response else "request_action": max(actions, key=actions.get),
                "action_probabilities": actions, "uncalibrated_action_probabilities": actions,
                "variant": "semantic_v3", "input_tokens": int(token_counts[index]), "evidence": [],
                "decision_protocol": "original Qwen Decision:; legal-set conditional probabilities; raw argmax",
                "timing": {"gpu_forward_seconds": gpu_seconds / len(values),
                    "action_only_seconds": elapsed / len(values), "action_only_gpu_seconds": gpu_seconds / len(values),
                    "input_and_forward_seconds": elapsed / len(values), "wall_seconds_with_serialization": elapsed / len(values),
                    "batch_wall_seconds": elapsed, "batch_size": len(values), "path": "action_only",
                    "semantics": "amortized batch service time, not isolated-request latency"}})
        return results

    @torch.inference_mode()
    def predict(self, model_input, sample_id="input", include_evidence=True, include_diagnostics=True):
        started = time.perf_counter()
        response = model_input["response_candidate"] is not None
        record = {"model_input": model_input}
        names = ["action"]
        stages = ("S5",) if response else ("S1", "S2", "S3", "S4")
        if include_diagnostics:
            names += [variable + stage[1:] for stage in stages
                      for variable in (("E", "F") if response else ("R", "E", "F"))]
            names += ["behavior"] if response else ["intent", "policy_id", "policy_action", "behavior"]
        probabilities, gpu_seconds, tokens = {}, 0., 0
        gpu = str(self.device).startswith("cuda")
        batches = [["action"]] + [names[start:start + self.view_batch_size]
                                   for start in range(1, len(names), self.view_batch_size)]
        for selected in batches:
            inputs, choices = self.encoder.texts([(record, name) for name in selected])
            tokens += int(inputs["attention_mask"].sum())
            inputs = move_tensors(inputs, self.device)
            if gpu:
                before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                before.record()
            logits = self.model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
            for index, name in enumerate(selected):
                probabilities[name] = dict(zip(map(str, labels_for_view(model_input, name)),
                                               logits[index, choices[index]].softmax(-1).tolist()))
            if gpu:
                after.record()
                after.synchronize()
                gpu_seconds += before.elapsed_time(after) / 1000
            if selected == ["action"]:
                action_seconds = time.perf_counter() - started
                action_gpu_seconds = gpu_seconds
        action = probabilities["action"]
        result = {"sample_id": sample_id, "task": "response_audit" if response else "request_decision",
            "response_publishability" if response else "request_action": max(action, key=action.get),
            "action_probabilities": action, "uncalibrated_action_probabilities": action,
            "variant": "semantic_v3", "input_tokens": tokens,
            "decision_protocol": "original Qwen Decision:; legal-set conditional probabilities; raw argmax"}
        if include_diagnostics:
            states, state_probs = {stage: None for stage in ("S1", "S2", "S3", "S4", "S5")}, {}
            for stage in stages:
                states[stage], state_probs[stage] = {}, {}
                for variable in (("E", "F") if response else ("R", "E", "F")):
                    values = probabilities[variable + stage[1:]]
                    choice = max(values, key=values.get)
                    states[stage][variable] = int(choice) if choice in ("0", "1") else choice
                    state_probs[stage][variable] = values
            if response:
                states["S5"]["R"] = 1
                state_probs["S5"]["R"] = {"0": 0., "1": 1., "uncertain": 0.}
            else:
                states["S5"] = {"R": 0, "E": "absent", "F": "not_applicable"}
            result.update({"stage_states": states, "stage_probabilities": state_probs,
                           "stage_F_contrasts": {stage: state_probs[stage]["F"]["1"] for stage in stages},
                           "behavior_probabilities": probabilities["behavior"],
                           "safe_response_behavior": max(probabilities["behavior"], key=probabilities["behavior"].get)})
            if not response:
                result.update({"intent_probabilities": probabilities["intent"],
                    "intent_class": max(probabilities["intent"], key=probabilities["intent"].get),
                    "policy_id_probabilities": probabilities["policy_id"],
                    "policy_id": max(probabilities["policy_id"], key=probabilities["policy_id"].get),
                    "policy_action_probabilities": probabilities["policy_action"],
                    "policy_action": max(probabilities["policy_action"], key=probabilities["policy_action"].get)})
        evidence = []
        if include_evidence and include_diagnostics:
            fields = ["query"] + (["image"] if model_input["images"] and not response else []) \
                + (["ocr_text"] if model_input["ocr_text"] is not None else []) \
                + (["policy_rule"] if model_input["policy_deck"] else []) \
                + (["response_candidate"] if response else [])
            views = [(model_input, stage, field) for stage in stages for field in fields]
            grouped = {stage: {"stage": stage, "spans": [], "regions": [], "errors": [], "field_results": []}
                       for stage in stages}
            for start in range(0, len(views), self.view_batch_size):
                selected = views[start:start + self.view_batch_size]
                inputs = move_tensors(self.encoder.evidence_prompts(selected), self.device)
                length = inputs["input_ids"].shape[1]
                if gpu:
                    before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    before.record()
                output = self.model.generate(**inputs, do_sample=False, max_new_tokens=768,
                                             use_cache=True, logits_to_keep=1)
                if gpu:
                    after.record()
                    after.synchronize()
                    gpu_seconds += before.elapsed_time(after) / 1000
                for index, (_, stage, field) in enumerate(selected):
                    emitted = output[index, length:]
                    answer = self.processor.tokenizer.decode(emitted, skip_special_tokens=True)
                    item = parse_evidence(answer, model_input, stage, field)
                    item["generated_tokens"] = int((emitted != self.processor.tokenizer.pad_token_id).sum())
                    item["hit_token_limit"] = item["generated_tokens"] == 768
                    for key in ("spans", "regions", "errors"):
                        grouped[stage][key].extend(item[key])
                    grouped[stage]["field_results"].append(item)
            evidence = list(grouped.values())
        result["evidence"] = evidence
        result["timing"] = {"gpu_forward_seconds": gpu_seconds,
            "action_only_seconds": action_seconds, "action_only_gpu_seconds": action_gpu_seconds,
            "input_and_forward_seconds": time.perf_counter() - started,
            "wall_seconds_with_serialization": time.perf_counter() - started,
            "path": "complete_audit" if include_diagnostics else "action_only"}
        return result

    def decide(self, model_input):
        if model_input["response_candidate"] is None and not model_input["policy_deck"]:
            return {"request_action": "review", "policy_id": "insufficient_policy",
                    "policy_action": "insufficient_policy", "reason": "empty-deck data contract"}
        return self.predict(model_input, include_evidence=False, include_diagnostics=False)
