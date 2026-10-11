"""Direct diagnostic distributions and evidence coordinates; no generation or gold input."""

from __future__ import annotations

import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers import AutoProcessor

from .data_contract_v1 import BEHAVIORS_V1, DECISIONS_V1, INTENTS_V1
from .models.diagsafe_v1 import DiagSafeV1, FIELDS
from .training.v1_data import EvidenceCollator


def move_tensors(value, device):
    if torch.is_tensor(value):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_tensors(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_tensors(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_tensors(item, device) for item in value)
    return value


def visible_packet(model_input, sample_id):
    record = {"model_input": model_input, "sample_id": sample_id,
              "task_type": "response_audit" if model_input["response_candidate"] is not None else "request_decision"}
    return {"units": [{"id": sample_id, "kind": "response" if record["task_type"] == "response_audit" else "request",
                        "records": [record], "edges": [], "quartet": None}]}


def probabilities(logits, names, temperature=1.):
    values = (logits.float() / temperature).softmax(-1).tolist()
    return {str(name): value for name, value in zip(names, values)}


def evidence_coordinates(weights, metadata, visible):
    located = []
    offsets = metadata["offsets"]
    for k in range(5):
        fields = {}
        for m, field in enumerate(FIELDS):
            masses = weights[k, m].float().cpu()
            item = {"null_probability": masses[-1].item()}
            if field == "image":
                images = []
                for index, (grid, positions) in enumerate(zip(metadata["image_grids"], metadata["image_positions"])):
                    _, h, w = grid
                    rows, cols = h // 2, w // 2
                    values = masses[positions]
                    peak = int(values.argmax())
                    images.append({"media_index": index, "grid_rows": rows, "grid_cols": cols,
                                   "patch_probabilities": values.tolist(),
                                   "peak_patch_box": [(peak % cols) / cols, (peak // cols) / rows, 1 / cols, 1 / rows]})
                item["images"] = images
            else:
                spans = []
                ranges = metadata["field_ranges"].get(field, [])
                for rule_index, (start, end) in enumerate(ranges):
                    indices = [i for i, (a, b) in enumerate(offsets) if a < end and b > start and b > a]
                    source = visible["policy_deck"][rule_index]["rule_text"] if field == "policy_rule" else visible[field]
                    policy_id = visible["policy_deck"][rule_index]["policy_id"] if field == "policy_rule" else None
                    for i in indices:
                        a, b = offsets[i]
                        a, b = max(a, start) - start, min(b, end) - start
                        spans.append({"start": a, "end": b, "text": source[a:b],
                                      "policy_id": policy_id, "probability": masses[i].item()})
                item["tokens"] = spans
            fields[field] = item
        located.append({"stage": f"S{k + 1}", "fields": fields})
    return located


class DiagnosticPredictor:
    def __init__(self, model_path, adapter_path, device="cuda"):
        self.device = device
        self.model = DiagSafeV1(model_path, adapter_path).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(adapter_path, local_files_only=True)
        self.collator = EvidenceCollator(self.processor, include_alignment_metadata=True)
        temperature_path = Path(adapter_path) / "temperatures.json"
        self.temperatures = json.loads(temperature_path.read_text()) if temperature_path.exists() else {}

    @torch.inference_mode()
    def raw(self, model_input, sample_id="input"):
        started = time.perf_counter()
        batch = self.collator([visible_packet(model_input, sample_id)])
        metadata = batch.pop("metadata")[0]
        batch.pop("supervision")
        batch = move_tensors(batch, self.device)
        gpu = str(self.device).startswith("cuda")
        if gpu:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=gpu):
            output = self.model(**batch)
        if gpu:
            end.record()
            end.synchronize()
        self.last_timing = {"gpu_forward_seconds": start.elapsed_time(end) / 1000 if gpu else None,
                            "input_and_forward_seconds": time.perf_counter() - started}
        return output, metadata

    def predict(self, model_input, sample_id="input", include_evidence=True):
        output, metadata = self.raw(model_input, sample_id)
        response = model_input["response_candidate"] is not None
        key = "response_publishability" if response else "request_action"
        temperature = self.temperatures.get(key, {}).get("temperature", 1.)
        actions = probabilities(output[key][0], DECISIONS_V1, temperature)
        raw_actions = probabilities(output[key][0], DECISIONS_V1)
        native = {task: probabilities(logits[0], ("unharmful", "harmful"))
                  for task, logits in output["native"].items()}
        if self.model.variant == "final_only":
            return {"sample_id": sample_id, "task": "response_audit" if response else "request_decision",
                    key: max(actions, key=actions.get), "action_probabilities": actions,
                    "input_tokens": metadata["tokens"], "variant": "final_only",
                    "uncalibrated_action_probabilities": raw_actions, "timing": self.last_timing,
                    "native_harm_probabilities": native}
        stages = {}
        for k in range(5):
            if response and k < 4:
                stages[f"S{k + 1}"] = None
                continue
            if not response and k == 4:
                stages["S5"] = {"R": {"0": 1., "1": 0., "uncertain": 0.},
                                "E": {"absent": 1., "present": 0., "uncertain": 0.},
                                "F": {"not_applicable": 1.}}
                continue
            stages[f"S{k + 1}"] = {
                variable: probabilities(output["stage"][variable][0, k],
                                        ("absent", "present", "uncertain") if variable == "E" else (0, 1, "uncertain"))
                for variable in ("R", "E", "F")}
        result = {"sample_id": sample_id, "task": "response_audit" if response else "request_decision",
                  key: max(actions, key=actions.get), "action_probabilities": actions,
                  "stage_probabilities": stages, "input_tokens": metadata["tokens"],
                  "uncalibrated_action_probabilities": raw_actions, "timing": self.last_timing,
                  "native_harm_probabilities": native,
                  "stage_F_contrasts": {f"S{k + 1}": (output["stage"]["F"][0, k, 1]
                      - output["stage"]["F"][0, k, 0]).item()
                      for k in (range(4) if not response else (4,))}}
        if not response:
            candidates = [(card["policy_id"], action) for card in model_input["policy_deck"] for action in DECISIONS_V1]
            candidates += [("no_match", "no_match"), ("insufficient_policy", "insufficient_policy")]
            values = output["policy_joint"][0].softmax(-1).tolist()
            joint = [{"policy_id": policy, "policy_action": action, "probability": value}
                     for (policy, action), value in zip(candidates, values)]
            chosen = max(joint, key=lambda row: row["probability"])
            result.update({"policy_id": chosen["policy_id"], "policy_action": chosen["policy_action"],
                           "policy_joint": joint,
                           "intent_probabilities": probabilities(output["intent"][0], INTENTS_V1),
                           "behavior_probabilities": probabilities(output["safe_response_behavior"][0], BEHAVIORS_V1)})
        if include_evidence:
            result["evidence"] = [item for item in evidence_coordinates(output["evidence_weights"][0], metadata, model_input)
                                  if (item["stage"] == "S5") == response]
        return result


def fit_temperature(logits, labels):
    logits = torch.stack(logits).detach().double().cpu()
    labels = torch.tensor(labels, dtype=torch.long)
    log_temperature = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([log_temperature], max_iter=100, line_search_fn="strong_wolfe")

    def scaled():
        finite = torch.isfinite(logits)
        return (logits.masked_fill(~finite, 0) / log_temperature.exp()).masked_fill(~finite, -torch.inf)

    def closure():
        optimizer.zero_grad()
        loss = F.cross_entropy(scaled(), labels)
        loss.backward()
        return loss

    before = closure().item()
    optimizer.step(closure)
    temperature = log_temperature.detach().exp().item()
    after = F.cross_entropy(scaled(), labels).item()
    return {"temperature": temperature, "NLL_before": before, "NLL_after": after, "denominator": len(labels)}
