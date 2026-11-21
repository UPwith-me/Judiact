"""Expose the selected audit's trained interfaces without conflating their targets."""

import math
import time

import torch

from .data_contract import DECISIONS
from .eval_batching import EvalBatcher, kv_bytes_per_token
from .checkpoint import PROTOCOL_VERSION, predictor_for_checkpoint
from .models.backbone import load_model, load_processor


def trace_outcome(trace, visible, native_harm=False):
    result = {"status": "ok" if trace["valid"] else "invalid_prediction",
              "harmful": None, "score": None, "complete_audit": trace}
    if not trace["valid"]:
        return result
    if native_harm:
        node = trace["nodes"][trace["task_type"]]
        result.update(harmful=node["value"] == "harmful",
            score=math.exp(node["log_probabilities"][1]),
            score_kind="trained_source_native_harmfulness_not_four_action_projection")
    else:
        node = trace["nodes"]["action"]
        probabilities = ({label: float(label == node["forced_by_contract"]) for label in DECISIONS}
            if node["forced_by_contract"] is not None else {label: math.exp(value)
            for label, value in zip(DECISIONS, node["log_probabilities"])})
        action = trace["values"]["action"]
        response = visible["response_candidate"] is not None
        result.update(action=action, action_probabilities=probabilities,
            harmful=action != "allow" if response else action in ("review", "refuse"),
            score=1 - probabilities["allow"] if response else probabilities["review"] + probabilities["refuse"],
            score_kind="policy_automatic_hold_not_source_native_harmfulness")
    return result


class JudiActGuard:
    def __init__(self, model, adapter, *, native_harm=False, device="cuda:0", include_evidence=True,
                 attention_backend="flash_attention_2"):
        from accelerate import Accelerator

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.device, self.native_harm = device, native_harm
        self.include_evidence = include_evidence
        self.processor = load_processor(model)
        loaded = load_model(model, adapter, training=False,
                               attn_implementation=attention_backend).to(device).eval()
        self.accelerator = Accelerator(mixed_precision="bf16")
        self.model = self.accelerator.prepare_model(loaded, device_placement=False, evaluation_mode=True)
        predictor_type = predictor_for_checkpoint(adapter, short_batch_size=64, short_max_batch_tokens=131072)
        self.audit_protocol = PROTOCOL_VERSION
        self.predictor = predictor_type(self.model, self.processor, 4096,
            class_batch_size=64, text_batch_size=32, max_batch_tokens=65536,
            max_decode_tokens=327680, source_cache={})
        self.family = "judiact"
        self.protocol = "native_harm" if native_harm else "shared_deck"
        self.eval_batcher = EvalBatcher(devices=[device], initial_batch_size=32,
            max_batch_size=128 if native_harm else 32,
            generation_tokens=0 if native_harm else getattr(self.predictor.encoder, "short_max_new_tokens", 0),
            kv_bytes_per_token=kv_bytes_per_token(self.model), include_policy=not native_harm,
            image_max_pixels=self.processor.image_processor.size["longest_edge"])

    @torch.inference_mode()
    def predict_batch(self, visibles, tasks=None):
        started = time.perf_counter()
        if self.native_harm:
            task_types = ["native_response_harm" if v["response_candidate"] is not None
                          else "native_prompt_harm" for v in visibles]
        else:
            task_types = ["response_audit" if v["response_candidate"] is not None
                          else "request_decision" for v in visibles]
        traces = self.predictor.predict_batch(visibles, task_types, sample=False,
                                               include_evidence=self.include_evidence)
        elapsed = time.perf_counter() - started
        results = []
        for visible, task, trace in zip(visibles, task_types, traces):
            result = trace_outcome(trace, visible, self.native_harm)
            result.update(audit_protocol=self.audit_protocol,
                postdecision_evidence_omitted=not self.include_evidence,
                timing={"batch_wall_seconds": elapsed, "batch_size": len(visibles),
                    "wall_seconds": elapsed / len(visibles),
                    "semantics": "amortized declared interface service time, not isolated latency"})
            results.append(result)
        return results
