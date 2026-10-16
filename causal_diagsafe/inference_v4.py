"""One visible-only generated audit followed by its conditional processing action."""

from __future__ import annotations

from .audit_v4 import PROTOCOL_VERSION, audit_slots, task_for_input, visible_only
from .data_contract_v1 import DECISIONS_V1, STAGES_V1
from .eval_batching import EvalBatcher, kv_bytes_per_token
from .models.diagsafe_v3 import load_model_v3, load_processor_v3
from .training.v4_sequence import AuditEncoder


def trajectory_prediction(trajectory, sample_id, tokenizer):
    values, visible = trajectory.values, trajectory.visible
    slots = audit_slots(visible, trajectory.task_type)
    probabilities, raw_probabilities = {}, {}
    for slot in slots:
        if slot.kind != "classify":
            continue
        raw = dict(zip(map(str, slot.labels), trajectory.rollout_log_probabilities[slot.name].exp().tolist()))
        raw_probabilities[slot.name] = raw
        probabilities[slot.name] = {key: float(key == str(slot.forced_value)) for key in raw} \
            if slot.forced_value is not None else raw
    result = {"sample_id": sample_id,
        "task": trajectory.task_type if trajectory.task_type.startswith("native_") else task_for_input(visible),
        "input_task_type": trajectory.task_type, "variant": "contextual_v4",
        "protocol_version": PROTOCOL_VERSION, **trajectory.counts, "timing": trajectory.timing,
        "decoding_settings": {"do_sample": False, "num_beams": 1, "use_model_defaults": False},
        "audit_record": tokenizer.decode(trajectory.input_ids[0, trajectory.prompt_length:], skip_special_tokens=False),
        "audit_slots": trajectory.entries, "evidence": trajectory.evidence,
        "model_slot_probabilities": raw_probabilities,
        "decision_protocol": "one greedy generated audit; legal-set conditional action probabilities; "
                             "empty request deck follows the fixed contract",
        "probability_semantics": "conditional on this generated prefix, not marginalized over all audits"}
    result["output_tokens"] = trajectory.counts["generated_tokens"]
    if trajectory.task_type.startswith("native_"):
        name = trajectory.task_type
        result.update({"native_harm_label": values[name], "native_harm_probabilities": probabilities[name]})
        return result
    response = trajectory.task_type == "response_audit"
    result.update({"response_publishability" if response else "request_action": values["action"],
        "action_probabilities": probabilities["action"],
        "uncalibrated_action_probabilities": probabilities["action"],
        "model_action_probabilities": raw_probabilities["action"],
        "safe_response_behavior": values["behavior"], "behavior_probabilities": probabilities["behavior"]})
    states, state_probs = {stage: None for stage in STAGES_V1}, {}
    for stage in (("S5",) if response else ("S1", "S2", "S3", "S4")):
        variables = ("E", "F") if response else ("R", "E", "F")
        states[stage] = {variable: values[variable + stage[1:]] for variable in variables}
        state_probs[stage] = {variable: probabilities[variable + stage[1:]] for variable in variables}
        if response:
            states[stage]["R"] = 1
            state_probs[stage]["R"] = {"0": 0., "1": 1., "uncertain": 0.}
    if not response:
        states["S5"] = {"R": 0, "E": "absent", "F": "not_applicable"}
        result.update({"intent_class": values["intent"], "intent_probabilities": probabilities["intent"],
            "policy_id": values["policy_id"], "policy_id_probabilities": probabilities["policy_id"],
            "policy_action": values["policy_action"], "policy_action_probabilities": probabilities["policy_action"]})
        result["legal_policy_pair"] = (
            values["policy_id"] in ("no_match", "insufficient_policy") and
            values["policy_action"] == values["policy_id"]) or (
            values["policy_id"] not in ("no_match", "insufficient_policy") and
            values["policy_action"] in DECISIONS_V1)
    result.update({"stage_states": states, "stage_probabilities": state_probs,
        "stage_F_contrasts": {stage: state_probs[stage]["F"]["1"] for stage in state_probs}})
    return result


class ContextualAuditPredictor:
    def __init__(self, model_path, adapter_path=None, device="cuda", max_evidence_tokens=512, use_cache=True):
        self.model = load_model_v3(model_path, adapter_path).to(device).eval()
        self.processor = load_processor_v3(adapter_path or model_path)
        self.encoder = AuditEncoder(self.processor, max_evidence_tokens, use_cache, optimize=True)

    @classmethod
    def from_model(cls, model, processor, max_evidence_tokens=512, use_cache=True):
        predictor = cls.__new__(cls)
        predictor.model, predictor.processor = model, processor
        predictor.encoder = AuditEncoder(processor, max_evidence_tokens, use_cache, optimize=True)
        return predictor

    def predict(self, model_input, sample_id="input", task_type=None):
        visible = visible_only(model_input)
        trajectory = self.encoder.rollout(self.model, visible, task_type or task_for_input(visible))
        return trajectory_prediction(trajectory, sample_id, self.processor.tokenizer)

    def predict_batch(self, model_inputs, sample_ids=None, task_types=None):
        identifiers = sample_ids or ["input" for _ in model_inputs]
        assert len(identifiers) == len(model_inputs)
        trajectories = self.encoder.rollout_batch(self.model, model_inputs, task_types)
        return [trajectory_prediction(trajectory, sid, self.processor.tokenizer)
                for trajectory, sid in zip(trajectories, identifiers)]

    def iter_predictions(self, rows, initial_batch_size=4, max_batch_size=64):
        """Measured shape buckets keep every diagnostic slot and original input."""
        from .audit_v4 import task_for_input
        device = next(self.model.parameters()).device
        self.eval_batcher = EvalBatcher(devices=[device], initial_batch_size=initial_batch_size,
            max_batch_size=max_batch_size, generation_tokens=4 * self.encoder.max_evidence_tokens,
            kv_bytes_per_token=kv_bytes_per_token(self.model),
            image_max_pixels=self.processor.image_processor.size["longest_edge"])
        entries = [{"visible": visible_only(row["model_input"]), "sid": row["sample_id"],
                    "task_type": row.get("task_type") or task_for_input(row["model_input"]),
                    "batch_key": row.get("task_type") or task_for_input(row["model_input"])} for row in rows]
        for _, prediction in self.eval_batcher.run(entries, lambda batch: self.predict_batch(
                [entry["visible"] for entry in batch], [entry["sid"] for entry in batch],
                [entry["task_type"] for entry in batch])):
            yield prediction

    def decide(self, model_input):
        return self.predict(model_input)
