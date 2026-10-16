"""One evidence workspace; categorical decisions never re-enter the model context."""

from .audit_v6 import PROTOCOL_VERSION
from .audit_constraints import DECODER_VERSION
from .inference_v4 import ContextualAuditPredictor
from .models.diagsafe_v3 import load_model_v3, load_processor_v3
from .training.v6_sequence import GroundedAuditEncoder


def tag_prediction(prediction):
    prediction.update(variant="grounded_v6", protocol_version=PROTOCOL_VERSION, label_context="blind",
                      evidence_decoder=DECODER_VERSION)
    prediction["model_context_record"] = prediction["audit_record"]
    prediction["audit_record"] = "".join("\n" + entry["name"] + ":" + entry["raw"]
                                        for entry in prediction["audit_slots"])
    prediction["decision_protocol"] = "original input and generated evidence; categorical outputs never fed back"
    prediction["probability_semantics"] = (
        "conditional on this generated evidence, not marginalized or calibrated; "
        "the risk/action reference measure follows the run's declared class_balance setting")
    return prediction


class GroundedAuditPredictor(ContextualAuditPredictor):
    def __init__(self, model_path, adapter_path=None, device="cuda", max_evidence_tokens=512, use_cache=True):
        self.model = load_model_v3(model_path, adapter_path).to(device).eval()
        self.processor = load_processor_v3(adapter_path or model_path)
        self.encoder = GroundedAuditEncoder(self.processor, max_evidence_tokens, use_cache,
                                           optimize=True, label_context="blind")

    @classmethod
    def from_model(cls, model, processor, max_evidence_tokens=512, use_cache=True):
        predictor = cls.__new__(cls)
        predictor.model, predictor.processor = model, processor
        predictor.encoder = GroundedAuditEncoder(processor, max_evidence_tokens, use_cache,
                                                 optimize=True, label_context="blind")
        return predictor

    def predict(self, model_input, sample_id="input", task_type=None):
        return tag_prediction(super().predict(model_input, sample_id, task_type))

    def predict_batch(self, model_inputs, sample_ids=None, task_types=None):
        return [tag_prediction(p) for p in super().predict_batch(model_inputs, sample_ids, task_types)]
