"""The coherent model uses the same single free-generated audit at deployment."""

from .audit_v5 import PROTOCOL_VERSION
from .inference_v4 import ContextualAuditPredictor
from .models.diagsafe_v3 import load_model_v3, load_processor_v3
from .training.v5_sequence import CoherentAuditEncoder


def tag_prediction(prediction):
    prediction.update(variant="coherent_v5", protocol_version=PROTOCOL_VERSION)
    return prediction


class CoherentAuditPredictor(ContextualAuditPredictor):
    def __init__(self, model_path, adapter_path=None, device="cuda", max_evidence_tokens=512, use_cache=True):
        self.model = load_model_v3(model_path, adapter_path).to(device).eval()
        self.processor = load_processor_v3(adapter_path or model_path)
        self.encoder = CoherentAuditEncoder(self.processor, max_evidence_tokens, use_cache, optimize=True)

    @classmethod
    def from_model(cls, model, processor, max_evidence_tokens=512, use_cache=True):
        predictor = cls.__new__(cls)
        predictor.model, predictor.processor = model, processor
        predictor.encoder = CoherentAuditEncoder(processor, max_evidence_tokens, use_cache, optimize=True)
        return predictor

    def predict(self, model_input, sample_id="input", task_type=None):
        return tag_prediction(super().predict(model_input, sample_id, task_type))

    def predict_batch(self, model_inputs, sample_ids=None, task_types=None):
        return [tag_prediction(p) for p in super().predict_batch(model_inputs, sample_ids, task_types)]
