"""Identical fixed-placeholder audit in autonomous train-selection evaluation."""

from .shared_audit import SharedAuditNode, shared_judgment_node
from .source_predictor import SourcePredictor
from .training.shared_sequence import SharedAuditEncoder


class SharedAuditPredictor(SourcePredictor):
    judgment_node = staticmethod(shared_judgment_node)

    def __init__(self, model, processor, max_new_tokens=4096, encoder=None, include_basis=True,
                 action_objective="natural", **kwargs):
        self.include_basis = include_basis
        self.action_objective = action_objective
        encoder = encoder or SharedAuditEncoder(processor, max_new_tokens, cache_size=512,
            witness_frequency_penalty=0., compact_witness_json=True)
        encoder.vision.maximum = 128
        super().__init__(model, processor, max_new_tokens, encoder=encoder, **kwargs)

    def predict_batch(self, *args, **kwargs):
        kwargs.setdefault("include_basis", self.include_basis)
        return super().predict_batch(*args, **kwargs)

    def forward_context_key(self, node):
        return self.encoder.context_key(node) if isinstance(node, SharedAuditNode) else None
