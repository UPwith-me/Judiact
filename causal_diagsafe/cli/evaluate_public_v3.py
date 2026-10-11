"""Shared locked lmms-eval tasks, action-only when no diagnostic gold exists."""

from .evaluate_public_v1 import main
from ..inference_v3 import SemanticPredictor


class PublicSemanticPredictor(SemanticPredictor):
    def predict(self, model_input, sample_id="input", include_evidence=False):
        return super().predict(model_input, sample_id, include_evidence=False, include_diagnostics=False)


if __name__ == "__main__":
    main(PublicSemanticPredictor)
