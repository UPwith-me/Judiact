"""Full dev inference with the shared Accelerate evaluation entry point."""

from .evaluate_v1 import main
from ..evaluation_v3 import summarize
from ..inference_v3 import SemanticPredictor


if __name__ == "__main__":
    main(SemanticPredictor, summarize)
