"""Generate complete v4 audits; the public base is usable without a trained adapter."""

import argparse
import json
from pathlib import Path

from ..inference_v4 import ContextualAuditPredictor
from ..io_utils import read_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-evidence-tokens", type=int, default=512)
    parser.add_argument("--protocol", choices=("v4", "v6"), default="v4")
    args = parser.parse_args()
    predictor_class = ContextualAuditPredictor
    if args.protocol == "v6":
        from ..inference_v6 import GroundedAuditPredictor
        predictor_class = GroundedAuditPredictor
    predictor = predictor_class(args.model, args.adapter, args.device, args.max_evidence_tokens)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        for index, row in enumerate(read_jsonl(args.inputs)):
            visible = row.get("model_input", row)
            result = predictor.predict(visible, row.get("sample_id", str(index)), row.get("task_type"))
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()


if __name__ == "__main__":
    main()
