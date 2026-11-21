"""Run autonomous JudiAct action, diagnostic and evidence evaluation."""

import argparse
from datetime import timedelta
import json
from pathlib import Path

from ..checkpoint import PROTOCOL_VERSION, predictor_for_checkpoint
from ..external_eval_io import record_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "adapter", "release", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="test")
    parser.add_argument("--include-mechanism", action="store_true",
                        help="Score published pairs and quartets without changing the natural denominator")
    parser.add_argument("--attention-backend", choices=("flash_attention_2", "sdpa"), default="flash_attention_2")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    import torch
    from accelerate import Accelerator, PartialState
    from transformers import set_seed

    from ..evaluation import evaluate_natural
    from ..models.backbone import load_model, load_processor

    state = PartialState(timeout=timedelta(seconds=7200))
    if state.is_main_process:
        record_settings(args.output, {"command": "evaluate", "protocol": PROTOCOL_VERSION,
            "world_size": state.num_processes, **{k: str(v.resolve()) if isinstance(v, Path) else v
                                                 for k, v in vars(args).items() if k != "resume"}})
    state.wait_for_everyone()
    set_seed(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    predictor = predictor_for_checkpoint(args.adapter, short_batch_size=64, short_max_batch_tokens=131072)
    processor = load_processor(args.model)
    model = load_model(args.model, args.adapter, attn_implementation=args.attention_backend).to(state.device).eval()
    model = Accelerator(mixed_precision="bf16").prepare_model(model, device_placement=False, evaluation_mode=True)
    metrics = evaluate_natural(model, processor, args.release, args.output, state, 4096,
        pool_size=args.batch_size, witness_frequency_penalty=0., predictor_type=predictor,
        protocol=PROTOCOL_VERSION, resume=args.resume, split="test_hidden" if args.split == "test" else "dev",
        include_mechanism=args.include_mechanism)
    if state.is_main_process:
        print(json.dumps(metrics), flush=True)


if __name__ == "__main__":
    main()
