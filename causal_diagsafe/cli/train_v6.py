"""Completed-epoch V6 training from the public backbone; no chained A/B/C runs."""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import TrainingArguments, set_seed

from ..audit_v6 import PROTOCOL_VERSION
from ..audit_constraints import DECODER_VERSION
from ..io_utils import read_jsonl, write_json
from ..models.diagsafe_v3 import load_model_v3, load_processor_v3
from ..training.v3_data import packet_collator
from ..training.v5_data import CoherentAuditData
from ..training.v6_formal import FormalAuditTrainer, RecoverySnapshotCallback
from ..training.v6_trainer import record_normalizer


def main():
    p = argparse.ArgumentParser()
    for name in ("model", "release", "output"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--effective-roots", type=int, default=64)
    p.add_argument("--record-batch-size", type=int, default=4)
    p.add_argument("--native-batch-size", type=int, default=16)
    p.add_argument("--rollout-batch-size", type=int, default=4)
    p.add_argument("--native-checkpoint-layer-stride", type=int, choices=(1, 2), default=1)
    p.add_argument("--fact-dispatch", choices=("local", "balanced"), default="balanced")
    p.add_argument("--recovery-save-steps", type=int, default=100)
    p.add_argument("--max-batch-tokens", type=int, default=16384)
    p.add_argument("--max-evidence-tokens", type=int, default=512)
    p.add_argument("--matching-coefficient", type=float, default=1.)
    p.add_argument("--class-balance", choices=("none", "risk_action", "determinate_risk_action"),
                   default="determinate_risk_action")
    p.add_argument("--resume-from-checkpoint")
    args = p.parse_args()
    set_seed(args.seed)
    torch.set_num_threads(4)
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    data = CoherentAuditData(Path(args.release))
    dataset, sampling = data.schedule(world, args.seed, args.effective_roots)
    processor = load_processor_v3(args.model)
    normalization = record_normalizer(data, processor.tokenizer)
    model = load_model_v3(args.model, training=True)
    output = Path(args.output)
    trainer = FormalAuditTrainer(model=model, processing_class=processor, train_dataset=dataset,
        data_collator=packet_collator, audit_data=data, update_scale=sampling["update_scale"],
        evidence_normalization=normalization["evidence_field_mean_tokens"],
        class_balance=args.class_balance, record_batch_size=args.record_batch_size,
        native_checkpoint_layer_stride=args.native_checkpoint_layer_stride,
        fact_dispatch=args.fact_dispatch,
        resume_facts=bool(args.resume_from_checkpoint),
        native_batch_size=args.native_batch_size, rollout_batch_size=args.rollout_batch_size,
        max_batch_tokens=args.max_batch_tokens, max_evidence_tokens=args.max_evidence_tokens,
        matching_coefficient=args.matching_coefficient,
        callbacks=[RecoverySnapshotCallback(args.recovery_save_steps)],
        eval_dataset=read_jsonl(data.release / "data/natural_request/dev.jsonl"),
        args=TrainingArguments(output_dir=str(output), num_train_epochs=args.epochs,
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            learning_rate=2e-5, weight_decay=.01, adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8,
            max_grad_norm=1., lr_scheduler_type="cosine", warmup_ratio=.03, bf16=True, tf32=True,
            logging_steps=1, eval_strategy="epoch", save_strategy="epoch", save_total_limit=None,
            load_best_model_at_end=True, metric_for_best_model="eval_loss", greater_is_better=False,
            remove_unused_columns=False, ddp_find_unused_parameters=False, dataloader_num_workers=0,
            ddp_timeout=7200,
            accelerator_config={"split_batches": False, "dispatch_batches": False, "use_seedable_sampler": False},
            report_to=[], disable_tqdm=True, seed=args.seed, data_seed=args.seed))
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "sampling.json", sampling)
        write_json(output / "normalization.json", normalization)
        write_json(output / "risk_action_balance.json", trainer.balance)
        write_json(output / "run_settings.json", {"arguments": vars(args), "method": PROTOCOL_VERSION,
            "evidence_decoder": DECODER_VERSION,
            "inventory": data.inventory, "initialization": "public Qwen base; zero-increment language LoRA",
            "optimizer": "one continuing HF AdamW and full-run cosine schedule; no epoch optimizer reset",
            "precision": "BF16 base/activations; FP32 losses; no quantization",
            "activation_recomputation": {"record_layer_stride": 1,
                "native_layer_stride": args.native_checkpoint_layer_stride},
            "training_fact_readouts": "skip unused categorical readouts; keep their exact fixed placeholder context",
            "fact_dispatch": "balance same-update inference over equal-parameter replicas; gradient packets unchanged"
                if args.fact_dispatch == "balanced" else "owner-local evidence generation",
            "recovery_snapshots": "periodic model/optimizer/RNG recovery only; select on completed-epoch dev only",
            "context": "exact generated evidence tokens; fixed categorical placeholders",
            "evidence": "compatible gold evidence only, field-fixed length denominator; no gold-prefix class losses",
            "classification": "per-task/risk/action observed-class balance when enabled; other fields unchanged",
            "probability_semantics": "conditional reference-measure scores, not natural-prior or calibrated probabilities",
            "fact_refresh": "first use each epoch at the current parameter state, detached; reuse only within that epoch",
            "selection": "completed-epoch natural dev; equal task and action-class NLL when class balanced",
            "dev_batching": "fixed shape-bucket batches of 8, with recorded tail/actual memory reductions",
            "hidden_tests_loaded": False, "human_final_status": "not_certified",
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "torch": torch.__version__})
        print(json.dumps({"inventory": data.inventory, "sampling": sampling}), flush=True)
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(str(output / "selected"))
    trainer.save_state()
    write_json(output / f"work_counts.rank{rank}.json", dict(trainer.work_counts))
    if trainer.is_world_process_zero():
        write_json(output / "training_result.json", {"metrics": result.metrics,
            "actual_steps": trainer.state.global_step, "actual_epoch": trainer.state.epoch,
            "best_checkpoint": trainer.state.best_model_checkpoint,
            "best_selection_NLL": trainer.state.best_metric})


if __name__ == "__main__":
    main()
