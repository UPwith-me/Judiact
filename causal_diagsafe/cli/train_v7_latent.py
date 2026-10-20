"""Full epochs of weak SFT followed by latent marginal learning, with fixed dev selection."""

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import TrainingArguments, set_seed

from ..audit_v7 import PROTOCOL_VERSION
from ..io_utils import read_jsonl, write_json
from ..models.diagsafe_v3 import load_model_v3, load_processor_v3
from ..training.v3_data import packet_collator
from ..training.v7_latent_data import LatentAuditData
from ..training.v7_latent_trainer import LatentAuditTrainer
from ..training.v7_objective import EpochAnchor


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "release", "bank", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--phase", choices=("sft", "latent"), required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--effective-roots", type=int, default=64)
    parser.add_argument("--student-draws", type=int, default=2)
    parser.add_argument("--rollout-batch-size", type=int, default=16)
    parser.add_argument("--class-batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=2)
    parser.add_argument("--native-batch-size", type=int, default=32)
    parser.add_argument("--max-batch-tokens", type=int, default=24576)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--checkpoint-stride", type=int, choices=(0, 1, 2), default=1)
    parser.add_argument("--trial", action="store_true")
    args = parser.parse_args()
    if args.phase == "latent" and not args.adapter:
        raise ValueError("latent phase requires the SFT adapter")
    set_seed(args.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    data = LatentAuditData(args.release, args.bank)
    if args.trial:
        data.restrict_trial()
    dataset, sampling = data.schedule(world, args.seed, args.effective_roots, locality=True)
    processor = load_processor_v3(args.model)
    model = load_model_v3(args.model, args.adapter, training=True)
    if args.phase == "latent":
        # A single explicitly defined policy density, not different random dropout
        # networks for proposal, posterior scoring and the M-step.
        for name, module in model.named_modules():
            if "lora_dropout" in name and isinstance(module, torch.nn.Dropout):
                module.p = 0.
        model.peft_config["default"].lora_dropout = 0.
    trainable = {name: str(value.dtype) for name, value in model.named_parameters() if value.requires_grad}
    assert set(trainable.values()) == {"torch.float32"}, "LoRA precision changed"
    anchor = EpochAnchor(model) if args.phase == "latent" else None
    output = Path(args.output)
    trainer = LatentAuditTrainer(
        model=model, processing_class=processor, train_dataset=dataset, data_collator=packet_collator,
        audit_data=data, update_scale=sampling["update_scale"], phase=args.phase, anchor=anchor,
        student_draws=args.student_draws, rollout_batch_size=args.rollout_batch_size,
        class_batch_size=args.class_batch_size, text_batch_size=args.text_batch_size,
        native_batch_size=args.native_batch_size, max_batch_tokens=args.max_batch_tokens,
        max_new_tokens=args.max_new_tokens, checkpoint_stride=args.checkpoint_stride,
        eval_dataset=read_jsonl(data.release / "data/natural_request/dev.jsonl"),
        args=TrainingArguments(
            output_dir=str(output), num_train_epochs=args.epochs, per_device_train_batch_size=1,
            gradient_accumulation_steps=1, learning_rate=2e-5 if args.phase == "sft" else 1e-5,
            weight_decay=.01, adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8,
            optim="adamw_torch_fused", max_grad_norm=1., lr_scheduler_type="cosine",
            warmup_ratio=0. if args.trial else .03, bf16=True, tf32=False,
            logging_steps=1, eval_strategy="no" if args.trial else "epoch", save_strategy="epoch",
            load_best_model_at_end=False, metric_for_best_model="eval_loss", greater_is_better=False,
            remove_unused_columns=False, ddp_find_unused_parameters=False, dataloader_num_workers=0,
            ddp_timeout=7200, accelerator_config={
                "split_batches": False, "dispatch_batches": False, "use_seedable_sampler": False},
            report_to=[], disable_tqdm=True, seed=args.seed, data_seed=args.seed))
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        write_json(output / "run_settings.json", {
            "arguments": vars(args), "protocol": PROTOCOL_VERSION, "inventory": data.inventory,
            "sampling": sampling, "action_root_balance": data.action_balance,
            "precision": "BF16 base/activations; FP32 LoRA, probabilities and loss; TF32 disabled",
            "optimizer": "continuous fused AdamW within each phase; reset at SFT->latent",
            "dropout": .05 if args.phase == "sft" else 0.,
            "trainable": trainable, "human_final_status": "not_certified",
            "hidden_tests_loaded": False,
            "likelihood": "actual grammar-normalized sequence probability including EOS; no token average",
            "bootstrap": "weak per-node normalized SFT; existing locations; teacher facts remain noisy",
            "latent": "SNIS Fisher-identity gradient with a prior-score control variate",
            "loss_note": "latent training loss is a gradient surrogate, not evidence of improved marginal NLL"})
        print(json.dumps({"inventory": data.inventory, "sampling": sampling}), flush=True)
    result = trainer.train()
    trainer.save_model(str(output / "final"))
    trainer.save_state()
    write_json(output / f"work_counts.rank{rank}.json", dict(trainer.work_counts))
    if trainer.is_world_process_zero():
        write_json(output / "training_result.json", {
            "metrics": result.metrics, "actual_steps": trainer.state.global_step,
            "actual_epoch": trainer.state.epoch, "continuation_adapter": str(output / "final")})


if __name__ == "__main__":
    main()
