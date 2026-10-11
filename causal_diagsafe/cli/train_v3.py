"""Fresh, paired three-epoch semantic training with the mature HF/PEFT stack."""

import argparse
import json
import os
import shutil
from pathlib import Path

import torch
from accelerate import PartialState
from transformers import TrainingArguments, set_seed

from ..io_utils import read_jsonl, write_json, write_jsonl
from ..models.diagsafe_v3 import load_model_v3, load_processor_v3
from ..training.v3_data import V3TrainingData, packet_collator
from ..training.v3_trainer import SemanticTrainer, evaluate_dev
from ..training.v3_views import classification_views, evidence_views


def main():
    parser = argparse.ArgumentParser()
    for name in ("model", "release", "reviews", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--variant", choices=("v3", "no_relation"), default="v3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--effective-roots", type=int, default=64)
    parser.add_argument("--view-batch-size", type=int, default=4)
    parser.add_argument("--base-dev", action="store_true")
    parser.add_argument("--resume")
    parser.add_argument("--max-steps", type=int, default=-1, help="bounded implementation diagnostic, not formal checkpoint selection")
    args = parser.parse_args()
    set_seed(args.seed)
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    output = Path(args.output)
    data = V3TrainingData(args.release, args.reviews)
    dataset, sampling = data.schedule(world, args.seed, args.effective_roots)
    processor = load_processor_v3(args.model)
    model = load_model_v3(args.model, training=True)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        for name in ("confirmed_core_image_aliases.jsonl", "confirmed_image_root_families.jsonl"):
            shutil.copyfile(Path(args.reviews) / name, output / name)
        write_json(output / "sampling.json", sampling)
        write_json(output / "run_settings.json", {
            "arguments": vars(args), "method_version": "semantic_v3_20261008", "schema_version": "diagsafe_sample_v1",
            "inventory": data.inventory, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "lm_head": "original frozen weight; gradient retained through it", "initialization": "public base; PEFT default A/B",
            "optimizer": "AdamW", "lr": 2e-5, "warmup_ratio": .03, "scheduler": "one cosine over full three epochs",
            "contrast_coefficient": 1. if args.variant == "v3" else 0.,
            "task_weights": {"request": 1/3, "response": 1/3, "auxiliary": 1/3},
            "absolute_relation_member_weight": 0., "processor_size": processor.image_processor.size,
            "checkpoint_selection": "completed epochs only; equal natural request/response action NLL",
            "image_coordinates": "gold normalized original-image xywh; Qwen normalized 0-1000 xyxy, rounded",
            "evidence_mask": "positive marks at E=present only; absent/uncertain legacy marks retained but not localization targets",
            "human_final_status": "not_certified", "torch": torch.__version__,
            "backbone_source": json.loads((Path(args.model) / "download_source.json").read_text()),
        })
        write_jsonl(output / "view_manifest.jsonl", ({
            "kind": kind, "root_id": root, "sample_id": record["sample_id"], "member_weight": 1/len(records),
            "classification_views": classification_views(record),
            "evidence_stage_fields": [[stage, field] for stage, field, _ in evidence_views(record)],
        } for kind, roots in data.populations.items() for root, records in roots.items() for record in records))
        print(json.dumps({"inventory": data.inventory, "sampling": sampling}), flush=True)
    # Trainer initializes the actual mixed-precision accelerator; no additional training framework.
    trainer = SemanticTrainer(
        model=model, processing_class=processor, train_dataset=dataset, data_collator=packet_collator,
        eval_dataset=read_jsonl(Path(args.release) / "data/natural_request/dev.jsonl")
            + read_jsonl(Path(args.release) / "data/natural_response/dev.jsonl"),
        release=args.release, reviews=str(output), relation_coefficient=1. if args.variant == "v3" else 0.,
        view_batch_size=args.view_batch_size,
        args=TrainingArguments(output_dir=str(output), num_train_epochs=args.epochs, max_steps=args.max_steps,
            per_device_train_batch_size=1, gradient_accumulation_steps=1,
            learning_rate=2e-5, weight_decay=.01, adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8,
            max_grad_norm=1., lr_scheduler_type="cosine", warmup_ratio=.03, bf16=True, tf32=True,
            logging_steps=1, eval_strategy="no" if args.max_steps > 0 else "epoch",
            save_strategy="no" if args.max_steps > 0 else "epoch", save_total_limit=None,
            load_best_model_at_end=args.max_steps < 0, metric_for_best_model="eval_loss", greater_is_better=False,
            remove_unused_columns=False, ddp_find_unused_parameters=False, dataloader_num_workers=0,
            accelerator_config={"split_batches": False, "dispatch_batches": False, "use_seedable_sampler": False},
            report_to=[], disable_tqdm=True, seed=args.seed, data_seed=args.seed))
    if args.base_dev and not args.resume:
        trainer.accelerator.wait_for_everyone()
        evaluate_dev(model.to(trainer.args.device), processor, args.release, str(output),
                     output.parent / "qwen_base_dev", trainer.accelerator, diagnostics=False,
                     view_batch_size=args.view_batch_size)
    result = trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output / ("bounded_diagnostic" if args.max_steps > 0 else "selected")))
    trainer.save_state()
    budget = torch.tensor([trainer.encoder.input_tokens, trainer.encoder.forward_calls, trainer.encoder.forward_views],
                          dtype=torch.int64, device=trainer.args.device)
    if world > 1:
        torch.distributed.all_reduce(budget)
    if trainer.is_world_process_zero():
        write_json(output / "training_result.json", {"metrics": result.metrics,
            "actual_steps": trainer.state.global_step, "actual_epoch": trainer.state.epoch,
            "best_checkpoint": trainer.state.best_model_checkpoint, "best_natural_dev_NLL": trainer.state.best_metric,
            "actual_training_input_tokens": int(budget[0]), "actual_training_forward_calls": int(budget[1]),
            "actual_training_forward_views": int(budget[2]), "bounded_diagnostic": args.max_steps > 0})


if __name__ == "__main__":
    main()
