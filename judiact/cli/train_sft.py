"""Stage 1: supervised JudiAct training with autonomous short reviews."""

import argparse
import json
import math
import os
from pathlib import Path

from ..short_review import PROTOCOL_VERSION
from ..checkpoint import read_checkpoint
from ..io_utils import read_jsonl, write_json


def select_checkpoint(output, epochs):
    candidates = []
    for checkpoint in output.glob("checkpoint-*"):
        state_path = checkpoint / "trainer_state.json"
        if not state_path.is_file():
            continue
        state = json.loads(state_path.read_text(encoding="utf-8"))
        step, epoch = state["global_step"], state.get("epoch")
        if epoch is None or not 1 <= epoch <= epochs or int(epoch) != epoch:
            continue
        metric_path = output / "dev_selection" / f"epoch{int(epoch)}" / "selection_metrics.json"
        if not metric_path.is_file():
            continue
        score = json.loads(metric_path.read_text(encoding="utf-8"))["eval_macro_F1"]
        logs = [r["eval_macro_F1"] for r in state["log_history"]
                if r.get("epoch") == epoch and r.get("step") == step and "eval_macro_F1" in r]
        if checkpoint.name != f"checkpoint-{step}" or not math.isfinite(score) or score not in logs:
            raise ValueError(f"Dev score does not belong to {checkpoint}")
        candidates.append({"epoch": int(epoch), "step": step, "checkpoint": str(checkpoint.resolve()),
                           "dev_macro_F1": score})
    if sorted(r["epoch"] for r in candidates) != list(range(1, epochs + 1)):
        raise ValueError("Selection needs one complete, scored checkpoint for each SFT epoch")
    best = min(candidates, key=lambda r: (-r["dev_macro_F1"], r["epoch"]))
    return {"selected_checkpoint": best["checkpoint"], "selected_epoch": best["epoch"],
            "criterion": "Mean four-action Macro-F1 of request and answer dev views; earliest epoch breaks a tie",
            "candidates": sorted(candidates, key=lambda r: r["epoch"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Local Qwen3-VL-4B-Instruct snapshot")
    parser.add_argument("--release", type=Path, required=True, help="Prepared train/dev bundle")
    parser.add_argument("--bank", type=Path, help="Train-only reference bank (defaults to the bundle's bank)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path, help="Full Trainer checkpoint; keep the same output directory")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--effective-roots", type=int, default=64)
    parser.add_argument("--max-batch-tokens", type=int, default=28672)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--attention-backend", choices=("flash_attention_2", "sdpa"), default="flash_attention_2")
    options = parser.parse_args()

    import torch
    from transformers import TrainerCallback, TrainingArguments, set_seed

    from ..models.backbone import load_model, load_processor
    from ..training.packets import packet_collator
    from ..training.exposure import ExposureEpochs
    from ..training.short_data import ShortReviewData
    from ..training.short_trainer import ShortReviewTrainer

    class SaveUpdates(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if options.save_every and state.global_step % options.save_every == 0:
                control.should_save = True
            return control

    if options.resume:
        read_checkpoint(options.resume)
        if options.resume.resolve().parent != options.output.resolve():
            parser.error("Resume in the original output directory to retain this epoch's short-review history.")
    set_seed(options.seed)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    world, rank = int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("RANK", 0))
    bank = options.bank or options.release / "process_references/bank.rank0.jsonl"
    data = ShortReviewData(options.release, bank, action_objective="target_budget",
                           local_boundary_conditions=True, student_source_reads=False,
                           action_budget=dict.fromkeys(("allow", "warn", "review", "refuse"), .25),
                           stratified_exposure=True, reference_seed=options.seed)
    dataset, sampling = data.schedule(world, options.seed, options.effective_roots, locality=True)
    processor = load_processor(options.model)
    model = load_model(options.model, training=True, attn_implementation=options.attention_backend)
    assert {p.dtype for p in model.parameters() if p.requires_grad} == {torch.float32}
    trainer = ShortReviewTrainer(
        model=model, processing_class=processor, train_dataset=dataset, data_collator=packet_collator,
        audit_data=data, update_scale=sampling["update_scale"], phase="sft",
        rollout_batch_size=64, rollout_max_batch_tokens=131072, frozen_image_cache_size=128,
        class_batch_size=64, text_batch_size=8, native_batch_size=64,
        max_batch_tokens=options.max_batch_tokens, max_target_positions=8192, probability_chunk_size=16,
        max_new_tokens=4096, short_max_new_tokens=512, witness_frequency_penalty=0.,
        checkpoint_layers=24, text_checkpoint_layers=30, packed_jobs=True, balanced_jobs=True,
        time_balanced_jobs=True,
        callbacks=[ExposureEpochs(data, dataset, world, options.seed, options.effective_roots), SaveUpdates()],
        eval_dataset=read_jsonl(data.release / "data/natural_request/dev.jsonl"),
        args=TrainingArguments(
            output_dir=str(options.output), num_train_epochs=options.epochs, per_device_train_batch_size=1,
            gradient_accumulation_steps=1, learning_rate=options.learning_rate, weight_decay=.01,
            adam_beta1=.9, adam_beta2=.999, adam_epsilon=1e-8, optim="adamw_torch_fused", max_grad_norm=1.,
            lr_scheduler_type="cosine", warmup_ratio=.03, bf16=True, tf32=False,
            logging_steps=1, logging_nan_inf_filter=False, eval_strategy="epoch", save_strategy="epoch",
            load_best_model_at_end=False, metric_for_best_model="eval_macro_F1", greater_is_better=True,
            remove_unused_columns=False, ddp_find_unused_parameters=False, dataloader_num_workers=0,
            ddp_timeout=7200, accelerator_config={"split_batches": False, "dispatch_batches": False,
                                                 "use_seedable_sampler": False},
            report_to=[], disable_tqdm=True, seed=options.seed, data_seed=options.seed))
    if rank == 0:
        write_json(options.output / "run_settings.json", {
            "protocol": PROTOCOL_VERSION, "arguments": {k: str(v) if isinstance(v, Path) else v
                                                          for k, v in vars(options).items()},
            "inventory": data.inventory, "sampling": sampling})
    result = trainer.train(resume_from_checkpoint=str(options.resume) if options.resume else None)
    trainer.accelerator.wait_for_everyone()
    if trainer.is_world_process_zero():
        selection = select_checkpoint(options.output, options.epochs)
        write_json(options.output / "selection.json", selection)
        write_json(options.output / "training_result.json", {
            "updates": trainer.state.global_step, "metrics": result.metrics, **selection})
        print(json.dumps(selection), flush=True)


if __name__ == "__main__":
    main()
