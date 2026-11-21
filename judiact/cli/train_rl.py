"""Conditional short-review phase two, one fixed epoch and no dev selection."""

import argparse
from collections import Counter
from datetime import timedelta
import faulthandler
import json
import os
from pathlib import Path
import random
import signal
import time

import numpy as np
import torch
import torch.distributed as dist
from transformers import AutoProcessor

from ..checkpoint import CHECKPOINT_CONFIG, read_checkpoint
from ..io_utils import write_json
from ..training.conditional_sequence import ConditionalReviewEncoder
from ..training.conditional_trainer import ALGORITHM, COMPONENTS, ConditionalReviewPlanner
from ..training.execution import job_tokens
from ..training.multipath_data import MultipathReviewData


def gather(value):
    rows = [None] * dist.get_world_size()
    dist.all_gather_object(rows, value)
    return rows


def make_engine(args, total):
    from verl.trainer.config import CheckpointConfig
    from verl.workers.config import HFModelConfig, FSDPEngineConfig, FSDPOptimizerConfig
    from torch.distributed.fsdp import register_fsdp_forward_method
    from ..training.verl_audit_engine import JudiActFSDPEngine
    model = HFModelConfig(path=str(args.model), tokenizer_path=str(args.checkpoint),
        lora_rank=16, lora_alpha=32, lora_adapter_path=str(args.resume or args.checkpoint),
        use_remove_padding=False, enable_gradient_checkpointing=True)
    engine = JudiActFSDPEngine(model, FSDPEngineConfig(strategy="fsdp2", fsdp_size=1,
        use_remove_padding=False, use_torch_compile=False, reshard_after_forward=False,
        wrap_policy={"transformer_layer_cls_to_wrap": ["Qwen3VLTextDecoderLayer"]}),
        FSDPOptimizerConfig(lr=args.learning_rate, lr_warmup_steps=20, total_training_steps=total,
            lr_scheduler_type="cosine", min_lr_ratio=0., betas=(.9, .999), weight_decay=.01,
            clip_grad=1., override_optimizer_config={"fused": True, "eps": 1e-8}), CheckpointConfig())
    engine.initialize()
    register_fsdp_forward_method(engine.module, "generate")
    for config in engine.module.peft_config.values():
        config.lora_dropout = 0.
    return engine


def save_checkpoint(args, engine, processor, initial, planner, step, total):
    temporary = args.output / f"checkpoint-{step}.incomplete"
    destination = args.output / f"checkpoint-{step}"
    state = {}
    for name, parameter in engine.module.named_parameters():
        if parameter.requires_grad:
            value = parameter.full_tensor() if hasattr(parameter, "full_tensor") else parameter
            state[name] = value.detach().cpu().float()
    if dist.get_rank() == 0:
        temporary.mkdir(parents=True, exist_ok=True)
        engine.module.save_pretrained(temporary, state_dict=state, save_embedding_layers=False)
        processor.save_pretrained(temporary)
        config = dict(initial, phase="conditional_short_review_rl", stage2=ALGORITHM, stage2_step=step,
            stage2_total=total, stage2_costs=str(args.costs),
            stage2_selection="fixed final endpoint; test never chooses checkpoint",
            sft_contexts="fresh greedy plus two replacements of one uniformly chosen short field",
            rollout_uses=1, dropout=0., context_weights=[.5, .25, .25],
            candidate_head_projection=args.candidate_head_projection,
            short_compact_after=args.short_compact_after,
            adaptive_recomputation=args.adaptive_recomputation,
            policy_gradient=args.policy_gradient,
            action_objective=planner.data.inventory["shared_action_objective"])
        write_json(temporary / CHECKPOINT_CONFIG, config)
        write_json(temporary / "action_costs.json", planner.data.multipath_costs)
    dist.barrier()
    torch.save({"optimizer": engine.optimizer.state_dict(), "scheduler": engine.lr_scheduler.state_dict(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "step": step, "total": total, "world_size": dist.get_world_size(), "algorithm": ALGORITHM,
        "policy_gradient": args.policy_gradient,
        "checkpoint": str(args.checkpoint), "release": str(args.release), "bank": str(args.bank),
        "work_counts": dict(planner.work_counts), "execution_rate": planner.execution_seconds_per_token,
        "rollout_rates": planner.rollout_seconds_per_token}, temporary / f"training_state.rank{dist.get_rank()}.pt")
    dist.barrier()
    if dist.get_rank() == 0:
        temporary.rename(destination)
        print(json.dumps({"phase": "checkpoint_saved", "step": step, "path": str(destination)}), flush=True)
    dist.barrier()
    return destination


def restore(args, engine, planner, total):
    if args.resume is None:
        return 0
    state = torch.load(args.resume / f"training_state.rank{dist.get_rank()}.pt",
                       map_location=torch.device("cuda", torch.cuda.current_device()), weights_only=False)
    assert (state["algorithm"], state["world_size"], state["total"]) == (ALGORITHM, dist.get_world_size(), total)
    assert state.get("policy_gradient", True) == args.policy_gradient
    assert (state["checkpoint"], state["release"], state["bank"]) == (str(args.checkpoint), str(args.release), str(args.bank))
    assert json.loads((args.resume / "action_costs.json").read_text()) == planner.data.multipath_costs
    engine.optimizer.load_state_dict(state["optimizer"])
    engine.lr_scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"].cpu())
    torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    planner.work_counts = Counter(state["work_counts"])
    planner.execution_seconds_per_token = state["execution_rate"]
    planner.rollout_seconds_per_token = state["rollout_rates"]
    return state["step"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "checkpoint", "release", "costs", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--bank", type=Path, help="Train-only reference bank (defaults to the bundle's bank)")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--learning-rate", type=float, default=2e-6)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--max-batch-tokens", type=int, default=28672)
    parser.add_argument("--rollout-batch-size", type=int, default=64)
    parser.add_argument("--rollout-max-batch-tokens", type=int, default=131072)
    parser.add_argument("--checkpoint-layers", type=int, default=24)
    parser.add_argument("--text-checkpoint-layers", type=int, default=30)
    parser.add_argument("--candidate-head-projection", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--short-compact-after", type=int, default=192)
    parser.add_argument("--adaptive-recomputation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--policy-gradient", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--effective-roots", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.bank = args.bank or args.release / "process_references/bank.rank0.jsonl"
    rank, local, world = (int(os.environ[key]) for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE"))
    torch.cuda.set_device(local)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", local), timeout=timedelta(hours=2))
    args.output.mkdir(parents=True, exist_ok=True)
    config = read_checkpoint(args.checkpoint)
    if config.get("phase") != "sft":
        raise ValueError("Stage 2 starts from a selected SFT adapter, not a Stage 2 endpoint")
    costs = json.loads(args.costs.read_text())
    if costs.get("synthetic_implementation_fixture"):
        raise ValueError("The cost table must come from a full training-set readout")
    assert Path(costs["source_adapter"]).resolve() == args.checkpoint.resolve()
    assert Path(costs["release"]).resolve() == args.release.resolve()
    data = MultipathReviewData(args.release, args.bank, costs, reference_seed=args.seed,
                               stratified_exposure=bool(config.get("stratified_exposure")))
    data.inventory["multipath"].update(algorithm=ALGORITHM,
        contexts="fresh greedy and two one-field replacements, weights .5/.25/.25; references once")
    data.inventory["shared_action_objective"]["training_objective"] = ALGORITHM
    schedule, sampling = data.schedule(world, args.seed, args.effective_roots, locality=True)
    total = sampling["steps_per_epoch"]
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    encoder = ConditionalReviewEncoder(processor, max_new_tokens=4096, cache_size=1024,
        probability_chunk_size=16, witness_frequency_penalty=0., compact_witness_json=True,
        short_max_new_tokens=config["short_max_new_tokens"],
        candidate_head_projection=args.candidate_head_projection, short_compact_after=args.short_compact_after)
    encoder.vision.maximum = 128
    engine = make_engine(args, total)
    planner = ConditionalReviewPlanner(data, encoder, engine.module, sampling["update_scale"],
        args.max_batch_tokens, args.rollout_batch_size, args.rollout_max_batch_tokens,
        policy_gradient=args.policy_gradient)
    planner.checkpoint_layers = args.checkpoint_layers
    planner.text_checkpoint_layers = args.text_checkpoint_layers
    planner.adaptive_recomputation = args.adaptive_recomputation
    first = restore(args, engine, planner, total)
    end = total
    if rank == 0:
        write_json(args.output / ("run_settings.json" if args.resume is None else f"resume.{first}.json"), {
            "algorithm": ALGORITHM, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "sampling": sampling, "inventory": data.inventory, "total_updates": total,
            "framework": "verl 0.7 FSDP2 engine, native multimodal forward, existing packed audit losses",
            "precision": "BF16 frozen base/activations, FP32 LoRA/loss/reduction, TF32 off, dropout off",
            "action": "equal task/class gold CE plus bounded full-information error cost; exposure compensated once",
            "risk": "V*exp(-H), two iid candidates of one uniformly drawn short field; " + (
                "LOO actor plus live reader" if args.policy_gradient else
                "live reader only; explicit score-function/CISPO actor term removed; shared parameters still update"),
            "policy": "same compact grammar/condition frequency 0.1/stop as deployment; T=1, top-p=1, top-k=0",
            "selection": "one fixed epoch endpoint, then test only; no dev or test selection",
            "test_loaded_by_training": False})
    stop = [False]
    faulthandler.enable()
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    signal.signal(signal.SIGUSR2, lambda *_: stop.__setitem__(0, True))
    for step in range(first, end):
        started = time.perf_counter()
        write_json(args.output / f"current.rank{rank}.json", {"step": step + 1, "phase": "conditional_rollout", "unix_time": time.time()})
        planner.loss_values.zero_()
        planner.policy_stats.zero_()
        planner.action_stats.zero_()
        torch.cuda.reset_peak_memory_stats()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            jobs = planner.packet_jobs(schedule[step * world + rank])
        prepared = time.perf_counter()
        # Preserve generated JSON escapes, including unpaired Unicode surrogates, in diagnostics.
        (args.output / f"current_paths.rank{rank}.json").write_text(
            json.dumps({"step": step + 1, "paths": planner.last_paths}, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8")
        write_json(args.output / f"current.rank{rank}.json", {"step": step + 1, "phase": "backward", "jobs": len(jobs), "unix_time": time.time()})
        engine.optimizer_zero_grad()
        events = []
        with engine.train_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            engine.module.set_requires_gradient_sync(False)
            for i, job in enumerate(jobs):
                engine.module.set_requires_gradient_sync(i == len(jobs) - 1)
                begin, finish = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                planner.compute_loss(engine.module, job).backward()
                finish.record()
                if i < len(jobs) - 1:
                    events.append((begin, finish, job_tokens(job)))
            norm = engine.optimizer_step()
            engine.lr_scheduler_step()
        planner.observe_execution(events)
        values = planner.loss_values.detach().clone()
        dist.all_reduce(values)
        values /= world
        action_stats = planner.action_stats.detach().clone()
        dist.all_reduce(action_stats)
        action_stats /= world
        policy = planner.policy_stats.cpu().tolist()
        paths = planner.last_paths
        q = [value for row in paths for value in row["Q"]]
        observation = {"step": step + 1, "total": total, "seconds": time.perf_counter() - started,
            "prepare_seconds": prepared - started, "grad_norm": float(norm),
            "lr": engine.optimizer.param_groups[0]["lr"], "losses": dict(zip(COMPONENTS, values.cpu().tolist())),
            "policy_logp_max_abs": policy[0], "policy_logp_abs_sum": policy[1], "policy_tokens": policy[2],
            "policy_clipped_tokens": policy[3], "Q_sum": sum(q), "Q_count": len(q),
            "action_components": {task: {label: dict(zip(("coefficient", "gold_nll", "expected_cost", "correct_mass"),
                action_stats[t, c].cpu().tolist())) for c, label in enumerate(costs["labels"])}
                for t, task in enumerate(("request_decision", "response_audit"))},
            "bounded_risk_mean": 1 - sum(q) / len(q) if q else None,
            "selected_kinds": dict(Counter(row["kind"] for row in paths)),
            "sampled_invalid": sum(not value for row in paths for value in row["valid"]),
            "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
            "rollout": planner.rollout_timing, "execution": planner.execution_report,
            "work_counts": dict(planner.work_counts), "stop_requested": stop[0]}
        with (args.output / f"train.rank{rank}.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(observation) + "\n")
        if step < 2:
            with (args.output / f"paths.rank{rank}.jsonl").open("a", encoding="utf-8") as handle:
                for path in paths:
                    handle.write(json.dumps({"step": step + 1, **path}, ensure_ascii=True) + "\n")
        rows = gather(observation)
        if rank == 0:
            print(json.dumps({"phase": "conditional_joint_update", "step": step + 1, "total": total,
                "seconds": max(row["seconds"] for row in rows), "grad_norm": float(norm),
                "losses": observation["losses"], "Q_mean": sum(row["Q_sum"] for row in rows) / sum(row["Q_count"] for row in rows),
                "policy_logp_max_abs": max(row["policy_logp_max_abs"] for row in rows),
                "peak_allocated_GiB": [row["peak_allocated_GiB"] for row in rows]}), flush=True)
        should_stop = any(row["stop_requested"] for row in rows)
        if (step + 1) % args.save_every == 0 or step + 1 == end or should_stop:
            checkpoint = save_checkpoint(args, engine, processor, config, planner, step + 1, total)
        if should_stop:
            end = step + 1
            break
    if rank == 0:
        write_json(args.output / "training_result.json", {"actual_steps": end, "total_updates": total,
            "complete": end == total,
            "policy_gradient": args.policy_gradient,
            "endpoint": str(checkpoint), "test_used_for_selection": False})
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
