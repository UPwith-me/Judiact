"""Current-policy facts on first use each epoch, with one continuing HF optimizer."""

import json
import math
import time
from collections import defaultdict
from pathlib import Path

import torch
from transformers import TrainerCallback

from ..audit_v6 import PROTOCOL_VERSION
from ..audit_constraints import DECODER_VERSION
from ..io_utils import read_jsonl, write_json, write_jsonl
from .v4_trainer import evaluate_natural
from .v6_sequence import trajectory_facts
from .v6_trainer import GroundedTrainer


def configure_activation_recomputation(model, stride):
    layers = model.get_base_model().model.language_model.layers
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = bool(stride and index % stride == 0)
    return [index for index, layer in enumerate(layers) if layer.gradient_checkpointing]


def distribute_fact_requests(requests, world):
    """Balance evidence blocks first, then real prompt tokens; never inspect targets."""
    unique = {row["sample_id"]: row for rows in requests for row in rows}
    work = sorted(unique.values(), key=lambda r: (-r["blocks"], -r["prompt_tokens"], r["sample_id"]))
    assignments, loads = [[] for _ in range(world)], [[0, 0, 0] for _ in range(world)]
    for row in work:
        rank = min(range(world), key=lambda r: (*loads[r], r))
        assignments[rank].append(row["sample_id"])
        loads[rank][0] += row["blocks"]
        loads[rank][1] += row["prompt_tokens"]
        loads[rank][2] += 1
    return assignments


class RecoverySnapshotCallback(TrainerCallback):
    def __init__(self, interval):
        self.interval = interval

    def on_step_end(self, args, state, control, **kwargs):
        if self.interval and (state.global_step == 1 or state.global_step % self.interval == 0):
            control.should_save = True
        return control


class FormalAuditTrainer(GroundedTrainer):
    def __init__(self, *args, rollout_batch_size=4, resume_facts=False,
                 native_checkpoint_layer_stride=1, fact_dispatch="balanced", **kwargs):
        super().__init__(*args, record_objective="field_fixed", matching_mode="legacy",
                         label_context="blind", fact_bank={}, **kwargs)
        self.rollout_batch_size = rollout_batch_size
        self.fact_epoch = None
        self.resume_facts = resume_facts
        self.native_checkpoint_layer_stride = native_checkpoint_layer_stride
        self.fact_dispatch = fact_dispatch
        self.latest_fact_work = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        native = all(r.trajectory.task_type.startswith("native_") for r in inputs["records"])
        configure_activation_recomputation(self.accelerator.unwrap_model(model),
                                          self.native_checkpoint_layer_stride if native else 1)
        return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

    def generate_fact_rows(self, model, sample_ids):
        groups = defaultdict(list)
        for sid in sample_ids:
            case = self.data.cases[sid]
            groups[case["task_type"], bool(case["model_input"]["images"])].append(sid)
        rows = {}
        for ids in groups.values():
            lengths = {sid: self.encoder.prepare(self.data.cases[sid]["model_input"],
                self.data.cases[sid]["task_type"])["input_ids"].numel() for sid in ids}
            ids.sort(key=lambda sid: (-lengths[sid], sid))
            batches, batch = [], []
            for sid in ids:
                if batch and (len(batch) >= self.rollout_batch_size or
                              (len(batch) + 1) * lengths[batch[0]] > self.max_batch_tokens):
                    batches.append(batch)
                    batch = []
                batch.append(sid)
            if batch:
                batches.append(batch)
            for batch in batches:
                cases = [self.data.cases[sid] for sid in batch]
                trajectories = self.encoder.rollout_batch(model,
                    [c["model_input"] for c in cases], [c["task_type"] for c in cases],
                    evidence_only=True)
                for sid, trajectory in zip(batch, trajectories):
                    rows[sid] = {"sample_id": sid, "facts": trajectory_facts(trajectory),
                        "generation_rank": self.accelerator.process_index,
                        "evidence_decoder": DECODER_VERSION,
                        "hit_token_limit": any(e["hit_token_limit"] for e in trajectory.evidence),
                        "evidence_errors": [e["errors"] for e in trajectory.evidence]}
        return rows

    @torch.no_grad()
    def prepare_packet_facts(self, model, packet):
        epoch = math.floor(self.state.epoch or 0.)
        directory = Path(self.args.output_dir) / "training_facts" / f"epoch{epoch + 1}"
        path = directory / f"rank{self.accelerator.process_index}.jsonl"
        if self.fact_epoch != epoch:
            directory.mkdir(parents=True, exist_ok=True)
            if path.exists() and not self.resume_facts:
                raise FileExistsError(f"Training facts already exist in this new run: {path}")
            # A recovery checkpoint cannot import evidence from later parameters.
            rows = [r for r in read_jsonl(path) if r["generated_at_step"] < self.state.global_step] \
                if path.exists() else []
            if path.exists():
                write_jsonl(path, rows)
            self.fact_bank = {r["sample_id"]: r["facts"] for r in rows}
            self.fact_epoch = epoch
        required = set(packet["absolute_ids"]) | {
            sid for experiment in packet["experiments"] for sid in experiment.sample_ids}
        missing = sorted(required - self.fact_bank.keys())
        world, rank = self.accelerator.num_processes, self.accelerator.process_index
        started = time.perf_counter()
        assignments = [missing]
        shared = self.fact_dispatch == "balanced" and world > 1
        if shared:
            local = []
            for sid in missing:
                case = self.data.cases[sid]
                base = self.encoder.prepare(case["model_input"], case["task_type"])
                local.append({"sample_id": sid, "prompt_tokens": base["input_ids"].numel(),
                    "blocks": sum(s.kind == "evidence" for s in
                                  self.encoder.protocol.audit_slots(case["model_input"], case["task_type"]))})
            requests = [None] * world
            # Every rank participates, including native-only packets with no requests.
            torch.distributed.all_gather_object(requests, local)
            assignments = distribute_fact_requests(requests, world)
        assigned = assignments[rank] if shared else missing
        decode_started = time.perf_counter()
        generated = self.generate_fact_rows(self.accelerator.unwrap_model(model), assigned)
        decode_seconds = time.perf_counter() - decode_started
        if shared:
            returned = [None] * world
            torch.distributed.all_gather_object(returned, generated)
            generated = {sid: row for worker in returned for sid, row in worker.items()}
        if missing:
            with path.open("a", encoding="utf-8") as handle:
                for sid in missing:
                    row = generated[sid]
                    self.fact_bank[sid] = row["facts"]
                    handle.write(json.dumps({**row, "generated_at_step": self.state.global_step,
                        "epoch": epoch + 1}, ensure_ascii=False) + "\n")
        self.latest_fact_work = {"requested_members": len(missing), "executed_members": len(assigned),
            "assigned_members_by_rank": [len(ids) for ids in assignments],
            "local_decode_seconds": decode_seconds}
        self.work_counts["training_fact_members"] += len(missing)
        self.work_counts["executed_fact_members"] += len(assigned)
        self.work_counts["training_fact_seconds"] += time.perf_counter() - started

    def training_step(self, model, inputs, num_items_in_batch=None):
        started = time.perf_counter()
        self.prepare_packet_facts(model, inputs)
        prepared = time.perf_counter()
        loss = super().training_step(model, inputs, num_items_in_batch)
        elapsed = time.perf_counter() - started
        path = Path(self.args.output_dir) / f"step_timing.rank{self.accelerator.process_index}.jsonl"
        with path.open("a") as handle:
            handle.write(json.dumps({"step": self.state.global_step + 1,
                "fact_seconds": prepared - started, "forward_backward_and_sync_seconds": elapsed - (prepared - started),
                "total_before_optimizer_seconds": elapsed,
                **self.latest_fact_work,
                "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3}) + "\n")
        return loss

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        if self.is_world_process_zero():
            write_json(Path(output_dir or self.args.output_dir) / "v6_config.json", {
                "protocol": PROTOCOL_VERSION, "class_balance": self.class_balance,
                "evidence_decoder": DECODER_VERSION,
                "fact_dispatch": self.fact_dispatch,
                "balance": self.balance, "evidence_normalization": self.evidence_normalization,
                "context": "generated evidence and fixed categorical placeholders",
                "fact_refresh": "current policy at first use per epoch; stopped gradient; reused within epoch",
                "matching_coefficient": self.matching_coefficient,
                "max_evidence_tokens": self.encoder.max_evidence_tokens})

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        from ..inference_v6 import GroundedAuditPredictor
        model = self.accelerator.unwrap_model(self.model_wrapped)
        epoch = round(self.state.epoch or 0)
        output = Path(self.args.output_dir) / "dev_selection" / f"epoch{epoch}"
        metrics = evaluate_natural(model, self.processing_class, self.data.release,
            output, self.accelerator,
            self.encoder.max_evidence_tokens, predictor_class=GroundedAuditPredictor,
            method_version=PROTOCOL_VERSION, initial_batch_size=8, max_batch_size=8)
        if self.class_balance != "none":
            if self.is_world_process_zero():
                predictions = {p["sample_id"]: p for p in read_jsonl(output / "predictions.jsonl")}
                per_task, denominators = [], {}
                for task, field in (("natural_request", "request_action"),
                                    ("natural_response", "response_publishability")):
                    classes = defaultdict(list)
                    for row in read_jsonl(self.data.release / f"data/{task}/dev.jsonl"):
                        gold = row["gold"][field]
                        probability = predictions[row["sample_id"]]["action_probabilities"][gold]
                        classes[gold].append(-math.log(max(probability, 1e-300)))
                    per_task.append(sum(sum(v) / len(v) for v in classes.values()) / len(classes))
                    denominators[task] = {k: len(v) for k, v in classes.items()}
                metrics.update(eval_population_NLL=metrics["eval_loss"],
                               eval_balanced_request_NLL=per_task[0],
                               eval_balanced_response_NLL=per_task[1],
                               eval_loss=sum(per_task) / len(per_task))
                write_json(output / "selection_metrics.json", {**metrics,
                    "criterion": "mean task, then mean observed action class, then mean conditional NLL",
                    "class_denominators": denominators})
            if self.accelerator.num_processes > 1:
                shared = [metrics]
                torch.distributed.broadcast_object_list(shared, src=0)
                metrics = shared[0]
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        model.train()
        return metrics
