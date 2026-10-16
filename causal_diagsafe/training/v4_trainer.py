"""HF Trainer owns AdamW and checkpoints; stream the exact v4 objective backwards."""

import json
import math
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler
from transformers import Trainer

from ..audit_v4 import audit_slots
from ..evaluation_v3 import summarize
from ..inference_v4 import ContextualAuditPredictor
from ..io_utils import read_jsonl, write_json, write_jsonl
from .v1_data import root_assignments
from .v4_events import field_event_nll
from .v4_losses import absolute_loss, encode_evidence, evidence_nll, member_losses
from .v4_sequence import AuditEncoder


def evaluate_natural(model, processor, release, output, state, max_evidence_tokens=512,
                     predictor_class=ContextualAuditPredictor, method_version="contextual_v4",
                     initial_batch_size=4, max_batch_size=64):
    """Only the predeclared natural dev determines the completed-epoch checkpoint."""
    release, output = Path(release), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = [row for task in ("natural_request", "natural_response")
            for row in read_jsonl(release / f"data/{task}/dev.jsonl")]
    local = rows[state.process_index::state.num_processes]
    predictor = predictor_class.from_model(model, processor, max_evidence_tokens)
    started = time.perf_counter()
    with (output / f"predictions.rank{state.process_index}.jsonl").open("w", encoding="utf-8") as handle:
        for prediction in predictor.iter_predictions(local, initial_batch_size, max_batch_size):
            handle.write(json.dumps(prediction, ensure_ascii=False) + "\n")
            handle.flush()
    write_json(output / f"batching.rank{state.process_index}.json", predictor.eval_batcher.summary())
    state.wait_for_everyone()
    metrics = None
    if state.is_main_process:
        predictions = {row["sample_id"]: row for rank in range(state.num_processes)
                       for row in read_jsonl(output / f"predictions.rank{rank}.jsonl")}
        samples = {row["sample_id"]: row for row in rows}
        assert predictions.keys() == samples.keys()
        report, _ = summarize(samples, predictions, set(samples), root_assignments(rows, release / "reviews"), [], [])
        report["method"] = method_version + "; complete generated audit; conditional action probabilities"
        report["subsets"]["natural"]["evidence"]["decoder"] = "continuous greedy audit; exact quotes/occurrences and original-image grounding"
        request, response = (report["subsets"]["natural"][key]["NLL"]
                             for key in ("request_action", "response_publishability"))
        metrics = {"eval_loss": .5 * (request["value"] + response["value"]),
            "eval_request_NLL": request["value"], "eval_response_NLL": response["value"],
            "eval_request_denominator": request["denominator"], "eval_response_denominator": response["denominator"],
            "eval_runtime": time.perf_counter() - started}
        write_json(output / "summary.json", report)
        write_json(output / "selection_metrics.json", metrics)
        write_jsonl(output / "predictions.jsonl", (predictions[row["sample_id"]] for row in rows))
        print(json.dumps({"completed_epoch_dev": str(output), **metrics}), flush=True)
    if state.num_processes > 1:
        shared = [metrics]
        torch.distributed.broadcast_object_list(shared, src=0)
        metrics = shared[0]
    return metrics


class ContextualTrainer(Trainer):
    def __init__(self, *args, audit_data, update_scale, native_batch_size=16, rollout_batch_size=4,
                 max_evidence_tokens=512, optimize=False, memory_profile=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.data, self.update_scale = audit_data, update_scale
        self.encoder = AuditEncoder(self.processing_class, max_evidence_tokens, optimize=optimize)
        self.native_batch_size, self.rollout_batch_size = native_batch_size, rollout_batch_size
        self.model_accepts_loss_kwargs = False
        self.work_counts = defaultdict(int)
        self.observed_terms = {}
        self.activation_bytes_per_token = None
        if memory_profile:
            observations = json.loads(Path(memory_profile).read_text())
            measured = [(row["peak_allocated_GiB"] - row["baseline_GiB"]) * 1024**3 / row["padded_tokens"]
                        for row in observations["measurements"]
                        if not row["checkpointed"] and "seconds" in row]
            # This is a hardware sizing margin on measured activation memory, not a metric gate.
            self.activation_bytes_per_token = max(measured) * 1.15

    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        kind = inputs["kind"]
        if kind == "native":
            cases = [self.data.cases[sid] for sid in inputs["ids"]]
            scores = self.encoder.read_native_batch(model, cases)
            loss = sum(-score[case["task_type"]][("safe", "harmful").index(case["targets"][case["task_type"]])]
                       * self.data.absolute_weights[case["sample_id"]] * .1
                       for case, score in zip(cases, scores))
            self.work_counts["native_members"] += len(cases)
            term = "L_N_times_0.1"
        elif kind == "evidence":
            sid, name = inputs["sid"], inputs["name"]
            case = self.data.cases[sid]
            nll, _ = evidence_nll(self.encoder, model, inputs["trajectory"], name, case["targets"][name])
            diagnostic_fields = len(case["targets"]) - 1
            loss = nll * self.data.absolute_weights[sid] / diagnostic_fields
            self.work_counts["evidence_forwards"] += 1
            term = "L_D_evidence"
        elif kind == "evidence_batch":
            examples = inputs["examples"]
            # Reuse/group graphs without changing a Core member's floating-point padding path.
            nlls = [evidence_nll(self.encoder, model, trajectory, name, self.data.cases[sid]["targets"][name])[0]
                    for sid, name, trajectory in examples]
            loss = sum(nll * self.data.absolute_weights[sid] / (len(self.data.cases[sid]["targets"]) - 1)
                       for nll, (sid, _, _) in zip(nlls, examples))
            self.work_counts["evidence_forwards"] += len(examples)
            self.work_counts["evidence_backward_groups"] += 1
            term = "L_D_evidence"
        elif kind == "absolute":
            sid, trajectory = inputs["sid"], inputs["trajectory"]
            scores = self.encoder.read_classifications(model, trajectory)
            placeholders = {name: scores["action"].sum() * 0 for name in self.data.cases[sid]["targets"]
                            if name.startswith("evidence_")}
            member = member_losses(self.data.cases[sid], trajectory, scores, placeholders)
            absolute = absolute_loss({sid: member}, self.data.absolute_weights)
            loss = absolute["L_A"] + absolute["L_D"]
            self.work_counts["absolute_members"] += 1
            term = "L_A_plus_L_D_classify"
        elif kind == "relation":
            experiment = inputs["experiment"]
            scores = {sid: self.encoder.read_classifications(model, inputs["trajectories"][sid])
                      for sid in experiment.sample_ids}
            loss = sum(field_event_nll(scores, field) * self.data.event_weights[experiment.key, field.slot]
                       for field in experiment.fields)
            self.work_counts["relation_experiments"] += 1
            self.work_counts["relation_member_forwards"] += len(scores)
            term = "L_G"
        elif kind == "classifications":
            ids = inputs["ids"]
            readouts = [self.encoder.read_classifications(model, inputs["trajectories"][sid]) for sid in ids]
            scores = dict(zip(ids, readouts))
            members = {}
            for sid in inputs["absolute_ids"]:
                placeholders = {name: scores[sid]["action"].sum() * 0 for name in self.data.cases[sid]["targets"]
                                if name.startswith("evidence_")}
                members[sid] = member_losses(self.data.cases[sid], inputs["trajectories"][sid], scores[sid], placeholders)
            if members:
                absolute = absolute_loss(members, self.data.absolute_weights)
                loss = absolute["L_A"] + absolute["L_D"]
            else:
                loss = readouts[0]["action"].sum() * 0
            loss += sum(field_event_nll(scores, field) * self.data.event_weights[experiment.key, field.slot]
                        for experiment in inputs["experiments"] for field in experiment.fields)
            self.work_counts["absolute_members"] += len(members)
            self.work_counts["relation_experiments"] += len(inputs["experiments"])
            self.work_counts["relation_member_forwards"] += sum(len(item.sample_ids) for item in inputs["experiments"])
            self.work_counts["actual_classification_members"] += len(ids)
            term = "L_A_plus_L_D_classify_plus_L_G"
        else:
            raise ValueError(kind)
        loss = loss * self.update_scale
        # Keep observations on device; a CPU scalar here synchronizes every small job.
        if term in self.observed_terms:
            self.observed_terms[term].add_(loss.detach())
        else:
            self.observed_terms[term] = loss.detach().clone()
        return (loss, {"loss": loss}) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        started = time.perf_counter()
        model.train()
        if hasattr(self.optimizer, "train") and callable(self.optimizer.train):
            self.optimizer.train()
        required = list(dict.fromkeys(inputs["absolute_ids"] +
            [sid for experiment in inputs["experiments"] for sid in experiment.sample_ids]))
        buckets = defaultdict(list)
        for sid in required:
            case = self.data.cases[sid]
            text = sum(len(case["model_input"].get(k) or "") for k in ("query", "response_candidate"))
            buckets[(case["task_type"], len(case["model_input"]["images"]), int(math.log2(max(1, text))))].append(sid)
        trajectories = {}
        with self.compute_loss_context_manager():
            raw_model = self.accelerator.unwrap_model(model)
            for ids in buckets.values():
                for start in range(0, len(ids), self.rollout_batch_size):
                    selected = ids[start:start + self.rollout_batch_size]
                    cases = [self.data.cases[sid] for sid in selected]
                    generated = self.encoder.rollout_batch(raw_model,
                        [case["model_input"] for case in cases], [case["task_type"] for case in cases])
                    trajectories.update(zip(selected, generated))
                    self.work_counts["audit_rollouts"] += len(generated)
                    for trajectory in generated:
                        self.work_counts["rollout_input_tokens"] += trajectory.prompt_length
                        self.work_counts["rollout_generated_tokens"] += trajectory.counts["generated_tokens"]
                        self.work_counts["rollout_processed_tokens"] += trajectory.counts["processed_tokens"]
        rollout_seconds = time.perf_counter() - started
        jobs = self.classification_jobs(inputs, trajectories) if self.encoder.optimize else []
        evidence = defaultdict(list)
        for sid in inputs["absolute_ids"]:
            trajectory = trajectories[sid]
            if not self.encoder.optimize:
                jobs.append({"kind": "absolute", "sid": sid, "trajectory": trajectory})
            for name in self.data.cases[sid]["targets"]:
                if not name.startswith("evidence_"):
                    continue
                if self.encoder.optimize:
                    entry = next(item for item in trajectory.entries if item["name"] == name)
                    evidence[(len(trajectory.visible["images"]), int(math.log2(entry["answer_start"])))].append(
                        (sid, name, trajectory))
                else:
                    jobs.append({"kind": "evidence", "sid": sid, "name": name, "trajectory": trajectory})
        for examples in evidence.values():
            for start in range(0, len(examples), 4):
                jobs.append({"kind": "evidence_batch", "examples": examples[start:start + 4]})
        if not self.encoder.optimize:
            jobs.extend({"kind": "relation", "experiment": experiment,
                         "trajectories": {sid: trajectories[sid] for sid in experiment.sample_ids}}
                        for experiment in inputs["experiments"])
        native = defaultdict(list)
        for sid in inputs["native_ids"]:
            case = self.data.cases[sid]
            text = sum(len(case["model_input"].get(k) or "") for k in ("query", "response_candidate"))
            if self.encoder.optimize:
                native[len(case["model_input"]["images"])].append(sid)
            else:
                native[(case["task_type"], bool(case["model_input"]["images"]), int(math.log2(max(1, text))))].append(sid)
        for ids in native.values():
            if self.encoder.optimize:
                lengths = {sid: self.native_length(sid) for sid in ids}
                ids.sort(key=lengths.__getitem__, reverse=True)
                free = self.activation_capacity()
                selected = []
                for sid in ids:
                    tokens = (len(selected) + 1) * lengths[(selected or [sid])[0]]
                    fits = self.activation_bytes_per_token is None or tokens * self.activation_bytes_per_token < free
                    if selected and (len(selected) == self.native_batch_size or not fits):
                        jobs.append({"kind": "native", "ids": selected})
                        selected = []
                    selected.append(sid)
                if selected:
                    jobs.append({"kind": "native", "ids": selected})
            else:
                for start in range(0, len(ids), self.native_batch_size):
                    jobs.append({"kind": "native", "ids": ids[start:start + self.native_batch_size]})
        total = torch.zeros((), device=self.args.device)
        phases = defaultdict(float)
        raw_model = self.accelerator.unwrap_model(model)
        for index, job in enumerate(jobs):
            job_started = time.perf_counter()
            if self.encoder.optimize and self.activation_bytes_per_token is not None:
                tokens = self.job_tokens(job)
                checkpointed = tokens * self.activation_bytes_per_token >= self.activation_capacity()
                if checkpointed != raw_model.is_gradient_checkpointing:
                    if checkpointed:
                        raw_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                    else:
                        raw_model.gradient_checkpointing_disable()
                self.work_counts["checkpointed_jobs" if checkpointed else "no_recompute_jobs"] += 1
            context = model.no_sync() if index < len(jobs) - 1 and hasattr(model, "no_sync") else nullcontext()
            with context:
                total += super().training_step(model, job, num_items_in_batch=None)
            phases[job["kind"]] += time.perf_counter() - job_started
        if self.encoder.optimize and self.state.global_step % 25 == 0:
            profile = {"step": self.state.global_step + 1, "rank": self.accelerator.process_index,
                "rollout_seconds": rollout_seconds, "backward_seconds_by_kind": dict(phases),
                "step_seconds_before_optimizer": time.perf_counter() - started,
                "audit_members": len(required), "backward_jobs": len(jobs),
                "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
                "visual_cache": self.encoder.visual_cache.summary()}
            with (Path(self.args.output_dir) / f"throughput.rank{self.accelerator.process_index}.jsonl").open("a") as handle:
                handle.write(json.dumps(profile) + "\n")
        return total

    def activation_capacity(self):
        free, _ = torch.cuda.mem_get_info(self.args.device)
        reusable = torch.cuda.memory_reserved(self.args.device) - torch.cuda.memory_allocated(self.args.device)
        return free + reusable - 3 * 1024**3

    def native_length(self, sid):
        case = self.data.cases[sid]
        base = self.encoder.prepare(case["model_input"], case["task_type"])
        slot, = audit_slots(case["model_input"], case["task_type"])
        return base["input_ids"].numel() + len(self.encoder.token_ids(slot.header))

    def job_tokens(self, job):
        if job["kind"] == "native":
            lengths = [self.native_length(sid) for sid in job["ids"]]
        elif job["kind"] == "classifications":
            lengths = [job["trajectories"][sid].input_ids.numel() for sid in job["ids"]]
        else:
            lengths = [next(item["answer_start"] for item in trajectory.entries if item["name"] == name)
                       + len(encode_evidence(self.data.cases[sid]["targets"][name],
                                             self.processing_class.tokenizer).token_ids)
                       for sid, name, trajectory in job["examples"]]
        return len(lengths) * max(lengths)

    def classification_jobs(self, inputs, trajectories):
        """Reuse a member's differentiable readout within small connected experiments."""
        absolute = set(inputs["absolute_ids"])
        remaining = list(inputs["experiments"])
        jobs, covered = [], set()

        def job(ids, experiments):
            ordered = sorted(ids)
            local_absolute = sorted(absolute.intersection(ids) - covered)
            covered.update(local_absolute)
            return {"kind": "classifications", "ids": ordered, "absolute_ids": local_absolute,
                    "experiments": experiments, "trajectories": {sid: trajectories[sid] for sid in ordered}}

        while remaining:
            connected = [remaining.pop(0)]
            ids = set(connected[0].sample_ids)
            while True:
                touching = [item for item in remaining if ids.intersection(item.sample_ids)]
                if not touching:
                    break
                connected.extend(touching)
                ids.update(sid for item in touching for sid in item.sample_ids)
                remaining = [item for item in remaining if item not in touching]
            if len(ids) <= 4:
                jobs.append(job(ids, connected))
            else:
                # Keep large components streamed rather than retaining many long graphs.
                for item in connected:
                    jobs.append(job(set(item.sample_ids), [item]))
        buckets = defaultdict(list)
        for sid in sorted(absolute - covered):
            item = trajectories[sid]
            buckets[(item.task_type, len(item.visible["images"]), int(math.log2(item.input_ids.numel())))].append(sid)
        for ids in buckets.values():
            for start in range(0, len(ids), 4):
                jobs.append(job(set(ids[start:start + 4]), []))
        assert covered == absolute
        return jobs

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        model = self.accelerator.unwrap_model(self.model_wrapped)
        epoch = round(self.state.epoch or 0)
        metrics = evaluate_natural(model, self.processing_class, self.data.release,
            Path(self.args.output_dir) / "dev_selection" / f"epoch{epoch}", self.accelerator,
            self.encoder.max_evidence_tokens)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        model.train()
        return metrics
