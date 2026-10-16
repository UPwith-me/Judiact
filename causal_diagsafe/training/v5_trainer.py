"""HF Trainer owns updates; stream root-weighted coherent record/matching losses."""

import json
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler
from transformers import Trainer

from ..audit_v5 import PROTOCOL_VERSION
from .v4_losses import absolute_loss
from .v5_matching import conditional_matching_nll
from .v5_sequence import CoherentAuditEncoder


class CoherentTrainer(Trainer):
    def __init__(self, *args, audit_data, update_scale, matching_coefficient=1.,
                 native_coefficient=.1, record_batch_size=4, native_batch_size=16,
                 max_batch_tokens=16384, max_evidence_tokens=512, **kwargs):
        super().__init__(*args, **kwargs)
        self.data, self.update_scale = audit_data, update_scale
        self.matching_coefficient, self.native_coefficient = matching_coefficient, native_coefficient
        self.record_batch_size, self.native_batch_size = record_batch_size, native_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.encoder = CoherentAuditEncoder(self.processing_class, max_evidence_tokens, optimize=True)
        self.model_accepts_loss_kwargs = False
        self.work_counts = defaultdict(int)
        self.observed_terms = {}

    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        scores, members = self.encoder.read_records(model, inputs["records"], inputs["absolute_ids"])
        absolute = absolute_loss(members, self.data.absolute_weights) if members else None
        exemplar = next(iter(next(iter(scores.values())).values()))
        terms = {"L_A": exemplar.sum() * 0, "L_D": exemplar.sum() * 0, "L_N": exemplar.sum() * 0,
                 "L_match": exemplar.sum() * 0}
        if absolute:
            terms.update(absolute)
        if self.matching_coefficient:
            for name, component, weight in inputs["matching_terms"]:
                terms["L_match"] = terms["L_match"] + weight * conditional_matching_nll(
                    [scores[sid][name] for sid in component.sample_ids], component.gold_indices)
        loss = self.update_scale * (terms["L_A"] + terms["L_D"] +
            self.native_coefficient * terms["L_N"] + self.matching_coefficient * terms["L_match"])
        for name, value in terms.items():
            observed = value.detach() * self.update_scale
            if name in self.observed_terms:
                self.observed_terms[name].add_(observed)
            else:
                self.observed_terms[name] = observed.clone()
        return (loss, {"loss": loss}) if return_outputs else loss

    def packet_jobs(self, packet):
        """Same member exposures for matched no-relation; no generated training prefixes."""
        required = list(dict.fromkeys(packet["absolute_ids"] + packet["native_ids"] +
            [sid for experiment in packet["experiments"] for sid in experiment.sample_ids]))
        encoded = {sid: self.encoder.encode_case(self.data.cases[sid]) for sid in required}
        absolute, covered, jobs = set(packet["absolute_ids"]), set(), []

        def append(ids, terms=()):
            ids = sorted(ids)
            own = set(ids) & absolute - covered
            covered.update(own)
            jobs.append({"records": [encoded[sid] for sid in ids],
                         "absolute_ids": sorted(own), "matching_terms": tuple(terms)})

        # A single-factor field component has at most four members in this release.
        # Group equal member sets so the same forward serves absolute and relation supervision.
        grouped = defaultdict(list)
        for experiment in packet["experiments"]:
            for field in experiment.fields:
                weight = self.data.event_weights[experiment.key, field.slot]
                if not weight:
                    continue
                for component in field.components:
                    if not component.is_constant:
                        grouped[tuple(sorted(component.sample_ids))].append((field.slot, component, weight))
        for ids, terms in grouped.items():
            append(ids, terms)

        def batches(ids, maximum):
            buckets = defaultdict(list)
            for sid in ids:
                record = encoded[sid]
                buckets[bool(record.trajectory.visible["images"])].append(sid)
            for values in buckets.values():
                values.sort(key=lambda sid: encoded[sid].trajectory.input_ids.numel(), reverse=True)
                batch = []
                for sid in values:
                    proposed = (len(batch) + 1) * encoded[batch[0] if batch else sid].trajectory.input_ids.numel()
                    if batch and (len(batch) >= maximum or proposed > self.max_batch_tokens):
                        yield batch
                        batch = []
                    batch.append(sid)
                if batch:
                    yield batch

        for ids in batches(sorted(absolute - covered), self.record_batch_size):
            append(ids)
        assert covered == absolute, "absolute members must receive exactly one population contribution"
        for ids in batches(packet["native_ids"], self.native_batch_size):
            jobs.append({"records": [encoded[sid] for sid in ids],
                         "absolute_ids": ids, "matching_terms": ()})
        assert jobs and all(item.trajectory.task_type.startswith("native_") for item in jobs[-1]["records"])
        return jobs

    def training_step(self, model, inputs, num_items_in_batch=None):
        started = time.perf_counter()
        jobs = self.packet_jobs(inputs)
        total = torch.zeros((), device=self.args.device)
        for index, job in enumerate(jobs):
            context = model.no_sync() if index < len(jobs) - 1 and hasattr(model, "no_sync") else nullcontext()
            with context:
                total += super().training_step(model, job, num_items_in_batch=None)
            self.work_counts["forward_members"] += len(job["records"])
            self.work_counts["absolute_members"] += len(job["absolute_ids"])
            self.work_counts["directional_components"] += len(job["matching_terms"])
            self.work_counts["backward_batches"] += 1
        if self.state.global_step % 10 == 0:
            profile = {"step": self.state.global_step + 1, "rank": self.accelerator.process_index,
                       "seconds_before_optimizer": time.perf_counter() - started,
                       "batch_count": len(jobs),
                       "training_fact_members_cumulative": self.work_counts.get("training_fact_members", 0),
                       "training_fact_seconds_cumulative": self.work_counts.get("training_fact_seconds", 0.),
                       "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3,
                       "loss_terms_population_scaled": {k: float(v) for k, v in self.observed_terms.items()}}
            self.observed_terms.clear()
            with (Path(self.args.output_dir) / f"throughput.rank{self.accelerator.process_index}.jsonl").open("a") as handle:
                handle.write(json.dumps(profile) + "\n")
        return total

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        from ..inference_v5 import CoherentAuditPredictor
        from .v4_trainer import evaluate_natural
        model = self.accelerator.unwrap_model(self.model_wrapped)
        epoch = round(self.state.epoch or 0)
        metrics = evaluate_natural(model, self.processing_class, self.data.release,
            Path(self.args.output_dir) / "dev_selection" / f"epoch{epoch}", self.accelerator,
            self.encoder.max_evidence_tokens, predictor_class=CoherentAuditPredictor, method_version=PROTOCOL_VERSION)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        model.train()
        return metrics
