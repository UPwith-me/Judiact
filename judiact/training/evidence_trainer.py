"""Stage-one evidence-indexed SFT on the existing root-weighted training loop."""

import json
from pathlib import Path

import torch

from ..evidence_audit import PROTOCOL_VERSION, MAIN_SFT_PROTOCOL_VERSION
from .evidence_data import evidence_views
from .latent_trainer import LatentAuditTrainer
from .execution import job_tokens, partition_by_time


class EvidenceAuditTrainer(LatentAuditTrainer):
    evaluation_protocol = MAIN_SFT_PROTOCOL_VERSION

    def __init__(self, *args, indexed=True, text_checkpoint_layers=None, time_balanced_jobs=False,
                 main_sft=False, dev_python=None, dev_memory_utilization=.78, **kwargs):
        self.main_sft = main_sft
        self.dev_python = dev_python
        self.dev_memory_utilization = dev_memory_utilization
        self.indexed = indexed
        self.text_checkpoint_layers = text_checkpoint_layers
        self.time_balanced_jobs = time_balanced_jobs
        self.execution_seconds_per_token = None
        self.execution_report = {}
        super().__init__(*args, **kwargs)
        # Large same-root packets revisit hundreds of identical CPU inputs.
        self.encoder.maximum = 1024
        if self.phase != "sft":
            raise ValueError("This supervision is stage-one SFT, not the legacy latent likelihood")

    def balance_packet_jobs(self, jobs):
        if not self.time_balanced_jobs or self.accelerator.num_processes == 1:
            return super().balance_packet_jobs(jobs)
        gathered = [None] * self.accelerator.num_processes
        torch.distributed.all_gather_object(gathered, (self.execution_seconds_per_token, jobs))
        rates, workers = zip(*gathered)
        bins = partition_by_time(workers, rates)
        self.execution_report = {
            "seconds_per_padded_token": list(rates),
            "assigned_padded_tokens": [sum(job_tokens(job) for job in batch) for batch in bins],
            "assigned_jobs": [len(batch) for batch in bins]}
        return bins[self.accelerator.process_index]

    def observe_execution(self, events):
        if not events:
            return
        # Exclude the last DDP backward: its time includes other ranks' waiting.
        events[-1][1].synchronize()
        seconds = sum(begin.elapsed_time(end) for begin, end, _ in events) / 1000
        tokens = sum(size for _, _, size in events)
        measured = seconds / tokens
        old = self.execution_seconds_per_token
        self.execution_seconds_per_token = measured if old is None else .8 * old + .2 * measured
        self.execution_report.update(measured_local_seconds=seconds, measured_local_tokens=tokens,
                                     updated_seconds_per_token=self.execution_seconds_per_token)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        previous = self.checkpoint_layers
        if self.text_checkpoint_layers is not None and not all(e.node.labels for e in inputs["examples"]):
            self.checkpoint_layers = self.text_checkpoint_layers
        try:
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        finally:
            self.checkpoint_layers = previous

    @torch.no_grad()
    def prepare_local_members(self, missing):
        for sid in missing:
            views = evidence_views(self.data.cases[sid], self.data.bank[sid]["candidate"], self.indexed,
                                   self.data.action_semantics)
            self.members[sid] = {"views": views, "weights": {
                "posterior": torch.tensor([1., 0.]),
                # Relations compare source-only views, not annotation-selected hints.
                "prior": torch.tensor([0., 1.]), "latent_score": torch.tensor([1., 0.])}}

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        if self.is_world_process_zero():
            path = Path(output_dir or self.args.output_dir) / "judiact_config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config.update(protocol=PROTOCOL_VERSION, indexed=self.indexed,
                prepared_source_cache_size=self.encoder.maximum,
                text_checkpoint_layers=self.text_checkpoint_layers,
                time_balanced_jobs=self.time_balanced_jobs,
                information_flow="X->W; X,W,P->L; X,P->locations; D=(original X,P,verified source locations,fallible W/L); D->C,A,behavior independently; no categorical gold parents",
                sft_contexts="equal per-field mass on indexed reference and unindexed original source; localization/W/L supervised once",
                relation_scope="source-only view; existing verified relations only",
                limitations="Index is not sufficient or certified faithful; discrete location/W/L sampling has no action gradient in SFT; full autonomous dev required",
                latent_gradient=None, observation_gradient=None, proposal=None,
                latent_root_normalization=None)
            if self.main_sft:
                config.update(protocol=MAIN_SFT_PROTOCOL_VERSION, source_total=True,
                    action_semantics="action_semantics", action_measure=self.data.action_measure,
                    root_action_balance=None,
                    dev_backend="vllm" if self.dev_python else "transformers",
                    dev_memory_utilization=self.dev_memory_utilization)
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
