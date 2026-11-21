"""Reuse the exact SFT job construction; verl owns parameters and updates."""

from collections import Counter
from types import SimpleNamespace

import torch.distributed as dist

from .shared_trainer import SharedAuditTrainer


class SourceReadReplay(SharedAuditTrainer):
    def __init__(self, data, encoder, update_scale, max_batch_tokens=28672):
        # Only the existing pure job/loss methods are used, not HF Trainer's
        # accelerator, optimizer, data loader, training loop or checkpointing.
        self.data, self.encoder = data, encoder
        self.update_scale, self.phase = update_scale, "sft"
        self.class_batch_size = self.native_batch_size = 64
        self.text_batch_size = 8
        self.max_batch_tokens = max_batch_tokens
        self.max_target_positions = 8192
        self.checkpoint_stride, self.checkpoint_layers = 1, 24
        self.text_checkpoint_layers = 30
        self.packed_jobs, self.balanced_jobs = True, True
        self.time_balanced_jobs = True
        self.execution_seconds_per_token = None
        self.execution_report = {}
        self.witness_recovery, self.sft_source_roll_in = False, False
        self.sft_objective = "field_budget"
        self.current_epoch, self.members = 0, {}
        self.work_counts, self.diagnostic_context_counts = Counter(), {}
        self.accelerator = SimpleNamespace(unwrap_model=lambda model: model,
            num_processes=dist.get_world_size(), process_index=dist.get_rank())
