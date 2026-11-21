"""Shared audit SFT on the existing root-weighted DDP training loop."""

from collections import defaultdict
from functools import partial
import json
from pathlib import Path

import torch

from ..audit_schema import audit_slots
from ..evidence_audit import materialize_index
from ..shared_audit import PROTOCOL_VERSION, SharedAuditNode
from ..source_audit import source_judgment_node
from ..shared_predictor import SharedAuditPredictor
from .shared_sequence import SharedAuditEncoder
from .action_loss import action_nll
from .source_trainer import SourceAuditTrainer


class SharedAuditTrainer(SourceAuditTrainer):
    evaluation_protocol = PROTOCOL_VERSION
    predictor_type = SharedAuditPredictor

    def __init__(self, *args, rollout_max_batch_tokens=65536, frozen_image_cache_size=32, **kwargs):
        super().__init__(*args, **kwargs)
        self.rollout_max_batch_tokens = rollout_max_batch_tokens
        previous = self.encoder
        self.encoder = SharedAuditEncoder(self.processing_class,
            max_new_tokens=previous.max_new_tokens, cache_size=previous.maximum,
            probability_chunk_size=previous.probability_chunk_size,
            witness_frequency_penalty=0., compact_witness_json=True)
        self.encoder.vision.maximum = frozen_image_cache_size
        self.read_history_epoch = None
        self.predictor_type = partial(SharedAuditPredictor,
            include_basis=not self.data.local_boundary_conditions, action_objective=self.data.action_objective)

    def supervised_nll(self, example, result):
        if example.node.name != "action":
            return super().supervised_nll(example, result)
        labels = example.node.labels
        prior = self.data.action_priors[example.node.task]
        budget = self.data.action_budget
        return action_nll(result[1], labels.index(example.target),
                          [prior[label] for label in labels], self.data.action_objective,
                          [budget[label] for label in labels] if budget is not None else None)

    def remember_source_read(self, sid, index):
        views = self.data.shared_views(sid, index)
        weights = torch.zeros(len(views))
        weights[0] = 1.
        self.members[sid] = {"views": views, "weights": {
            "posterior": weights, "prior": weights, "latent_score": weights}}
        return views

    def restore_source_reads(self):
        if self.read_history_epoch == self.current_epoch:
            return
        self.read_history_epoch = self.current_epoch
        # Trainer restores optimizer/RNG, but not this epoch's first-visit reads.
        # Reuse recorded actual indices rather than silently changing roll-in on resume.
        from ..io_utils import read_jsonl
        paths = Path(self.args.output_dir).glob(f"source_reads.epoch{self.current_epoch + 1}.rank*.jsonl")
        rows = [row for path in sorted(paths) for row in read_jsonl(path)
                if row["step"] < self.state.global_step]
        for row in sorted(rows, key=lambda row: row["step"]):
            if row["sample_id"] not in self.members:
                self.remember_source_read(row["sample_id"], row["source_index"])

    @torch.no_grad()
    def prepare_members(self, ids):
        if not self.data.student_source_reads:
            for sid in sorted(set(ids) - self.members.keys()):
                self.remember_source_read(sid, None)
                self.work_counts["direct_source_members"] += 1
            return
        self.restore_source_reads()
        missing = sorted(set(ids) - self.members.keys())
        if self.balanced_jobs and self.accelerator.num_processes > 1:
            self.prepare_balanced_members(missing)
        else:
            self.prepare_local_members(missing)

    @torch.no_grad()
    def prepare_balanced_members(self, missing):
        world, rank = self.accelerator.num_processes, self.accelerator.process_index
        requested = [None] * world
        torch.distributed.all_gather_object(requested, missing)
        unique = sorted({sid for group in requested for sid in group})
        if not unique:
            return
        costs = []
        for sid in unique:
            case = self.data.cases[sid]
            cost = 0
            for slot in audit_slots(case["model_input"], case["task_type"]):
                if slot.kind == "evidence":
                    node = source_judgment_node(case["model_input"], case["task_type"], slot.name)
                    cost += self.encoder.prepare(node)["input_ids"].numel() + self.encoder.max_new_tokens
            costs.append((cost, sid))
        assigned, loads = [[] for _ in range(world)], [0] * world
        for cost, sid in sorted(costs, reverse=True):
            owner = min(range(world), key=loads.__getitem__)
            assigned[owner].append(sid)
            loads[owner] += cost
        produced = self.prepare_local_members(assigned[rank])
        returned = [None] * world
        # Rebuild identical supervised views locally from compact actual indices.
        # The old exchange repeated source text and gold targets across all ranks.
        torch.distributed.all_gather_object(returned, produced)
        for worker in returned:
            for sid, index in worker.items():
                if sid not in self.members:
                    self.remember_source_read(sid, index)
        self.work_counts["executed_latent_members"] += len(assigned[rank])

    @torch.no_grad()
    def prepare_local_members(self, missing):
        if not missing:
            return {}
        nodes, owners, identities = [], [], {}
        # Each selected observation is generated by the current student. No gold
        # positions, W facts, predicted class labels or adjudication notes enter it.
        for sid in missing:
            case = self.data.cases[sid]
            for slot in audit_slots(case["model_input"], case["task_type"]):
                if slot.kind == "evidence":
                    node = source_judgment_node(case["model_input"], case["task_type"], slot.name)
                    key = node.text, node.media
                    if key not in identities:
                        identities[key] = len(nodes)
                        nodes.append(node)
                        owners.append([])
                    owners[identities[key]].append((sid, slot.stage))
        model = self.accelerator.unwrap_model(self.model_wrapped)
        outputs = defaultdict(dict)
        for batch in self.batches(nodes, rollout=True):
            generated = self.encoder.generate(model, [nodes[i] for i in batch])
            for index, output in zip(batch, generated):
                for sid, stage in owners[index]:
                    outputs[sid][stage] = output
        self.work_counts["student_read_forward_nodes"] += len(nodes)
        history = Path(self.args.output_dir) / (
            f"source_reads.epoch{(self.current_epoch or 0) + 1}.rank{self.accelerator.process_index}.jsonl")
        indices = {}
        with history.open("a", encoding="utf-8") as handle:
            for sid in missing:
                generated = outputs[sid]
                index, errors = materialize_index(self.data.cases[sid]["model_input"],
                    {stage: row.get("raw", "null") if not row.get("error") else "null"
                     for stage, row in generated.items()})
                views = self.remember_source_read(sid, index)
                indices[sid] = index
                self.work_counts["student_read_members"] += 1
                self.work_counts["student_read_distinct_contexts"] += len(views) - 1
                self.work_counts["student_read_invalid_nodes"] += sum(bool(row.get("error")) for row in generated.values())
                handle.write(json.dumps({"sample_id": sid, "step": self.state.global_step,
                    "source_index": index, "index_errors": errors,
                    "nodes": {stage: {**{key: row.get(key) for key in ("tokens", "error", "terminated", "generation_controls")},
                                      **({"raw": row.get("raw")} if row.get("error") else {})}
                              for stage, row in generated.items()},
                    "distinct_classification_contexts": len(views),
                    "scope": "current student first visit per epoch; no gold selection; original sources retained"},
                    ensure_ascii=False) + "\n")
        return indices

    def batches(self, examples, rollout=False, native=False):
        if rollout:
            lengths = [self.encoder.prepare(node)["input_ids"].numel() for node in examples]
            group = []
            # Frozen vision is already indexed by the actual image sequence.
            # Text-only and visual rows can share decoding instead of paying
            # their longest autoregressive tails in two serial batches.
            for index in sorted(range(len(examples)), key=lambda i: -lengths[i]):
                if group and (len(group) >= self.rollout_batch_size or
                        (len(group) + 1) * lengths[group[0]] > self.rollout_max_batch_tokens):
                    yield group
                    group = []
                group.append(index)
            if group:
                yield group
            return
        if native:
            yield from super().batches(examples, rollout, native)
            return
        shared, ordinary = defaultdict(list), []
        for i, example in enumerate(examples):
            if isinstance(example.node, SharedAuditNode):
                shared[self.encoder.context_key(example.node)].append(i)
            else:
                ordinary.append(i)
        for batch in super().batches([examples[i] for i in ordinary]):
            yield [ordinary[i] for i in batch]
        groups = sorted(shared.values(), key=lambda indices:
            -self.encoder.prepare(examples[indices[0]].node)["input_ids"].numel())
        result, count, longest = [], 0, 0
        for indices in groups:
            length = self.encoder.prepare(examples[indices[0]].node)["input_ids"].numel()
            if result and (count + 1 > self.class_batch_size or (count + 1) * longest > self.max_batch_tokens):
                yield result
                result, count, longest = [], 0, 0
            longest = max(longest, length)
            result.extend(indices)
            count += 1
        if result:
            yield result

    def pack_core_jobs(self, jobs):
        ordinary, shared = [], []
        for job in jobs:
            (shared if isinstance(job["examples"][0].node, SharedAuditNode) else ordinary).append(job)
        result = super().pack_core_jobs(ordinary)
        # Coalesce all readouts of the same actual input. Relations keep their
        # original rows/weights; repeated CE occurrences are added exactly once.
        pending = None
        contexts, key_indices = {}, {}

        def flush():
            nonlocal pending, contexts, key_indices
            if pending is not None:
                result.append(pending)
            pending, contexts, key_indices = None, {}, {}

        shared.sort(key=lambda job: -max(self.encoder.prepare(e.node)["input_ids"].numel()
                                        for e in job["examples"]))
        for job in shared:
            proposed = {self.encoder.context_key(e.node): self.encoder.prepare(e.node)["input_ids"].numel()
                        for e in job["examples"]}
            combined = contexts | proposed
            if pending is not None and (len(combined) > self.class_batch_size or
                                       len(combined) * max(combined.values()) > self.max_batch_tokens):
                flush()
            if pending is None:
                pending = {"examples": [], "keys": [], "coefficients": [], "relations": []}
            contexts.update(proposed)
            positions = []
            for key, example, coefficient in zip(job["keys"], job["examples"], job["coefficients"]):
                if key not in key_indices:
                    key_indices[key] = len(pending["keys"])
                    pending["keys"].append(key)
                    pending["examples"].append(example)
                    pending["coefficients"].append(0.)
                index = key_indices[key]
                pending["coefficients"][index] += coefficient
                positions.append(index)
            if job["relation"] is not None:
                gold, weight, groups, mixing = job["relation"]
                pending["relations"].append((gold, weight,
                    [[positions[i] for i in group] for group in groups], mixing))
        flush()
        return result

    def balance_packet_jobs(self, jobs):
        for job in jobs:
            if not isinstance(job["examples"][0].node, SharedAuditNode):
                continue
            contexts = {self.encoder.context_key(e.node): self.encoder.shared_input(e.node)
                        for e in job["examples"]}
            width = max(base["input_ids"].numel() for base, _ in contexts.values())
            positions = {width - base["input_ids"].numel() + position
                         for base, slots in contexts.values() for position in slots.values()}
            job["tensor_shape"] = len(contexts), width, len(positions)
        return super().balance_packet_jobs(jobs)

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        if self.is_world_process_zero():
            path = Path(output_dir or self.args.output_dir) / "judiact_config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config.update(protocol=PROTOCOL_VERSION,
                sft_ablation=getattr(self, "sft_ablation", "full"),
                conditional_matching=getattr(self, "conditional_matching", True),
                action_measure="root with declared task-specific action objective",
                action_objective=self.data.inventory["shared_action_objective"],
                basis_output=not self.data.local_boundary_conditions,
                student_source_reads=self.data.student_source_reads,
                information_flow="original X/P + fixed category scaffold -> shared continuous hidden computation "
                    "and independent LM readouts; raw sources remain visible; no categorical gold or sampled answers as inputs",
                sft_contexts=self.data.inventory["shared_audit_sft"]["read_context"],
                sft_budgets=self.data.inventory["shared_audit_sft"]["budgets"],
                local_semantic_targets=self.data.inventory["shared_audit_sft"]["local_target_counts"],
                stratified_exposure=self.data.inventory.get("stratified_exposure"),
                rollout_execution={"max_batch_tokens": self.rollout_max_batch_tokens,
                    "max_sequences": self.rollout_batch_size,
                    "frozen_image_cache_size": self.encoder.vision.maximum,
                    "active_during_sft": self.data.student_source_reads,
                    "distribution": "actual evidence prefixes plus declared output budgets; exchange indices only"},
                location_decoding="compact grammar; exact repeated object/free-description sentence ends as "
                    "explicit failed location, not repaired JSON; quoted source words are never penalized; "
                    "impossible exact-quote prefixes fail against the selected real source without completing JSON; "
                    "output frequency 0.1 inside each free region description only; "
                    "batched grammar masks; greedy sequence likelihood not computed",
                limitations="shared computation permits but does not prove causal diagnostic reliance; "
                    "local semantic targets are not certified action explanations; online read/stop policy not trained",
                stage2="not scheduled; requires the separate online correction design")
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
