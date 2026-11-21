"""Shared-LoRA weak SFT and finite-candidate marginal learning on the same DAG."""

from collections import Counter, defaultdict
from contextlib import nullcontext
import json
import math
from pathlib import Path
import time

import torch
from torch.utils.data import SequentialSampler
from transformers import Trainer

from ..reference_audit import DEFAULT_MAX_NEW_TOKENS, PROTOCOL_VERSION, diagnostic_node, materialize_witness, source_node
from .conditional_matching import conditional_matching_nll
from .latent_data import candidate_views, identity, source_context_views
from .latent_objective import candidate_weights, stratified_log_measure
from .sequence import NodeEncoder


def configure_activation_recomputation(model, stride, checkpoint_layers=None):
    layers = model.get_base_model().model.language_model.layers
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = bool(stride and index % stride == 0) if checkpoint_layers is None else \
            (index + 1) * checkpoint_layers // len(layers) > index * checkpoint_layers // len(layers)
    return [index for index, layer in enumerate(layers) if layer.gradient_checkpointing]


class LatentAuditTrainer(Trainer):
    def __init__(self, *args, audit_data, update_scale, phase, anchor=None, student_draws=2,
                 rollout_batch_size=16, class_batch_size=16, text_batch_size=2,
                 native_batch_size=32, max_batch_tokens=24576, max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                 checkpoint_stride=1, packed_jobs=False, balanced_jobs=False,
                 probability_chunk_size=16, max_target_positions=0, checkpoint_layers=None,
                 witness_recovery=False, recovery_seed=42, witness_frequency_penalty=.1,
                 compact_witness_json=True, sft_objective="field_budget",
                 sft_source_roll_in=False, **kwargs):
        Trainer.__init__(self, *args, **kwargs)
        self.data, self.update_scale, self.phase, self.anchor = audit_data, update_scale, phase, anchor
        self.encoder = NodeEncoder(self.processing_class, max_new_tokens=max_new_tokens, cache_size=256,
                                   probability_chunk_size=probability_chunk_size,
                                   witness_frequency_penalty=witness_frequency_penalty,
                                   compact_witness_json=compact_witness_json)
        self.student_draws, self.rollout_batch_size = student_draws, rollout_batch_size
        self.class_batch_size, self.text_batch_size = class_batch_size, text_batch_size
        self.native_batch_size, self.max_batch_tokens = native_batch_size, max_batch_tokens
        self.max_target_positions = max_target_positions
        self.checkpoint_stride = checkpoint_stride
        self.checkpoint_layers = checkpoint_layers
        self.packed_jobs = packed_jobs
        self.balanced_jobs = balanced_jobs
        self.witness_recovery, self.recovery_seed = witness_recovery, recovery_seed
        self.sft_objective = sft_objective
        self.sft_source_roll_in = sft_source_roll_in
        if sft_source_roll_in and (phase != "sft" or sft_objective != "field_budget"):
            raise ValueError("source-context SFT uses the field budget, not a latent density")
        if witness_recovery and phase != "sft":
            raise ValueError("reference-derived prefix recovery is weak SFT, not a latent density")
        self.model_accepts_loss_kwargs = False
        self.current_epoch, self.members = None, {}
        self.work_counts = Counter()
        self.last_posterior = []
        self.diagnostic_context_counts = {}

    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        from ..evaluation import evaluate_natural
        model = self.accelerator.unwrap_model(self.model_wrapped)
        output = Path(self.args.output_dir) / "dev_selection" / f"epoch{round(self.state.epoch or 0)}"
        metrics = evaluate_natural(model, self.processing_class, self.data.release, output,
                                   self.accelerator, self.encoder.max_new_tokens,
                                   witness_frequency_penalty=self.encoder.witness_frequency_penalty,
                                   compact_witness_json=self.encoder.compact_witness_json)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        model.train()
        return metrics

    def batches(self, examples, rollout=False, native=False):
        buckets, lengths, target_lengths = defaultdict(list), {}, {}
        for index, example in enumerate(examples):
            node = example if rollout else example.node
            size = self.encoder.prepare(node)["input_ids"].numel()
            if not rollout:
                size += len(example.prefix_ids or [])
            target_size = 0 if rollout else len(self.encoder.target_ids(example))
            lengths[index] = size + target_size
            target_lengths[index] = target_size
            buckets[bool(node.labels), bool(node.media),
                    0 if node.labels else target_size.bit_length()].append(index)
        for (categorical, _, _), ids in sorted(buckets.items()):
            maximum = self.rollout_batch_size if rollout else self.native_batch_size if native else \
                self.class_batch_size if categorical else self.text_batch_size
            ids.sort(key=lambda i: -lengths[i])
            group, longest_target = [], 0
            for index in ids:
                if group and (len(group) >= maximum or
                              (len(group) + 1) * lengths[group[0]] > self.max_batch_tokens or
                              (not rollout and self.max_target_positions and
                               (len(group) + 1) * max(longest_target, target_lengths[index]) > self.max_target_positions)):
                    yield group
                    group = []
                    longest_target = 0
                group.append(index)
                longest_target = max(longest_target, target_lengths[index])
            if group:
                yield group

    def propose(self, model, ids, sample=True, draws=None):
        pending, nodes, generated = [], [], defaultdict(list)
        for sid in ids:
            case = self.data.cases[sid]
            for draw in range(self.student_draws if draws is None else draws):
                pending.append((sid, draw))
                nodes.append(source_node(case["model_input"], case["task_type"]))
                generated[sid].append(None)
        w_outputs = [None] * len(nodes)
        if self.sft_source_roll_in and not sample:
            # Policy variants share the exact policy-blind W input. Stochastic
            # latent proposals must remain independent and never use this cache.
            keys = [(node.text, node.media) for node in nodes]
            missing = {}
            for key, node in zip(keys, nodes):
                missing.setdefault(key, node)
            unique = list(missing.items())
            source_outputs = {}
            for batch in self.batches([node for _, node in unique], rollout=True):
                values = self.encoder.generate(model, [unique[i][1] for i in batch], sample=False)
                for i, value in zip(batch, values):
                    source_outputs[unique[i][0]] = value
            w_outputs = [source_outputs[key] for key in keys]
            self.work_counts["sft_witness_generated_sources"] += len(unique)
            self.work_counts["sft_witness_source_reuses"] += len(nodes) - len(unique)
        else:
            for batch in self.batches(nodes, rollout=True):
                values = self.encoder.generate(model, [nodes[i] for i in batch], sample=sample)
                for i, value in zip(batch, values):
                    w_outputs[i] = value
        link_nodes, link_indices = [], []
        for (sid, draw), output in zip(pending, w_outputs):
            case = self.data.cases[sid]
            generated[sid][draw] = {"witness": output, "links": None}
            if output["error"]:
                continue
            try:
                witness = materialize_witness(case["model_input"], output["value"])
            except (ValueError, KeyError, TypeError):
                continue
            link_indices.append((sid, draw))
            link_nodes.append(diagnostic_node(case["model_input"], case["task_type"], "links", witness))
        for batch in self.batches(link_nodes, rollout=True):
            values = self.encoder.generate(model, [link_nodes[i] for i in batch], sample=sample)
            for i, value in zip(batch, values):
                sid, draw = link_indices[i]
                generated[sid][draw]["links"] = value
        return generated

    @torch.no_grad()
    def prepare_members(self, ids):
        missing = sorted(set(ids) - self.members.keys())
        if (self.phase == "latent" or self.sft_source_roll_in) and self.balanced_jobs and self.accelerator.num_processes > 1:
            self.prepare_balanced_members(missing)
            return
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
            length = self.encoder.prepare(source_node(case["model_input"], case["task_type"]))["input_ids"].numel()
            row = self.data.bank.get(sid)
            if row and row.get("candidate"):
                length += sum(len(self.encoder.helper.token_ids(json.dumps(output["value"], ensure_ascii=False)))
                              for output in row["candidate"].values() if output and output.get("value") is not None)
            costs.append((length, sid))
        assigned, loads = [[] for _ in range(world)], [0] * world
        for cost, sid in sorted(costs, reverse=True):
            owner = min(range(world), key=loads.__getitem__)
            assigned[owner].append(sid)
            loads[owner] += cost
        self.prepare_local_members(assigned[rank])
        produced = {sid: self.members[sid] for sid in assigned[rank]}
        returned = [None] * world
        torch.distributed.all_gather_object(returned, produced)
        # The epoch-wide CPU cache is identical on every rank: one sampled
        # candidate set and posterior per member, even when several fields reuse it.
        for worker in returned:
            self.members.update(worker)
        self.work_counts["executed_latent_members"] += len(assigned[rank])

    @torch.no_grad()
    def prepare_local_members(self, missing):
        if not missing:
            return
        if self.phase == "sft":
            for sid in missing:
                row = self.data.bank.get(sid)
                if row is None or row.get("candidate") is None:
                    self.members[sid] = None
                    self.work_counts["missing_bank_members"] += 1
                    continue
                prior, observed, error = candidate_views(self.data.cases[sid], row["candidate"])
                if error is None and any(len(self.encoder.target_ids(e)) > self.encoder.max_new_tokens
                                         for e in prior):
                    error = "reference_path_outside_student_budget"
                if error:
                    self.members[sid] = None
                    self.work_counts["unsupported_bank/" + error] += 1
                    continue
                self.members[sid] = {"views": [(prior, observed)], "weights": {
                    "posterior": torch.ones(1), "prior": torch.ones(1), "latent_score": torch.ones(1)}}
            supported = [sid for sid in missing if self.members[sid] is not None]
            if self.sft_source_roll_in and supported:
                model = self.accelerator.unwrap_model(self.model_wrapped)
                generated = self.propose(model, supported, sample=False, draws=1)
                path = Path(self.args.output_dir) / f"sft_contexts.epoch{self.current_epoch + 1}.rank{self.accelerator.process_index}.jsonl"
                with path.open("a", encoding="utf-8") as handle:
                    for sid in supported:
                        candidate = generated[sid][0]
                        observed, error = source_context_views(self.data.cases[sid], candidate)
                        if observed:
                            member = self.members[sid]
                            member["views"].append(([], observed))
                            member["weights"] = {name: torch.tensor([1., 0.]) for name in member["weights"]}
                        self.work_counts["sft_student_context_members"] += 1
                        self.work_counts["sft_student_context_fields"] += len(observed)
                        if error:
                            self.work_counts["sft_student_context_error/" + error] += 1
                        handle.write(json.dumps({"sample_id": sid, "step": self.state.global_step,
                            "candidate": candidate, "supervised_fields": [e.node.name for e in observed],
                            "error": error, "scope": "first-visit student context cached within epoch; not gold"},
                            ensure_ascii=False) + "\n")
            return
        with self.anchor.reference() as frozen:
            generated = self.propose(frozen, missing)
            all_examples, keys, descriptions = [], [], {}
            for sid in missing:
                row = self.data.bank.get(sid)
                bank = row.get("candidate") if row and not row.get("error") else None
                if bank is not None:
                    bank_prior, _, bank_error = candidate_views(self.data.cases[sid], bank)
                    if bank_error or any(len(self.encoder.target_ids(e)) > self.encoder.max_new_tokens
                                         for e in bank_prior):
                        bank = None
                        self.work_counts["bank_atom_outside_support"] += 1
                candidates = ([bank] if bank is not None else []) + generated[sid]
                views, errors = [], []
                for index, candidate in enumerate(candidates):
                    prior, observed, error = candidate_views(self.data.cases[sid], candidate)
                    views.append((prior, observed if not error else []))
                    errors.append(error)
                    for role, examples in (("prior", prior), ("observed", observed if not error else [])):
                        all_examples.extend(examples)
                        keys.extend((sid, index, role) for _ in examples)
                descriptions[sid] = (candidates, views, errors, bank is not None)
            logps = defaultdict(float)
            for batch in self.batches(all_examples):
                scored = self.encoder.probability_batch(frozen, [all_examples[i] for i in batch])
                for i, (value, _) in zip(batch, scored):
                    logps[keys[i]] += float(value)
            reports = []
            for sid in missing:
                candidates, views, errors, has_bank = descriptions[sid]
                lp = torch.tensor([logps[sid, k, "prior"] for k in range(len(candidates))], dtype=torch.float64)
                ly = torch.tensor([logps[sid, k, "observed"] if not errors[k] else -math.inf
                                   for k in range(len(candidates))], dtype=torch.float64)
                if has_bank:
                    bank_key = identity(candidates[0], self.encoder, self.data.cases[sid])
                    matches = [identity(c, self.encoder, self.data.cases[sid]) == bank_key
                               for c in candidates[1:]]
                else:
                    matches = [False] * len(candidates)
                stochastic = candidates[1:] if has_bank else candidates
                proposal = [sum(c[name]["log_probability"] for name in ("witness", "links")
                                if c.get(name) is not None) for c in stochastic]
                measure = stratified_log_measure(proposal, matches, has_bank)
                for k, (prior, _) in enumerate(views):
                    if any(len(self.encoder.target_ids(e)) > self.encoder.max_new_tokens for e in prior):
                        lp[k] = -math.inf
                        errors[k] = "outside_declared_generation_budget"
                weights = candidate_weights(lp, ly, log_measure=measure)
                self.members[sid] = None if weights is None else {"views": views, "weights": weights}
                self.work_counts["candidate_paths"] += len(candidates)
                self.work_counts["invalid_candidate_paths"] += sum(error is not None for error in errors)
                self.work_counts["no_observed_support_members"] += weights is None
                reports.append({"sample_id": sid,
                    "log_prior": [x if math.isfinite(x) else None for x in lp.tolist()],
                    "log_observed": [x if math.isfinite(x) else None for x in ly.tolist()],
                    "sampled_log_q": proposal,
                    "log_quadrature_measure": [x if math.isfinite(x) else None for x in measure.tolist()],
                    "errors": errors, "posterior": weights["posterior"].tolist() if weights else None,
                    "prior_weights": weights["prior"].tolist() if weights else None,
                    "effective_candidates": weights["effective_candidates"] if weights else 0,
                    "max_weight": weights["max_weight"] if weights else None})
                # Records are retained as machine proposals, never human-final annotations.
                path = Path(self.args.output_dir) / f"candidates.epoch{self.current_epoch + 1}.rank{self.accelerator.process_index}.jsonl"
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps({"sample_id": sid, "candidates": candidates}, ensure_ascii=False) + "\n")
            path = Path(self.args.output_dir) / f"posterior.epoch{self.current_epoch + 1}.rank{self.accelerator.process_index}.jsonl"
            with path.open("a", encoding="utf-8") as handle:
                for report in reports:
                    handle.write(json.dumps(report, ensure_ascii=False, allow_nan=False) + "\n")
            self.last_posterior = reports

    def packet_jobs(self, packet):
        required = set(packet["absolute_ids"]) | {sid for e in packet["experiments"] for sid in e.sample_ids}
        self.prepare_members(required)
        views, coefficients, marginal_weights = {}, {}, {}
        for sid in sorted(required):
            member = self.members[sid]
            if member is None:
                continue
            weights = member["weights"]
            root_mass = self.data.root_coefficient(sid)
            reference_prior, reference_observed = member["views"][0]
            context_counts = Counter(e.node.name for p, o in member["views"] for e in p + o)
            for k, (prior, observed) in enumerate(member["views"]):
                # Invalid paths still participate in the prior-score control variate.
                for role, examples in (("prior", prior), ("observed", observed)):
                    for example in examples:
                        key = (sid, k, example.node.name)
                        views[key] = example
                        if self.phase == "sft":
                            tokens = 1 if example.node.labels else len(self.encoder.target_ids(example))
                            if self.sft_objective == "field_budget":
                                coefficient = self.data.sft_coefficient(
                                    sid, example.node.name, len(reference_prior),
                                    sum(e.node.name != "action" for e in reference_observed)) / tokens
                            else:
                                coefficient = root_mass / (len(examples) * tokens)
                        else:
                            coefficient = root_mass * float(weights["latent_score" if role == "prior" else "posterior"][k])
                        coefficients[key] = coefficient if sid in packet["absolute_ids"] else 0.
                        marginal_weights[key] = float(weights["prior"][k])
                        # Visited-context CE reuses the online categorical forward.
                        # Reference relations retain their original context and mass.
                        if self.phase == "sft":
                            coefficients[key] /= context_counts[example.node.name]
                            coefficients[key] /= 1 + self.diagnostic_context_counts.get(
                                (sid, example.node.name), 0)
                        if self.witness_recovery and role == "prior" and example.node.name == "witness":
                            from .recovery import recovery_views
                            recovered = recovery_views(example, self.encoder, self.recovery_seed,
                                                       self.current_epoch or 0, length_match=True)
                            if recovered:
                                original_coefficient = coefficients[key]
                                coefficients[key] *= .8
                                for extra in recovered:
                                    extra_key = (sid, k, extra.view)
                                    views[extra_key] = extra
                                    coefficients[extra_key] = (
                                        original_coefficient * len(self.encoder.target_ids(example)) * .2
                                        / (len(recovered) * len(self.encoder.target_ids(extra)))
                                        if sid in packet["absolute_ids"] else 0.)
                                    marginal_weights[extra_key] = float(weights["prior"][k])
        absolute = {key for key, coefficient in coefficients.items() if coefficient != 0.}
        jobs, covered = [], set()

        def append(keys, relation=None):
            own = set(keys) & absolute - covered
            covered.update(own)
            jobs.append({"examples": [views[key] for key in keys], "keys": keys,
                         "coefficients": [coefficients[key] if key in own else 0. for key in keys],
                         "relation": relation})

        for experiment in packet["experiments"] if getattr(self, "conditional_matching", True) else []:
            for field in experiment.fields:
                weight = self.data.event_weights[experiment.key, field.slot]
                if not weight:
                    continue
                for component in field.components:
                    if component.is_constant:
                        continue
                    rows, groups, mix = [], [], []
                    for sid in component.sample_ids:
                        keys = sorted(key for key in views if key[0] == sid and key[2] == field.slot)
                        keys = [key for key in keys if marginal_weights[key] > 0]
                        if not keys:
                            break
                        groups.append(list(range(len(rows), len(rows) + len(keys))))
                        rows.extend(keys)
                        values = torch.tensor([marginal_weights[key] for key in keys], dtype=torch.float64)
                        mix.append((values / values.sum()).log().tolist())
                    if len(groups) == len(component.sample_ids):
                        append(rows, (component.gold_indices, weight, groups, mix))
                    else:
                        self.work_counts["relations_without_candidate_context"] += 1
        remaining = sorted(absolute - covered)
        for indices in self.batches([views[key] for key in remaining]):
            append([remaining[i] for i in indices])
        assert covered == absolute
        if self.packed_jobs:
            jobs = self.pack_core_jobs(jobs)
        # Keep the real native batch last for a common DDP synchronization boundary.
        natives = sorted(packet["native_ids"])
        examples = [self.data.native_example(sid) for sid in natives]
        for batch in self.batches(examples, native=True):
            ids = [natives[i] for i in batch]
            jobs.append({"examples": [examples[i] for i in batch],
                         "keys": [(sid, 0, "native") for sid in ids],
                         "coefficients": [self.data.root_coefficient(sid) for sid in ids], "relation": None})
        for job in jobs:
            targets = [len(self.encoder.target_ids(e)) for e in job["examples"]]
            lengths = [self.encoder.prepare(e.node)["input_ids"].numel() + len(e.prefix_ids or []) + target
                       for e, target in zip(job["examples"], targets)]
            job["tensor_shape"] = (len(lengths), max(lengths), max(targets))
        return self.balance_packet_jobs(jobs) if self.balanced_jobs else jobs

    def balance_packet_jobs(self, jobs):
        """Redistribute complete loss components within the same global update."""
        world, rank = self.accelerator.num_processes, self.accelerator.process_index
        if world == 1:
            return jobs
        native, local = [], []
        for job in jobs:
            if all(e.node.name.startswith("native_") for e in job["examples"]):
                native.append(job)
                continue
            rows, length, _ = job["tensor_shape"]
            local.append((rows * length, job))
        gathered = [None] * world
        torch.distributed.all_gather_object(gathered, local)
        bins, costs = [[] for _ in range(world)], [0] * world
        for cost, job in sorted((item for worker in gathered for item in worker), key=lambda x: -x[0]):
            owner = min(range(world), key=costs.__getitem__)
            bins[owner].append(job)
            costs[owner] += cost
        # Native work remains local and last, preserving the real DDP sync call.
        return bins[rank] + native

    def pack_core_jobs(self, jobs):
        """Batch independent components; retain every row, coefficient and relation."""
        buckets, sizes, targets = defaultdict(list), {}, {}
        for index, job in enumerate(jobs):
            lengths = [self.encoder.prepare(e.node)["input_ids"].numel() + len(e.prefix_ids or [])
                       + len(self.encoder.target_ids(e))
                       for e in job["examples"]]
            categorical = all(e.node.labels for e in job["examples"])
            targets[index] = max(len(self.encoder.target_ids(e)) for e in job["examples"])
            buckets[categorical, any(e.node.media for e in job["examples"]),
                    0 if categorical else targets[index].bit_length()].append(index)
            sizes[index] = max(lengths)
        result = []
        for (categorical, _, _), indices in sorted(buckets.items()):
            maximum = self.class_batch_size if categorical else self.text_batch_size
            indices.sort(key=lambda i: -sizes[i])
            merged = None
            longest = longest_target = 0
            for index in indices:
                job = jobs[index]
                rows = len(job["examples"])
                if merged is not None and (len(merged["examples"]) + rows > maximum or
                                           (len(merged["examples"]) + rows) * longest > self.max_batch_tokens or
                                           (self.max_target_positions and
                                            (len(merged["examples"]) + rows) * max(longest_target, targets[index]) > self.max_target_positions)):
                    result.append(merged)
                    merged = None
                if merged is None:
                    merged = {"examples": [], "keys": [], "coefficients": [], "relations": []}
                    longest = sizes[index]
                    longest_target = 0
                longest_target = max(longest_target, targets[index])
                offset = len(merged["examples"])
                for field in ("examples", "keys", "coefficients"):
                    merged[field].extend(job[field])
                if job["relation"] is not None:
                    gold, weight, groups, mixing = job["relation"]
                    merged["relations"].append((gold, weight,
                                                [[i + offset for i in group] for group in groups], mixing))
            if merged is not None:
                result.append(merged)
        return result

    def relation_nll(self, scores, gold):
        return conditional_matching_nll(scores, gold)

    def supervised_nll(self, example, result):
        return -result[0]

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        stride, count = self.checkpoint_stride, self.checkpoint_layers
        shape = inputs.get("tensor_shape")
        if count is not None and (shape is None or shape[0] * shape[1] > self.max_batch_tokens or
                                  (self.max_target_positions and shape[0] * shape[2] > self.max_target_positions)):
            stride, count = 1, None
        layers = configure_activation_recomputation(self.accelerator.unwrap_model(model), stride, count)
        self.work_counts[f"recompute_layers/{len(layers)}"] += 1
        results = self.encoder.probability_batch(model, inputs["examples"])
        loss = results[0][0] * 0
        for example, result, coefficient in zip(inputs["examples"], results, inputs["coefficients"]):
            if coefficient:
                loss = loss + coefficient * self.supervised_nll(example, result)
                self.work_counts["trained_node/" + example.node.name] += 1
                if example.prefix_ids is not None:
                    self.work_counts["trained_view/" + example.view] += 1
        relations = inputs.get("relations")
        if relations is None:
            relations = [] if inputs["relation"] is None else [inputs["relation"]]
        for gold, weight, groups, mixing in relations:
            scores = []
            for indices, logweights in zip(groups, mixing):
                values = torch.stack([results[i][1] for i in indices])
                scores.append((values + values.new_tensor(logweights)[:, None]).logsumexp(0))
            loss = loss + weight * self.relation_nll(scores, gold)
            self.work_counts["directional_components"] += 1
        loss *= self.update_scale
        return (loss, {"loss": loss}) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        epoch = math.floor(self.state.epoch or 0)
        if epoch != self.current_epoch:
            self.current_epoch, self.members = epoch, {}
            if self.phase == "latent":
                self.anchor.refresh()
        started = time.perf_counter()
        with self.compute_loss_context_manager():
            jobs = self.packet_jobs(inputs)
        prepared = time.perf_counter()
        total = torch.zeros((), device=self.args.device)
        timed = []
        for i, job in enumerate(jobs):
            context = model.no_sync() if i < len(jobs) - 1 and hasattr(model, "no_sync") else nullcontext()
            measure = getattr(self, "time_balanced_jobs", False) and i < len(jobs) - 1
            if measure:
                begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                begin.record()
            with context:
                total += Trainer.training_step(self, model, job, num_items_in_batch=None)
            if measure:
                end.record()
                rows, length, _ = job["tensor_shape"]
                timed.append((begin, end, rows * length))
        if timed:
            self.observe_execution(timed)
        path = Path(self.args.output_dir) / f"work.rank{self.accelerator.process_index}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": self.state.global_step + 1, "epoch": epoch + 1,
                "phase": self.phase, "preparation_seconds": prepared - started,
                "forward_backward_seconds": time.perf_counter() - prepared, "backward_batches": len(jobs),
                "cumulative": dict(self.work_counts), "vision_cache": self.encoder.vision.summary(),
                **({"rollout_timing": self.rollout_timing} if hasattr(self, "rollout_timing") else {}),
                **({"execution_balance": self.execution_report} if timed else {}),
                "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3}) + "\n")
        return total

    def save_model(self, output_dir=None, _internal_call=False):
        Trainer.save_model(self, output_dir, _internal_call)
        if self.is_world_process_zero():
            path = Path(output_dir or self.args.output_dir)
            (path / "judiact_config.json").write_text(json.dumps({
                "protocol": PROTOCOL_VERSION, "phase": self.phase, "student_draws": self.student_draws,
                "proposal": "exact fixed-reference atom plus importance-sampled complement; sampling-time probabilities",
                "latent_gradient": "detached SNIS posterior minus prior score control variate",
                "observation_gradient": "detached SNIS posterior weighted joint observed likelihood",
                "latent_root_normalization": "root-weighted joint likelihood; no per-field divisor",
                "root_action_balance": self.data.action_balance,
                "sft_objective": self.sft_objective,
                "sft_source_roll_in": self.sft_source_roll_in,
                "sft_contexts": "equal field mass across reference and first-visit greedy student W/L; source labels only; reference-only action and relations"
                    if self.sft_source_roll_in else "reference contexts only",
                "sft_source_cache": "same-source greedy W shared within a single current-model proposal call; no cross-update reuse beyond existing member cache; L remains policy-specific; stochastic proposals remain independent"
                    if self.sft_source_roll_in else None,
                "sft_field_balance": self.data.sft_balance,
                "max_new_tokens": self.encoder.max_new_tokens, "checkpoint_stride": self.checkpoint_stride,
                "checkpoint_layers": self.checkpoint_layers,
                "checkpoint_scope": "partial recomputation within configured tensor budgets; full for oversized components",
                "packed_jobs": self.packed_jobs,
                "balanced_jobs": self.balanced_jobs,
                "witness_recovery": self.witness_recovery,
                "recovery_weight": "80% full W; 10% stop; 10% continue" if self.witness_recovery else None,
                "probability_chunk_size": self.encoder.probability_chunk_size,
                "max_target_positions": self.max_target_positions,
                "visual_cache": "canonical single-image frozen readouts; missing images only",
                "information_flow": "X->W; X,W,P->rule links and source-grounded diagnostics; W,L,C,selected rule text->D->A; no raw-source action input or final-action feedback",
                "bank_inventory": self.data.inventory["reference_bank"],
                "relation_scope": "prior-weighted available candidate contexts; existing conditional matching",
                "limitations": "finite-candidate bias; teacher weak supervision; no EM monotonicity guarantee",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
