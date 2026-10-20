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

from ..audit_v7 import PROTOCOL_VERSION, diagnostic_node, materialize_witness, source_node
from .v5_matching import conditional_matching_nll
from .v7_latent_data import candidate_views, identity
from .v7_latent_objective import candidate_weights, stratified_log_measure
from .v7_sequence import NodeEncoder


def configure_activation_recomputation(model, stride):
    layers = model.get_base_model().model.language_model.layers
    for index, layer in enumerate(layers):
        layer.gradient_checkpointing = bool(stride and index % stride == 0)
    return [index for index, layer in enumerate(layers) if layer.gradient_checkpointing]


class LatentAuditTrainer(Trainer):
    def __init__(self, *args, audit_data, update_scale, phase, anchor=None, student_draws=2,
                 rollout_batch_size=16, class_batch_size=16, text_batch_size=2,
                 native_batch_size=32, max_batch_tokens=24576, max_new_tokens=2048,
                 checkpoint_stride=1, **kwargs):
        Trainer.__init__(self, *args, **kwargs)
        self.data, self.update_scale, self.phase, self.anchor = audit_data, update_scale, phase, anchor
        self.encoder = NodeEncoder(self.processing_class, max_new_tokens=max_new_tokens, cache_size=256)
        self.student_draws, self.rollout_batch_size = student_draws, rollout_batch_size
        self.class_batch_size, self.text_batch_size = class_batch_size, text_batch_size
        self.native_batch_size, self.max_batch_tokens = native_batch_size, max_batch_tokens
        self.checkpoint_stride = checkpoint_stride
        self.model_accepts_loss_kwargs = False
        self.current_epoch, self.members = None, {}
        self.work_counts = Counter()
        self.last_posterior = []

    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        from ..evaluation_v7 import evaluate_natural
        model = self.accelerator.unwrap_model(self.model_wrapped)
        output = Path(self.args.output_dir) / "dev_selection" / f"epoch{round(self.state.epoch or 0)}"
        metrics = evaluate_natural(model, self.processing_class, self.data.release, output,
                                   self.accelerator, self.encoder.max_new_tokens)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        model.train()
        return metrics

    def batches(self, examples, rollout=False, native=False):
        buckets, lengths = defaultdict(list), {}
        for index, example in enumerate(examples):
            node = example if rollout else example.node
            size = self.encoder.prepare(node)["input_ids"].numel()
            target_size = 0 if rollout else len(self.encoder.target_ids(example))
            lengths[index] = size + target_size
            buckets[bool(node.labels), bool(node.media),
                    0 if node.labels else target_size.bit_length()].append(index)
        for (categorical, _, _), ids in sorted(buckets.items()):
            maximum = self.rollout_batch_size if rollout else self.native_batch_size if native else \
                self.class_batch_size if categorical else self.text_batch_size
            ids.sort(key=lambda i: -lengths[i])
            group = []
            for index in ids:
                if group and (len(group) >= maximum or
                              (len(group) + 1) * lengths[group[0]] > self.max_batch_tokens):
                    yield group
                    group = []
                group.append(index)
            if group:
                yield group

    def propose(self, model, ids):
        pending, nodes, generated = [], [], defaultdict(list)
        for sid in ids:
            case = self.data.cases[sid]
            for draw in range(self.student_draws):
                pending.append((sid, draw))
                nodes.append(source_node(case["model_input"], case["task_type"]))
                generated[sid].append(None)
        w_outputs = [None] * len(nodes)
        for batch in self.batches(nodes, rollout=True):
            values = self.encoder.generate(model, [nodes[i] for i in batch], sample=True)
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
            values = self.encoder.generate(model, [link_nodes[i] for i in batch], sample=True)
            for i, value in zip(batch, values):
                sid, draw = link_indices[i]
                generated[sid][draw]["links"] = value
        return generated

    @torch.no_grad()
    def prepare_members(self, ids):
        missing = sorted(set(ids) - self.members.keys())
        if not missing:
            return
        if self.phase == "sft":
            for sid in missing:
                row = self.data.bank.get(sid)
                if row is None:
                    self.members[sid] = None
                    self.work_counts["missing_bank_members"] += 1
                    continue
                prior, observed, error = candidate_views(self.data.cases[sid], row["candidate"])
                if error:
                    self.members[sid] = None
                    self.work_counts["unsupported_bank/" + error] += 1
                    continue
                self.members[sid] = {"views": [(prior, observed)], "weights": {
                    "posterior": torch.ones(1), "prior": torch.ones(1), "latent_score": torch.ones(1)}}
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
            observed_count = max(1, sum(not key.startswith("evidence_") for key in self.data.cases[sid]["targets"]))
            for k, (prior, observed) in enumerate(member["views"]):
                # Invalid paths still participate in the prior-score control variate.
                for role, examples in (("prior", prior), ("observed", observed)):
                    for example in examples:
                        key = (sid, k, example.node.name)
                        views[key] = example
                        if self.phase == "sft":
                            denominator = len(examples) * (1 if example.node.labels else len(self.encoder.target_ids(example)))
                            coefficient = root_mass / max(1, denominator)
                        else:
                            coefficient = root_mass * float(weights["latent_score" if role == "prior" else "posterior"][k]) / observed_count
                        coefficients[key] = coefficient if sid in packet["absolute_ids"] else 0.
                        marginal_weights[key] = float(weights["prior"][k])
        absolute = {key for key, coefficient in coefficients.items() if coefficient != 0.}
        jobs, covered = [], set()

        def append(keys, relation=None):
            own = set(keys) & absolute - covered
            covered.update(own)
            jobs.append({"examples": [views[key] for key in keys], "keys": keys,
                         "coefficients": [coefficients[key] if key in own else 0. for key in keys],
                         "relation": relation})

        for experiment in packet["experiments"]:
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
        # Keep the real native batch last for a common DDP synchronization boundary.
        natives = sorted(packet["native_ids"])
        examples = [self.data.native_example(sid) for sid in natives]
        for batch in self.batches(examples, native=True):
            ids = [natives[i] for i in batch]
            jobs.append({"examples": [examples[i] for i in batch],
                         "keys": [(sid, 0, "native") for sid in ids],
                         "coefficients": [self.data.root_coefficient(sid) for sid in ids], "relation": None})
        return jobs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        configure_activation_recomputation(self.accelerator.unwrap_model(model), self.checkpoint_stride)
        results = self.encoder.probability_batch(model, inputs["examples"])
        loss = results[0][0] * 0
        for example, (logp, _), coefficient in zip(inputs["examples"], results, inputs["coefficients"]):
            if coefficient:
                loss = loss - coefficient * logp
                self.work_counts["trained_node/" + example.node.name] += 1
        if inputs["relation"] is not None:
            gold, weight, groups, mixing = inputs["relation"]
            scores = []
            for indices, logweights in zip(groups, mixing):
                values = torch.stack([results[i][1] for i in indices])
                scores.append((values + values.new_tensor(logweights)[:, None]).logsumexp(0))
            loss = loss + weight * conditional_matching_nll(scores, gold)
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
        for i, job in enumerate(jobs):
            context = model.no_sync() if i < len(jobs) - 1 and hasattr(model, "no_sync") else nullcontext()
            with context:
                total += Trainer.training_step(self, model, job, num_items_in_batch=None)
        path = Path(self.args.output_dir) / f"work.rank{self.accelerator.process_index}.jsonl"
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"step": self.state.global_step + 1, "epoch": epoch + 1,
                "phase": self.phase, "preparation_seconds": prepared - started,
                "forward_backward_seconds": time.perf_counter() - prepared, "backward_batches": len(jobs),
                "cumulative": dict(self.work_counts), "vision_cache": self.encoder.vision.summary(),
                "peak_allocated_GiB": torch.cuda.max_memory_allocated() / 1024**3}) + "\n")
        return total

    def save_model(self, output_dir=None, _internal_call=False):
        Trainer.save_model(self, output_dir, _internal_call)
        if self.is_world_process_zero():
            path = Path(output_dir or self.args.output_dir)
            (path / "v7_config.json").write_text(json.dumps({
                "protocol": PROTOCOL_VERSION, "phase": self.phase, "student_draws": self.student_draws,
                "proposal": "exact teacher-bank atom plus importance-sampled complement; sampling-time probabilities",
                "latent_gradient": "detached SNIS posterior minus prior score control variate",
                "observation_gradient": "detached SNIS posterior weighted joint observed likelihood",
                "root_action_balance": self.data.action_balance,
                "max_new_tokens": self.encoder.max_new_tokens, "checkpoint_stride": self.checkpoint_stride,
                "information_flow": "X->W; W,P->rule links; observed diagnostics and D->A; no final-action feedback",
                "bank_inventory": self.data.inventory["v7_latent_bank"],
                "relation_scope": "prior-weighted available candidate contexts; existing conditional matching",
                "limitations": "finite-candidate bias; teacher weak supervision; no EM monotonicity guarantee",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
