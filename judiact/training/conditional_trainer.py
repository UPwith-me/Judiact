"""One-field conditional policy credit with live, supervised audit readouts."""

from collections import Counter
import math
import random
import time

import torch
from torch.nn.utils.rnn import pad_sequence

from ..shared_audit import SharedAuditNode
from ..short_review import REVIEW_KINDS, short_review_node
from .latent_data import generated_target
from .latent_trainer import configure_activation_recomputation
from .multipath_data import multipath_views
from .multipath_objective import action_loss, cispo_token_sum
from .read_replay import SourceReadReplay
from .short_sequence import partition_review_members, review_batches


ALGORITHM = "judiact-conditional-replacement"
CONTEXT_WEIGHTS = (.5, .25, .25)
COMPONENTS = ("action", "diagnostics", "local_fact", "local_rule", "local_boundary",
              "evidence", "short_review_evidence", "short_review_condition",
              "short_review_rule", "native", "relations", "risk_reader", "risk_actor")


class ConditionalReviewPlanner(SourceReadReplay):
    def __init__(self, data, encoder, model, update_scale, max_batch_tokens=28672,
                 rollout_batch_size=64, rollout_max_batch_tokens=131072, policy_gradient=True):
        super().__init__(data, encoder, update_scale, max_batch_tokens)
        self.model_wrapped = model
        self.rollout_batch_size = rollout_batch_size
        self.rollout_max_batch_tokens = rollout_max_batch_tokens
        self.policy_gradient = policy_gradient
        self.rollout_seconds_per_token = None
        self.cost_rows = {task: torch.tensor(table["normalized_cost"], device="cuda", dtype=torch.float32)
                          for task, table in data.multipath_tables.items()}
        self.loss_values = torch.zeros(len(COMPONENTS), device="cuda")
        self.policy_stats = torch.zeros(4, device="cuda")
        self.action_stats = torch.zeros(2, 4, 4, device="cuda")
        self.last_paths = []
        self.adaptive_recomputation = False

    def action_spec(self, sid):
        case = self.data.cases[sid]
        y = self.data.multipath_costs["labels"].index(case.get("sft_targets", case["targets"])["action"])
        task = case["task_type"]
        return task, y, self.data.multipath_tables[task]["gold_mass"][y]

    @torch.no_grad()
    def prepare_members(self, ids):
        started = time.perf_counter()
        world, rank = self.accelerator.num_processes, self.accelerator.process_index
        workers = [None] * world
        torch.distributed.all_gather_object(workers, sorted(ids))
        required = sorted({sid for worker in workers for sid in worker})
        nodes, costs = {}, []
        for sid in required:
            case = self.data.cases[sid]
            nodes[sid] = {kind: short_review_node(case["model_input"], case["task_type"], kind)
                          for kind in REVIEW_KINDS}
            costs.append((sum(self.encoder.prepare(n)["input_ids"].numel() +
                              self.encoder.short_max_new_tokens for n in nodes[sid].values()), sid))
        owners, loads = partition_review_members(costs, world, self.rollout_seconds_per_token)
        local = {sid: {"kind": random.choice(REVIEW_KINDS), "outputs": [{}, {}, {}]}
                 for sid in owners[rank]}
        model, previous = self.model_wrapped, self.model_wrapped.training
        model.eval()
        batch_sizes = []
        try:
            greedy = [(sid, kind, node) for sid in owners[rank] for kind, node in nodes[sid].items()]
            pending = [job[2] for job in greedy]
            for batch in review_batches(self.encoder, pending, self.rollout_batch_size,
                                        self.rollout_max_batch_tokens):
                batch_sizes.append(len(batch))
                for index, output in zip(batch, self.encoder.generate(model, [pending[i] for i in batch])):
                    sid, kind, _ = greedy[index]
                    local[sid]["outputs"][0][kind] = output
                    self.work_counts["greedy_short_items"] += 1
            for record in local.values():
                record["outputs"][1:] = [dict(record["outputs"][0]), dict(record["outputs"][0])]
            sampled = [(sid, k, nodes[sid][local[sid]["kind"]]) for sid in owners[rank] for k in (1, 2)]
            pending = [job[2] for job in sampled]
            for batch in review_batches(self.encoder, pending, self.rollout_batch_size,
                                        self.rollout_max_batch_tokens):
                batch_sizes.append(len(batch))
                for index, output in zip(batch, self.encoder.sample_reviews(model, [pending[i] for i in batch])):
                    sid, k, _ = sampled[index]
                    local[sid]["outputs"][k][local[sid]["kind"]] = output
                    self.work_counts["sampled_short_items"] += 1
                    self.work_counts["failed_sampled_short_items"] += bool(output.get("error"))
            generated = time.perf_counter()
            examples, keys = [], []
            for sid, record in local.items():
                record.update(H=[0., 0., 0.], action_logps=[None, None, None])
                for k, (_, observed) in enumerate(multipath_views(self.data, sid, record["outputs"], self.current_epoch)):
                    for example in observed:
                        if isinstance(example.node, SharedAuditNode):
                            examples.append(example)
                            keys.append((sid, k))
            for batch in self.batches(examples):
                results = self.encoder.probability_batch(model, [examples[i] for i in batch])
                for index, result in zip(batch, results):
                    sid, k = keys[index]
                    task, y, mass = self.action_spec(sid)
                    name = examples[index].node.name
                    if name == "action":
                        term = action_loss(result[1], y, self.cost_rows[task][y])
                        local[sid]["action_logps"][k] = result[1].cpu().tolist()
                    else:
                        a = self.data.member_sft_coefficient(sid, "action", 0, 0) / (4 * mass)
                        b = self.data.member_sft_coefficient(sid, name, 0, 0)
                        term = -result[0] * (b / a)
                    local[sid]["H"][k] += float(term)
            for sid, record in local.items():
                kind = record["kind"]
                # Validity describes only the replacement. Other frozen fields
                # may be wrong; a legal null is valid, never fabricated evidence.
                valid = [not record["outputs"][k][kind].get("error") and
                         record["outputs"][k][kind]["terminated"] for k in (1, 2)]
                q = torch.tensor(record["H"][1:], dtype=torch.float64).neg().exp()
                q *= torch.tensor(valid)
                record.update(Q=q.tolist(), valid=valid, advantages=[float(q[0] - q[1]), float(q[1] - q[0])])
        finally:
            model.train(previous)
        elapsed = time.perf_counter() - started
        workers = [None] * world
        torch.distributed.all_gather_object(workers, (local, elapsed, loads[rank]))
        old_rates = self.rollout_seconds_per_token or [None] * world
        self.rollout_seconds_per_token = [old if not n else t / n if old is None else .8 * old + .2 * t / n
            for old, (_, t, n) in zip(old_rates, workers)]
        self.members, self.last_paths = {}, []
        for rows, _, _ in workers:
            for sid, record in rows.items():
                if sid not in ids:
                    continue
                _, y, _ = self.action_spec(sid)
                prediction = torch.tensor(record["action_logps"]).argmax(-1)
                self.work_counts["retained_all_wrong_groups"] += bool((prediction != y).all())
                weights = torch.tensor(CONTEXT_WEIGHTS)
                self.members[sid] = {**record,
                    "views": multipath_views(self.data, sid, record["outputs"], self.current_epoch),
                    "weights": {name: weights for name in ("posterior", "prior", "latent_score")}}
                self.last_paths.append({"sample_id": sid, **record})
        self.rollout_timing = {"seconds": elapsed, "generation_seconds": generated - started,
            "audit_readout_seconds": time.perf_counter() - started - (generated - started),
            "rank_seconds": [t for _, t, _ in workers], "assigned_members": list(map(len, owners)),
            "batch_sizes": batch_sizes}

    def packet_jobs(self, packet):
        self.balanced_jobs = False
        try:
            jobs = super().packet_jobs(packet)
        finally:
            self.balanced_jobs = True
        for job in jobs:
            risk = []
            for i, (example, key, coefficient) in enumerate(zip(job["examples"], job["keys"], job["coefficients"])):
                if isinstance(example.node, SharedAuditNode):
                    sid, k, _ = key
                    # Generic packing divided categorical mass over 3 contexts.
                    job["coefficients"][i] = coefficient * 3 * CONTEXT_WEIGHTS[k]
                    risk.append(coefficient * 3 * .5 * self.members[sid]["Q"][k - 1] if k else 0.)
                else:
                    risk.append(0.)
            job["risk_coefficients"] = risk
        examples, old, advantages, coefficients, keys = [], [], [], [], []
        for sid in sorted(packet["absolute_ids"]) if self.policy_gradient else []:
            member, case = self.members[sid], self.data.cases[sid]
            _, _, mass = self.action_spec(sid)
            for k in (1, 2):
                advantage = member["advantages"][k - 1]
                if advantage == 0.:
                    continue  # Zero actor derivative, not exclusion from supervised learning.
                kind = member["kind"]
                output = member["outputs"][k][kind]
                examples.append(generated_target(sid, short_review_node(case["model_input"], case["task_type"], kind), output))
                old.append(output["sampling_token_log_probabilities"])
                advantages.append(advantage)
                coefficients.append(self.data.sft_coefficient(sid, "action", 0, 0) / (4 * mass * 2))
                keys.append((sid, k, "policy_" + kind))
        for batch in self.batches(examples):
            selected = [examples[i] for i in batch]
            targets = [len(self.encoder.target_ids(e)) for e in selected]
            lengths = [self.encoder.prepare(e.node)["input_ids"].numel() + n for e, n in zip(selected, targets)]
            jobs.append({"examples": selected, "keys": [keys[i] for i in batch],
                "coefficients": [coefficients[i] for i in batch], "relation": None,
                "path_old_logps": [old[i] for i in batch], "path_advantages": [advantages[i] for i in batch],
                "tensor_shape": (len(batch), max(lengths), max(targets))})
        self.members = {}
        return self.balance_packet_jobs(jobs)

    def supervised_nll(self, example, result):
        if example.node.name == "action":
            task, y, mass = self.action_spec(example.sample_id)
            return action_loss(result[1], y, self.cost_rows[task][y]) / (4 * mass)
        return -result[0]

    def compute_loss(self, model, job):
        count = self.checkpoint_layers if all(e.node.labels for e in job["examples"]) else self.text_checkpoint_layers
        rows, width, targets = job["tensor_shape"]
        if rows * width > self.max_batch_tokens or rows * targets > self.max_target_positions:
            count = None
        elif self.adaptive_recomputation:
            layers = len(model.get_base_model().model.language_model.layers)
            pressure = max(rows * width / self.max_batch_tokens,
                           rows * targets / self.max_target_positions)
            # Keep the existing full-batch uncheckpointed activation budget;
            # small source/target jobs can retain more layers instead of recomputing.
            count = max(0, layers - math.floor((layers - count) / pressure))
        self.work_counts["activation_recomputation_layers/" + str(count)] += 1
        configure_activation_recomputation(model, self.checkpoint_stride, count)
        if "path_old_logps" in job:
            current = self.encoder.token_log_probabilities_batch(model, job["examples"])
            old = [p.new_tensor(v) for p, v in zip(current, job["path_old_logps"])]
            assert all(p.shape == q.shape for p, q in zip(current, old))
            delta = torch.cat([p.detach() - q for p, q in zip(current, old)])
            self.policy_stats[0] = torch.maximum(self.policy_stats[0], delta.abs().max())
            self.policy_stats[1] += delta.abs().sum()
            self.policy_stats[2] += delta.numel()
            self.policy_stats[3] += (delta > .6931471805599453).sum()
            p, q = pad_sequence(current, batch_first=True), pad_sequence(old, batch_first=True)
            mask = pad_sequence([torch.ones_like(v, dtype=torch.bool) for v in current], batch_first=True)
            loss = cispo_token_sum(p, q, p.new_tensor(job["path_advantages"]), mask,
                                  p.new_tensor(job["coefficients"])) * self.update_scale
            self.loss_values[-1] += loss.detach()
            return loss
        results = self.encoder.probability_batch(model, job["examples"])
        loss = results[0][0] * 0.
        for example, result, weight, risk in zip(job["examples"], results, job["coefficients"], job["risk_coefficients"]):
            if not weight and not risk:
                continue
            value = self.supervised_nll(example, result)
            loss = loss + (weight + risk) * value
            name = example.node.name
            group = next((c for c in COMPONENTS if name == c or name.startswith(c + "_")), "diagnostics")
            self.loss_values[COMPONENTS.index(group)] += (weight * value * self.update_scale).detach()
            self.loss_values[-2] += (risk * value * self.update_scale).detach()
            self.work_counts["trained_node/" + name] += 1
            if name == "action":
                task, y, mass = self.action_spec(example.sample_id)
                coefficient = weight * self.update_scale / (4 * mass)
                stats = torch.stack((result[0].new_tensor(1.), -result[1][y],
                    (result[1].exp() * self.cost_rows[task][y]).sum(), (result[1].argmax() == y).float()))
                self.action_stats[int(task == "response_audit"), y] += coefficient * stats.detach()
        relations = job.get("relations", [] if job.get("relation") is None else [job["relation"]])
        for gold, weight, groups, mixing in relations:
            scores = [(torch.stack([results[i][1] for i in indices]) +
                       results[0][0].new_tensor(logweights)[:, None]).logsumexp(0)
                      for indices, logweights in zip(groups, mixing)]
            value = weight * self.relation_nll(scores, gold)
            loss = loss + value
            self.loss_values[COMPONENTS.index("relations")] += (value * self.update_scale).detach()
            self.work_counts["directional_components"] += 1
        return loss * self.update_scale
