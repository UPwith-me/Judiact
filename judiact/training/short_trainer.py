"""SFT on actual student short reviews, with separately supervised references."""

from functools import partial
import json
from pathlib import Path
import time

import torch

from ..short_review import PROTOCOL_VERSION, REVIEW_KINDS, short_review_node, review_context
from ..short_predictor import ShortReviewPredictor
from ..io_utils import read_jsonl, write_json
from .shared_trainer import SharedAuditTrainer
from .short_sequence import ShortReviewEncoder, partition_review_members, review_batches


class ShortReviewTrainer(SharedAuditTrainer):
    evaluation_protocol = PROTOCOL_VERSION

    def __init__(self, *args, short_max_new_tokens=512, **kwargs):
        super().__init__(*args, **kwargs)
        previous = self.encoder
        self.encoder = ShortReviewEncoder(self.processing_class, previous.max_new_tokens,
            cache_size=previous.maximum, probability_chunk_size=previous.probability_chunk_size,
            witness_frequency_penalty=0., compact_witness_json=True,
            short_max_new_tokens=short_max_new_tokens)
        self.encoder.vision.maximum = previous.vision.maximum
        self.predictor_type = partial(ShortReviewPredictor, short_max_new_tokens=short_max_new_tokens,
            action_objective=self.data.action_objective, short_batch_size=self.rollout_batch_size,
            short_max_batch_tokens=self.rollout_max_batch_tokens)
        self.rollout_seconds_per_token = None
        self.rollout_timing = {}
        self.short_history_epoch = None
        self.component_names = ("action", "diagnostics", "local_fact", "local_rule", "local_boundary",
                                "evidence", "short_review_evidence", "short_review_rule",
                                "short_review_condition", "native")
        self._component_values = torch.zeros(len(self.component_names), device=self.args.device)
        self._terms = []

    def remember_review(self, sid, outputs):
        views = self.data.short_views(sid, review_context(outputs), self.current_epoch or 0)
        weights = torch.tensor([1.])
        self.members[sid] = {"views": views, "weights": {
            "posterior": weights, "prior": weights, "latent_score": weights}}

    def restore_reviews(self):
        if self.short_history_epoch == self.current_epoch:
            return
        self.short_history_epoch = self.current_epoch
        for path in sorted(Path(self.args.output_dir).glob(
                f"short_reviews.epoch{self.current_epoch + 1}.rank*.jsonl")):
            for row in read_jsonl(path):
                if row["step"] < self.state.global_step and row["sample_id"] not in self.members:
                    self.remember_review(row["sample_id"], row["outputs"])

    @torch.no_grad()
    def prepare_members(self, ids):
        self.rollout_timing = {}
        self.restore_reviews()
        missing = sorted(set(ids) - self.members.keys())
        world, rank = self.accelerator.num_processes, self.accelerator.process_index
        if world > 1:
            requested = [None] * world
            torch.distributed.all_gather_object(requested, missing)
            missing = sorted({sid for group in requested for sid in group})
        if not missing:
            return
        nodes_by_sid, costs = {}, []
        for sid in missing:
            case = self.data.cases[sid]
            nodes = [short_review_node(case["model_input"], case["task_type"], kind) for kind in REVIEW_KINDS]
            nodes_by_sid[sid] = nodes
            costs.append((sum(self.encoder.prepare(n)["input_ids"].numel() +
                              self.encoder.short_max_new_tokens for n in nodes), sid))
        owners, loads = partition_review_members(costs, world,
            self.rollout_seconds_per_token if self.time_balanced_jobs else None)
        jobs = [(sid, kind, node) for sid in owners[rank]
                for kind, node in zip(REVIEW_KINDS, nodes_by_sid[sid])]
        nodes = [job[2] for job in jobs]
        outputs = {sid: {} for sid in owners[rank]}
        model = self.accelerator.unwrap_model(self.model_wrapped)
        started = time.perf_counter()
        batch_count = 0
        for batch in review_batches(self.encoder, nodes, self.rollout_batch_size, self.rollout_max_batch_tokens):
            generated = self.encoder.generate(model, [nodes[j] for j in batch])
            batch_count += 1
            for j, result in zip(batch, generated):
                sid, kind, _ = jobs[j]
                outputs[sid][kind] = result
                self.work_counts["short_review_generated/" + kind] += 1
                self.work_counts["short_review_tokens/" + kind] += result["tokens"]
                self.work_counts["short_review_errors/" + kind] += bool(result["error"])
                self.work_counts["short_review_null/" + kind] += result["value"] is None and not result["error"]
        seconds = time.perf_counter() - started
        history = Path(self.args.output_dir) / (
            f"short_reviews.epoch{self.current_epoch + 1}.rank{rank}.jsonl")
        with history.open("a", encoding="utf-8") as handle:
            for sid, result in outputs.items():
                handle.write(json.dumps({"sample_id": sid, "step": self.state.global_step,
                    "outputs": result, "origin": "current_student_no_gold_input"}, ensure_ascii=False) + "\n")
        if world > 1:
            gathered = [None] * world
            torch.distributed.all_gather_object(gathered, (outputs, seconds, loads[rank]))
        else:
            gathered = [(outputs, seconds, loads[rank])]
        old_rates = self.rollout_seconds_per_token or [None] * world
        rates = []
        for old, (_, elapsed, tokens) in zip(old_rates, gathered):
            measured = elapsed / tokens if tokens else old
            rates.append(measured if old is None or measured is None else .8 * old + .2 * measured)
        self.rollout_seconds_per_token = rates
        self.rollout_timing = {"local_seconds": seconds, "local_nodes": len(nodes),
            "local_batches": batch_count, "assigned_members": list(map(len, owners)),
            "rank_seconds": [item[1] for item in gathered], "seconds_per_token": rates}
        for worker, _, _ in gathered:
            for sid, result in worker.items():
                self.remember_review(sid, result)

    def supervised_nll(self, example, result):
        loss = super().supervised_nll(example, result)
        self._terms.append(loss.detach())
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._terms = []
        result = super().compute_loss(model, inputs, return_outputs, num_items_in_batch)
        selected = [(e, c) for e, c in zip(inputs["examples"], inputs["coefficients"]) if c]
        assert len(selected) == len(self._terms)
        if selected:
            names = []
            for example, _ in selected:
                name = example.node.name
                group = next((g for g in self.component_names if name == g or name.startswith(g + "_")),
                             "diagnostics")
                names.append(self.component_names.index(group))
            terms = torch.stack(self._terms)
            coefficients = terms.new_tensor([c * self.update_scale for _, c in selected])
            self._component_values.index_add_(0, torch.tensor(names, device=terms.device), terms * coefficients)
        self._terms = []
        return result

    def training_step(self, model, inputs, num_items_in_batch=None):
        self._component_values.zero_()
        loss = super().training_step(model, inputs, num_items_in_batch)
        values = torch.cat((self._component_values, loss.detach().reshape(1)))
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(values)
            values /= self.accelerator.num_processes
        if self.is_world_process_zero():
            values = values.cpu().tolist()
            row = dict(zip(self.component_names, values[:-1]))
            row.update(step=self.state.global_step + 1, total=values[-1],
                       relations=values[-1] - sum(values[:-1]),
                       meaning="actual weighted loss contributions in this update; dropout active")
            with (Path(self.args.output_dir) / "short_loss.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
        return loss

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        from ..evaluation import evaluate_natural
        self._save_checkpoint(self.model_wrapped, trial=None)
        self.accelerator.wait_for_everyone()
        model = self.accelerator.unwrap_model(self.model_wrapped)
        model.zero_grad(set_to_none=True)
        self.encoder.vision.values.clear()
        torch.cuda.empty_cache()
        model.eval()
        output = Path(self.args.output_dir) / "dev_selection" / f"epoch{round(self.state.epoch or 0)}"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            metrics = evaluate_natural(model, self.processing_class, self.data.release, output,
                self.accelerator, self.encoder.max_new_tokens, predictor_type=self.predictor_type,
                protocol=PROTOCOL_VERSION)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        model.train()
        return metrics

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        if self.is_world_process_zero():
            path = Path(output_dir or self.args.output_dir) / "judiact_config.json"
            config = json.loads(path.read_text())
            config.update(protocol=PROTOCOL_VERSION, short_review=self.data.inventory["short_review"],
                short_max_new_tokens=self.encoder.short_max_new_tokens,
                short_condition_decoding="kind before text; compact JSON; text-only additive frequency 0.1; "
                    "exact repeated complete condition sentence ends as error; no reference repair",
                sft_budgets=self.data.inventory["short_review"]["budgets"],
                information_flow="original X/P + own three short review proposals -> fixed categorical "
                    "scaffold; gold short references have a separate supervised forward only",
                sft_contexts="first-visit current student per epoch; original sources retained; "
                    "no filtering of null, malformed or truncated proposals",
                relation_scope="same autonomous short-review contexts; original relation targets/weights unchanged"
                    if getattr(self, "conditional_matching", True) else
                    "matching loss disabled; absolute member supervision and own-review contexts retained",
                dev_primary_path="original_source_and_own_short_review", dev_secondary_path=None,
                rollout_execution={"max_sequences": self.rollout_batch_size,
                    "max_prefill_and_decode_tokens": self.rollout_max_batch_tokens,
                    "time_balanced_members": self.time_balanced_jobs,
                    "active_during_sft": True, "compact_json": True,
                    "tail_compaction_after_tokens": 192,
                    "condition_counts": "sparse FP32, same output-only penalty",
                    "grammar_tokens": "reuse CPU tokens from stopping criteria"},
                stage2="not automatically scheduled; first-stage complete-epoch selection only")
            write_json(path, config)
