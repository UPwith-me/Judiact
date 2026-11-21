"""Source-first SFT using the existing weighted, packed DDP training loop."""

import json
from pathlib import Path

import torch

from ..source_audit import PROTOCOL_VERSION
from .evidence_trainer import EvidenceAuditTrainer
from .source_data import source_views


class SourceAuditTrainer(EvidenceAuditTrainer):
    evaluation_protocol = PROTOCOL_VERSION
    predictor_type = None

    @torch.no_grad()
    def prepare_local_members(self, missing):
        for sid in missing:
            views = source_views(self.data.cases[sid], read_views=self.data.read_views)
            weights = torch.zeros(len(views))
            weights[0] = 1.
            self.members[sid] = {"views": views, "weights": {
                "posterior": weights, "prior": weights, "latent_score": weights}}

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        from ..evaluation import evaluate_natural, summarize_natural
        from ..source_predictor import SourcePredictor
        from ..io_utils import read_jsonl, write_json
        # BF16 dense merging measurably changed this adapter's decisions. Keep
        # the exact live BF16-base/FP32-LoRA representation for epoch selection.
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
                self.accelerator, self.encoder.max_new_tokens,
                predictor_type=self.predictor_type or SourcePredictor,
                protocol=self.evaluation_protocol)
        if self.is_world_process_zero():
            rows = [row for task in ("natural_request", "natural_response")
                    for row in read_jsonl(self.data.release / f"data/{task}/dev.jsonl")]
            supervision = {row["sample_id"]: row for row in read_jsonl(
                self.data.release / "reviews/supervision_metadata.jsonl") if row["split"] == "dev"}
            predictions = {row["sample_id"]: {"trace": row["trace"]["readback"]}
                           for row in read_jsonl(output / "predictions.jsonl")}
            report = summarize_natural(rows, predictions, supervision)
            report.update(protocol=self.evaluation_protocol, primary=False,
                decision_context="autonomous_source_read", backend="transformers_unmerged",
                note="own locations; not per-example oracle selection; read policy not yet trained")
            write_json(output / "readback_summary.json", report)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        model.train()
        return metrics

    def save_model(self, output_dir=None, _internal_call=False):
        super().save_model(output_dir, _internal_call)
        if self.is_world_process_zero():
            path = Path(output_dir or self.args.output_dir) / "judiact_config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config.update(protocol=PROTOCOL_VERSION, source_total=True,
                action_semantics="action_semantics", action_measure="root",
                root_action_balance=None, indexed=False,
                information_flow="original X,P -> independent judgments, locations and reviewed information-gap/revision object; no W/L or categorical gold parents",
                sft_contexts="original X/P only" if not self.data.read_views else "matched diagnostic: direct/read average",
                sft_budgets=self.data.inventory["source_first_sft"]["budgets"],
                relation_scope="original-source view only",
                limitations="boundary object is auxiliary supervision, not a proven causal mediator; autonomous read/stop policy not yet trained",
                dev_backend="transformers_unmerged",
                dev_precision="live BF16 base/activations with FP32 LoRA; no dense merge",
                dev_primary_path="original_source", dev_secondary_path="autonomous_source_read",
                stage2="not scheduled; old trajectory objectives are incompatible")
            path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
