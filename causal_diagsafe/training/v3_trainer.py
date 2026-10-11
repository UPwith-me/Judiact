"""HF Trainer owns optimization/checkpoints; stream small semantic backwards."""

import json
import time
from contextlib import nullcontext
from pathlib import Path

import torch
from torch.utils.data import SequentialSampler
from transformers import Trainer

from ..data_contract_v1 import load_relations_v1, load_s2_quartets_v1
from ..evaluation_v3 import summarize
from ..inference_v3 import SemanticPredictor
from ..io_utils import read_jsonl, write_json, write_jsonl
from .v1_data import root_assignments
from .v3_losses import job_loss, stream_jobs
from .v3_views import ViewEncoder


def evaluate_dev(model, processor, release, reviews, output, state, diagnostics=True, view_batch_size=4):
    directory, release = Path(output), Path(release)
    directory.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(release / "data/core_request/dev.jsonl") + read_jsonl(release / "data/s5_response/dev.jsonl")
    predictor = SemanticPredictor.from_model(model, processor, view_batch_size)
    started = time.perf_counter()
    model.eval()
    with (directory / f"predictions.rank{state.process_index}.jsonl").open("w", encoding="utf-8") as handle:
        for local, index in enumerate(range(state.process_index, len(rows), state.num_processes), 1):
            row = rows[index]
            result = predictor.predict(row["model_input"], row["sample_id"], include_evidence=diagnostics,
                                       include_diagnostics=diagnostics)
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            if local % 25 == 0:
                handle.flush()
                print(json.dumps({"dev_rank": state.process_index, "processed": local,
                                  "elapsed_seconds": time.perf_counter() - started}), flush=True)
    state.wait_for_everyone()
    metrics = None
    if state.is_main_process:
        predictions = {row["sample_id"]: row for rank in range(state.num_processes)
                       for row in read_jsonl(directory / f"predictions.rank{rank}.jsonl")}
        samples = {row["sample_id"]: row for row in rows}
        assert predictions.keys() == samples.keys()
        natural = {row["sample_id"] for name in ("natural_request", "natural_response")
                   for row in read_jsonl(release / "data" / name / "dev.jsonl")}
        edges = list(load_relations_v1(str(release / "data/causal/verified_dev.jsonl"), samples))
        quartets = list(load_s2_quartets_v1(str(release / "data/causal/verified_s2_quartets_dev.jsonl"), samples))
        report, groups = summarize(samples, predictions, natural, root_assignments(rows, reviews), edges, quartets)
        report["settings"] = {"split": "dev", "hidden_tests_loaded": False, "calibration": "none",
            "actions": "raw legal-set argmax; empty-deck runtime override is not applied to these metrics",
            "complete_audit": diagnostics, "view_batch_size": view_batch_size}
        request, response = (report["subsets"]["natural"][key]["NLL"]
                             for key in ("request_action", "response_publishability"))
        metrics = {"eval_loss": .5 * (request["value"] + response["value"]),
                   "eval_request_NLL": request["value"], "eval_response_NLL": response["value"],
                   "eval_request_denominator": request["denominator"], "eval_response_denominator": response["denominator"],
                   "eval_runtime": time.perf_counter() - started}
        write_json(directory / "summary.json", report)
        write_json(directory / "selection_metrics.json", metrics)
        write_jsonl(directory / "predictions.jsonl", (predictions[row["sample_id"]] for row in rows))
        write_jsonl(directory / "mechanism_groups.jsonl", groups)
        errors = []
        for row in rows:
            key = "response_publishability" if row["task_type"] == "response_audit" else "request_action"
            if row["gold"][key] != predictions[row["sample_id"]][key]:
                errors.append({"sample_id": row["sample_id"], "task": key, "gold": row["gold"][key],
                               "prediction": predictions[row["sample_id"]][key]})
        write_jsonl(directory / "decision_errors.jsonl", errors)
        print(json.dumps({"dev_output": str(directory), **metrics}), flush=True)
    if state.num_processes > 1:
        values = [metrics]
        torch.distributed.broadcast_object_list(values, src=0)
        metrics = values[0]
    return metrics


class SemanticTrainer(Trainer):
    def __init__(self, *args, release, reviews, relation_coefficient=1., view_batch_size=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.release, self.reviews = release, reviews
        self.relation_coefficient, self.view_batch_size = relation_coefficient, view_batch_size
        self.encoder = ViewEncoder(self.processing_class)
        self.model_accepts_loss_kwargs = False
        self.forward_views = 0

    def _get_train_sampler(self, train_dataset=None):
        return SequentialSampler(train_dataset if train_dataset is not None else self.train_dataset)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = job_loss(model, inputs, self.encoder, self.relation_coefficient, self.view_batch_size)
        return (loss, {"loss": loss}) if return_outputs else loss

    def training_step(self, model, inputs, num_items_in_batch=None):
        jobs = list(stream_jobs(inputs, self.view_batch_size))
        total = torch.zeros((), device=self.args.device)
        # Forward AND backward stay in no_sync until the last real job. All language
        # LoRA parameters participate there; variable numbers of views cannot deadlock DDP.
        for index, job in enumerate(jobs):
            context = model.no_sync() if index < len(jobs) - 1 and hasattr(model, "no_sync") else nullcontext()
            with context:
                total += super().training_step(model, job, num_items_in_batch=None)
            self.forward_views += len(job["items"]) if job["kind"] == "classify_batch" \
                else len(job["specs"]) if job["kind"] == "contrast" else 1
        return total

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        model = self.accelerator.unwrap_model(self.model_wrapped)
        epoch = round(self.state.epoch or 0)
        metrics = evaluate_dev(model, self.processing_class, self.release, self.reviews,
            Path(self.args.output_dir) / "dev_eval" / f"epoch{epoch}", self.accelerator,
            diagnostics=True, view_batch_size=self.view_batch_size)
        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        model.train()
        return metrics
