"""Autonomous short review followed by the unchanged categorical audit scaffold."""

from .audit_schema import visible_only
from .reference_audit import canonical
from .short_review import REVIEW_KINDS, short_review_node, short_judgment_node, review_context
from .shared_predictor import SharedAuditPredictor
from .training.short_sequence import ShortReviewEncoder, review_batches


class ShortReviewPredictor(SharedAuditPredictor):
    def __init__(self, model, processor, max_new_tokens=4096, encoder=None,
                 short_max_new_tokens=512, short_batch_size=None, short_max_batch_tokens=None,
                 candidate_head_projection=False, short_compact_after=192, **kwargs):
        encoder = encoder or ShortReviewEncoder(processor, max_new_tokens, cache_size=512,
            witness_frequency_penalty=0., compact_witness_json=True,
            short_max_new_tokens=short_max_new_tokens, candidate_head_projection=candidate_head_projection,
            short_compact_after=short_compact_after)
        kwargs.pop("readback", None)
        kwargs.pop("include_basis", None)
        super().__init__(model, processor, max_new_tokens, encoder=encoder,
                         readback=False, include_basis=False, **kwargs)
        self.active_reviews = {}
        self.short_batch_size = short_batch_size
        self.short_max_batch_tokens = short_max_batch_tokens

    def judgment_node(self, visible, task, name, index=None):
        return short_judgment_node(visible, task, name, self.active_reviews[canonical(visible), task])

    def predict_batch(self, model_inputs, task_types=None, sample=False, **kwargs):
        visibles = [visible_only(value) for value in model_inputs]
        tasks = [task or ("response_audit" if v["response_candidate"] is not None else "request_decision")
                 for v, task in zip(visibles, task_types or [None] * len(visibles))]
        outputs = [{} for _ in visibles]
        jobs = [(i, kind, short_review_node(v, task, kind))
                for i, (v, task) in enumerate(zip(visibles, tasks))
                if not task.startswith("native_") for kind in REVIEW_KINDS]
        nodes = [job[2] for job in jobs]
        for batch in review_batches(self.encoder, nodes,
                self.short_batch_size or kwargs.get("text_batch_size") or self.text_batch_size,
                self.short_max_batch_tokens or kwargs.get("max_batch_tokens") or self.max_batch_tokens):
            generated = self.encoder.generate(self.model, [nodes[i] for i in batch], sample=sample)
            for j, result in zip(batch, generated):
                owner, kind, _ = jobs[j]
                outputs[owner][kind] = result
        self.active_reviews = {(canonical(v), task): review_context(output)
                               for v, task, output in zip(visibles, tasks, outputs) if output}
        try:
            traces = super().predict_batch(model_inputs, tasks, sample=sample, **kwargs)
            for trace, output in zip(traces, outputs):
                if output:
                    trace["short_review"] = output
                    trace["decision_context"] = "original_source_and_own_short_review"
                    errors = [{"node": "short_review_" + kind, "error": result["error"]}
                              for kind, result in output.items() if result.get("error")]
                    trace["errors"].extend(errors)
                    trace["audit_complete"] &= not errors
            return traces
        finally:
            self.active_reviews = {}
