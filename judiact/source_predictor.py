"""Direct adjudication plus a separately reported autonomous source-read path."""

from collections import defaultdict
import copy

from .audit_schema import audit_slots, visible_only
from .evidence_parser import parse_evidence
from .reference_audit import native_node
from .evidence_audit import materialize_index
from .source_audit import source_judgment_node
from .reference_predictor import AdjudicationPredictor


class SourcePredictor(AdjudicationPredictor):
    judgment_node = staticmethod(source_judgment_node)

    def forward_context_key(self, node):
        return None

    def __init__(self, *args, readback=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.readback = readback

    def predict_batch(self, model_inputs, task_types=None, sample=False, include_evidence=True,
                      text_batch_size=None, max_batch_tokens=None, include_behavior=True,
                      include_classifications=True, sample_classifications=True, include_basis=True):
        visibles = [visible_only(value) for value in model_inputs]
        tasks = [task or ("response_audit" if v["response_candidate"] is not None else "request_decision")
                 for v, task in zip(visibles, task_types or [None] * len(visibles))]
        traces = [{"task_type": task, "nodes": {}, "values": {}, "errors": [], "valid": False,
                   "decision_context": "original_source"} for task in tasks]
        text_batch_size = self.text_batch_size if text_batch_size is None else text_batch_size
        max_batch_tokens = self.max_batch_tokens if max_batch_tokens is None else max_batch_tokens

        def run(jobs, destinations):
            buckets = defaultdict(list)
            for index, node in jobs:
                length = self.encoder.prepare(node)["input_ids"].numel()
                buckets[bool(node.labels), bool(node.media), "" if node.labels else node.name].append((length, index, node))
            for (categorical, _, _), rows in buckets.items():
                rows.sort(key=lambda row: -row[0])
                maximum = self.class_batch_size if categorical else text_batch_size
                batches, batch, contexts = [], [], set()
                for row in rows:
                    key = self.forward_context_key(row[2])
                    key = key if key is not None else ("independent", id(row[2]))
                    count = len(contexts | {key})
                    if batch and (count > maximum or count * batch[0][0] > max_batch_tokens or
                            (not categorical and self.max_decode_tokens and
                             count * (batch[0][0] + self.encoder.max_new_tokens) > self.max_decode_tokens)):
                        batches.append(batch)
                        batch = []
                        contexts = set()
                    batch.append(row)
                    contexts.add(key)
                if batch:
                    batches.append(batch)
                for batch in batches:
                    outputs = self.encoder.generate(self.model, [row[2] for row in batch],
                        sample and (not categorical or sample_classifications))
                    for (_, index, node), result in zip(batch, outputs):
                        destinations[index]["nodes"][node.name] = result
                        if node.labels:
                            destinations[index]["values"][node.name] = result["value"]

        active = [i for i, task in enumerate(tasks) if not task.startswith("native_")]
        native = [i for i, task in enumerate(tasks) if task.startswith("native_")]
        run([(i, native_node(visibles[i], tasks[i])) for i in native], traces)
        for i in native:
            traces[i]["valid"] = traces[i]["nodes"][tasks[i]]["error"] is None
        classifications = [(i, slot.name) for i in active for slot in audit_slots(visibles[i], tasks[i])
                           if slot.kind == "classify" and (include_behavior or slot.name != "behavior")] \
            if include_classifications else []
        # No generated basis or annotation-selected location is a direct decision parent.
        run([(i, self.judgment_node(visibles[i], tasks[i], name)) for i, name in classifications], traces)
        locations = [(i, slot) for i in active for slot in audit_slots(visibles[i], tasks[i])
                     if slot.kind == "evidence"] if include_evidence else []
        run([(i, self.judgment_node(visibles[i], tasks[i], slot.name)) for i, slot in locations], traces)
        if include_basis:
            run([(i, self.judgment_node(visibles[i], tasks[i], "basis")) for i in active], traces)
        for i in active:
            trace = traces[i]
            trace["evidence"] = []
            outputs = {}
            for owner, slot in locations:
                if owner != i:
                    continue
                result = trace["nodes"][slot.name]
                outputs[slot.stage] = result.get("raw", "null") if not result["error"] else "null"
                parsed = parse_evidence(result.get("raw", "null"), visibles[i], slot.stage)
                if result["error"]:
                    parsed["errors"].append(result["error"])
                trace["evidence"].append(parsed)
            trace["evidence_index"], trace["index_errors"] = materialize_index(visibles[i], outputs)
            trace["valid"] = bool(include_classifications and trace["nodes"].get("action", {}).get("error") is None)
            trace["audit_complete"] = (trace["valid"] and not trace["index_errors"] and
                not any(node.get("error") for node in trace["nodes"].values()) and
                not any(item["errors"] for item in trace["evidence"]))
        if self.readback and include_evidence and include_classifications:
            read_traces = [copy.deepcopy(trace) for trace in traces]
            run([(i, self.judgment_node(visibles[i], tasks[i], name, traces[i]["evidence_index"]))
                 for i, name in classifications], read_traces)
            for i in active:
                read_traces[i]["decision_context"] = "autonomous_source_read"
                traces[i]["readback"] = read_traces[i]
        return traces
