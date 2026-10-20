"""Autonomous V7 inference. No annotation object is accepted by this interface."""

from __future__ import annotations

import copy
from collections import defaultdict

from .audit_v4 import audit_slots, visible_only
from .audit_v5 import parse_evidence
from .audit_v7 import (
    INITIAL_REQUEST, RULE_FIELDS, diagnostic_node, diagnostic_record,
    materialize_links, materialize_witness, native_node, restrict_evidence_to_parents,
    source_identity, source_node,
)
from .training.v7_sequence import NodeEncoder


class AdjudicationPredictor:
    def __init__(self, model, processor, max_new_tokens=2048, class_batch_size=4, encoder=None,
                 source_cache=None):
        self.model = model
        self.encoder = encoder or NodeEncoder(processor, max_new_tokens)
        self.class_batch_size = class_batch_size
        self.source_cache = source_cache

    def predict(self, model_input, task_type=None, sample=False, include_evidence=True):
        return self.predict_batch([model_input], [task_type], sample, include_evidence)[0]

    def predict_batch(self, model_inputs, task_types=None, sample=False, include_evidence=True,
                      text_batch_size=8, max_batch_tokens=24576):
        """Batch independent nodes; each dependent call still starts a fresh context."""
        visibles = [visible_only(value) for value in model_inputs]
        tasks = [task or ("response_audit" if v["response_candidate"] is not None else "request_decision")
                 for v, task in zip(visibles, task_types or [None] * len(visibles))]
        traces = [{"task_type": task, "nodes": {}, "values": {}, "errors": [], "valid": False}
                  for task in tasks]

        def run(jobs):
            buckets = defaultdict(list)
            for index, node in jobs:
                length = self.encoder.prepare(node)["input_ids"].numel()
                buckets[bool(node.labels), bool(node.media)].append((length, index, node))
            for (categorical, _), rows in buckets.items():
                rows.sort(key=lambda item: -item[0])
                maximum = self.class_batch_size if categorical else text_batch_size
                batches, batch = [], []
                for row in rows:
                    if batch and (len(batch) >= maximum or
                                  (len(batch) + 1) * batch[0][0] > max_batch_tokens):
                        batches.append(batch)
                        batch = []
                    batch.append(row)
                if batch:
                    batches.append(batch)
                for batch in batches:
                    outputs = self.encoder.generate(self.model, [row[2] for row in batch], sample)
                    for (_, index, node), value in zip(batch, outputs):
                        traces[index]["nodes"][node.name] = value
                        if node.labels:
                            traces[index]["values"][node.name] = value["value"]

        native = [i for i, task in enumerate(tasks) if task.startswith("native_")]
        run([(i, native_node(visibles[i], tasks[i])) for i in native])
        for i in native:
            traces[i]["valid"] = traces[i]["nodes"][tasks[i]]["error"] is None
        active = [i for i in range(len(tasks)) if i not in native]
        jobs = []
        for i in active:
            key = (tasks[i], source_identity(visibles[i]))
            cached = self.source_cache.get(key) if self.source_cache is not None and not sample else None
            traces[i]["source_reused"] = cached is not None
            if cached is not None:
                traces[i]["nodes"]["witness"] = copy.deepcopy(cached)
            else:
                jobs.append((i, source_node(visibles[i], tasks[i])))
        run(jobs)
        for i in active:
            trace, visible = traces[i], visibles[i]
            result = trace["nodes"]["witness"]
            if self.source_cache is not None and not sample:
                self.source_cache[tasks[i], source_identity(visible)] = copy.deepcopy(result)
            if result["error"]:
                trace["errors"].append("witness:" + result["error"])
                continue
            try:
                trace["witness"] = materialize_witness(visible, result["value"])
            except (ValueError, KeyError, TypeError) as error:
                trace["errors"].append("witness:" + str(error))
            trace["values"]["empty_request_deck"] = tasks[i] != "response_audit" and not visible["policy_deck"]
        active = [i for i in active if not traces[i]["errors"]]

        def node(i, name):
            return diagnostic_node(visibles[i], tasks[i], name, traces[i]["witness"],
                                   traces[i]["values"], traces[i].get("links"))

        run([(i, node(i, name)) for i in active
             for name in (("E5", "F5") if tasks[i] == "response_audit" else INITIAL_REQUEST)])
        run([(i, node(i, "links")) for i in active])
        for i in active:
            result = traces[i]["nodes"]["links"]
            if result["error"]:
                traces[i]["errors"].append("links:" + result["error"])
                continue
            try:
                traces[i]["links"] = materialize_links(
                    visibles[i]["policy_deck"], traces[i]["witness"], result["value"])
            except (ValueError, KeyError, TypeError) as error:
                traces[i]["errors"].append("links:" + str(error))
        active = [i for i in active if not traces[i]["errors"]]
        run([(i, node(i, name)) for i in active if tasks[i] != "response_audit" for name in RULE_FIELDS])
        run([(i, node(i, "action")) for i in active])
        run([(i, node(i, "behavior")) for i in active])
        evidence = [(i, slot) for i in active for slot in audit_slots(visibles[i], tasks[i])
                    if slot.kind == "evidence"] if include_evidence else []
        run([(i, node(i, slot.name)) for i, slot in evidence])
        for i in active:
            trace = traces[i]
            trace["diagnostic_record"] = diagnostic_record(
                trace["witness"], trace["links"], trace["values"], tasks[i])
            trace["evidence"] = []
            trace["valid"] = True
        for i, slot in evidence:
            result = traces[i]["nodes"][slot.name]
            parsed = parse_evidence(result.get("raw", "null"), visibles[i], slot.stage)
            if result["error"]:
                parsed["errors"].append(result["error"])
            traces[i]["evidence"].append(restrict_evidence_to_parents(parsed, traces[i]["witness"]))
        return traces
