"""Autonomous audit inference. No annotation object is accepted by this interface."""

from __future__ import annotations

import copy
from collections import defaultdict

from .audit_schema import audit_slots, visible_only
from .evidence_parser import parse_evidence
from .reference_audit import (
    DEFAULT_MAX_NEW_TOKENS, INITIAL_REQUEST, RULE_FIELDS, diagnostic_node, diagnostic_record,
    materialize_links, materialize_witness, native_node, restrict_evidence_to_parents,
    source_identity, source_node,
)
from .training.sequence import NodeEncoder


class AdjudicationPredictor:
    evidence_first = False
    source_total = False
    witness_repair = False
    action_semantics = False

    def __init__(self, model, processor, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, class_batch_size=64, encoder=None,
                 source_cache=None, text_batch_size=32, max_batch_tokens=65536, max_decode_tokens=327680,
                 track_greedy_probability=False, witness_frequency_penalty=.1, compact_witness_json=True):
        self.model = model
        self.encoder = encoder or NodeEncoder(processor, max_new_tokens, cache_size=512,
                                              witness_frequency_penalty=witness_frequency_penalty,
                                              compact_witness_json=compact_witness_json)
        if encoder is None:
            self.encoder.vision.maximum = 128
        self.encoder.track_greedy_probability = track_greedy_probability
        self.class_batch_size = class_batch_size
        self.text_batch_size = text_batch_size
        self.max_batch_tokens = max_batch_tokens
        self.max_decode_tokens = max_decode_tokens
        self.source_cache = source_cache

    def predict(self, model_input, task_type=None, sample=False, include_evidence=True):
        return self.predict_batch([model_input], [task_type], sample, include_evidence)[0]

    def predict_batch(self, model_inputs, task_types=None, sample=False, include_evidence=True,
                      text_batch_size=None, max_batch_tokens=None, include_behavior=True,
                      include_classifications=True, sample_classifications=True):
        """Batch independent nodes; each dependent call still starts a fresh context."""
        visibles = [visible_only(value) for value in model_inputs]
        tasks = [task or ("response_audit" if v["response_candidate"] is not None else "request_decision")
                 for v, task in zip(visibles, task_types or [None] * len(visibles))]
        traces = [{"task_type": task, "nodes": {}, "values": {}, "errors": [], "valid": False}
                  for task in tasks]
        text_batch_size = self.text_batch_size if text_batch_size is None else text_batch_size
        max_batch_tokens = self.max_batch_tokens if max_batch_tokens is None else max_batch_tokens

        def run(jobs):
            buckets = defaultdict(list)
            for index, node in jobs:
                length = self.encoder.prepare(node)["input_ids"].numel()
                buckets[bool(node.labels), bool(node.media), "" if node.labels else node.name].append((length, index, node))
            for (categorical, _, _), rows in buckets.items():
                rows.sort(key=lambda item: -item[0])
                maximum = self.class_batch_size if categorical else text_batch_size
                batches, batch = [], []
                for row in rows:
                    if batch and (len(batch) >= maximum or
                                  (len(batch) + 1) * batch[0][0] > max_batch_tokens or
                                  (not categorical and self.max_decode_tokens and
                                   (len(batch) + 1) * (batch[0][0] + self.encoder.max_new_tokens) > self.max_decode_tokens)):
                        batches.append(batch)
                        batch = []
                    batch.append(row)
                if batch:
                    batches.append(batch)
                for batch in batches:
                    outputs = self.encoder.generate(self.model, [row[2] for row in batch],
                        sample and (not categorical or sample_classifications))
                    for (_, index, node), value in zip(batch, outputs):
                        traces[index]["nodes"][node.name] = value
                        if node.labels:
                            traces[index]["values"][node.name] = value["value"]

        native = [i for i, task in enumerate(tasks) if task.startswith("native_")]
        run([(i, native_node(visibles[i], tasks[i])) for i in native])
        for i in native:
            traces[i]["valid"] = traces[i]["nodes"][tasks[i]]["error"] is None
        active = [i for i in range(len(tasks)) if i not in native]
        jobs, pending_sources, source_aliases = [], {}, {}
        for i in active:
            key = (tasks[i], source_identity(visibles[i]))
            cached = self.source_cache.get(key) if self.source_cache is not None and not sample else None
            traces[i]["source_reused"] = cached is not None
            if cached is not None:
                traces[i]["nodes"]["witness"] = copy.deepcopy(cached)
            elif self.source_cache is not None and not sample and key in pending_sources:
                source_aliases[i] = pending_sources[key]
                traces[i]["source_reused"] = True
            else:
                jobs.append((i, source_node(visibles[i], tasks[i])))
                pending_sources[key] = i
        run(jobs)
        for index, owner in source_aliases.items():
            traces[index]["nodes"]["witness"] = copy.deepcopy(traces[owner]["nodes"]["witness"])
        if self.witness_repair:
            from .evidence_audit import witness_repair_node
            for i in active:
                traces[i]["draft_witness_output"] = copy.deepcopy(traces[i]["nodes"]["witness"])
            run([(i, witness_repair_node(visibles[i], tasks[i], traces[i]["draft_witness_output"]))
                 for i in active])
        for i in active:
            trace, visible = traces[i], visibles[i]
            result = trace["nodes"]["witness"]
            trace["values"]["empty_request_deck"] = tasks[i] != "response_audit" and not visible["policy_deck"]
            if self.source_cache is not None and not sample:
                self.source_cache[tasks[i], source_identity(visible)] = copy.deepcopy(
                    trace.get("draft_witness_output", result))
            if result["error"]:
                trace["errors"].append("witness:" + result["error"])
                if self.source_total:
                    trace["witness"] = materialize_witness(visible, {"facts": []})
                continue
            try:
                trace["witness"] = materialize_witness(visible, result["value"])
            except (ValueError, KeyError, TypeError) as error:
                trace["errors"].append("witness:" + str(error))
                if self.source_total:
                    trace["witness"] = materialize_witness(visible, {"facts": []})
        active = [i for i in active if self.source_total or not traces[i]["errors"]]

        def node(i, name):
            if self.evidence_first and name != "links":
                from .evidence_audit import evidence_node
                if not self.indexed_reader:
                    return evidence_node(visibles[i], tasks[i], name,
                        materialize_witness(visibles[i], {"facts": []}), [], {"spans": [], "regions": []},
                        self.action_semantics)
                return evidence_node(visibles[i], tasks[i], name, traces[i]["witness"],
                    traces[i].get("links", []), traces[i].get("evidence_index", {"spans": [], "regions": []}),
                    self.action_semantics)
            return diagnostic_node(visibles[i], tasks[i], name, traces[i]["witness"],
                                   traces[i]["values"], traces[i].get("links"))

        if not self.evidence_first and include_classifications:
            run([(i, node(i, name)) for i in active
                 for name in (("E5", "F5") if tasks[i] == "response_audit" else INITIAL_REQUEST)])
        run([(i, node(i, "links")) for i in active])
        for i in active:
            result = traces[i]["nodes"]["links"]
            if result["error"]:
                traces[i]["errors"].append("links:" + result["error"])
                if self.source_total:
                    traces[i]["links"] = []
                continue
            try:
                traces[i]["links"] = materialize_links(
                    visibles[i]["policy_deck"], traces[i]["witness"], result["value"])
            except (ValueError, KeyError, TypeError) as error:
                traces[i]["errors"].append("links:" + str(error))
                if self.source_total:
                    traces[i]["links"] = []
        active = [i for i in active if self.source_total or not traces[i]["errors"]]
        evidence = [(i, slot) for i in active for slot in audit_slots(visibles[i], tasks[i])
                    if slot.kind == "evidence"] if include_evidence or self.evidence_first else []
        if self.evidence_first:
            from .evidence_audit import materialize_index
            run([(i, node(i, slot.name)) for i, slot in evidence])
            for i in active:
                outputs = {slot.stage: traces[i]["nodes"][slot.name].get("raw", "null")
                           for j, slot in evidence if i == j}
                index, errors = materialize_index(visibles[i], outputs)
                traces[i]["evidence_index"] = index
                traces[i]["index_errors"] = errors
            if include_classifications:
                run([(i, node(i, name)) for i in active
                     for name in (("E5", "F5") if tasks[i] == "response_audit" else INITIAL_REQUEST)])
        if include_classifications:
            run([(i, node(i, name)) for i in active if tasks[i] != "response_audit" for name in RULE_FIELDS])
            run([(i, node(i, "action")) for i in active])
        if include_behavior and include_classifications:
            run([(i, node(i, "behavior")) for i in active])
        if not self.evidence_first:
            run([(i, node(i, slot.name)) for i, slot in evidence])
        for i in active:
            trace = traces[i]
            if self.evidence_first:
                trace["diagnostic_record"] = node(i, "action").payload["record"]
            else:
                trace["diagnostic_record"] = diagnostic_record(
                    trace["witness"], trace["links"], trace["values"], tasks[i], visibles[i]["policy_deck"])
            trace["evidence"] = []
            trace["valid"] = include_classifications
        for i, slot in evidence:
            result = traces[i]["nodes"][slot.name]
            parsed = parse_evidence(result.get("raw", "null"), visibles[i], slot.stage)
            if result["error"]:
                parsed["errors"].append(result["error"])
            traces[i]["evidence"].append(restrict_evidence_to_parents(parsed, traces[i]["witness"]))
        if self.source_total:
            for trace in traces:
                trace["audit_complete"] = (trace["valid"] and not trace["errors"]
                    and not trace.get("index_errors")
                    and not any(node.get("error") for node in trace["nodes"].values())
                    and not any(item["errors"] for item in trace.get("evidence", [])))
        return traces
