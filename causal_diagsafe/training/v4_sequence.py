"""One official Qwen/Transformers generation context, with reusable field prefixes."""

from __future__ import annotations

import time
import json
from collections import OrderedDict
from dataclasses import dataclass
from contextlib import nullcontext

import torch
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

from ..audit_v4 import audit_prompt, audit_slots, evidence_schema, parse_evidence, visible_only
from .. import audit_v4
from ..data_contract_v1 import messages_v1
from ..eval_batching import completion_tokens
from ..inference_v1 import move_tensors
from .v3_views import open_rgb


class AuditConstraintProcessor(LogitsProcessor):
    """Batch the mask transfer; LM Format Enforcer still defines every legal JSON token."""
    def __init__(self, constraints, prompt_length, masks):
        self.constraints, self.prompt_length, self.masks = constraints, prompt_length, masks

    def __call__(self, input_ids, scores):
        bodies = input_ids[:, self.prompt_length:].tolist() \
            if any(not isinstance(c, list) for c in self.constraints) else None
        rows = []
        for index, constraint in enumerate(self.constraints):
            allowed = constraint if isinstance(constraint, list) else constraint.get_allowed_tokens(bodies[index]).allowed_tokens
            if not allowed:
                raise ValueError("Unsatisfiable audit token constraint")
            key = (id(allowed), scores.shape[-1], scores.dtype, scores.device)
            if key not in self.masks:
                mask = torch.full_like(scores[0], -torch.inf)
                mask[torch.tensor(allowed, device=scores.device)] = 0
                # Retaining the list prevents its Python ID being reused for another mask.
                self.masks[key] = (allowed, mask)
                if len(self.masks) > 128:
                    self.masks.popitem(last=False)
            self.masks.move_to_end(key)
            rows.append(self.masks[key][1])
        return scores + torch.stack(rows)


@dataclass
class AuditTrajectory:
    visible: dict
    task_type: str
    base_inputs: dict
    input_ids: torch.Tensor
    prompt_length: int
    entries: list
    values: dict
    rollout_log_probabilities: dict
    evidence: list
    timing: dict
    counts: dict

    def inputs(self, through=None, target_ids=None):
        """Re-read a detached model prefix; a gold evidence body cannot reach later slots."""
        end = next(entry["answer_start"] for entry in self.entries if entry["name"] == through) \
            if through is not None else self.input_ids.shape[1]
        ids = self.input_ids[:, :end]
        if target_ids is not None:
            ids = torch.cat((ids, torch.as_tensor(target_ids, dtype=torch.long).reshape(1, -1)), -1)
        return {**self.base_inputs, "input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def classification_readouts(self):
        """Positions predicting each slot, not positions after its generated answer."""
        return {entry["name"]: {"position": entry["answer_start"] - 1,
                               "candidate_token_ids": entry["candidate_token_ids"]}
                for entry in self.entries if entry["kind"] == "classify"}


class AuditEncoder:
    def __init__(self, processor, max_evidence_tokens=512, use_cache=True, optimize=False, audit_protocol=None):
        self.processor = processor
        self.protocol = audit_protocol or audit_v4
        self.processor.tokenizer.padding_side = "left"
        self.max_evidence_tokens, self.use_cache = max_evidence_tokens, use_cache
        self._token_enforcer_data = None
        self.optimize = optimize
        self.prepared = OrderedDict()
        if optimize:
            from .v4_visual_cache import FrozenVisualCache
            self.visual_cache = FrozenVisualCache()

    def visual_context(self, model, visibles):
        return self.visual_cache.context(model, [image["uri"] for visible in visibles for image in visible["images"]]) \
            if self.optimize else nullcontext()

    @staticmethod
    def qwen_core(model):
        raw = model.module if hasattr(model, "module") else model
        raw = raw.get_base_model() if hasattr(raw, "get_base_model") else raw
        return raw.model

    @staticmethod
    def compact_cache(cache, prefix_lengths, old_padding):
        """Remove per-row EOS/padding; leave one real token for the next official prefill."""
        lengths = [length - 1 for length in prefix_lengths]
        maximum = max(lengths)
        for layer in cache.layers:
            assert type(layer).__name__ == "DynamicLayer", "This path requires full-attention Qwen cache"
            for name in ("keys", "values"):
                value = getattr(layer, name)
                assert all(padding + length <= value.shape[-2] for padding, length in zip(old_padding, lengths))
                result = value.new_zeros(value.shape[0], value.shape[1], maximum, value.shape[3])
                for row, (padding, length) in enumerate(zip(old_padding, lengths)):
                    result[row, :, maximum - length:, :] = value[row, :, padding:padding + length, :]
                setattr(layer, name, result)
        return cache

    def prepare(self, model_input, task_type=None):
        visible = visible_only(model_input)
        key = (task_type, json.dumps(visible, sort_keys=True)) if self.optimize else None
        if key in self.prepared:
            self.prepared.move_to_end(key)
            return self.prepared[key]
        prompt = self.protocol.audit_prompt(visible, task_type)
        text = self.processor.apply_chat_template(messages_v1(visible, prompt),
                                                  tokenize=False, add_generation_prompt=True)
        images = [open_rgb(item["uri"]) for item in visible["images"]]
        batch = dict(self.processor(text=[text], images=images or None, padding=True,
                                    truncation=False, return_tensors="pt"))
        batch.pop("token_type_ids", None)
        assert batch["input_ids"].shape[0] == 1 and bool(batch["attention_mask"].all())
        if self.optimize:
            self.prepared[key] = batch
            if len(self.prepared) > 128:
                self.prepared.popitem(last=False)
        return batch

    def token_ids(self, text):
        return self.processor.tokenizer(text, add_special_tokens=False)["input_ids"]

    def context_tokens(self, slot, emitted):
        return emitted

    def candidate_ids(self, slot):
        ids = [self.token_ids(text) for text in slot.continuations]
        assert all(len(value) == 1 for value in ids), (slot.name, ids)
        values = [value[0] for value in ids]
        assert len(set(values)) == len(values), (slot.name, values)
        return values

    def batch_inputs(self, bases, prefixes):
        """Left-pad complete prefixes; concatenate media in the same row/image order."""
        length = max(ids.numel() for ids in prefixes)
        ids = torch.full((len(prefixes), length), self.processor.tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, prefix in enumerate(prefixes):
            ids[index, -prefix.numel():] = prefix.flatten()
            mask[index, -prefix.numel():] = 1
        result = {"input_ids": ids, "attention_mask": mask}
        for name in {key for base in bases for key in base} - {"input_ids", "attention_mask"}:
            assert name in ("pixel_values", "image_grid_thw"), (name, "unhandled media layout")
            result[name] = torch.cat([base[name] for base in bases if name in base], 0)
        return result

    def read_native_batch(self, model, cases):
        """Native supervision has a single initial slot and needs no generated prefix."""
        bases, prefixes, slots = [], [], []
        for case in cases:
            slot, = self.protocol.audit_slots(case["model_input"], case["task_type"])
            assert case["task_type"].startswith("native_") and slot.kind == "classify"
            base = self.prepare(case["model_input"], case["task_type"])
            bases.append(base)
            prefixes.append(torch.cat((base["input_ids"].flatten(), torch.tensor(self.token_ids(slot.header)))))
            slots.append(slot)
        inputs = move_tensors(self.batch_inputs(bases, prefixes), next(model.parameters()).device)
        with self.visual_context(model, [case["model_input"] for case in cases]):
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        return [{slot.name: logits[index, self.candidate_ids(slot)].log_softmax(-1)}
                for index, slot in enumerate(slots)]

    @torch.no_grad()
    def rollout_batch(self, model, model_inputs, task_types=None, evidence_only=False):
        """Same greedy audit, batched by slot layout; no padding enters a later prefix."""
        if evidence_only:
            assert getattr(self, "label_context", None) == "blind"
        if self.optimize and self.use_cache and len(model_inputs) == 1:
            trajectory = self.rollout(model, model_inputs[0], task_types[0] if task_types else None,
                                      evidence_only=evidence_only)
            trajectory.timing.update(batch_wall_seconds=trajectory.timing["wall_seconds_with_serialization"],
                                     batch_size=1, semantics="amortized batch service time, not isolated latency")
            return [trajectory]
        visibles = [visible_only(value) for value in model_inputs]
        tasks = task_types or ["response_audit" if v["response_candidate"] is not None
                               else "request_decision" for v in visibles]
        layouts = [self.protocol.audit_slots(v, task) for v, task in zip(visibles, tasks)]
        assert len(visibles) == len(tasks) and visibles
        assert all([(s.name, s.kind) for s in layout] == [(s.name, s.kind) for s in layouts[0]]
                   for layout in layouts), "batch contains different audit protocols"
        started = time.perf_counter()
        bases = [self.prepare(v, task) for v, task in zip(visibles, tasks)]
        prefixes = [base["input_ids"].flatten() for base in bases]
        prompt_lengths = [ids.numel() for ids in prefixes]
        values, probabilities = [{} for _ in visibles], [{} for _ in visibles]
        evidence, entries = [[] for _ in visibles], [[] for _ in visibles]
        counts = [{"input_tokens": length, "generated_tokens": 0, "scaffold_tokens": 0,
                   "processed_tokens": 0, "generation_forward_calls": 0} for length in prompt_lengths]
        device, tokenizer = next(model.parameters()).device, self.processor.tokenizer
        was_training = model.training
        model.eval()
        config = GenerationConfig(do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id, use_cache=self.use_cache)
        gpu_seconds = 0.
        cache, intrinsic_rope_offset = None, None
        enforcers, masks = {}, OrderedDict()
        try:
            for index in range(len(layouts[0])):
                current = [layout[index] for layout in layouts]
                if evidence_only and current[0].kind == "classify":
                    # These readouts never enter a V6 prefix or its training fact bank.
                    # Append exactly the fixed context and prefill it at the next evidence slot.
                    for row, slot in enumerate(current):
                        scaffold = torch.tensor(self.token_ids(slot.header) + [self.placeholder_id])
                        prefixes[row] = torch.cat((prefixes[row], scaffold))
                        counts[row]["scaffold_tokens"] += scaffold.numel()
                    continue
                starts, constraints, candidates = [], [], []
                for row, slot in enumerate(current):
                    header = torch.tensor(self.token_ids(slot.header), dtype=torch.long)
                    prefixes[row] = torch.cat((prefixes[row], header))
                    starts.append(prefixes[row].numel())
                    counts[row]["scaffold_tokens"] += header.numel()
                    if slot.kind == "classify":
                        candidate = self.candidate_ids(slot)
                        allowed = [candidate[slot.labels.index(slot.forced_value)]] \
                            if slot.forced_value is not None else candidate
                        constraints.append(allowed if self.optimize else lambda _batch, _ids, options=allowed: options)
                        candidates.append(candidate)
                    else:
                        constraints.append(self.shared_enforcer(visibles[row], slot.stage, enforcers) if self.optimize
                                           else self.evidence_constraint(visibles[row], slot.stage))
                        candidates.append(None)
                # Compact stopped rows before continuing; padded cache reuse without
                # rebasing would change the real tokens' Qwen MRoPE positions.
                inputs = move_tensors(self.batch_inputs(bases, prefixes), device)
                padded_length = inputs["input_ids"].shape[1]
                padding = [padded_length - start for start in starts]
                cached_length = cache.get_seq_length() if cache is not None else 0
                if cache is not None:
                    inputs["past_key_values"] = cache
                    # Moving pad keys does not move real rotary positions. Rebase Qwen's
                    # physical-cache offset to each row's new left padding.
                    self.qwen_core(model).rope_deltas = intrinsic_rope_offset - \
                        torch.tensor(padding, device=device).reshape(-1, 1)
                budget = 1 if current[0].kind == "classify" else self.max_evidence_tokens
                if device.type == "cuda":
                    before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    before.record()
                with self.visual_context(model, visibles):
                    constraint_args = {"logits_processor": LogitsProcessorList([
                        AuditConstraintProcessor(constraints, padded_length, masks)])} if self.optimize else {
                        "prefix_allowed_tokens_fn": lambda row, ids: constraints[row](row, ids)}
                    output = model.generate(**inputs, generation_config=config, use_model_defaults=False,
                        do_sample=False, num_beams=1, max_new_tokens=budget,
                        **constraint_args,
                        return_dict_in_generate=True, output_logits=current[0].kind == "classify", logits_to_keep=1)
                if device.type == "cuda":
                    after.record()
                    after.synchronize()
                    gpu_seconds += before.elapsed_time(after) / 1000
                for row, slot in enumerate(current):
                    emitted = completion_tokens(output.sequences[row, padded_length:], tokenizer.eos_token_id)
                    emitted_length = emitted.numel()
                    stopped = bool(emitted_length and emitted[-1].item() == tokenizer.eos_token_id)
                    counts[row]["generated_tokens"] += emitted_length
                    counts[row]["generation_forward_calls"] += emitted_length
                    counts[row]["processed_tokens"] += starts[row] - max(0, cached_length - padding[row]) \
                        + max(emitted_length - 1, 0)
                    if stopped:
                        emitted = emitted[:-1]
                    raw = tokenizer.decode(emitted, skip_special_tokens=False)
                    prefixes[row] = torch.cat((prefixes[row], self.context_tokens(slot, emitted).cpu()))
                    entry = {"name": slot.name, "kind": slot.kind, "answer_start": starts[row],
                        "answer_end": prefixes[row].numel(), "raw": raw,
                        "candidate_token_ids": candidates[row], "forced_by_contract": slot.forced_value}
                    if slot.kind == "classify":
                        picked = candidates[row].index(emitted[0].item())
                        values[row][slot.name] = slot.labels[picked]
                        probabilities[row][slot.name] = output.logits[0][row, candidates[row]].float().log_softmax(-1).cpu()
                        entry.update(labels=list(slot.labels), value=values[row][slot.name])
                    else:
                        parsed = self.protocol.parse_evidence(raw, visibles[row], slot.stage)
                        parsed.update(generated_tokens=emitted_length,
                                      hit_token_limit=emitted_length == budget and not stopped)
                        if parsed["hit_token_limit"]:
                            parsed["errors"].append("evidence_token_budget_exhausted")
                        evidence[row].append(parsed)
                    entries[row].append(entry)
                if self.optimize and self.use_cache:
                    if intrinsic_rope_offset is None:
                        intrinsic_rope_offset = self.qwen_core(model).rope_deltas + \
                            torch.tensor(padding, device=device).reshape(-1, 1)
                    cache = self.compact_cache(output.past_key_values,
                        [prefix.numel() for prefix in prefixes], padding)
                del output, inputs
        finally:
            model.train(was_training)
        elapsed = time.perf_counter() - started
        return [AuditTrajectory(v, task, base, ids.reshape(1, -1), length, marks, labels, scores, ev,
                    {"gpu_forward_seconds": gpu_seconds / len(visibles),
                     "input_and_forward_seconds": elapsed / len(visibles),
                     "wall_seconds_with_serialization": elapsed / len(visibles),
                     "batch_wall_seconds": elapsed, "batch_size": len(visibles),
                     "semantics": "amortized batch service time, not isolated latency",
                     "path": "continuous_audit_batched", "cache": "compacted_continuous" if self.optimize and self.use_cache
                     else "within_slot_only"}, count)
                for v, task, base, ids, length, marks, labels, scores, ev, count in zip(
                    visibles, tasks, bases, prefixes, prompt_lengths, entries, values, probabilities, evidence, counts)]

    def evidence_constraint(self, visible, stage):
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import (
            build_token_enforcer_tokenizer_data, build_transformers_prefix_allowed_tokens_fn,
        )
        if self._token_enforcer_data is None:
            self._token_enforcer_data = build_token_enforcer_tokenizer_data(self.processor.tokenizer)
        return build_transformers_prefix_allowed_tokens_fn(
            self._token_enforcer_data, JsonSchemaParser(self.protocol.evidence_schema(visible, stage)))

    def shared_enforcer(self, visible, stage, enforcers):
        schema = json.dumps(self.protocol.evidence_schema(visible, stage), sort_keys=True)
        if schema not in enforcers:
            enforcers[schema] = self.evidence_constraint(visible, stage).token_enforcer
        return enforcers[schema]

    def read_classifications(self, model, trajectory):
        """Differentiable field readouts; detached rollout scores are observations only."""
        readouts = trajectory.classification_readouts()
        device = next(model.parameters()).device
        positions = torch.tensor([item["position"] for item in readouts.values()], device=device)
        inputs = move_tensors(trajectory.inputs(), device)
        with self.visual_context(model, [trajectory.visible]):
            logits = model(**inputs, use_cache=False, logits_to_keep=positions).logits[0].float()
        return {name: logits[index, item["candidate_token_ids"]].log_softmax(-1)
                for index, (name, item) in enumerate(readouts.items())}

    def read_classifications_batch(self, model, trajectories):
        """Read each real prefix at its own left-padding-adjusted prediction positions."""
        length = max(item.input_ids.shape[1] for item in trajectories)
        maps = [item.classification_readouts() for item in trajectories]
        positions = [[readout["position"] + length - item.input_ids.shape[1] for readout in mapping.values()]
                     for item, mapping in zip(trajectories, maps)]
        unique = sorted({position for row in positions for position in row})
        columns = {position: column for column, position in enumerate(unique)}
        device = next(model.parameters()).device
        inputs = move_tensors(self.batch_inputs([item.base_inputs for item in trajectories],
                                               [item.input_ids.flatten() for item in trajectories]), device)
        with self.visual_context(model, [item.visible for item in trajectories]):
            logits = model(**inputs, use_cache=False,
                           logits_to_keep=torch.tensor(unique, device=device)).logits.float()
        return [{name: logits[row, columns[position], entry["candidate_token_ids"]].log_softmax(-1)
                 for position, (name, entry) in zip(positions[row], mapping.items())}
                for row, mapping in enumerate(maps)]

    @torch.no_grad()
    def rollout(self, model, model_input, task_type=None, evidence_only=False):
        """Greedy short audit; all generated members are independent and gold-free."""
        if evidence_only:
            assert getattr(self, "label_context", None) == "blind"
        visible = visible_only(model_input)
        task = task_type or ("response_audit" if visible["response_candidate"] is not None
                             else "request_decision")
        slots = self.protocol.audit_slots(visible, task)
        started = time.perf_counter()
        base = self.prepare(visible, task)
        device = next(model.parameters()).device
        model_inputs = move_tensors(base, device)
        ids, cache = model_inputs["input_ids"], None
        prompt_length = ids.shape[1]
        values, probabilities, evidence, entries = {}, {}, [], []
        generated_tokens, scaffold_tokens, forward_tokens, forward_calls = 0, 0, 0, 0
        was_training = model.training
        model.eval()
        tokenizer = self.processor.tokenizer
        config = GenerationConfig(do_sample=False, num_beams=1, temperature=None, top_p=None, top_k=None,
            pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
            bos_token_id=tokenizer.bos_token_id, use_cache=self.use_cache)
        gpu_seconds = 0.
        enforcers, masks = {}, OrderedDict()
        try:
            for slot in slots:
                header = torch.tensor([self.token_ids(slot.header)], device=device)
                scaffold_tokens += header.numel()
                ids = torch.cat((ids, header), -1)
                if evidence_only and slot.kind == "classify":
                    ids = torch.cat((ids, torch.tensor([[self.placeholder_id]], device=device)), -1)
                    scaffold_tokens += 1
                    continue
                answer_start = ids.shape[1]
                inputs = {**model_inputs, "input_ids": ids, "attention_mask": torch.ones_like(ids)}
                cached_length = cache.get_seq_length() if cache is not None else 0
                if cache is not None:
                    inputs["past_key_values"] = cache
                if slot.kind == "classify":
                    candidates = self.candidate_ids(slot)
                    allowed = [candidates[slot.labels.index(slot.forced_value)]] \
                        if slot.forced_value is not None else candidates
                    constraint = allowed if self.optimize else lambda _batch, _ids, options=allowed: options
                    budget = 1
                else:
                    candidates = None
                    constraint = self.shared_enforcer(visible, slot.stage, enforcers) if self.optimize \
                        else self.evidence_constraint(visible, slot.stage)
                    budget = self.max_evidence_tokens
                gpu = device.type == "cuda"
                if gpu:
                    before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    before.record()
                with self.visual_context(model, [visible]):
                    constraint_args = {"logits_processor": LogitsProcessorList([
                        AuditConstraintProcessor([constraint], answer_start, masks)])} if self.optimize else {
                        "prefix_allowed_tokens_fn": constraint}
                    output = model.generate(**inputs, generation_config=config, use_model_defaults=False,
                        do_sample=False, num_beams=1, max_new_tokens=budget,
                        **constraint_args, return_dict_in_generate=True,
                        output_logits=slot.kind == "classify", logits_to_keep=1)
                if gpu:
                    after.record()
                    after.synchronize()
                    gpu_seconds += before.elapsed_time(after) / 1000
                emitted = output.sequences[:, answer_start:]
                emitted_length = emitted.shape[1]
                generated_tokens += emitted_length
                forward_calls += emitted_length
                forward_tokens += ids.shape[1] - cached_length + max(emitted_length - 1, 0)
                cache = output.past_key_values if self.use_cache else None
                # im_end terminates a JSON generation call, not the continuous audit turn.
                if emitted_length and emitted[0, -1].item() == tokenizer.eos_token_id:
                    emitted = emitted[:, :-1]
                raw = tokenizer.decode(emitted[0], skip_special_tokens=False)
                ids = torch.cat((ids, self.context_tokens(slot, emitted)), -1)
                entry = {"name": slot.name, "kind": slot.kind, "answer_start": answer_start,
                         "answer_end": ids.shape[1], "raw": raw,
                         "candidate_token_ids": candidates, "forced_by_contract": slot.forced_value}
                if slot.kind == "classify":
                    selected = candidates.index(emitted[0, 0].item())
                    values[slot.name] = slot.labels[selected]
                    probabilities[slot.name] = output.logits[0][0, candidates].float().log_softmax(-1).cpu()
                    entry["labels"] = list(slot.labels)
                    entry["value"] = values[slot.name]
                else:
                    parsed = self.protocol.parse_evidence(raw, visible, slot.stage)
                    parsed["generated_tokens"] = emitted_length
                    parsed["hit_token_limit"] = emitted_length == budget and \
                        output.sequences[0, -1].item() != tokenizer.eos_token_id
                    if parsed["hit_token_limit"]:
                        parsed["errors"].append("evidence_token_budget_exhausted")
                    evidence.append(parsed)
                entries.append(entry)
        finally:
            model.train(was_training)
        return AuditTrajectory(visible, task, base, ids.cpu(), prompt_length, entries, values,
            probabilities, evidence,
            {"gpu_forward_seconds": gpu_seconds, "input_and_forward_seconds": time.perf_counter() - started,
             "wall_seconds_with_serialization": time.perf_counter() - started,
             "path": "continuous_audit", "cache": self.use_cache},
            {"input_tokens": prompt_length, "generated_tokens": generated_tokens,
             "scaffold_tokens": scaffold_tokens, "processed_tokens": forward_tokens,
             "generation_forward_calls": forward_calls})
