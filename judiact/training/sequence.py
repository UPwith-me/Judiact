"""Fresh-context audit node calls using the official Qwen processor and LM head."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import inspect
import math

import torch
from transformers import GenerationConfig

from ..audit_schema import AuditSlot
from ..reference_audit import DEFAULT_MAX_NEW_TOKENS, Node, canonical
from ..tensor_inputs import move_tensors
from .multimodal_views import open_rgb
from .audit_sequence import AuditEncoder
from .visual_cache import FrozenVisualCache
from .objective import correction_ce


@dataclass
class NodeTarget:
    sample_id: str
    node: Node
    target: object
    view: str = "reference"
    hard: bool = True
    continuation_ids: list[int] | None = None
    terminated: bool = True
    prefix_ids: list[int] | None = None


class SampleLogProbability:
    """Record the distribution actually sampled, after grammar masking."""

    def __init__(self, eos, record_tokens=False):
        self.eos = eos
        self.previous = self.total = self.active = None
        self.record_tokens = record_tokens
        self.token_values, self.token_masks = [], []

    def consume(self, token):
        if self.previous is not None:
            value = self.previous.gather(1, token[:, None]).squeeze(1).double()
            if self.record_tokens:
                self.token_values.append(torch.where(self.active, value, 0.))
                self.token_masks.append(self.active.clone())
            self.total += torch.where(self.active, value, 0.)
            self.active &= token != self.eos

    def __call__(self, input_ids, scores):
        if self.total is None:
            self.total = torch.zeros(scores.shape[0], dtype=torch.float64, device=scores.device)
            self.active = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
        else:
            self.consume(input_ids[:, -1])
        self.previous = scores.float().log_softmax(-1).detach()
        return scores

    def finish(self, sequences):
        self.consume(sequences[:, -1])
        return self.total.cpu().tolist()

    def sampled_token_logps(self):
        values = torch.stack(self.token_values, 1).cpu()
        masks = torch.stack(self.token_masks, 1).cpu()
        return [row[mask].tolist() for row, mask in zip(values, masks)]


class GeneratedTokenPenalties:
    """Output-only additive presence/frequency controls, never prompt penalties."""

    def __init__(self, presence=0., frequency=0.):
        self.presence, self.frequency = presence, frequency
        self.prompt_length = None

    def __call__(self, input_ids, scores):
        if self.prompt_length is None:
            self.prompt_length = input_ids.shape[1]
        generated = input_ids[:, self.prompt_length:]
        counts = torch.zeros_like(scores)
        counts.scatter_add_(1, generated, torch.ones_like(generated, dtype=scores.dtype))
        return scores - (counts > 0).to(scores.dtype) * self.presence - counts * self.frequency


def grammar_masked_logits(logits, bitmask):
    """FP32 grammar normalization without a dense vocabulary-sized bool mask."""
    import xgrammar as xgr
    values = logits.float().clone()
    # The only consumers normalize these logits and gather legal targets, so
    # their gradients at the overwritten, illegal coordinates are exactly zero.
    xgr.apply_token_bitmask_inplace(values, bitmask, vocab_size=values.shape[-1])
    return values


class NodeEncoder:
    def __init__(self, processor, max_new_tokens=DEFAULT_MAX_NEW_TOKENS, cache_size=64, cache_vision=True,
                 probability_chunk_size=16, witness_frequency_penalty=.1, compact_witness_json=True,
                 fast_generation=True):
        self.processor = processor
        self.helper = AuditEncoder(processor)
        self.maximum = cache_size
        self.prepared = OrderedDict()
        self.image_prepared = OrderedDict()
        self.vision = FrozenVisualCache(max_images=32, canonical_images=True) if cache_vision else None
        self.max_new_tokens = max_new_tokens
        self._grammar_compiler = None
        self._grammar_vocab_size = None
        self.probability_chunk_size = probability_chunk_size
        self.track_greedy_probability = True
        self.record_sampling_token_logps = False
        self.witness_frequency_penalty = witness_frequency_penalty
        self.compact_witness_json = compact_witness_json
        self.fast_generation = fast_generation
        self.reference_tokens = OrderedDict()
        self.grammar_masks = OrderedDict()
        self.grammar_mask_bytes = 0
        self.grammar_mask_budget = 128 * 1024**2
        self.candidate_tokens = {}

    @staticmethod
    def media_keys(nodes):
        return [canonical([uri, None if box == (0, 0, 1000, 1000) else box])
                for node in nodes for uri, box in node.media]

    def visual_context(self, model, nodes):
        from contextlib import nullcontext
        return self.vision.context(model, self.media_keys(nodes)) if self.vision else nullcontext()

    def prepare(self, node):
        key = (node.text, node.media)
        if key in self.prepared:
            self.prepared.move_to_end(key)
            return self.prepared[key]
        media_key = (node.media, canonical(self.processor.image_processor.size)) if node.media else None
        cached_images = self.image_prepared.get(media_key)
        if cached_images is not None:
            self.image_prepared.move_to_end(media_key)
            images = cached_images[0]
        else:
            images = []
            for uri, box in node.media:
                image = open_rgb(uri)
                if box is not None:
                    width, height = image.size
                    # Preserve pixels. The mapping is explicit and identical for train/eval.
                    pixel_box = (math.floor(box[0] * width / 1000), math.floor(box[1] * height / 1000),
                                 math.ceil(box[2] * width / 1000), math.ceil(box[3] * height / 1000))
                    image = image.crop(pixel_box)
                images.append(image)
        content = [{"type": "image", "image": image} for image in images]
        content.append({"type": "text", "text": node.text})
        messages = [{"role": "user", "content": content}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_processor = self.processor.image_processor
        had_preprocess = "preprocess" in vars(image_processor)
        original = vars(image_processor).get("preprocess")
        if cached_images is not None:
            # The official processor still expands image tokens and tokenizes
            # each actual prompt; only identical immutable pixels are reused.
            image_processor.preprocess = lambda *args, **kwargs: dict(cached_images[1])
        try:
            batch = dict(self.processor(text=[text], images=images or None, padding=False,
                                        truncation=False, return_tensors="pt"))
        finally:
            if cached_images is not None:
                if had_preprocess:
                    image_processor.preprocess = original
                else:
                    del image_processor.preprocess
        if node.media and cached_images is None:
            self.image_prepared[media_key] = (images, {name: batch[name] for name in
                                                       ("pixel_values", "image_grid_thw")})
            while len(self.image_prepared) > 128:
                self.image_prepared.popitem(last=False)
        batch.pop("token_type_ids", None)
        if self.maximum:
            self.prepared[key] = batch
            while len(self.prepared) > self.maximum:
                self.prepared.popitem(last=False)
        return batch

    def candidate_ids(self, node):
        key = (node.name, node.labels)
        if key not in self.candidate_tokens:
            self.candidate_tokens[key] = tuple(self.helper.candidate_ids(
                AuditSlot(node.name, "classify", node.labels)))
        return list(self.candidate_tokens[key])

    def target_ids(self, example):
        if example.continuation_ids is not None:
            assert not example.node.labels
            return list(example.continuation_ids) + (
                [self.processor.tokenizer.eos_token_id] if example.terminated else [])
        if example.node.labels:
            return [self.candidate_ids(example.node)[example.node.labels.index(example.target)]]
        text = example.target if isinstance(example.target, str) else canonical(example.target)
        key = (text, example.node.schema is not None)
        if key in self.reference_tokens:
            self.reference_tokens.move_to_end(key)
            return list(self.reference_tokens[key])
        if example.node.schema is not None:
            # Fixed references have a declared serialization; sampled paths above
            # keep their actual token sequence and proposal probability unchanged.
            text = canonical(json.loads(text))
        ids = self.helper.token_ids(text) + [self.processor.tokenizer.eos_token_id]
        self.reference_tokens[key] = tuple(ids)
        while len(self.reference_tokens) > 8192:
            self.reference_tokens.popitem(last=False)
        return ids

    def target_grammar_mask(self, example, ids, vocab_size):
        """Cache only model-independent CPU masks for the exact token history."""
        import xgrammar as xgr
        compact = example.node.compact_json or (self.compact_witness_json and example.node.name == "witness")
        prefix = tuple(example.prefix_ids or [])
        key = (canonical(example.node.schema), compact, prefix, tuple(ids), vocab_size)
        if key in self.grammar_masks:
            self.grammar_masks.move_to_end(key)
            return self.grammar_masks[key]
        matcher = xgr.GrammarMatcher(self.compiled_grammar(
            example.node.schema, vocab_size, compact=compact))
        for position, token in enumerate(prefix):
            if not matcher.accept_token(token):
                raise ValueError((example.sample_id, example.node.name,
                                  "conditioning prefix outside grammar", position, token))
        mask = xgr.allocate_token_bitmask(len(ids), vocab_size)
        for position, token in enumerate(ids):
            matcher.fill_next_token_bitmask(mask, position)
            if not matcher.accept_token(token):
                raise ValueError((example.sample_id, example.node.name,
                                  "target outside generation grammar", position, token))
        size = mask.numel() * mask.element_size()
        if size <= self.grammar_mask_budget:
            while self.grammar_masks and self.grammar_mask_bytes + size > self.grammar_mask_budget:
                _, removed = self.grammar_masks.popitem(last=False)
                self.grammar_mask_bytes -= removed.numel() * removed.element_size()
            self.grammar_masks[key] = mask
            self.grammar_mask_bytes += size
        return mask

    def supervised_batch(self, examples):
        bases = [self.prepare(example.node) for example in examples]
        targets = [self.target_ids(example) for example in examples]
        prefixes = [torch.cat((base["input_ids"].flatten(),
                               torch.tensor((example.prefix_ids or []) + ids[:-1], dtype=torch.long)))
                    for base, example, ids in zip(bases, examples, targets)]
        batch = self.helper.batch_inputs(bases, prefixes)
        # Targets end at the last true input position, so only this suffix needs LM projection.
        count = max(map(len, targets))
        batch["logits_to_keep"] = count
        return batch, targets

    def score_batch(self, model, examples, anchor=None, anchoring=0.):
        inputs, targets = self.supervised_batch(examples)
        device = next(model.parameters()).device
        inputs = move_tensors(inputs, device)
        nodes = [example.node for example in examples]
        old = None
        if anchoring:
            # Finish the detached reference call before any student graph exists.
            with anchor.reference() as reference, self.visual_context(reference, nodes):
                old = reference(**inputs, use_cache=False).logits.detach()
        with self.visual_context(model, nodes):
            logits = model(**inputs, use_cache=False).logits
        losses, probabilities = [], []
        for row, (example, ids) in enumerate(zip(examples, targets)):
            local = logits[row, -len(ids):]
            previous = old[row, -len(ids):] if old is not None else None
            if example.node.labels:
                candidates = self.candidate_ids(example.node)
                local = local[:, candidates]
                previous = previous[:, candidates] if previous is not None else None
                labels = torch.tensor([example.node.labels.index(example.target)], device=device)
                probabilities.append(local[0].float().log_softmax(-1))
            else:
                labels = torch.tensor(ids, device=device)
                probabilities.append(None)
            hard = torch.full(labels.shape, example.hard, device=device, dtype=torch.bool)
            losses.append(correction_ce(local, labels, previous, anchoring, hard).sum())
        return losses, probabilities

    def probability_batch(self, model, examples):
        """Normalized node log probabilities under the target grammar policy.

        No length averaging or top-k vocabulary approximation is used here.
        Free-text EOS and censored, token-limit paths retain their distinct mass.
        Proposal-only repetition controls are intentionally absent from p_theta.
        """
        inputs, targets = self.supervised_batch(examples)
        inputs = move_tensors(inputs, next(model.parameters()).device)
        with self.visual_context(model, [e.node for e in examples]):
            logits = model(**inputs, use_cache=False).logits
        result = []
        for row, (example, ids) in enumerate(zip(examples, targets)):
            local = logits[row, -len(ids):]
            if example.node.labels:
                logp = local[0, self.candidate_ids(example.node)].float().log_softmax(-1)
                if example.node.forced_value is not None:
                    forced = example.node.labels.index(example.node.forced_value)
                    fixed = torch.full_like(logp, -math.inf)
                    fixed[forced] = 0.
                    logp = logp.sum() * 0 + fixed
                result.append((logp[example.node.labels.index(example.target)], logp))
                continue
            mask = self.target_grammar_mask(example, ids, local.shape[-1]).to(local.device)
            token_ids = torch.tensor(ids, device=local.device)
            total = local[0, 0].float() * 0
            chunk = self.probability_chunk_size
            for start in range(0, len(ids), chunk):
                values = local[start:start + chunk].float()
                masked = grammar_masked_logits(values, mask[start:start + chunk])
                total = total + (values.gather(1, token_ids[start:start + chunk, None]).squeeze(1)
                                 - masked.logsumexp(-1)).sum()
            result.append((total, None))
        return result

    def compiled_grammar(self, schema, vocab_size, compact=False):
        import xgrammar as xgr
        if self._grammar_compiler is None:
            info = xgr.TokenizerInfo.from_huggingface(
                self.processor.tokenizer, vocab_size=vocab_size,
                stop_token_ids=[self.processor.tokenizer.eos_token_id])
            self._grammar_compiler = xgr.GrammarCompiler(info, max_threads=4, cache_limit_bytes=512 * 1024**2)
            self._grammar_vocab_size = vocab_size
            # Main train/dev share 0.2.6rc2 and explicitly ordered properties.
            # Historical 0.1 environments do not expose this keyword.
            self._grammar_order_kwargs = {"any_order": False} if "any_order" in inspect.signature(
                self._grammar_compiler.compile_json_schema).parameters else {}
        assert vocab_size == self._grammar_vocab_size
        return self._grammar_compiler.compile_json_schema(
            json.loads(canonical(schema)), **self._grammar_order_kwargs, any_whitespace=not compact,
            separators=(",", ":") if compact else None)

    @torch.no_grad()
    def generate(self, model, nodes, sample=False, generator=None, text_temperature=1.,
                 text_presence_penalty=0., text_frequency_penalty=None):
        """No gold input and no cross-node KV reuse. Invalid output is an error record."""
        bases = [self.prepare(node) for node in nodes]
        inputs = self.helper.batch_inputs(bases, [base["input_ids"].flatten() for base in bases])
        inputs = move_tensors(inputs, next(model.parameters()).device)
        was_training = model.training
        model.eval()
        try:
            if text_frequency_penalty is None:
                text_frequency_penalty = self.witness_frequency_penalty if all(
                    node.name == "witness" for node in nodes) else 0.
            if all(node.labels for node in nodes):
                with self.visual_context(model, nodes):
                    logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                distributions, picks = [], []
                for row, node in enumerate(nodes):
                    logp = logits[row, self.candidate_ids(node)].log_softmax(-1)
                    picked = torch.multinomial(logp.exp(), 1, generator=generator).squeeze(0) \
                        if sample else logp.argmax()
                    distributions.append(logp)
                    picks.append(picked)
                # Keep the same per-row RNG calls, but transfer their results
                # together instead of synchronizing twice for every row.
                sizes = [len(p) for p in distributions]
                flat = torch.cat(distributions).cpu().tolist()
                picks = torch.stack(picks).cpu().tolist()
                result, offset = [], 0
                for node, picked, size in zip(nodes, picks, sizes):
                    value = node.forced_value if node.forced_value is not None else node.labels[picked]
                    result.append({"value": value, "log_probabilities": flat[offset:offset + size],
                                   "forced_by_contract": node.forced_value, "error": None})
                    offset += size
                return result
            assert not any(node.labels for node in nodes), "batch text and categorical calls separately"
            vocab_size = model.get_output_embeddings().weight.shape[0]
            grammars = [self.compiled_grammar(
                node.schema, vocab_size, compact=node.compact_json or (self.compact_witness_json and node.name == "witness"))
                for node in nodes]
            penalty_type = GeneratedTokenPenalties
            if self.fast_generation:
                from .decode import IncrementalTokenPenalties
                penalty_type = IncrementalTokenPenalties
            grammar, cycle_stop = self.generation_components(nodes, grammars, sample=sample)
            processors = [grammar]
            if text_presence_penalty or text_frequency_penalty:
                processors.append(penalty_type(text_presence_penalty, text_frequency_penalty))
            if sample and text_temperature != 1.:
                from transformers import TemperatureLogitsWarper
                processors.append(TemperatureLogitsWarper(text_temperature))
            config = GenerationConfig(
                do_sample=sample, temperature=1. if sample else None, top_p=1. if sample else None,
                top_k=0 if sample else None, num_beams=1, use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id)
            track_probability = sample or self.track_greedy_probability
            sampler = SampleLogProbability(self.processor.tokenizer.eos_token_id,
                record_tokens=sample and self.record_sampling_token_logps) if track_probability else None
            if sampler:
                # Proposal log q must include temperature, not just grammar masking.
                processors.append(sampler)
            with self.visual_context(model, nodes):
                outputs = model.generate(
                    **inputs, generation_config=config, use_model_defaults=False,
                    synced_gpus=getattr(self, "generation_synced_gpus", None),
                    max_new_tokens=self.max_new_tokens, logits_to_keep=1,
                    logits_processor=processors,
                    stopping_criteria=[cycle_stop] if cycle_stop is not None else [])
            sampling_logps = sampler.finish(outputs) if sampler else [None] * len(nodes)
            token_logps = sampler.sampled_token_logps() if sample and self.record_sampling_token_logps else None
            result = []
            sequences = outputs[:, inputs["input_ids"].shape[1]:].cpu().tolist()
            for row, (node, ids, sampling_logp) in enumerate(zip(nodes, sequences, sampling_logps)):
                cycle = cycle_stop.reasons[row] if cycle_stop is not None else None
                if cycle:
                    ids = ids[:cycle_stop.lengths[row]]
                eos = self.processor.tokenizer.eos_token_id
                stopped = eos in ids
                ids = ids[:ids.index(eos)] if stopped else ids
                raw = self.processor.tokenizer.decode(ids, skip_special_tokens=False)
                error = cycle or (None if stopped else "node_token_budget_exhausted")
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    value, error = None, error or "invalid_node_json"
                result.append({"value": value, "raw": raw, "token_ids": ids,
                               "terminated": stopped,
                               "log_probability": sampling_logp if sample else 0.,
                               "policy_log_probability": sampling_logp, "sampled": sample,
                               "error": error, "tokens": len(ids),
                               "generation_controls": {"temperature": text_temperature if sample else None,
                                   "presence_penalty": text_presence_penalty,
                                   "frequency_penalty": text_frequency_penalty,
                                   "compact_json": node.compact_json or (self.compact_witness_json and node.name == "witness"),
                                   "log_probability_semantics": "sampling proposal q, including constraints and EOS"}})
                if token_logps is not None:
                    result[-1]["sampling_token_log_probabilities"] = token_logps[row]
            return result
        finally:
            model.train(was_training)

    def generation_components(self, nodes, grammars, sample=False):
        from xgrammar.contrib.hf import LogitsProcessor
        return LogitsProcessor(grammars), None
