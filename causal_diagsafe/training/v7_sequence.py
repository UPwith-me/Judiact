"""Fresh-context V7 node calls using the official Qwen processor and LM head."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import json
import math

import torch
from transformers import GenerationConfig

from ..audit_v4 import AuditSlot
from ..audit_v7 import Node, canonical
from ..inference_v1 import move_tensors
from .v3_views import open_rgb
from .v4_sequence import AuditEncoder
from .v4_visual_cache import FrozenVisualCache
from .v7_objective import correction_ce


@dataclass
class NodeTarget:
    sample_id: str
    node: Node
    target: object
    view: str = "reference"
    hard: bool = True
    continuation_ids: list[int] | None = None
    terminated: bool = True


class SampleLogProbability:
    """Record the distribution actually sampled, after grammar masking."""

    def __init__(self, eos):
        self.eos = eos
        self.previous = self.total = self.active = None

    def consume(self, token):
        if self.previous is not None:
            value = self.previous.gather(1, token[:, None]).squeeze(1).double()
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


class NodeEncoder:
    def __init__(self, processor, max_new_tokens=2048, cache_size=64, cache_vision=True):
        self.processor = processor
        self.helper = AuditEncoder(processor)
        self.maximum = cache_size
        self.prepared = OrderedDict()
        self.vision = FrozenVisualCache(max_images=32) if cache_vision else None
        self.max_new_tokens = max_new_tokens
        self._grammar_compiler = None
        self._grammar_vocab_size = None

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
        batch = dict(self.processor(text=[text], images=images or None, padding=False,
                                    truncation=False, return_tensors="pt"))
        batch.pop("token_type_ids", None)
        if self.maximum:
            self.prepared[key] = batch
            while len(self.prepared) > self.maximum:
                self.prepared.popitem(last=False)
        return batch

    def candidate_ids(self, node):
        return self.helper.candidate_ids(AuditSlot(node.name, "classify", node.labels))

    def target_ids(self, example):
        if example.continuation_ids is not None:
            assert not example.node.labels
            return list(example.continuation_ids) + (
                [self.processor.tokenizer.eos_token_id] if example.terminated else [])
        if example.node.labels:
            return [self.candidate_ids(example.node)[example.node.labels.index(example.target)]]
        text = example.target if isinstance(example.target, str) else canonical(example.target)
        if example.node.schema is not None:
            # Fixed references have a declared serialization; sampled paths above
            # keep their actual token sequence and proposal probability unchanged.
            text = canonical(json.loads(text))
        return self.helper.token_ids(text) + [self.processor.tokenizer.eos_token_id]

    def supervised_batch(self, examples):
        bases = [self.prepare(example.node) for example in examples]
        targets = [self.target_ids(example) for example in examples]
        prefixes = [torch.cat((base["input_ids"].flatten(), torch.tensor(ids[:-1], dtype=torch.long)))
                    for base, ids in zip(bases, targets)]
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
        """Normalized node log probabilities under the actual constrained policy.

        No length averaging or top-k vocabulary approximation is used here.
        Free-text EOS and censored, token-limit paths retain their distinct mass.
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
            import xgrammar as xgr
            matcher = xgr.GrammarMatcher(self.compiled_grammar(example.node.schema, local.shape[-1]))
            mask = xgr.allocate_token_bitmask(len(ids), local.shape[-1])
            for position, token in enumerate(ids):
                matcher.fill_next_token_bitmask(mask, position)
                if not matcher.accept_token(token):
                    raise ValueError((example.sample_id, example.node.name,
                                      "target outside generation grammar", position, token))
            mask = mask.to(local.device)
            columns = torch.arange(local.shape[-1], device=local.device)
            total = local[0, 0].float() * 0
            for start in range(0, len(ids), 16):
                values = local[start:start + 16].float()
                allowed = ((mask[start:start + 16, columns // 32] >> (columns % 32)) & 1).bool()
                token_ids = torch.tensor(ids[start:start + 16], device=local.device)
                total = total + (values.gather(1, token_ids[:, None]).squeeze(1)
                                 - values.masked_fill(~allowed, -math.inf).logsumexp(-1)).sum()
            result.append((total, None))
        return result

    def compiled_grammar(self, schema, vocab_size):
        import xgrammar as xgr
        if self._grammar_compiler is None:
            info = xgr.TokenizerInfo.from_huggingface(
                self.processor.tokenizer, vocab_size=vocab_size,
                stop_token_ids=[self.processor.tokenizer.eos_token_id])
            self._grammar_compiler = xgr.GrammarCompiler(info, max_threads=4, cache_limit_bytes=512 * 1024**2)
            self._grammar_vocab_size = vocab_size
        assert vocab_size == self._grammar_vocab_size
        return self._grammar_compiler.compile_json_schema(json.loads(canonical(schema)), any_order=False)

    @torch.no_grad()
    def generate(self, model, nodes, sample=False, generator=None):
        """No gold input and no cross-node KV reuse. Invalid output is an error record."""
        bases = [self.prepare(node) for node in nodes]
        inputs = self.helper.batch_inputs(bases, [base["input_ids"].flatten() for base in bases])
        inputs = move_tensors(inputs, next(model.parameters()).device)
        was_training = model.training
        model.eval()
        try:
            if all(node.labels for node in nodes):
                with self.visual_context(model, nodes):
                    logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[:, -1].float()
                result = []
                for row, node in enumerate(nodes):
                    logp = logits[row, self.candidate_ids(node)].log_softmax(-1)
                    picked = torch.multinomial(logp.exp(), 1, generator=generator).item() \
                        if sample else logp.argmax().item()
                    value = node.forced_value if node.forced_value is not None else node.labels[picked]
                    result.append({"value": value, "log_probabilities": logp.cpu().tolist(),
                                   "forced_by_contract": node.forced_value, "error": None})
                return result
            assert not any(node.labels for node in nodes), "batch text and categorical calls separately"
            from xgrammar.contrib.hf import LogitsProcessor
            vocab_size = model.get_output_embeddings().weight.shape[0]
            grammar = LogitsProcessor([self.compiled_grammar(node.schema, vocab_size) for node in nodes])
            config = GenerationConfig(
                do_sample=sample, temperature=1. if sample else None, top_p=1. if sample else None,
                top_k=0 if sample else None, num_beams=1, use_cache=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id)
            sampler = SampleLogProbability(self.processor.tokenizer.eos_token_id)
            with self.visual_context(model, nodes):
                outputs = model.generate(
                    **inputs, generation_config=config, use_model_defaults=False,
                    max_new_tokens=self.max_new_tokens, logits_to_keep=1,
                    logits_processor=[grammar, sampler])
            sampling_logps = sampler.finish(outputs)
            result = []
            for node, ids, sampling_logp in zip(nodes, outputs[:, inputs["input_ids"].shape[1]:], sampling_logps):
                stopped = (ids == self.processor.tokenizer.eos_token_id).nonzero()
                ids = ids[:stopped[0].item()] if len(stopped) else ids
                raw = self.processor.tokenizer.decode(ids, skip_special_tokens=False)
                error = None if len(stopped) else "node_token_budget_exhausted"
                try:
                    value = json.loads(raw)
                except json.JSONDecodeError:
                    value, error = None, "invalid_node_json"
                result.append({"value": value, "raw": raw, "token_ids": ids.cpu().tolist(),
                               "terminated": bool(len(stopped)),
                               "log_probability": sampling_logp if sample else 0.,
                               "policy_log_probability": sampling_logp, "sampled": sample,
                               "error": error, "tokens": len(ids)})
            return result
        finally:
            model.train(was_training)
