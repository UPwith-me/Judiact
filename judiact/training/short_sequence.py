"""Bounded compact review generation, retaining the shared classification path."""

import json
import re
import statistics
from contextlib import nullcontext

import torch

from .location_decode import LocationCycleStop

from .shared_sequence import SharedAuditEncoder


class ShortReviewEncoder(SharedAuditEncoder):
    def __init__(self, *args, short_max_new_tokens=512, short_compact_after=192, **kwargs):
        super().__init__(*args, **kwargs)
        self.short_max_new_tokens = short_max_new_tokens
        self.short_compact_after = short_compact_after
        self.compact_finished_rows = True
        self._short_decode_state = None

    def generation_components(self, nodes, grammars, sample=False):
        if all(node.name.startswith("short_review_") for node in nodes):
            import xgrammar as xgr
            from .decode import BatchedGrammarLogitsProcessor
            if sample:
                return BatchedGrammarLogitsProcessor(grammars, xgr.BatchGrammarMatcher(max_threads=2)), None
            tokenizer = self.processor.tokenizer
            state = ShortConditionStop(grammars[0].tokenizer_info.decoded_vocab,
                tokenizer.eos_token_id, tokenizer.pad_token_id, len(nodes))
            self._short_decode_state = state
            penalty = ShortConditionFrequency(state, .1)

            class ReviewGrammar(BatchedGrammarLogitsProcessor):
                def __call__(self, input_ids, scores):
                    return penalty(input_ids, super().__call__(input_ids, scores))

                def advance(self, tokens):
                    self.active = [i for i in self.active if state.reasons[i] is None]
                    super().advance(tokens)

            return ReviewGrammar(grammars, xgr.BatchGrammarMatcher(max_threads=2),
                                 token_history=lambda: state.last_tokens), state
        return super().generation_components(nodes, grammars, sample)

    def generate(self, model, nodes, *args, **kwargs):
        short = [node.name.startswith("short_review_") for node in nodes]
        if not any(short):
            return super().generate(model, nodes, *args, **kwargs)
        assert all(short)
        previous = self.max_new_tokens
        self.max_new_tokens = self.short_max_new_tokens
        self._short_decode_state = None
        sampled = kwargs.get("sample", args[0] if args else False)
        execution = ActiveShortRows(model, lambda: self._short_decode_state, self.short_compact_after) \
            if self.compact_finished_rows and not sampled else nullcontext()
        try:
            with execution:
                outputs = super().generate(model, nodes, *args, **kwargs)
            for node, output in zip(nodes, outputs):
                output["generation_controls"].update(
                    condition_text_frequency_penalty=.1 if node.name == "short_review_condition"
                    and not output["sampled"] else 0.,
                    condition_frequency_scope="current free condition text only; excludes facts, IDs, kind and structure",
                    repetition_stop="exact repeated complete condition sentence is an explicit failed proposal",
                    finished_row_compaction=self.compact_finished_rows and not sampled,
                    finished_row_compaction_after=self.short_compact_after,
                    short_max_new_tokens=self.short_max_new_tokens)
            return outputs
        finally:
            self.max_new_tokens = previous


def condition_text_start(raw):
    marker = '"text":'
    offset = raw.find(marker)
    if offset < 0:
        return None
    start = offset + len(marker)
    if raw[start:start + 1] != '"':
        return None
    try:
        json.decoder.scanstring(raw, start + 1)
    except ValueError:
        return start
    return None


class ShortConditionStop(LocationCycleStop):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.finished = [False] * len(self.text)
        self.last_tokens = None

    def __call__(self, input_ids, scores, **kwargs):
        from partial_json_parser import loads
        self.emitted += 1
        self.last_tokens = input_ids[:, -1].tolist()
        for i, token in enumerate(self.last_tokens):
            if self.reasons[i] or token in (self.eos, self.pad):
                self.finished[i] = True
                continue
            try:
                chunk = self.decoders[i].decode(self.vocab[token])
            except UnicodeDecodeError:
                # Keep malformed model bytes as a failed proposal, not repaired facts.
                self.reasons[i], self.lengths[i] = "invalid_short_review_utf8", self.emitted
                self.finished[i] = True
                continue
            self.text[i] += chunk
            raw = self.text[i]
            if '"text":' not in raw or not any(c in chunk for c in (".", "!", "?", "\u3002", "\uff01", "\uff1f")):
                continue
            value = loads(raw)
            text = value.get("text", "") if isinstance(value, dict) else ""
            sentences = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+|(?<=[\u3002\uff01\uff1f])", text)
            complete = [s.strip() for s in sentences if s.strip().endswith(tuple(".!?\u3002\uff01\uff1f"))
                        and any(c.isalpha() for c in s)]
            if len(complete) != len(set(complete)):
                self.reasons[i], self.lengths[i] = "repeated_short_condition_sentence", self.emitted
                self.finished[i] = True
        return torch.tensor([reason is not None for reason in self.reasons], device=input_ids.device)


class ActiveShortRows:
    """Remove ended rows from Qwen decode, leaving HF's sequence bookkeeping intact."""

    def __init__(self, model, state, compact_after=192):
        self.model = model.get_base_model() if hasattr(model, "get_base_model") else model
        self.state = state
        self.rows = None
        self.prefilled = False
        self.compact_after = compact_after

    def __enter__(self):
        self.before = self.model.register_forward_pre_hook(self.prepare, with_kwargs=True)
        self.after = self.model.register_forward_hook(self.restore_logits, with_kwargs=True)
        return self

    def __exit__(self, *exc):
        self.before.remove()
        self.after.remove()

    def prepare(self, module, args, kwargs):
        ids = kwargs["input_ids"]
        if not self.prefilled:
            self.size = ids.shape[0]
            self.rows = list(range(self.size))
            self.prefilled = True
            self.index = None
            return args, kwargs
        state = self.state()
        # All reviewed short references fit within 171 tokens. Avoid changing
        # small-batch GEMM shapes during ordinary short answers; compact tails.
        if state.emitted < self.compact_after:
            return args, kwargs
        active = [i for i in self.rows if not state.finished[i]]
        assert active, "HF must stop once every short proposal has ended"
        if active != self.rows:
            local = torch.tensor([self.rows.index(i) for i in active], device=ids.device)
            kwargs["past_key_values"].batch_select_indices(local)
            # Qwen stores per-example multimodal offsets outside the KV cache.
            module.model.rope_deltas = module.model.rope_deltas.index_select(0, local)
            self.rows = active
        if len(self.rows) == self.size:
            return args, kwargs
        self.index = torch.tensor(self.rows, device=ids.device)
        kwargs["input_ids"] = ids.index_select(0, self.index)
        mask = kwargs["attention_mask"]
        assert mask.ndim == 2 and kwargs.get("position_ids") is None
        kwargs["attention_mask"] = mask.index_select(0, self.index)
        return args, kwargs

    def restore_logits(self, module, args, kwargs, output):
        if self.index is not None:
            full = output.logits.new_zeros((self.size, *output.logits.shape[1:]))
            full.index_copy_(0, self.index, output.logits)
            output.logits = full
        return output


class ShortConditionFrequency:
    def __init__(self, state, frequency):
        self.state, self.frequency = state, frequency
        self.counts = [{} for _ in state.text]
        self.starts = [None] * len(state.text)

    def __call__(self, input_ids, scores):
        rows, tokens, counts = self.sparse_counts()
        if rows:
            row_ids = torch.tensor(rows, device=scores.device)
            token_ids = torch.tensor(tokens, device=scores.device)
            values = torch.tensor(counts, device=scores.device, dtype=scores.dtype)
            scores[row_ids, token_ids] -= self.frequency * values
        return scores

    def sparse_counts(self):
        """The same prefix-dependent counts for generation and policy replay."""
        starts = [condition_text_start(raw) for raw in self.state.text]
        rows, tokens, counts = [], [], []
        for i, (start, previous) in enumerate(zip(starts, self.starts)):
            if start != previous:
                self.counts[i].clear()
            if start is not None and start == previous:
                token = self.state.last_tokens[i]
                self.counts[i][token] = self.counts[i].get(token, 0) + 1
            for token, count in self.counts[i].items():
                rows.append(i)
                tokens.append(token)
                counts.append(count)
        self.starts = starts
        return rows, tokens, counts


def review_batches(encoder, nodes, max_sequences, max_tokens):
    lengths = [encoder.prepare(node)["input_ids"].numel() for node in nodes]
    batch = []
    for i in sorted(range(len(nodes)), key=lambda i: -lengths[i]):
        if batch and (len(batch) >= max_sequences or
                      (len(batch) + 1) * (lengths[batch[0]] + encoder.short_max_new_tokens) > max_tokens):
            yield batch
            batch = []
        batch.append(i)
    if batch:
        yield batch


def partition_review_members(costs, world, seconds_per_token=None):
    """Move complete members between GPUs, preserving every proposal and target."""
    measured = seconds_per_token or [None] * world
    known = [value for value in measured if value is not None]
    default = statistics.median(known) if known else 1.
    rates = [default if value is None else value for value in measured]
    owners, loads = [[] for _ in range(world)], [0] * world
    for cost, sid in sorted(costs, reverse=True):
        owner = min(range(world), key=lambda r: (loads[r] + cost) * rates[r])
        owners[owner].append(sid)
        loads[owner] += cost
    return owners, loads
