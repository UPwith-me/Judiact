"""Control free-description repetition without penalizing copied source words."""

import codecs
import json
import re

import torch
from transformers import StoppingCriteria


def complete_evidence_items(raw, name):
    marker = '"' + name + '":['
    start = raw.find(marker)
    if start < 0:
        return []
    offset = start + len(marker)
    decoder, items = json.JSONDecoder(), []
    while offset < len(raw):
        try:
            item, offset = decoder.raw_decode(raw, offset)
        except json.JSONDecodeError:
            break
        items.append(item)
        if raw[offset:offset + 1] != ",":
            break
        offset += 1
    return items


def location_cycle(raw, inspect_description=True):
    for name in ("quotes", "regions"):
        items = complete_evidence_items(raw, name)
        keys = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in items]
        if len(keys) != len(set(keys)):
            return "repeated_evidence_item"
    if not inspect_description or '"description":' not in raw:
        return None
    from partial_json_parser import loads
    value = loads(raw)
    if not isinstance(value, dict):
        return None
    for region in value.get("regions", []):
        description = region.get("description", "") if isinstance(region, dict) else ""
        # Only free visual descriptions: never inspect quoted source text for
        # lexical repetition. Decimal points and unfinished sentences do not count.
        sentences = re.split(r"(?<=[.!?\u3002\uff01\uff1f])\s+|(?<=[\u3002\uff01\uff1f])", description)
        complete = [s.strip() for s in sentences if s.strip().endswith(tuple(".!?\u3002\uff01\uff1f"))
                    and any(c.isalpha() for c in s)]
        if len(complete) != len(set(complete)):
            return "repeated_region_sentence"
    return None


def impossible_quote_prefix(raw, sources):
    """An exact quote cannot recover after its prefix leaves the selected source."""
    marker = '"quotes":['
    start = raw.find(marker)
    if start < 0:
        return None
    offset, decoder = start + len(marker), json.JSONDecoder()
    while offset < len(raw) and raw[offset:offset + 1] == "{":
        fragment = raw[offset:]
        try:
            item, end = decoder.raw_decode(raw, offset)
        except json.JSONDecodeError:
            position = fragment.find('"text":')
            if position < 0:
                return None
            try:
                item = json.loads(fragment[:position] + '"text":""}')
                tail = fragment[position + len('"text":'):]
                try:
                    text, _ = decoder.raw_decode(tail)
                except json.JSONDecodeError:
                    # A partial escape/Unicode pair must wait for more bytes.
                    text = json.loads(tail + '"')
            except json.JSONDecodeError:
                return None
            if text and 0xD800 <= ord(text[-1]) <= 0xDBFF:
                return None
            item["text"], end = text, None
        text = item.get("text")
        key = item.get("field"), item.get("policy_id")
        if isinstance(text, str) and text and not any(text in source for source in sources.get(key, [])):
            return "quote_prefix_outside_selected_source"
        if end is None or raw[end:end + 1] != ",":
            return None
        offset = end + 1
    return None


class LocationCycleStop(StoppingCriteria):
    """A detected cycle remains an error, not a synthetically completed JSON."""

    def __init__(self, decoded_vocab, eos, pad, rows, sources=None):
        self.vocab, self.eos, self.pad = decoded_vocab, eos, pad
        self.text = [""] * rows
        self.decoders = [codecs.getincrementaldecoder("utf-8")() for _ in range(rows)]
        self.reasons, self.lengths = [None] * rows, [None] * rows
        self.emitted = 0
        self.sources = sources

    def __call__(self, input_ids, scores, **kwargs):
        self.emitted += 1
        for i, token in enumerate(input_ids[:, -1].tolist()):
            if self.reasons[i] or token in (self.eos, self.pad):
                continue
            chunk = self.decoders[i].decode(self.vocab[token])
            self.text[i] += chunk
            reason = impossible_quote_prefix(self.text[i], self.sources[i]) if self.sources is not None else None
            if not reason and any(c in chunk for c in ("}", ".", "!", "?", "\u3002", "\uff01", "\uff1f")):
                reason = location_cycle(self.text[i])
            if reason:
                self.reasons[i], self.lengths[i] = reason, self.emitted
        return torch.tensor([reason is not None for reason in self.reasons], device=input_ids.device)


def open_description_start(raw):
    marker = '"description":'
    offset = raw.rfind(marker)
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


class DescriptionFrequencyPenalty:
    """Count only tokens wholly inside the current free-description string."""

    def __init__(self, state, frequency):
        self.state, self.frequency = state, frequency
        self.counts = None
        self.starts = [None] * len(state.text)

    def __call__(self, input_ids, scores):
        starts = [open_description_start(raw) for raw in self.state.text]
        if self.counts is None:
            self.counts = torch.zeros_like(scores)
        changed = torch.tensor([a != b for a, b in zip(starts, self.starts)], device=scores.device)
        self.counts[changed] = 0
        continuing = torch.tensor([a is not None and a == b for a, b in zip(starts, self.starts)],
                                  device=scores.device, dtype=scores.dtype)
        self.counts.scatter_add_(1, input_ids[:, -1:], continuing[:, None])
        self.starts = starts
        # Closing/opening JSON tokens are not counted. Outside a description the
        # row has zero counts, including quoted text, IDs, coordinates and EOS.
        return scores - self.frequency * self.counts


def location_generation_components(grammars, encoder, nodes):
    import xgrammar as xgr
    from .decode import BatchedGrammarLogitsProcessor

    if not hasattr(encoder, "_location_vocab"):
        encoder._location_vocab = grammars[0].tokenizer_info.decoded_vocab
    tokenizer = encoder.processor.tokenizer
    sources = []
    for node in nodes:
        row = {}
        for span in node.payload["witness"]["spans"]:
            row.setdefault((span["field"], None), []).append(span["text"])
        for card in node.payload.get("policy_deck", []):
            row["policy_rule", card["policy_id"]] = [card["rule_text"]]
        sources.append(row)
    stop = LocationCycleStop(encoder._location_vocab, tokenizer.eos_token_id,
                             tokenizer.pad_token_id, len(grammars), sources)
    frequency = encoder.location_description_frequency
    penalty = DescriptionFrequencyPenalty(stop, frequency) if frequency else None

    class LocationGrammar(BatchedGrammarLogitsProcessor):
        def __call__(self, input_ids, scores):
            scores = super().__call__(input_ids, scores)
            return penalty(input_ids, scores) if penalty is not None else scores

        def advance(self, tokens):
            # HF pads rows stopped by our criterion; padding is not a JSON token.
            self.active = [i for i in self.active if stop.reasons[i] is None]
            super().advance(tokens)

    return LocationGrammar(grammars, xgr.BatchGrammarMatcher(max_threads=2)), stop
