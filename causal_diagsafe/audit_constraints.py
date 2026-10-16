"""XGrammar token constraints for the existing continuous audit controller."""

import copy
import json
from types import SimpleNamespace

import numpy as np
import xgrammar as xgr


DECODER_VERSION = "xgrammar_0_2_8_audit_v3_compact"


class AuditGrammar:
    def __init__(self, tokenizer):
        self.eos = tokenizer.eos_token_id
        self.vocab_size = max(tokenizer.get_vocab().values()) + 1
        info = xgr.TokenizerInfo.from_huggingface(tokenizer, vocab_size=self.vocab_size,
                                                stop_token_ids=[self.eos])
        self.compiler = xgr.GrammarCompiler(info, max_threads=2)

    def constraint(self, schema):
        # Match the actual target serialization, including description-before-box.
        schema = copy.deepcopy(schema)

        def order(node):
            if isinstance(node, dict):
                # XGrammar 0.2.8's length-constrained string grammar rejects JSON
                # escapes. The existing evidence parser still rejects empty quotes.
                if node.get("type") == "string" and node.get("minLength") == 1:
                    node.pop("minLength")
                props = node.get("properties")
                if props and set(props) == {"description", "media_index", "bbox"}:
                    node["properties"] = {name: props[name] for name in ("description", "media_index", "bbox")}
                for value in node.values():
                    order(value)
            elif isinstance(node, list):
                for value in node:
                    order(value)

        order(schema)
        # Match compact training JSON. Unbounded structural whitespace can keep
        # greedy decoding in a newline/space loop without completing the value.
        grammar = self.compiler.compile_json_schema(json.dumps(schema), any_whitespace=False,
                                                    indent=None, separators=(",", ":"))
        return AuditTokenConstraint(grammar, self.eos, self.vocab_size)


class AuditTokenConstraint:
    """Replay/rollback handles the controller's interleaved batch prefixes."""
    def __init__(self, grammar, eos, vocab_size):
        self.matcher = xgr.GrammarMatcher(grammar, override_stop_tokens=[eos])
        self.eos, self.vocab_size = eos, vocab_size
        self.mask = xgr.allocate_token_bitmask(1, vocab_size)
        self.prefix = []
        self.token_enforcer = self
        self.prompt_length = None

    def get_allowed_tokens(self, tokens):
        tokens = list(tokens)
        if self.eos in tokens:
            tokens = tokens[:tokens.index(self.eos) + 1]
        common = 0
        while common < min(len(tokens), len(self.prefix)) and tokens[common] == self.prefix[common]:
            common += 1
        if common < len(self.prefix):
            self.matcher.rollback(len(self.prefix) - common)
        for token in tokens[common:]:
            if not self.matcher.accept_token(token):
                raise ValueError(f"Audit decoder accepted an illegal token: {token}")
        self.prefix = tokens
        if self.matcher.is_terminated():
            allowed = [self.eos]
        else:
            self.matcher.fill_next_token_bitmask(self.mask)
            bits = np.unpackbits(self.mask.numpy().view(np.uint8), bitorder="little")[:self.vocab_size]
            allowed = np.flatnonzero(bits).tolist()
        return SimpleNamespace(allowed_tokens=allowed)

    def __call__(self, _row, input_ids):
        if self.prompt_length is None:
            self.prompt_length = len(input_ids)
        return self.get_allowed_tokens(input_ids[self.prompt_length:].tolist()).allowed_tokens
