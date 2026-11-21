"""Exact history counts and an experimental batched grammar comparison."""

import torch
import xgrammar as xgr


class BatchedGrammarLogitsProcessor:
    """Batch official XGrammar masks and optionally reuse returned CPU tokens."""

    def __init__(self, grammars, batch_matcher, token_history=None):
        self.matchers = [xgr.GrammarMatcher(grammar) for grammar in grammars]
        self.batch_matcher = batch_matcher
        self.vocab_size = grammars[0].tokenizer_info.vocab_size
        self.active = list(range(len(grammars)))
        self.mask = self.device_mask = None
        self.prefilled = False
        self.token_history = token_history

    def fill(self):
        if self.active:
            self.batch_matcher.batch_fill_next_token_bitmask(
                [self.matchers[i] for i in self.active], self.mask, self.active)

    def advance(self, tokens):
        accepted = self.batch_matcher.batch_accept_token(
            [self.matchers[i] for i in self.active], [tokens[i] for i in self.active])
        assert all(accepted), "sampled token outside grammar"
        self.active = [i for i in self.active if not self.matchers[i].is_terminated()]
        self.fill()

    def __call__(self, input_ids, scores):
        if self.mask is None:
            self.mask = xgr.allocate_token_bitmask(len(self.matchers), self.vocab_size)
            if scores.is_cuda:
                self.mask = self.mask.pin_memory()
                self.device_mask = torch.empty_like(self.mask, device=scores.device)
            else:
                self.device_mask = self.mask
        if self.prefilled:
            self.advance(self.token_history() if self.token_history is not None else input_ids[:, -1].tolist())
        else:
            self.prefilled = True
            self.fill()
        # Finished rows keep their last mask, exactly as the official HF
        # processor does; HF replaces their sampled token with padding.
        if scores.is_cuda:
            # The sampled-token transfer before advance completes this
            # copy before the CPU reuses its pinned buffer for the next step.
            self.device_mask.copy_(self.mask, non_blocking=True)
        xgr.apply_token_bitmask_inplace(scores, self.device_mask)
        return scores


class IncrementalTokenPenalties:
    """Same output-only counts and FP32 arithmetic, without history rescans."""

    def __init__(self, presence=0., frequency=0.):
        self.presence, self.frequency = presence, frequency
        self.counts = None
        self.length = None

    def __call__(self, input_ids, scores):
        if self.counts is None:
            self.counts = torch.zeros_like(scores)
        else:
            added = input_ids[:, self.length:]
            self.counts.scatter_add_(1, added, torch.ones_like(added, dtype=scores.dtype))
        self.length = input_ids.shape[1]
        return scores - (self.counts > 0).to(scores.dtype) * self.presence - self.counts * self.frequency
