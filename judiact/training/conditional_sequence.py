"""Short-policy sampling and replay with the deployed stopping and penalties."""

import torch

from ..tensor_inputs import move_tensors
from .sequence import grammar_masked_logits
from .short_sequence import (ActiveShortRows, ShortConditionFrequency,
                                ShortConditionStop, ShortReviewEncoder)


class ConditionalReviewEncoder(ShortReviewEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.record_sampling_token_logps = True
        self.track_greedy_probability = False
        # The stage-two engine uses full replicas (fsdp_size=1), with unequal
        # local rollout counts. HF's automatic FSDP completion all-reduce hangs.
        self.generation_synced_gpus = False

    def generation_components(self, nodes, grammars, sample=False):
        return super().generation_components(nodes, grammars, sample=False)

    def sample_reviews(self, model, nodes):
        with ActiveShortRows(model, lambda: self._short_decode_state, self.short_compact_after):
            outputs = super().generate(model, nodes, sample=True, text_temperature=1.,
                                       text_presence_penalty=0., text_frequency_penalty=0.)
        for node, output in zip(nodes, outputs):
            # HF pads a row after a custom stop while other rows continue. Those
            # padding draws are not actions in this row's stopped trajectory.
            count = len(output['token_ids']) + int(output['terminated'])
            output['sampling_token_log_probabilities'] = output['sampling_token_log_probabilities'][:count]
            assert len(output['sampling_token_log_probabilities']) == count
            output['log_probability'] = output['policy_log_probability'] = sum(
                output['sampling_token_log_probabilities'])
            output['generation_controls'].update(
                condition_text_frequency_penalty=.1 if node.name == 'short_review_condition' else 0.,
                finished_row_compaction=True,
                log_probability_semantics='actual grammar and short-condition penalty, sampled tokens/EOS only')
        return outputs

    def penalty_history(self, example, ids, vocab_size):
        grammar = self.compiled_grammar(example.node.schema, vocab_size, compact=True)
        tokenizer = self.processor.tokenizer
        state = ShortConditionStop(grammar.tokenizer_info.decoded_vocab,
                                   tokenizer.eos_token_id, tokenizer.pad_token_id, 1)
        penalty = ShortConditionFrequency(state, .1)
        history = []
        for token in ids:
            _, tokens, counts = penalty.sparse_counts()
            history.append((tokens, counts))
            state(torch.tensor([[token]]), None)
        return history

    def token_log_probabilities_batch(self, model, examples):
        inputs, targets = self.supervised_batch(examples)
        inputs = move_tensors(inputs, next(model.parameters()).device)
        with self.visual_context(model, [e.node for e in examples]):
            logits = model(**inputs, use_cache=False).logits
        result = []
        for row, (example, ids) in enumerate(zip(examples, targets)):
            local = logits[row, -len(ids):]
            masks = self.target_grammar_mask(example, ids, local.shape[-1]).to(local.device)
            history = self.penalty_history(example, ids, local.shape[-1])
            tokens = torch.tensor(ids, device=local.device)
            pieces = []
            for start in range(0, len(ids), self.probability_chunk_size):
                end = min(start + self.probability_chunk_size, len(ids))
                values = grammar_masked_logits(local[start:end], masks[start:end])
                rows, penalized, counts = [], [], []
                for offset, (keys, numbers) in enumerate(history[start:end]):
                    rows.extend([offset] * len(keys))
                    penalized.extend(keys)
                    counts.extend(numbers)
                if rows:
                    values[torch.tensor(rows, device=values.device),
                           torch.tensor(penalized, device=values.device)] -= values.new_tensor(counts) * .1
                pieces.append(values.gather(1, tokens[start:end, None]).squeeze(1) - values.logsumexp(-1))
            result.append(torch.cat(pieces))
        return result
