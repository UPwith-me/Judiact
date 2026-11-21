"""One differentiable source prefill for all fixed-placeholder audit readouts."""

from collections import OrderedDict
from contextlib import contextmanager, nullcontext

import torch
from torch.nn import functional as F

from ..shared_audit import SharedAuditNode
from ..tensor_inputs import move_tensors
from .sequence import NodeEncoder


@contextmanager
def candidate_head(model, token_ids):
    """Project only the candidates used by the conditional classification loss."""
    head = model.get_output_embeddings()
    assert isinstance(head, torch.nn.Linear) and not any(p.requires_grad for p in head.parameters())
    indices = torch.tensor(token_ids, device=head.weight.device)
    original = head.forward

    def project(hidden):
        # Read inside forward, after FSDP's normal parameter materialization.
        weight = head.weight.index_select(0, indices)
        bias = head.bias.index_select(0, indices) if head.bias is not None else None
        return F.linear(hidden, weight, bias)

    head.forward = project
    try:
        yield
    finally:
        head.forward = original


class SharedAuditEncoder(NodeEncoder):
    def __init__(self, *args, location_description_frequency=.1, candidate_head_projection=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.candidate_head_projection = candidate_head_projection
        self.location_description_frequency = location_description_frequency
        placeholder = self.helper.token_ids(" ?")
        assert len(placeholder) == 1, "The audit placeholder must occupy one token"
        self.placeholder_id = placeholder[0]
        self.shared_prepared = OrderedDict()
        self.track_greedy_probability = False

    def generation_components(self, nodes, grammars, sample=False):
        if not sample and all(node.name.startswith("evidence_") for node in nodes):
            from .location_decode import location_generation_components
            return location_generation_components(grammars, self, nodes)
        return super().generation_components(nodes, grammars, sample=sample)

    @staticmethod
    def context_key(node):
        return node.text, node.media

    def shared_input(self, node):
        key = self.context_key(node)
        if key in self.shared_prepared:
            self.shared_prepared.move_to_end(key)
            return self.shared_prepared[key]
        base = super().prepare(node)
        parts = [base["input_ids"].flatten()]
        length = parts[0].numel()
        positions = {}
        for slot in node.slots:
            header = torch.tensor(self.helper.token_ids(slot.header), dtype=torch.long)
            parts.append(header)
            length += header.numel()
            positions[slot.name] = length - 1
            parts.append(torch.tensor([self.placeholder_id]))
            length += 1
        inputs = dict(base, input_ids=torch.cat(parts).reshape(1, -1),
                      attention_mask=torch.ones(1, length, dtype=torch.long))
        result = (inputs, positions)
        self.shared_prepared[key] = result
        while len(self.shared_prepared) > self.maximum:
            self.shared_prepared.popitem(last=False)
        return result

    def prepare(self, node):
        if isinstance(node, SharedAuditNode):
            return self.shared_input(node)[0]
        return super().prepare(node)

    def shared_log_probabilities(self, model, nodes):
        owners, unique, mapping = {}, [], []
        for node in nodes:
            key = self.context_key(node)
            if key not in owners:
                owners[key] = len(unique)
                unique.append(node)
            mapping.append(owners[key])
        prepared = [self.shared_input(node) for node in unique]
        bases = [item[0] for item in prepared]
        inputs = self.helper.batch_inputs(bases, [base["input_ids"].flatten() for base in bases])
        width = inputs["input_ids"].shape[1]
        positions = [width - bases[owner]["input_ids"].numel() + prepared[owner][1][node.name]
                     for node, owner in zip(nodes, mapping)]
        selected = sorted(set(positions))
        output_index = {position: i for i, position in enumerate(selected)}
        device = next(model.parameters()).device
        inputs = move_tensors(inputs, device)
        # Select LM projection positions, not attention positions. The complete
        # causal prefix remains in the autograd graph, including earlier slots.
        inputs["logits_to_keep"] = torch.tensor(selected, device=device)
        candidates = [self.candidate_ids(node) for node in nodes]
        vocabulary = sorted({token for ids in candidates for token in ids})
        projection = candidate_head(model, vocabulary) if self.candidate_head_projection else nullcontext()
        if self.candidate_head_projection:
            remap = {token: i for i, token in enumerate(vocabulary)}
            candidates = [[remap[token] for token in ids] for ids in candidates]
        with self.visual_context(model, unique), projection:
            logits = model(**inputs, use_cache=False).logits
        return [logits[owner, output_index[position], ids].float().log_softmax(-1)
                for ids, owner, position in zip(candidates, mapping, positions)]

    def probability_batch(self, model, examples):
        shared = [isinstance(e.node, SharedAuditNode) for e in examples]
        if not any(shared):
            return super().probability_batch(model, examples)
        assert all(shared), "Shared audit and standalone targets require separate forward jobs"
        probabilities = self.shared_log_probabilities(model, [e.node for e in examples])
        result = []
        for example, logp in zip(examples, probabilities):
            if example.node.forced_value is not None:
                forced = example.node.labels.index(example.node.forced_value)
                fixed = torch.full_like(logp, -torch.inf)
                fixed[forced] = 0.
                logp = logp.sum() * 0 + fixed
            result.append((logp[example.node.labels.index(example.target)], logp))
        return result

    @torch.no_grad()
    def generate(self, model, nodes, sample=False, generator=None, **kwargs):
        if not any(isinstance(node, SharedAuditNode) for node in nodes):
            outputs = super().generate(model, nodes, sample=sample, generator=generator, **kwargs)
            if not sample and all(node.name.startswith("evidence_") for node in nodes):
                for output in outputs:
                    output["generation_controls"]["description_frequency_penalty"] = self.location_description_frequency
                    output["generation_controls"]["description_frequency_scope"] = "current free description only; no quote/structure/prompt counts"
            return outputs
        assert all(isinstance(node, SharedAuditNode) for node in nodes)
        was_training = model.training
        model.eval()
        try:
            probabilities = self.shared_log_probabilities(model, nodes)
            picks = [torch.multinomial(p.exp(), 1, generator=generator).squeeze(0) if sample
                     else p.argmax() for p in probabilities]
            picks = torch.stack(picks).cpu().tolist()
            flat = torch.cat(probabilities).cpu().tolist()
            result, offset = [], 0
            for node, pick, logp in zip(nodes, picks, probabilities):
                size = logp.numel()
                value = node.forced_value if node.forced_value is not None else node.labels[pick]
                result.append({"value": value, "log_probabilities": flat[offset:offset + size],
                               "forced_by_contract": node.forced_value, "error": None})
                offset += size
            return result
        finally:
            model.train(was_training)
