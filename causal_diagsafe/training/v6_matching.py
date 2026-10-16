"""Match real inputs to a shared bank of candidate prefixes, not row-specific gold."""

from itertools import permutations

import torch

from ..audit_v5 import audit_slots


def candidate_log_odds(log_probabilities, index):
    """Finite even when exp(logp[index]) rounds to one; invariant to a common logit shift."""
    others = torch.cat((log_probabilities[:index], log_probabilities[index + 1:]))
    return log_probabilities[index] - torch.logsumexp(others, 0)


def crossed_matching_nll(scores, gold_indices):
    """Sum all correct-label allocations, including indistinguishable-label candidates.

    Every allocation uses every input and candidate once. Input-independent
    candidate scores therefore cancel, even when the prefixes reveal the labels.
    The reviewed components in this experiment contain at most four members.
    """
    if scores.dtype in (torch.float16, torch.bfloat16):
        scores = scores.float()
    n = len(gold_indices)
    assert scores.shape == (n, n) and 1 <= n <= 4
    if len(set(gold_indices)) == 1:
        return scores.sum() * 0
    allocations = list(permutations(range(n)))
    indices = torch.tensor(allocations, device=scores.device)
    relative = scores - scores.diag()[:, None]
    values = relative[torch.arange(n, device=scores.device)[None, :], indices].sum(-1)
    correct = torch.tensor([all(gold_indices[j] == gold_indices[i] for i, j in enumerate(p))
                            for p in allocations], device=scores.device)
    return torch.logsumexp(values, 0) - torch.logsumexp(values[correct], 0)


def compatible_prefix_codebooks(cases, field):
    """A transferred letter must still denote the same preceding categorical value."""
    signatures = []
    for case in cases:
        signature = []
        for slot in audit_slots(case["model_input"], case["task_type"]):
            signature.append((slot.name, slot.kind, slot.labels, slot.forced_value))
            if slot.name == field:
                break
        else:
            raise KeyError(field)
        signatures.append(tuple(signature))
    return all(value == signatures[0] for value in signatures)
