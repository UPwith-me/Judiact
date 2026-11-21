"""Field-local four-action objectives; inference and matching use raw scores."""

import torch


def action_nll(log_probabilities, target_index, prior, objective, target_budget=None):
    if objective == "natural":
        return -log_probabilities[target_index]
    prior = torch.as_tensor(prior, dtype=log_probabilities.dtype,
                            device=log_probabilities.device)
    if objective == "equal_class":
        return -log_probabilities[target_index] / (len(prior) * prior[target_index])
    if objective == "target_budget":
        return -log_probabilities[target_index] * target_budget[target_index] / prior[target_index]
    if objective == "logit_adjusted":
        adjusted = log_probabilities + prior.log()
        return adjusted.logsumexp(0) - adjusted[target_index]
    raise ValueError(objective)
