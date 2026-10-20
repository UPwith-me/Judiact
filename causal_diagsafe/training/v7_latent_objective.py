"""Finite-proposal marginal learning; proposal probabilities are not teacher opinions."""

import math

import torch


def stratified_log_measure(student_logq, matches_bank, has_bank):
    """Sum the fixed bank atom exactly; importance-sample its complement.

    No probability under the teacher is invented. Student proposal probabilities
    come directly from sampling, including constraints/EOS. Draws matching the
    bank atom contribute zero, because that atom is already included exactly.
    """
    m = len(student_logq)
    values = -torch.as_tensor(student_logq, dtype=torch.float64) - math.log(m)
    if has_bank:
        values = values.masked_fill(torch.as_tensor(matches_bank, dtype=torch.bool), -math.inf)
        values = torch.cat((values.new_zeros(1), values))
    return values


def candidate_weights(log_prior, log_observed, log_q=None, *, log_measure=None):
    """Detached SNIS weights, with a zero-mean prior-score control variate.

    In the population limit E_prior[grad log p(z)] = 0. Subtracting this score
    avoids reinforcing random sampled prose when observed likelihood is flat.
    Self-normalization is a finite-sample approximation, not exact EM.
    """
    with torch.no_grad():
        ratio = log_prior.detach().double() + (
            log_measure.detach().double() if log_measure is not None else -log_q.detach().double())
        prior = ratio.softmax(0)
        joint = ratio + log_observed.detach().double()
        supported = torch.isfinite(joint)
        if not supported.any():
            return None
        posterior = joint.softmax(0)
        return {
            "posterior": posterior,
            "prior": prior,
            "latent_score": posterior - prior,
            "effective_candidates": float(1 / posterior.square().sum()),
            "posterior_entropy": float(-(posterior * posterior.clamp_min(1e-300).log()).sum()),
            "supported_candidates": int(supported.sum()),
            "max_weight": float(posterior.max()),
        }


def marginal_surrogate(log_prior, log_observed, weights):
    """Gradient surrogate; its scalar value is not a marginal NLL."""
    p = weights["posterior"].to(log_observed)
    c = weights["latent_score"].to(log_prior)
    # Invalid paths have zero posterior, not 0 * -inf in the objective.
    return -(c * log_prior).sum() - (p[p > 0] * log_observed[p > 0]).sum()
