"""Frozen Macro-F1 costs, enumerated actions and a token-sum CISPO surrogate."""

import torch


ALGORITHM = "short_multipath_gold_anchored_cost"
CISPO_REFERENCE = "https://verl.readthedocs.io/en/latest/_modules/verl/trainer/ppo/core_algos.html#compute_policy_loss_cispo"


def multipath_cost_table(confusion):
    c = torch.as_tensor(confusion, dtype=torch.float64).detach().clone()
    assert c.shape == (4, 4) and torch.isfinite(c).all() and (c >= 0).all()
    m, s = c.sum(1), c.sum(0)
    assert (m > 0).all()
    torch.testing.assert_close(m.sum(), m.new_tensor(1.))
    denominator = m + s
    gain = .5 * (torch.diag(denominator.reciprocal()) -
                 (c.diag() / denominator.square())[None, :])
    cost = gain.diag()[:, None] - gain
    scaled = 4 * m[:, None] * cost
    kappa = scaled.max()
    return {"confusion": c.tolist(), "gold_mass": m.tolist(), "predicted_mass": s.tolist(),
            "gain": gain.tolist(), "cost": cost.tolist(), "kappa": kappa.item(),
            "normalized_cost": (scaled / kappa).tolist()}


def action_loss(log_probabilities, target_index, cost_row):
    """All four actions, with a live graph through both CE and expected cost."""
    return -log_probabilities[..., target_index] + (log_probabilities.exp() * cost_row).sum(-1)


def loo_advantages(utilities):
    count = utilities.shape[-1]
    assert count >= 2
    values = utilities.detach()
    return values - (values.sum(-1, keepdim=True) - values) / (count - 1)


def cispo_token_sum(logp, behavior_logp, advantages, policy_mask, trajectory_weights,
                    *, ratio_min=0., ratio_max=2.):
    """verl CISPO token kernel; fixed member weights replace its batch reduction.

    This matches clip_ratio_low=1-ratio_min, clip_ratio_high=ratio_max-1,
    including the upstream log-ratio clamp. No PPO min, length mean, or std.
    Token IS is a local surrogate, not full-trajectory importance sampling.
    """
    assert logp.shape == behavior_logp.shape == policy_mask.shape
    assert trajectory_weights.shape == advantages.shape == logp.shape[:1]
    assert 0 <= ratio_min <= 1 <= ratio_max
    active = policy_mask.bool()
    current = torch.where(active, logp, 0.)
    previous = torch.where(active, behavior_logp.detach(), 0.)
    weights = (current - previous).clamp(-20., 20.).exp().clamp(ratio_min, ratio_max).detach()
    scores = (current * weights).sum(-1)
    return -(trajectory_weights * advantages.detach() * scores).sum()
