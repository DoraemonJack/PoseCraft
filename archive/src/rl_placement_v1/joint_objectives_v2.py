"""Separate candidate and conditional-residual training objectives.

The caller must compute candidate_baseline from the preselection task state,
without using the sampled candidate. This semantic independence cannot be
inferred from a tensor's shape or autograd graph.
"""
import math

import numpy as np
import torch


def valid_set_loss(scores, positive_mask):
    """Negative log probability of selecting any annotated valid candidate."""
    mask = torch.as_tensor(positive_mask, dtype=torch.bool, device=scores.device)
    if scores.ndim != 1 or mask.shape != scores.shape:
        raise ValueError("scores and positive_mask must be matching vectors")
    if not mask.any():
        return scores.sum() * 0
    return torch.logsumexp(scores, 0) - torch.logsumexp(scores[mask], 0)


def terminal_task_quality(distance_m, scale=.1):
    """Absolute terminal proximity quality; annotations are reward/metric only."""
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    if torch.is_tensor(distance_m):
        return torch.exp(-distance_m / scale)
    if np.ndim(distance_m):
        return np.exp(-np.asarray(distance_m, dtype=float) / scale)
    return math.exp(-float(distance_m) / scale)


def discounted_returns(rewards, gamma=1):
    """Return-to-go as Python floats, for list, NumPy, or tensor rewards."""
    if not math.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0, 1]")
    result = []
    running = 0.0
    for reward in reversed(list(rewards)):
        if torch.is_tensor(reward):
            reward = reward.detach().item()
        running = float(reward) + gamma * running
        result.append(running)
    return result[::-1]


def policy_loss(candidate_logprob, candidate_baseline, candidate_return,
                residual_logprobs, residual_values, residual_returns):
    """Actor/critic losses with distinct preselection and residual baselines.

    Residual terms are averaged over executed decisions (including stop).
    Value losses use half squared error. Returns are detached supervision;
    policy advantages are detached to avoid critic gradients through actors.
    """
    def scalar(value):
        return torch.as_tensor(value, device=candidate_logprob.device,
                               dtype=candidate_logprob.dtype).reshape(())

    candidate_target = scalar(candidate_return).detach()
    candidate_advantage = candidate_target - scalar(candidate_baseline)
    rank_policy = -scalar(candidate_logprob) * candidate_advantage.detach()
    rank_value = .5 * candidate_advantage.square()
    if not (len(residual_logprobs) == len(residual_values) == len(residual_returns)):
        raise ValueError("residual log-probabilities, values, and returns must align")
    if len(residual_logprobs):
        logprobs = torch.stack([scalar(x) for x in residual_logprobs])
        values = torch.stack([scalar(x) for x in residual_values])
        targets = torch.stack([scalar(x).detach() for x in residual_returns])
        advantages = targets - values
        residual_policy = -(logprobs * advantages.detach()).mean()
        residual_value = .5 * advantages.square().mean()
    else:
        residual_policy = scalar(candidate_logprob) * 0
        residual_value = scalar(candidate_baseline) * 0
    total = rank_policy + rank_value + residual_policy + residual_value
    return dict(total=total, rank_policy=rank_policy, rank_value=rank_value,
                residual_policy=residual_policy, residual_value=residual_value)
