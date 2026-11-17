"""Separate task-return and collision-cost actor critics for the joint treatment.

This is a sampled Lagrangian actor-critic, not a CPO implementation or a
per-output guarantee. Unknown geometry is masked, with coverage reported.
"""
import numpy as np
import torch
from torch.nn import functional as F


def collision_target(stats):
    if bool(stats.get('collision_cost_known', False)):
        return float(np.clip(stats['collision_cost'], 0., 1.)), True, 'continuous_asset_interior'
    if bool(stats.get('scene_geometry_available', False)) and stats['collision_noncontact_pairs'] > 0:
        return 1., True, 'observed_surface_crossing_fallback'
    if bool(stats.get('scene_geometry_available', False)) and bool(stats.get('collision_complete', False)):
        # This is a zero OBSERVED CROSSING cost, not a signed-distance target or
        # certification of unobserved free space/absence of full containment.
        return 0., True, 'observed_surface_nonintersection_proxy'
    return 0., False, 'unknown_not_a_clearance_label'


def support_bonus(stats):
    if stats['scene_geometry_available'] and stats['support_known_fraction'] >= .65:
        return .1 * (stats['support_contact_fraction'] - stats['support_gap_fraction'])
    return 0.


def constrained_rl_losses(results, qualities, *, collision_multiplier, rl_weight=.5):
    """Use column 0 for task return and column 1 for collision cost-to-go."""
    losses = []
    collision_costs, known_masks, modes = [], [], []
    for result, quality in zip(results, qualities):
        c, known, mode = collision_target(result['geometry'])
        collision_costs.append(c); known_masks.append(known); modes.append(mode)
        values = torch.stack(result['residual_values'])
        local_logp = torch.stack(result['residual_logprobs'])
        entropy = result['candidate_entropy'] + torch.stack(result['entropies']).mean()
        costs = np.asarray(result['step_costs'], float)
        remaining_action_cost = np.cumsum(costs[::-1])[::-1].copy()
        terminal = float(quality) + support_bonus(result['geometry'])
        task_returns = values.new_tensor(terminal - remaining_action_cost)
        task_adv = task_returns - values[:, 0]
        scene_task_adv = task_returns[0] - result['scene_values'][0]
        cost_adv = values.new_tensor(c) - values[:, 1]
        scene_cost_adv = result['scene_values'].new_tensor(c) - result['scene_values'][1]
        mult = float(collision_multiplier) if known else 0.
        actor = -result['candidate_logprob'] * (scene_task_adv - mult * scene_cost_adv).detach()
        actor = actor - (local_logp * (task_adv - mult * cost_adv).detach()).sum()
        critic = .5 * (scene_task_adv.square() + task_adv.square().mean())
        if known:
            critic = critic + .5 * (scene_cost_adv.square() + cost_adv.square().mean())
        losses.append(rl_weight * (actor + critic - .005 * entropy))
    return dict(rl_loss=torch.stack(losses), collision_costs=collision_costs,
                collision_known=known_masks, collision_modes=modes)


def update_multiplier(value, costs, known, *, budget=.05, rate=.05):
    selected = np.asarray(costs, float)[np.asarray(known, bool)]
    if not len(selected):
        return float(value)
    return float(np.clip(value + rate * (selected.mean() - budget), .1, 10.))
