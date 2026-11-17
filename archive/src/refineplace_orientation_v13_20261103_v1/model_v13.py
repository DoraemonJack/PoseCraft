"""Frozen V11 placement followed by an isolated action-conditioned yaw value model.

The new task value predicts training-position heading quality, NOT official
full placement success. Collision evidence availability is separately predicted;
neither that prediction nor the selector certifies physical safety.
"""
import math
from pathlib import Path
import sys
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
V11 = HERE.parent / 'refineplace_orientation_v11_20261103_v1'
if not V11.is_dir():
    V11 = Path('/path/to/research300/refineplace_orientation_v11_20261103_v1')
sys.path.insert(0, str(V11))
from model_v11 import RelationBoundPoseResidual, model_from_checkpoint as make_v11

DEFAULT_ACTIONS = (-math.pi / 4, 0., math.pi / 4)


def _head(hidden, outputs=1):
    result = nn.Sequential(nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, outputs))
    nn.init.zeros_(result[-1].weight)
    nn.init.zeros_(result[-1].bias)
    return result


class RefinePlaceV13(nn.Module):
    def __init__(self, baseline, relation_count, hidden=64):
        super().__init__()
        self.baseline = baseline
        self.baseline.requires_grad_(False)
        self.baseline.eval()
        self.value_state = nn.Sequential(nn.LayerNorm(self.local_state_dim), nn.Linear(self.local_state_dim, hidden), nn.SiLU())
        self.value_action = nn.Sequential(nn.Linear(8, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.value_relation = RelationBoundPoseResidual(relation_count, self.local_state_dim, outputs=hidden)
        with torch.no_grad():
            self.value_relation.numeric_frame.bias.copy_(torch.tensor([0., 1.]))
        self.value_body = nn.Sequential(nn.LayerNorm(hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.task_value = _head(hidden)
        self.collision_value = _head(hidden)
        self.collision_availability = _head(hidden)
        # This auxiliary classifier supervises the same state representation.
        # The inference selector uses action values, not its class argmax.
        self.mode_teacher = _head(hidden, 2)

    @property
    def local_state_dim(self):
        return self.baseline.local_state_dim

    @property
    def reference_head(self):
        return self.baseline.reference_head

    @property
    def region_dim(self):
        return self.baseline.region_dim

    def train(self, mode=True):
        super().train(mode)
        self.baseline.eval()
        return self

    def reference_batch(self, **kwargs):
        return self.baseline.reference_batch(**kwargs)

    def region_batch(self, region_features):
        return self.baseline.region_batch(region_features)

    def candidates_batch(self, *args, **kwargs):
        return self.baseline.candidates_batch(*args, **kwargs)

    def action_values(self, state, centers, yaws, task_indices, action_delta=None):
        """Action-conditioned values for N current states and A relative angles.

        ``state`` is [N, local_state_dim] (160 in production), with the current
        frozen fused feature plus current geometric evidence. ``task_indices``
        indexes the batch whose candidates already bound the reference context.
        Positions never move. Actions are radians; stop is the unique zero.
        """
        base = self.baseline
        if base._bound_context is None or base._bound_assets is None:
            raise RuntimeError('Candidate forward must bind references/assets before action values')
        n = len(state)
        if state.ndim != 2 or state.shape[1] != self.local_state_dim or centers.shape != (n, 3) or yaws.shape != (n,):
            raise ValueError('Expected state[N,D], centers[N,3], yaws[N]')
        if task_indices.shape != (n,) or task_indices.dtype != torch.long:
            raise ValueError('task_indices must be long[N]')
        if action_delta is None:
            action_delta = state.new_tensor(DEFAULT_ACTIONS)[None].expand(n, -1)
        if action_delta.ndim != 2 or action_delta.shape[0] != n or action_delta.shape[1] < 2:
            raise ValueError('action_delta must be [N,A], including one stop')
        if any(not bool(torch.isfinite(t).all()) for t in (state, centers, yaws, action_delta)):
            raise ValueError('Finite states and poses required')
        stop = action_delta.abs() <= 1e-7
        if not bool((stop.sum(-1) == 1).all()):
            raise ValueError('Exactly one stop angle per state required')
        if bool((action_delta.abs() > math.pi / 4 + 1e-6).any()):
            raise ValueError('Refinement action exceeds 45 degrees')
        ctx = {k: v.index_select(0, task_indices) for k, v in base._bound_context.items()
               if k not in ('candidate_xyz', 'candidate_mask')}
        ctx['candidate_mask'] = torch.ones((n, 1), device=state.device, dtype=torch.bool)
        next_yaw = yaws[:, None] + action_delta
        current = yaws[:, None].expand_as(action_delta)
        encoded_action = torch.stack((current.cos(), current.sin(), next_yaw.cos(), next_yaw.sin(),
            action_delta.cos(), action_delta.sin(), action_delta / (math.pi / 4), action_delta.abs() / (math.pi / 4)), -1)
        relation = base._bound_forward(self.value_relation, state[:, None], base._bound_assets.index_select(0, task_indices),
            centers[:, None], next_yaw[:, None], ctx, task_indices)[:, 0]
        latent = self.value_body(self.value_state(state)[:, None] + self.value_action(encoded_action) + relation)
        task_logits = self.task_value(latent).squeeze(-1)
        cost_logits = self.collision_value(latent).squeeze(-1)
        known_logits = self.collision_availability(latent).squeeze(-1)
        stop_index = stop.long().argmax(-1)
        mode_logits = self.mode_teacher(latent[torch.arange(n, device=state.device), stop_index])
        # A directional auxiliary target trains the same discrete action scores;
        # no independently executed continuous mean competes with value selection.
        directional_weights = task_logits.masked_fill(stop, -torch.inf).softmax(-1)
        normalized_delta = (directional_weights * action_delta / (math.pi / 4)).sum(-1)
        return dict(task_q=task_logits.sigmoid(), task_q_logits=task_logits,
                    collision_cost=cost_logits.sigmoid(), collision_cost_logits=cost_logits,
                    collision_known_logits=known_logits, mode_logits=mode_logits,
                    normalized_delta=normalized_delta, action_delta=action_delta,
                    next_yaw=torch.remainder(next_yaw + math.pi, 2 * math.pi) - math.pi,
                    stop_index=stop_index)


def select_action(values, *, task_margin=.02, uncertain_margin=.05,
                  maximum_known_cost=.25, allowed_cost_increase=.02, known_threshold=.7):
    """Compare each action to STOP, retaining uncertainty rather than inventing zero risk.

    A predicted-known action must pass a cost comparison. An uncertain action
    needs a larger task improvement, may still be selected, and receives no
    claimed risk improvement. No geometry is queried and no safety is certified.
    """
    if not (0 <= task_margin <= uncertain_margin and 0 <= maximum_known_cost <= 1
            and allowed_cost_increase >= 0 and 0 < known_threshold < 1):
        raise ValueError('Invalid selector thresholds')
    q, cost, known_logits, delta = (values[k] for k in ('task_q', 'collision_cost', 'collision_known_logits', 'action_delta'))
    if q.ndim != 2 or any(t.shape != q.shape for t in (cost, known_logits, delta)):
        raise ValueError('Action value shapes disagree')
    if any(not bool(torch.isfinite(t).all()) for t in (q, cost, known_logits, delta)):
        raise ValueError('Finite predicted values required, including uncertain costs')
    stop = delta.abs() <= 1e-7
    if not bool((stop.sum(-1) == 1).all()):
        raise ValueError('Exactly one zero action required')
    stop_index = stop.long().argmax(-1)
    rows = torch.arange(len(q), device=q.device)
    stop_q, stop_cost = q[rows, stop_index], cost[rows, stop_index]
    predicted_known = known_logits.sigmoid() >= known_threshold
    stop_known = predicted_known[rows, stop_index]
    cost_limit = torch.maximum(cost.new_full(stop_cost.shape, maximum_known_cost),
                               torch.where(stop_known, stop_cost + allowed_cost_increase, cost.new_full(stop_cost.shape, maximum_known_cost)))
    acceptable_risk = ~predicted_known | (cost <= cost_limit[:, None])
    required_gain = torch.where(predicted_known, q.new_tensor(task_margin), q.new_tensor(uncertain_margin))
    gain = q - stop_q[:, None]
    eligible = (~stop) & (gain > required_gain) & acceptable_risk
    selected = q.masked_fill(~eligible, -torch.inf).argmax(-1)
    selected = torch.where(eligible.any(-1), selected, stop_index)
    selected_known = predicted_known[rows, selected]
    selected_cost = cost[rows, selected]
    # Zero denotes no credit, NOT zero collision cost.
    risk_improvement = torch.where(selected_known & stop_known, stop_cost - selected_cost, torch.zeros_like(stop_cost))
    return dict(action_index=selected, action_delta=delta[rows, selected],
                stopped=selected == stop_index, task_gain=gain[rows, selected],
                uncertain=~selected_known, selected_collision_cost=selected_cost,
                risk_improvement_credit=risk_improvement, predicted_known=predicted_known,
                eligible=eligible, required_gain=required_gain, cost_limit=cost_limit,
                safety_guarantee=False)


def select_actions(outputs, *, task_margin=.02, unknown_margin=.05,
                   maximum_known_cost=.25, allowed_cost_increase=.02, known_threshold=.7):
    """Return device action indices and an explicitly JSON-serializable audit."""
    result = select_action(outputs, task_margin=task_margin, uncertain_margin=unknown_margin,
        maximum_known_cost=maximum_known_cost, allowed_cost_increase=allowed_cost_increase,
        known_threshold=known_threshold)
    report = {key: value.detach().cpu().tolist() if isinstance(value,torch.Tensor) else value
              for key,value in result.items()}
    report.update(task_margin=task_margin, unknown_margin=unknown_margin,
                  known_threshold=known_threshold, maximum_known_cost=maximum_known_cost,
                  allowed_cost_increase=allowed_cost_increase,
                  risk_status_source='predicted availability; not verified no-collision or safety guarantee')
    return result['action_index'], report


def model_from_checkpoint(d, types, region_dim, checkpoint=None):
    is_v13 = checkpoint is not None and checkpoint.get('architecture') == 'RefinePlaceV13'
    if checkpoint is not None and not is_v13 and checkpoint.get('architecture') != 'RefinePlaceV11':
        raise ValueError('V13 starts from frozen V11, not a V12 refinement policy')
    baseline = make_v11(d, types, region_dim, None if is_v13 else checkpoint)
    model = RefinePlaceV13(baseline, len(types))
    if is_v13:
        model.load_state_dict(checkpoint['joint_policy'], strict=True)
    return model
