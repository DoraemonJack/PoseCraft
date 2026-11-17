"""Checkpoint-compatible wrapper with observation-only collision action inputs.

Load an old state into ``inner`` first, then wrap. New action outputs initialize
to zero so the initial candidate, offset and local policies are identical.
The original empirical numeric yaw model is preserved; no +X front is assumed.
"""
import torch
from torch import nn
from torch.nn import functional as F


DIRECTIONAL_FEATURE_NAMES = (
    'collision_cost', 'local_scene_evidence_mask', 'closed_asset_interior_proxy_mask',
    'proximity_pos_object_x', 'proximity_neg_object_x', 'proximity_pos_object_y',
    'proximity_neg_object_y', 'sin_yaw', 'cos_yaw', 'horizontal_shape_anisotropy')
DIRECTIONAL_DIM = len(DIRECTIONAL_FEATURE_NAMES)


class DirectionalReferencePosePolicy(nn.Module):
    def __init__(self, inner, local_state_dim=None, directional_dim=DIRECTIONAL_DIM, hidden=96):
        super().__init__()
        self.inner = inner
        self.directional_dim = int(directional_dim)
        if self.directional_dim != DIRECTIONAL_DIM:
            raise ValueError('directional feature schema requires exactly 10 values')
        if local_state_dim is None:
            local_state_dim = inner.base.actor_body[0].in_features
        self.local_state_dim = int(local_state_dim)
        self.collision_body = nn.Sequential(nn.Linear(self.local_state_dim + self.directional_dim + 8, hidden),
                                            nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.action_residual = nn.Linear(hidden, 6)
        self.collision_head = nn.Linear(hidden, 1)
        for module in (self.action_residual, self.collision_head):
            nn.init.zeros_(module.weight)
            nn.init.zeros_(module.bias)
        self.last_collision_prediction = None

    @property
    def base(self): return self.inner.base
    @property
    def reference_head(self): return self.inner.reference_head
    @property
    def yaw_reference(self): return self.inner.yaw_reference
    @property
    def local_reference(self): return self.inner.local_reference
    @property
    def log_std(self): return self.inner.log_std

    def reference_batch(self, **kwargs): return self.inner.reference_batch(**kwargs)
    def candidates_batch(self, *args, **kwargs): return self.inner.candidates_batch(*args, **kwargs)
    def offsets_batch(self, *args, **kwargs): return self.inner.offsets_batch(*args, **kwargs)
    def local_batch(self, state): return self.inner.local_batch(state)

    def local_batch_with_reference(self, state, centers, yaws, task_indices, *,
                                   directional_collision, object_dims):
        n = len(state)
        if state.shape != (n, self.local_state_dim) or centers.shape != (n, 3) or yaws.shape != (n,):
            raise ValueError('invalid current state/pose shapes')
        if directional_collision.shape != (n, self.directional_dim) or object_dims.shape != (n, 3):
            raise ValueError('invalid directional vector or object dimensions')
        if not all(torch.isfinite(x).all() for x in (state, centers, yaws, directional_collision, object_dims)):
            raise ValueError('collision actor inputs must be finite')
        if (object_dims <= 0).any():
            raise ValueError('object dimensions must be positive')
        logits, means, values = self.inner.local_batch_with_reference(state, centers, yaws, task_indices)
        # Observations are detached: auxiliary loss trains prediction, not the
        # measurement target or a geometry implementation that produced it.
        direction = directional_collision.detach()
        pose = torch.cat(((centers.detach() / object_dims.detach().clamp_min(.01)).clamp(-20., 20.),
                          object_dims.detach().clamp_min(.01).log(),
                          yaws.detach().sin()[:, None], yaws.detach().cos()[:, None]), -1)
        hidden = self.collision_body(torch.cat((state, direction, pose), -1))
        residual = self.action_residual(hidden) * direction[:, 1:2].clamp(0., 1.)
        # The actor may observe measured cost, but its auxiliary predictor must
        # not learn the identity map from the scalar target in feature zero.
        cost_direction = direction.clone()
        cost_direction[:, 0] = 0.
        cost_hidden = self.collision_body(torch.cat((state, cost_direction, pose), -1))
        self.last_collision_prediction = self.collision_head(cost_hidden).squeeze(-1).sigmoid()
        return logits + residual[:, :3], means + residual[:, 3:], values


def wrap_checkpoint_policy(inner, original_state, **kwargs):
    """Strict loading prevents silently dropping existing yaw/reference weights."""
    inner.load_state_dict(original_state, strict=True)
    return DirectionalReferencePosePolicy(inner, **kwargs)


def collision_auxiliary_loss(results):
    """Per-task masked cost regression at matching visited PRE-action states.

    A local observed surface alone is insufficient for an interior-cost target:
    the rollout mask also requires the certified closed-asset proxy validity.
    """
    if not results:
        raise ValueError('nonempty result batch required')
    losses = []
    for result in results:
        predictions = result.get('collision_predictions', [])
        targets = result.get('collision_targets', [])
        masks = result.get('collision_target_masks', [])
        if len(predictions) != len(targets) or len(predictions) != len(masks):
            raise ValueError('collision prediction/target identities differ')
        if not predictions:
            scores = result['scores']
            losses.append(scores.masked_fill(~torch.isfinite(scores), 0.).sum() * 0.)
            continue
        predicted = torch.stack(predictions)
        truth = predicted.new_tensor(targets)
        known = predicted.new_tensor(masks)
        if not torch.isfinite(truth).all() or ((truth < 0) | (truth > 1)).any() or ((known != 0) & (known != 1)).any():
            raise ValueError('collision targets must be [0,1] with boolean validity')
        loss = F.smooth_l1_loss(predicted, truth, reduction='none')
        losses.append((loss * known).sum() / known.sum().clamp_min(1.))
    return torch.stack(losses)
