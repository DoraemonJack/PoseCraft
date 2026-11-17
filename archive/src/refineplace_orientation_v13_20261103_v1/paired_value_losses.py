"""Paired GT-position yaw supervision; never official success/negative-position labels."""
import math
import torch
from torch.nn import functional as F


def _groups(group_ids, n, device):
    ids = torch.as_tensor(group_ids, device=device, dtype=torch.long)
    if ids.shape != (n,):
        raise ValueError('One same-position group id per state required')
    return ids


def _group_mean(losses, active, group_ids):
    terms = []
    for group in torch.unique(group_ids):
        mask = (group_ids == group)
        while mask.ndim < active.ndim:
            mask = mask[..., None]
        mask = mask & active
        if bool(mask.any()):
            terms.append(losses[mask].mean())
    return torch.stack(terms).mean() if terms else losses.sum() * 0.


def balanced_mode_loss(logits, stop_mask, group_ids, valid_mask=None):
    """Each position gives half mass to STOP and half to ROTATE when both exist.

    All-valid-heading groups have only STOP and receive their own full weight.
    No nonexistent ROTATE samples are fabricated for such groups.
    """
    if logits.ndim != 2 or logits.shape[1] != 2 or not bool(torch.isfinite(logits).all()):
        raise ValueError('Finite mode logits[N,2], STOP=0 ROTATE=1 required')
    stop = torch.as_tensor(stop_mask, device=logits.device, dtype=torch.bool)
    active = torch.ones(len(logits), device=logits.device, dtype=torch.bool) if valid_mask is None else torch.as_tensor(valid_mask, device=logits.device, dtype=torch.bool)
    if stop.shape != active.shape or active.shape != (len(logits),):
        raise ValueError('Mode masks must be [N]')
    ids = _groups(group_ids, len(logits), logits.device)
    losses = F.cross_entropy(logits, (~stop).long(), reduction='none')
    groups = []
    for group in torch.unique(ids):
        classes = []
        for flag in (True, False):
            mask = (ids == group) & active & (stop == flag)
            if bool(mask.any()):
                classes.append(losses[mask].mean())
        if classes:
            groups.append(torch.stack(classes).mean())
    return torch.stack(groups).mean() if groups else losses.sum() * 0.


def paired_value_losses(predictions, targets, group_ids, *, mode_weight=1., direction_weight=.5,
                        task_value_weight=1., collision_weight=.2, known_weight=.1):
    """Separate endpoint heading quality, known collision cost and availability losses.

    targets: stop_mask[N], normalized_delta[N], task_quality[N,A],
    collision_cost[N,A], collision_known[N,A]; optional valid_mask[N] and
    task_quality_known[N,A] and direction_known[N]. Unknown cost targets may be NaN; they are masked
    before arithmetic and never trained as zero-cost/no-collision outcomes.
    """
    q = predictions['task_q']
    if q.ndim != 2 or not bool(torch.isfinite(q).all()):
        raise ValueError('Finite task_q[N,A] required')
    n = len(q); device = q.device
    ids = _groups(group_ids, n, device)
    active = torch.as_tensor(targets.get('valid_mask', torch.ones(n)), device=device, dtype=torch.bool)
    stop = torch.as_tensor(targets['stop_mask'], device=device, dtype=torch.bool)
    direction = torch.as_tensor(targets['normalized_delta'], device=device, dtype=q.dtype)
    direction_known = torch.as_tensor(targets.get('direction_known', torch.ones(n)), device=device, dtype=torch.bool)
    quality = torch.as_tensor(targets['task_quality'], device=device, dtype=q.dtype)
    quality_known = torch.as_tensor(targets.get('task_quality_known', torch.ones_like(q)), device=device, dtype=torch.bool)
    cost = torch.as_tensor(targets['collision_cost'], device=device, dtype=q.dtype)
    known = torch.as_tensor(targets['collision_known'], device=device, dtype=torch.bool)
    if active.shape != (n,) or stop.shape != (n,) or direction.shape != (n,) or direction_known.shape != (n,) or any(t.shape != q.shape for t in (quality, quality_known, cost, known)):
        raise ValueError('Paired target dimensions disagree')
    if not bool(torch.isfinite(direction[active]).all()) or bool((direction[active].abs() > 1+1e-6).any()):
        raise ValueError('Normalized angular targets must be finite in [-1,1]')
    quality_active = quality_known & active[:, None]
    cost_active = known & active[:, None]
    if (not bool(torch.isfinite(quality[quality_active]).all()) or not bool(torch.isfinite(cost[cost_active]).all())
        or bool(((quality[quality_active]<0)|(quality[quality_active]>1)).any())
        or bool(((cost[cost_active]<0)|(cost[cost_active]>1)).any())):
        raise ValueError('Known endpoint qualities/costs must be finite in [0,1]')
    quality = torch.where(quality_active, quality, torch.zeros_like(quality))
    cost = torch.where(cost_active, cost, torch.zeros_like(cost))
    mode = balanced_mode_loss(predictions['mode_logits'], stop, ids, active)
    directional = F.smooth_l1_loss(predictions['normalized_delta'], torch.where(active, direction, torch.zeros_like(direction)), reduction='none')
    direction_active = active & ~stop & direction_known
    directional = _group_mean(directional, direction_active, ids)
    task_loss = _group_mean(F.smooth_l1_loss(q, quality, reduction='none'), quality_active, ids)
    collision = _group_mean(F.smooth_l1_loss(predictions['collision_cost'], cost, reduction='none'), cost_active, ids)
    availability = _group_mean(F.binary_cross_entropy_with_logits(predictions['collision_known_logits'], known.to(q.dtype), reduction='none'), active[:, None].expand_as(known), ids)
    loss = mode_weight*mode + direction_weight*directional + task_value_weight*task_loss + collision_weight*collision + known_weight*availability
    return dict(loss=loss, mode_loss=mode, direction_loss=directional, task_value_loss=task_loss,
                collision_loss=collision, collision_known_loss=availability,
                stop_count=int((active&stop).sum()), rotate_count=int((active&~stop).sum()),
                direction_supervised_states=int(direction_active.sum()),
                known_collision_actions=int(cost_active.sum()), unknown_collision_actions=int((active[:,None]&~known).sum()),
                groups=int(len(torch.unique(ids[active]))))


def paired_supervision_loss(outputs, valid_mask8, collision_cost, collision_known, *,
                            mode_weight=1., direction_weight=.5, task_value_weight=1.,
                            advantage_weight=.25, collision_weight=.2, known_weight=.1):
    """All eight starts at one TRAIN GT position, three actions per state.

    State order is position-major then yaw 0..7. Relative action order is
    [-45,0,+45] degrees. Endpoint proxy R = .5*valid + .5*angular_quality,
    where angular_quality compares only to valid headings AT THIS POSITION.
    R is a training reward proxy, not official placement-success probability.
    This retains a nonzero improvement signal even 135/180 degrees from a
    valid heading. It does not label arbitrary unannotated positions invalid.
    """
    q = outputs['task_q']; device = q.device
    valid = torch.as_tensor(valid_mask8, device=device, dtype=torch.bool)
    if valid.ndim == 3 and valid.shape[1:] == (1, 8):
        valid = valid[:, 0]
    if valid.ndim != 2 or valid.shape[1] != 8 or not bool(valid.any(-1).all()):
        raise ValueError('Each paired GT position needs a complete [8] valid-yaw mask with a positive')
    b = len(valid)
    if q.shape != (b*8, 3):
        raise ValueError('Expected outputs[B*8,3], position-major yaw order 0..7')
    actions = outputs['action_delta']
    expected = q.new_tensor([-math.pi/4, 0., math.pi/4])[None].expand(b*8, -1)
    if actions.shape != expected.shape or not torch.allclose(actions, expected, atol=1e-6, rtol=0):
        raise ValueError('Paired action order must be [-45,0,+45] degrees')
    cost = torch.as_tensor(collision_cost, device=device, dtype=q.dtype)
    known = torch.as_tensor(collision_known, device=device, dtype=torch.bool)
    if cost.shape != (b,8) or known.shape != (b,8):
        raise ValueError('Collision observations must be [B,8] at the same exact positions')
    angles = torch.arange(8, device=device, dtype=q.dtype)*(math.pi/4)
    difference = angles[None, None, :] - angles[None, :, None]
    difference = torch.remainder(difference+math.pi, 2*math.pi)-math.pi
    difference = difference.expand(b,-1,-1)
    angular_quality = (1+difference.cos().masked_fill(~valid[:,None,:],-torch.inf).max(-1).values)/2
    reward_by_yaw = .5*valid.to(q.dtype)+.5*angular_quality
    endpoint_bin = (torch.arange(8,device=device)[:,None]+torch.tensor([-1,0,1],device=device)[None])%8
    endpoint_rewards = reward_by_yaw[:,endpoint_bin].reshape(b*8,3)
    # Supervise direction only if its endpoint utility distinguishes left/right.
    # In particular, 180 degrees from a single valid heading has two equally
    # good moves. An arbitrary nearest-yaw tie-break must not oppose equal Qs.
    right_minus_left = endpoint_rewards[:,2]-endpoint_rewards[:,0]
    direction_known = right_minus_left.abs()>1e-6
    direction_target = torch.where(direction_known,right_minus_left.sign(),torch.zeros_like(right_minus_left))
    targets = dict(stop_mask=valid.reshape(-1), normalized_delta=direction_target,
                   direction_known=direction_known, task_quality=endpoint_rewards,
                   collision_cost=cost[:,endpoint_bin].reshape(b*8,3),
                   collision_known=known[:,endpoint_bin].reshape(b*8,3))
    groups = torch.arange(b,device=device).repeat_interleave(8)
    result = paired_value_losses(outputs, targets, groups, mode_weight=mode_weight,
        direction_weight=direction_weight, task_value_weight=task_value_weight,
        collision_weight=collision_weight, known_weight=known_weight)
    # Fit relative values against STOP as well as absolute soft rewards.
    predicted_advantage = q-q[:,1:2]
    desired_advantage = targets['task_quality']-targets['task_quality'][:,1:2]
    advantage = F.smooth_l1_loss(predicted_advantage[:,[0,2]], desired_advantage[:,[0,2]])
    result['advantage_loss'] = advantage
    result['loss'] = result['loss']+advantage_weight*advantage
    result['total_loss'] = result['loss']
    result['task_reward_targets'] = targets['task_quality'].detach()
    result['target_normalized_delta'] = targets['normalized_delta'].detach()
    result['target_direction_mask'] = (targets['direction_known'] & ~targets['stop_mask']).detach()
    result['target_stop_mask'] = targets['stop_mask'].detach()
    result['endpoint_bin_indices'] = endpoint_bin.detach()
    return result
