"""Full-eight-angle supervision and position-local numeric-yaw reward.

These objectives are for official, decoded PlaceIt3D training masks. A zero in
that mask means that the pose is not a labelled valid solution to the whole
task; it does NOT identify collision as the reason. In particular this module
must not be fed a sparse list of arbitrarily sampled positives as if it were a
complete mask. Nothing here reads labels in the model's observation forward.

The reward is a training proxy tied to numeric yaw annotations. It is not an
implementation of the official facing evaluator or a semantic-front detector.
"""
from __future__ import annotations

import math
import numpy as np
import torch
from torch.nn import functional as F


def _nonnegative_finite(name, value):
    if not math.isfinite(value) or value < 0:
        raise ValueError(f'{name} must be finite and nonnegative')


def supervised_full8(scores, offsets, labels, keys, candidate_mask=None, *,
                     yaw_weight=1., rank_weight=.5, offset_weight=1.,
                     positive_mass_weight=1., balanced_yaw=True):
    """Return (per_task_loss[B], diagnostics of per-task tensors).

    ``scores`` is [B,C,8], ``offsets`` [B,C,8,2]. Label dictionaries use the
    existing ``positive``, ``rank``, ``offset_xy``, ``offset_mask`` schema.
    ``known`` is retained for audit counts but never masks a numeric yaw.
    Labels may contain fewer than C rows; such trailing padding is excluded.
    If supplied, candidate_mask[B,C] must not activate rows absent in labels.

    Positive-set mass, soft ranking and offsets preserve the old terms, with
    the mass/rank competition now spanning all eight labelled angles. The new
    yaw BCE sees all unpadded candidates and all eight angles. Its default is
    class-balanced: one half of the mean positive BCE plus one half of the
    mean negative BCE (one existing class receives weight one). This prevents
    sparse positives from being swamped by candidate-count-dependent zeros.
    All coefficients are development hyperparameters, not benchmark rules.
    """
    if scores.ndim != 3 or scores.shape[-1] != 8:
        raise ValueError('scores must have shape [B,C,8]')
    b, c, _ = scores.shape
    if offsets.shape != (b, c, 8, 2) or len(keys) != b or b == 0:
        raise ValueError('offset/key batch shape mismatch or empty batch')
    if scores.device != offsets.device:
        raise ValueError('scores and offsets must share a device')
    for name, value in [('yaw_weight', yaw_weight), ('rank_weight', rank_weight),
                        ('offset_weight', offset_weight),
                        ('positive_mass_weight', positive_mass_weight)]:
        _nonnegative_finite(name, value)
    if candidate_mask is not None:
        candidate_mask = torch.as_tensor(candidate_mask, device=scores.device)
        if candidate_mask.dtype != torch.bool or candidate_mask.shape != (b, c):
            raise ValueError('candidate_mask must be bool [B,C]')
    losses = []
    report = {name: [] for name in (
        'positive_mass_loss', 'rank_loss', 'yaw_bce_loss', 'offset_loss',
        'positive_bce', 'negative_bce', 'positive_count', 'negative_count',
        'valid_candidate_count', 'old_known_yaw_count')}
    for i, key in enumerate(keys):
        z = labels[key]
        positive = np.asarray(z['positive'])
        if positive.dtype != bool or positive.ndim != 2 or positive.shape[1] != 8:
            raise ValueError('positive must be a decoded boolean [N,8] mask')
        n = positive.shape[0]
        if n > c:
            raise ValueError('more label candidates than model candidates')
        rank = np.asarray(z['rank'])
        target = np.asarray(z['offset_xy'])
        om = np.asarray(z['offset_mask'])
        if rank.shape != (n, 8) or target.shape != (n, 8, 2) or om.shape != (n, 8):
            raise ValueError('label arrays must share the candidate/angle axes')
        if om.dtype != bool:
            raise ValueError('offset_mask must be boolean')
        valid_rows = torch.arange(c, device=scores.device) < n
        if candidate_mask is not None:
            if bool((candidate_mask[i] & ~valid_rows).any()):
                raise ValueError('active candidate has no label row')
            valid_rows = valid_rows & candidate_mask[i]
        mask = valid_rows[:, None].expand(c, 8)
        if not bool(torch.isfinite(scores[i][mask]).all()):
            raise ValueError('valid candidate logits must be finite')
        # Padding is often -inf. Never use scores.sum()*0 on such arrays.
        zero = (scores[i].masked_fill(~torch.isfinite(scores[i]), 0.).sum()
                + offsets[i].masked_fill(~torch.isfinite(offsets[i]), 0.).sum()) * 0.
        mass_loss = rank_loss = yaw_loss = offset_loss = zero
        pos_bce = neg_bce = zero
        positive_count = negative_count = 0
        if bool(valid_rows.any()):
            row_selection = valid_rows[:n].detach().cpu().numpy()
            chosen_rank = rank[row_selection]
            if not np.isfinite(chosen_rank).all() or (chosen_rank < 0).any():
                raise ValueError('valid rank targets must be finite/nonnegative')
            chosen_positive = torch.as_tensor(positive[row_selection],
                                             device=scores.device).reshape(-1)
            logits = scores[i][mask]
            positive_count = int(chosen_positive.sum().item())
            negative_count = int(chosen_positive.numel() - positive_count)
            if positive_count:
                mass_loss = logits.logsumexp(0) - logits[chosen_positive].logsumexp(0)
            probabilities = torch.as_tensor(chosen_rank, device=scores.device,
                                            dtype=scores.dtype).reshape(-1)
            if bool(probabilities.sum() > 0):
                probabilities = probabilities / probabilities.sum()
                rank_loss = -(probabilities * logits.log_softmax(0)).sum()
            bce = F.binary_cross_entropy_with_logits(logits, chosen_positive.to(logits.dtype),
                                                     reduction='none')
            if positive_count:
                pos_bce = bce[chosen_positive].mean()
            if negative_count:
                neg_bce = bce[~chosen_positive].mean()
            if balanced_yaw:
                yaw_loss = (pos_bce + neg_bce) / ((positive_count > 0) + (negative_count > 0))
            else:
                yaw_loss = bce.mean()
            selected_om = torch.as_tensor(om[row_selection], device=scores.device)
            if bool(selected_om.any()):
                predicted_offsets = offsets[i, :n][valid_rows[:n]][selected_om]
                target_offsets = torch.as_tensor(target[row_selection], device=scores.device,
                                                 dtype=offsets.dtype)[selected_om]
                if not (bool(torch.isfinite(predicted_offsets).all())
                        and bool(torch.isfinite(target_offsets).all())):
                    raise ValueError('supervised offsets must be finite')
                offset_loss = F.smooth_l1_loss(predicted_offsets / .15, target_offsets / .15)
        loss = (positive_mass_weight * mass_loss + rank_weight * rank_loss
                + yaw_weight * yaw_loss + offset_weight * offset_loss)
        losses.append(loss)
        for name, value in [('positive_mass_loss', mass_loss), ('rank_loss', rank_loss),
                            ('yaw_bce_loss', yaw_loss), ('offset_loss', offset_loss),
                            ('positive_bce', pos_bce), ('negative_bce', neg_bce)]:
            report[name].append(value)
        known = np.asarray(z.get('known', np.ones(8, bool)), dtype=bool)
        if known.shape != (8,):
            raise ValueError('legacy known mask must have shape [8]')
        for name, value in [('positive_count', positive_count), ('negative_count', negative_count),
                            ('valid_candidate_count', int(valid_rows.sum())),
                            ('old_known_yaw_count', int(known.sum()))]:
            report[name].append(torch.tensor(value, device=scores.device, dtype=torch.int64))
    return torch.stack(losses), {key: torch.stack(value) for key, value in report.items()}


def dense_paired_pose_quality(center, yaw, positive_pairs, *, facing_active=False,
                             orientation_weight=.25, spatial_scale=.1,
                             angular_scale=1., max_orientation_distance=.5,
                             same_position_tolerance=1e-6):
    """Original paired quality plus a bounded position-local angular signal.

    The additional signal selects the nearest labelled position, independently
    of yaw, and compares only with yaw labels AT THAT POSITION. It must not
    select a better-fitting yaw from a different, distant valid position. Ties
    between positions are resolved lexicographically, independent of row order.
    Multiple yaws at the same position remain multiple acceptable targets.

    Angular shaping is enabled only for a caller-confirmed facing task, with a
    positive-pose witness within max_orientation_distance. It equals
    (1+cos(delta))/2 and is not multiplied by exp(-distance**2/.1**2).
    In consequence it remains useful at 0.3 m, while never claiming that the
    current location itself satisfies facing, support or collision. Use this
    as terminal shaping, or difference its potential across steps; do not
    accumulate the same positive state bonus indefinitely.

    Returns quality, paired_quality, angular_quality, enabled, distance, and
    witness diagnostics. No evidence or absent facing disables only the extra
    term. The base paired numeric-pose term remains unchanged.
    """
    center = np.asarray(center, dtype=float)
    pairs = np.asarray(positive_pairs, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all() or not math.isfinite(yaw):
        raise ValueError('finite center[3] and yaw required')
    if pairs.ndim != 2 or pairs.shape[1] != 4 or not np.isfinite(pairs).all():
        raise ValueError('positive_pairs must be finite [P,4]')
    for name, value in [('orientation_weight', orientation_weight), ('angular_scale', angular_scale),
                        ('max_orientation_distance', max_orientation_distance),
                        ('same_position_tolerance', same_position_tolerance)]:
        _nonnegative_finite(name, value)
    if not math.isfinite(spatial_scale) or spatial_scale <= 0:
        raise ValueError('spatial_scale must be positive and finite')
    result = dict(quality=0., paired_quality=0., angular_quality=0., enabled=False,
                  distance=None, has_positive=bool(len(pairs)),
                  nearest_position=None, local_yaw_count=0, angular_error_rad=None,
                  paired_witness_index=None,
                  evidence='no_confirmed_positive_pose_reward_evidence')
    if not len(pairs):
        return result
    d2 = np.square(pairs[:, :3] - center).sum(-1)
    angles = np.arctan2(np.sin(yaw - pairs[:, 3]), np.cos(yaw - pairs[:, 3]))
    costs = d2 / spatial_scale**2 + angular_scale * (1. - np.cos(angles))
    paired_winner = int(np.argmin(costs))
    paired_quality = float(np.exp(-costs[paired_winner]))
    nearest_d2 = float(d2.min())
    nearest_indices = np.flatnonzero(np.isclose(d2, nearest_d2, atol=1e-12, rtol=0.))
    candidates = pairs[nearest_indices, :3]
    index = int(nearest_indices[np.lexsort((candidates[:, 2], candidates[:, 1], candidates[:, 0]))[0]])
    nearest_position = pairs[index, :3]
    at_position = np.linalg.norm(pairs[:, :3] - nearest_position, axis=-1) <= same_position_tolerance
    error = float(np.min(np.abs(angles[at_position])))
    enabled = bool(facing_active and math.sqrt(nearest_d2) <= max_orientation_distance)
    angular_quality = float((1. + math.cos(error)) / 2.) if enabled else 0.
    result.update(quality=paired_quality + orientation_weight * angular_quality,
                  paired_quality=paired_quality, angular_quality=angular_quality,
                  enabled=enabled, distance=math.sqrt(nearest_d2),
                  nearest_position=nearest_position.tolist(),
                  local_yaw_count=int(len(np.unique(pairs[at_position, 3]))),
                  angular_error_rad=error if enabled else None,
                  paired_witness_index=paired_winner,
                  evidence='local_annotated_numeric_yaw_shaping' if enabled
                           else 'paired_pose_only_no_active_local_facing_evidence')
    return result
