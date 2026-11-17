"""V4 annotation targets. Centers are bottom-face points, dimensions stay XYZ.

Refer points are raw mesh minus its mean. Centering cancels axis-alignment
translation: aligned_centered = raw_centered @ R.T. A world bounding-box
center is not a bottom-face center and must never substitute for these points.
"""
from pathlib import Path
import numpy as np


def _ids(value, name):
    a = np.asarray(value)
    if a.ndim != 1 or a.dtype.kind not in 'iu' or np.any(a < 0):
        raise ValueError(name + ' must be nonnegative integer vector')
    return a


def _validate_rotation_masks(rotations, masks):
    r = np.asarray(rotations, dtype=float)
    m = np.asarray(masks)
    if r.ndim != 1 or len(r) > 8 or not np.isfinite(r).all():
        raise ValueError('rotations must contain at most eight finite degrees')
    bins = np.rint(r / 45.)
    if not np.allclose(r / 45., bins, atol=1e-7, rtol=0):
        raise ValueError('rotations must be multiples of 45 degrees')
    bins = bins.astype(np.int64) % 8
    if len(np.unique(bins)) != len(bins):
        raise ValueError('duplicate canonical rotations')
    if m.ndim != 1 or m.dtype.kind not in 'iu' or np.any(m < 0) or np.any(m > 255):
        raise ValueError('masks must be uint8-compatible integer vector')
    if np.any(m.astype(np.uint64) >= (1 << len(r))):
        raise ValueError('mask references missing local rotation')
    return bins, m.astype(np.uint8, copy=False)


def decode_valid_rotations(rotations, masks):
    """Decode bit i against rotations[i], never against canonical bin i."""
    bins, m = _validate_rotation_masks(rotations, masks)
    out = np.zeros((len(m), 8), dtype=bool)
    for i, b in enumerate(bins):
        out[:, b] = (m.astype(np.uint64) & (1 << i)) != 0
    return out


def match_candidate_pose_targets(candidate_indices, valid_point_ids, rotations, masks):
    """Join original scene point IDs exactly; duplicate annotations union labels."""
    candidates = _ids(candidate_indices, 'candidate_indices')
    ids = _ids(valid_point_ids, 'valid_point_ids')
    # Validate ALL annotations before matching, including unselected masks.
    _, local_masks = _validate_rotation_masks(rotations, masks)
    if len(ids) != len(local_masks):
        raise ValueError('point IDs and masks must have equal lengths')
    matched_masks = np.zeros(len(candidates), dtype=np.uint8)
    if len(ids) and len(candidates):
        # A shared unsigned type keeps exact integer equality even when one
        # source is uint64 and the other int64 (float promotion would lose IDs).
        ids = ids.astype(np.uint64, copy=False)
        candidates = candidates.astype(np.uint64, copy=False)
        order = np.argsort(ids)
        sorted_ids = ids[order]
        starts = np.r_[0, np.flatnonzero(sorted_ids[1:] != sorted_ids[:-1]) + 1]
        unique_ids = sorted_ids[starts]
        union_masks = np.bitwise_or.reduceat(local_masks[order], starts)
        positions = np.searchsorted(unique_ids, candidates)
        in_range = positions < len(unique_ids)
        matching = np.flatnonzero(in_range)
        matching = matching[unique_ids[positions[matching]] == candidates[matching]]
        matched_masks[matching] = union_masks[positions[matching]]
    return decode_valid_rotations(rotations, matched_masks)


def joint_valid_set_loss(logits, mask, return_stats=False):
    """Negative log probability of the joint valid set, globally over C x 8."""
    import torch
    mask = torch.as_tensor(mask, dtype=torch.bool, device=logits.device)
    if logits.ndim != 2 or logits.shape[1] != 8 or mask.shape != logits.shape:
        raise ValueError('logits and mask must have shape [C,8]')
    count = int(mask.sum().item())
    loss = (torch.logsumexp(logits.reshape(-1), 0) - torch.logsumexp(logits[mask], 0)
            if count else logits.sum() * 0.)
    stats = {'positive_count': count, 'has_positive': bool(count)}
    return (loss, stats) if return_stats else loss


def pose_quality_and_metrics(center, yaw_aligned, valid_xyz, valid_mask8,
                             spatial_scale=.1, angular_scale=1.):
    """Best JOINT pair: exp(-min(||delta/.1||^2 + scale*(1-cos(dyaw)))).

    Only N-sized work arrays are used. nearest_angle_rad refers to an allowed
    angle at the nearest labeled position, while joint_best_* share one pair.
    """
    xyz = np.asarray(valid_xyz, dtype=float)
    mask = np.asarray(valid_mask8)
    center = np.asarray(center, dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or mask.shape != (len(xyz), 8) or mask.dtype != bool:
        raise ValueError('expected xyz[N,3] and boolean mask[N,8]')
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(xyz).all() or not np.isfinite(yaw_aligned):
        raise ValueError('pose coordinates and yaw must be finite')
    if not np.isfinite(spatial_scale) or spatial_scale <= 0 or not np.isfinite(angular_scale) or angular_scale < 0:
        raise ValueError('invalid quality scales')
    d2 = np.sum((xyz-center)**2, axis=1)
    labeled = mask.any(axis=1)
    if not labeled.any():
        return dict(quality=0., nearest_distance_m=float('inf'), nearest_angle_rad=float('inf'),
                    joint_best_distance_m=float('inf'), joint_best_angle_rad=float('inf'),
                    joint_best_point_index=-1, joint_best_bin=-1, joint_cost=float('inf'),
                    angle_compatible_distance_m=None, has_positive=False)
    nearest = int(np.argmin(np.where(labeled, d2, np.inf)))
    angles = np.abs((float(yaw_aligned)-np.arange(8)*np.pi/4+np.pi) % (2*np.pi)-np.pi)
    # Existential success uses the distance to a position whose OWN allowed
    # yaw matches; never combine a nearby point with another point's angle.
    compatible = mask[:, angles <= np.pi/8 + 1e-8].any(axis=1)
    compatible_distance = float(np.sqrt(d2[compatible].min())) if compatible.any() else None
    best_cost, best_i, best_b = float('inf'), -1, -1
    for b in range(8):
        costs = np.where(mask[:, b], d2/spatial_scale**2 + angular_scale*(1-np.cos(angles[b])), np.inf)
        i = int(np.argmin(costs))
        if costs[i] < best_cost:
            best_cost, best_i, best_b = float(costs[i]), i, b
    return dict(quality=float(np.exp(-best_cost)), nearest_distance_m=float(np.sqrt(d2[nearest])),
                nearest_angle_rad=float(angles[mask[nearest]].min()), joint_best_distance_m=float(np.sqrt(d2[best_i])),
                joint_best_angle_rad=float(angles[best_b]), joint_best_point_index=best_i,
                joint_best_bin=best_b, joint_cost=best_cost,
                angle_compatible_distance_m=compatible_distance, has_positive=True)


def validate_pure_yaw_alignment(alignment):
    a = np.asarray(alignment, dtype=float)
    if (a.shape != (4, 4) or not np.isfinite(a).all()
            or not np.allclose(a[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(a[:3, :3].T @ a[:3, :3], np.eye(3), atol=1e-5)
            or not np.isclose(np.linalg.det(a[:3, :3]), 1., atol=1e-5)
            or not np.allclose(a[2, :3], [0, 0, 1], atol=1e-6)):
        raise ValueError('axis alignment must be a proper homogeneous pure-yaw rotation')
    return a


def load_axis_alignment(path):
    lines = [line for line in Path(path).read_text(encoding='utf-8').splitlines()
             if line.strip().startswith('axisAlignment')]
    if len(lines) != 1:
        raise ValueError('expected exactly one axisAlignment entry')
    values = np.fromstring(lines[0].split('=', 1)[1], sep=' ')
    if values.size != 16:
        raise ValueError('axisAlignment requires 16 numbers')
    return validate_pure_yaw_alignment(values.reshape(4, 4))


def raw_centered_to_aligned_centered(points, alignment):
    a = validate_pure_yaw_alignment(alignment)
    p = np.asarray(points, dtype=float)
    if p.ndim < 1 or p.shape[-1] != 3 or not np.isfinite(p).all():
        raise ValueError('points must have final dimension 3 and be finite')
    return p @ a[:3, :3].T


def raw_yaw_to_aligned(yaw, alignment):
    a = validate_pure_yaw_alignment(alignment)
    return (np.asarray(yaw) + np.arctan2(a[1, 0], a[0, 0]) + np.pi) % (2*np.pi) - np.pi
