"""Positive-only empirical numeric-yaw reward; no semantic-front assumption.

Each reward witness is one confirmed (position, yaw) pair. Missing annotations
are absence of reward evidence, never certified invalid or physical failure.
"""
import numpy as np


AXIS_CONVENTION = 'numeric canonical +X rotated about +Z; semantic asset front unverified'


def numeric_heading(yaw):
    """Numeric basis direction only; callers must not label this asset front."""
    if not np.isfinite(yaw):
        raise ValueError('finite yaw required')
    return np.array([np.cos(yaw), np.sin(yaw), 0.], dtype=float)


def make_positive_pose_pairs(scene_xyz, valid_ids, valid_mask8):
    """Expand canonical boolean masks, already decoded from local bit masks.

    The result [P,4] stores metric XYZ and numeric yaw radians. Sorting and
    deduplication make witness identity deterministic across annotation order.
    """
    xyz = np.asarray(scene_xyz, dtype=float)
    ids = np.asarray(valid_ids)
    masks = np.asarray(valid_mask8)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('scene_xyz must be finite [N,3]')
    if ids.ndim != 1 or ids.dtype.kind not in 'iu' or np.any(ids >= len(xyz)) or np.any(ids < 0):
        raise ValueError('valid_ids must index scene_xyz')
    if masks.shape != (len(ids), 8) or masks.dtype != bool:
        raise ValueError('valid_mask8 must be canonical boolean [len(valid_ids),8]')
    rows, bins = np.nonzero(masks)
    pairs = np.column_stack((xyz[ids[rows]], bins * (np.pi / 4)))
    return np.unique(pairs, axis=0)


def paired_pose_quality(center, yaw, positive_pairs, *, spatial_scale=.1, angular_scale=1.):
    """exp(-best paired spatial/circular cost), never independent minima.

    Angular attraction is continuous through +/-pi. Every angle in the
    witness set is an observed positive; unlisted angles receive no negative
    classification label. This is a reward proxy, not physical correctness.
    """
    center = np.asarray(center, dtype=float)
    pairs = np.asarray(positive_pairs, dtype=float)
    if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(yaw):
        raise ValueError('finite center[3] and yaw required')
    if pairs.ndim != 2 or pairs.shape[1] != 4 or not np.isfinite(pairs).all():
        raise ValueError('positive_pairs must be finite [P,4]')
    if not np.isfinite(spatial_scale) or spatial_scale <= 0 or not np.isfinite(angular_scale) or angular_scale < 0:
        raise ValueError('positive spatial_scale and nonnegative angular_scale required')
    if len(pairs) == 0:
        return dict(quality=0., has_positive=False, no_positive=True, witness_index=None,
                    joint_distance_m=None, joint_angle_rad=None, joint_cost=None,
                    evidence='no_confirmed_positive_pose_reward_evidence')
    d2 = np.square(pairs[:, :3] - center).sum(-1)
    angles = np.arctan2(np.sin(yaw - pairs[:, 3]), np.cos(yaw - pairs[:, 3]))
    costs = d2 / spatial_scale**2 + angular_scale * (1. - np.cos(angles))
    winner = int(np.argmin(costs))
    return dict(quality=float(np.exp(-costs[winner])), has_positive=True, no_positive=False,
                witness_index=winner, joint_distance_m=float(np.sqrt(d2[winner])),
                joint_angle_rad=float(abs(angles[winner])), joint_cost=float(costs[winner]),
                evidence='confirmed_positive_position_and_numeric_yaw_pair_proxy')


def paired_quality_from_annotations(center, yaw, scene_xyz, valid_ids, valid_mask8, **kwargs):
    return paired_pose_quality(center, yaw,
                               make_positive_pose_pairs(scene_xyz, valid_ids, valid_mask8), **kwargs)
