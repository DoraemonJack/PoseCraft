"""V5 auxiliary TRAINING targets; never supply annotation distances to a policy.

These targets do not alter the exact identity-based validity labels in V4.
Offsets use world XY metres and the nearest annotation of the SAME yaw bin.
"""
import numpy as np
from scipy.spatial import cKDTree


def _positive_scalar(value, name, allow_zero=False):
    if not np.isscalar(value) or not np.isfinite(value) or (value < 0 if allow_zero else value <= 0):
        raise ValueError(name + (' must be finite and nonnegative' if allow_zero else ' must be finite and positive'))
    return float(value)


def build_distance_targets(centers, valid_xyz, valid_mask8, temperature=.1,
                           offset_radius=.15, vertical_tolerance=.01):
    """Build global C x 8 soft ranking and nearby same-yaw offset supervision.

    Missing yaw bins have infinite distance, zero probability and no offsets.
    ``has_target`` means at least one candidate/annotated-yaw pair exists.
    Inputs are read only; outputs are newly allocated arrays.
    """
    centers = np.asarray(centers, dtype=np.float64)
    xyz = np.asarray(valid_xyz, dtype=np.float64)
    mask = np.asarray(valid_mask8)
    if centers.ndim != 2 or centers.shape[1] != 3 or not np.isfinite(centers).all():
        raise ValueError('centers must have shape [C,3] and be finite')
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not np.isfinite(xyz).all():
        raise ValueError('valid_xyz must have shape [N,3] and be finite')
    if mask.shape != (len(xyz), 8) or mask.dtype != bool:
        raise ValueError('valid_mask8 must be boolean with shape [N,8]')
    temperature = _positive_scalar(temperature, 'temperature')
    offset_radius = _positive_scalar(offset_radius, 'offset_radius')
    vertical_tolerance = _positive_scalar(vertical_tolerance, 'vertical_tolerance', allow_zero=True)
    distances = np.full((len(centers), 8), np.inf)
    offsets = np.zeros((len(centers), 8, 2), dtype=np.float64)
    eligible = np.zeros((len(centers), 8), dtype=bool)
    for b in range(8):
        points = xyz[mask[:, b]]
        if not len(points) or not len(centers):
            continue
        d, indices = cKDTree(points).query(centers, k=1)
        delta = points[indices] - centers
        distances[:, b] = d
        xy = delta[:, :2]
        # Cap even excluded far targets, so every emitted offset is bounded.
        norms = np.linalg.norm(xy, axis=1)
        offsets[:, b] = xy * np.minimum(1., offset_radius / np.maximum(norms, np.finfo(float).tiny))[:, None]
        eligible[:, b] = (d <= offset_radius) & (np.abs(delta[:, 2]) <= vertical_tolerance)
    finite = np.isfinite(distances)
    probabilities = np.zeros_like(distances)
    if finite.any():
        weights = np.exp(-(distances[finite] - distances[finite].min()) / temperature)
        probabilities[finite] = weights / weights.sum()
    return dict(distance_m=distances, rank_prob=probabilities,
                offset_xy_m=offsets, offset_mask=eligible, has_target=bool(finite.any()))


def distance_auxiliary_loss(logits, predicted_offset_xy, targets, offset_radius=.15):
    """Return (global soft ranking CE, eligible normalized smooth-L1).

    Targets are detached on conversion, including when tensors are supplied.
    Empty supervision produces finite differentiable zeros on the model device.
    """
    import torch
    import torch.nn.functional as F
    offset_radius = _positive_scalar(offset_radius, 'offset_radius')
    if logits.ndim != 2 or logits.shape[1] != 8 or not logits.is_floating_point():
        raise ValueError('logits must be floating point with shape [C,8]')
    if predicted_offset_xy.shape != (*logits.shape, 2) or not predicted_offset_xy.is_floating_point():
        raise ValueError('predicted_offset_xy must be floating point with shape [C,8,2]')
    if predicted_offset_xy.device != logits.device:
        raise ValueError('predictions must share a device')
    prob = torch.as_tensor(targets['rank_prob'], dtype=logits.dtype, device=logits.device).detach()
    offset = torch.as_tensor(targets['offset_xy_m'], dtype=predicted_offset_xy.dtype, device=logits.device).detach()
    mask = torch.as_tensor(targets['offset_mask'], dtype=torch.bool, device=logits.device).detach()
    if prob.shape != logits.shape or mask.shape != logits.shape or offset.shape != predicted_offset_xy.shape:
        raise ValueError('target shapes must match predictions')
    if not torch.isfinite(prob).all() or (prob < 0).any() or not torch.isfinite(offset).all():
        raise ValueError('probabilities and offsets must be finite and probabilities nonnegative')
    mass = prob.sum()
    if mass.item() != 0 and not torch.isclose(mass, mass.new_tensor(1.), atol=1e-5, rtol=1e-5):
        raise ValueError('rank_prob must sum to zero or one')
    rank_loss = -(prob.reshape(-1) * F.log_softmax(logits.reshape(-1), dim=0)).sum() if mass.item() else logits.sum() * 0.
    offset_loss = F.smooth_l1_loss(predicted_offset_xy[mask] / offset_radius,
                                 offset[mask] / offset_radius) if mask.any() else predicted_offset_xy.sum() * 0.
    return rank_loss, offset_loss
