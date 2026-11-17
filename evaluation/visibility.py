"""Visibility from an anchor, following PlaceIt3D supplement section 9.1.

The camera sits on the anchor vertex closest to the anchor centroid and looks
at the asset. The field of view is 60 degrees. A placement is visible when any
in-frame sample of the asset surface is not occluded by the scene. The
official renderer is pyrender at 256x256; this ray test applies the same
camera rule. ``samples`` is the maximum number of asset pixels tested.
"""
from __future__ import annotations

import numpy as np

from geometry import OBB, ray_min_t

FOV_DEGREES = 60.0
NEAR_M = 0.01


def _basis(forward: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = forward / np.linalg.norm(forward)
    world_up = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(forward, world_up))) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0])
    right = np.cross(forward, world_up)
    right = right / np.linalg.norm(right)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    return right, up, forward


def camera_eye(anchor_vertices: np.ndarray, asset_center: np.ndarray) -> np.ndarray:
    centroid = anchor_vertices.mean(0)
    index = int(np.linalg.norm(anchor_vertices - centroid, axis=1).argmin())
    return anchor_vertices[index]


def _in_fov(points: np.ndarray, eye: np.ndarray, right, up, forward, fov_degrees: float) -> np.ndarray:
    vectors = points - eye
    z = vectors @ forward
    x = vectors @ right
    y = vectors @ up
    half = np.tan(np.deg2rad(fov_degrees) / 2.0)
    return (z > NEAR_M) & (np.abs(x) <= z * half) & (np.abs(y) <= z * half)


def asset_is_visible(scene_triangles: np.ndarray, tri_min: np.ndarray, tri_max: np.ndarray,
                     asset_triangles: np.ndarray, anchor_vertices: np.ndarray, asset_box: OBB,
                     samples: int = 256, fov_degrees: float = FOV_DEGREES) -> bool:
    if anchor_vertices is None or len(anchor_vertices) == 0:
        raise RuntimeError("Anchor has no vertices for the visibility camera.")
    eye = camera_eye(anchor_vertices, asset_box.center)
    direction = asset_box.center - eye
    if np.linalg.norm(direction) < 1e-6:
        return False
    right, up, forward = _basis(direction)
    centroids = asset_triangles.mean(1)
    keep = _in_fov(centroids, eye, right, up, forward, fov_degrees)
    visible_points = centroids[keep]
    if len(visible_points) == 0:
        return False
    if len(visible_points) > samples:
        stride = int(np.ceil(len(visible_points) / samples))
        visible_points = visible_points[::stride][:samples]
    region_min = np.minimum(eye, visible_points.min(0)) - 0.05
    region_max = np.maximum(eye, visible_points.max(0)) + 0.05
    mask = np.all(tri_max >= region_min, axis=1) & np.all(tri_min <= region_max, axis=1)
    occluders = scene_triangles[mask]
    for point in visible_points:
        vector = point - eye
        distance = float(np.linalg.norm(vector))
        ray = vector / distance
        start = eye + ray * NEAR_M
        remain = distance - NEAR_M
        hit = ray_min_t(start, ray, occluders, t_min=1e-4)
        if hit + 0.005 >= remain:
            return True
    return False
