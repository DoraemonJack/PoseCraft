"""Upright oriented boxes and the geometric predicates used by PlaceIt3D.

ScanNet anchors and placed assets are gravity-aligned: only yaw about +Z
changes. Box distance is therefore the horizontal gap between the footprints
combined with the vertical gap between the z intervals.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class OBB:
    center: np.ndarray
    size: np.ndarray
    yaw: float

    def __post_init__(self):
        self.center = np.asarray(self.center, dtype=np.float64)
        self.size = np.asarray(self.size, dtype=np.float64)
        self.yaw = float(self.yaw)


def rotz(yaw: float) -> np.ndarray:
    """Right-handed rotation about +Z, matching VLA-3D ``rotz``."""
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def front_vector(yaw: float) -> np.ndarray:
    """Asset front is local +Y, then rotated by ``yaw``."""
    return rotz(yaw)[:, 1]


def obb_corners(box: OBB) -> np.ndarray:
    half = box.size / 2.0
    signs = np.array(
        [[x, y, z] for x in (-1.0, 1.0) for y in (-1.0, 1.0) for z in (-1.0, 1.0)],
        dtype=np.float64,
    )
    return (signs * half) @ rotz(box.yaw).T + box.center


def xy_polygon(box: OBB) -> np.ndarray:
    """Footprint rectangle, counter-clockwise."""
    c, s = np.cos(box.yaw), np.sin(box.yaw)
    rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
    hx, hy = box.size[0] / 2.0, box.size[1] / 2.0
    local = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]], dtype=np.float64)
    return local @ rotation.T + box.center[:2]


def z_interval(box: OBB) -> tuple[float, float]:
    half = box.size[2] / 2.0
    return float(box.center[2] - half), float(box.center[2] + half)


def polygon_area(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0
    x, y = poly[:, 0], poly[:, 1]
    return float(0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def _ensure_ccw(poly: np.ndarray) -> np.ndarray:
    if polygon_area(poly) < 0.0:
        return poly[::-1].copy()
    return poly


def _segment_intersection(p, q, a, b):
    def cross(u, v):
        return u[0] * v[1] - u[1] * v[0]

    r, s = q - p, b - a
    denom = cross(r, s)
    if abs(denom) < 1e-15:
        return p
    t = cross(a - p, s) / denom
    return p + t * r


def clip_polygon(subject: np.ndarray, clip: np.ndarray) -> np.ndarray:
    """Sutherland–Hodgman clip of ``subject`` against convex CCW ``clip``."""
    output = [np.asarray(p, dtype=np.float64) for p in subject]
    if not output:
        return np.zeros((0, 2))
    for i in range(len(clip)):
        if not output:
            break
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        edge = b - a
        incoming = output
        output = []
        start = incoming[-1]
        for end in incoming:
            end_in = edge[0] * (end[1] - a[1]) - edge[1] * (end[0] - a[0]) >= -1e-10
            start_in = edge[0] * (start[1] - a[1]) - edge[1] * (start[0] - a[0]) >= -1e-10
            if end_in:
                if not start_in:
                    output.append(_segment_intersection(start, end, a, b))
                output.append(end)
            elif start_in:
                output.append(_segment_intersection(start, end, a, b))
            start = end
    if not output:
        return np.zeros((0, 2))
    return np.vstack(output)


def intersection_over_minimum(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    a = _ensure_ccw(np.asarray(poly_a, dtype=np.float64))
    b = _ensure_ccw(np.asarray(poly_b, dtype=np.float64))
    area_a = abs(polygon_area(a))
    area_b = abs(polygon_area(b))
    minimum = min(area_a, area_b)
    if minimum <= 1e-12:
        return 0.0
    inter = clip_polygon(a, b)
    return abs(polygon_area(inter)) / minimum


def iom_1d(a0: float, a1: float, b0: float, b1: float) -> float:
    if a1 < a0:
        a0, a1 = a1, a0
    if b1 < b0:
        b0, b1 = b1, b0
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    minimum = min(a1 - a0, b1 - b0)
    if minimum <= 1e-12:
        return 0.0
    return inter / minimum


def _point_segment_distance(point, a, b) -> float:
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-16:
        return float(np.linalg.norm(point - a))
    t = float(np.clip(np.dot(point - a, ab) / denom, 0.0, 1.0))
    return float(np.linalg.norm(point - (a + t * ab)))


def polygon_distance(poly_a: np.ndarray, poly_b: np.ndarray) -> float:
    if intersection_over_minimum(poly_a, poly_b) > 0.0 and abs(polygon_area(clip_polygon(_ensure_ccw(poly_a), _ensure_ccw(poly_b)))) > 1e-10:
        return 0.0
    best = np.inf
    for source, target in ((poly_a, poly_b), (poly_b, poly_a)):
        for point in source:
            for i in range(len(target)):
                best = min(best, _point_segment_distance(point, target[i], target[(i + 1) % len(target)]))
    return float(best)


def box_gap(a: OBB, b: OBB) -> float:
    """Surface gap between two upright boxes. Zero when they touch or overlap."""
    horizontal = polygon_distance(xy_polygon(a), xy_polygon(b))
    az0, az1 = z_interval(a)
    bz0, bz1 = z_interval(b)
    vertical = max(0.0, max(az0, bz0) - min(az1, bz1))
    if horizontal == 0.0:
        return vertical
    if vertical == 0.0:
        return horizontal
    return float(np.hypot(horizontal, vertical))


def xy_iom(a: OBB, b: OBB) -> float:
    return intersection_over_minimum(xy_polygon(a), xy_polygon(b))


def lateral_iom(asset: OBB, anchor: OBB) -> float:
    front = front_vector(asset.yaw)
    lateral = np.array([-front[1], front[0]], dtype=np.float64)

    def project(box: OBB) -> tuple[float, float]:
        coords = obb_corners(box)[:, :2] @ lateral
        return float(coords.min()), float(coords.max())

    return iom_1d(*project(asset), *project(anchor))


def forward_angle(asset: OBB, anchor: OBB) -> float:
    """Smallest angle, in radians, from the asset front to the anchor box.

    The main paper checks whether the anchor intersects a cone around the
    frontal vector, so the angle is measured to the closest of the anchor
    center and its corners, and only toward the forward half-space.
    """
    front = front_vector(asset.yaw)
    points = np.vstack([anchor.center, obb_corners(anchor)])
    vectors = points - asset.center
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 1e-8
    if not np.any(valid):
        return 0.0
    directions = vectors[valid] / norms[valid, None]
    dots = np.clip(directions @ front, -1.0, 1.0)
    forward = dots > 0.0
    if not np.any(forward):
        return float(np.pi)
    return float(np.arccos(dots[forward]).min())


def point_in_obb(points: np.ndarray, box: OBB, tol: float = 0.0) -> np.ndarray:
    local = (np.asarray(points, dtype=np.float64) - box.center) @ rotz(box.yaw)
    half = box.size / 2.0 + tol
    return np.all(np.abs(local) <= half, axis=1)


def ray_min_t(origin: np.ndarray, direction: np.ndarray, triangles: np.ndarray, t_min: float = 1e-5) -> float:
    """Closest hit distance of a unit ray against ``(T, 3, 3)`` triangles."""
    if triangles is None or len(triangles) == 0:
        return np.inf
    v0 = triangles[:, 0, :]
    v1 = triangles[:, 1, :]
    v2 = triangles[:, 2, :]
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(det) > 1e-8
    if not np.any(valid):
        return np.inf
    inverse = np.zeros_like(det)
    inverse[valid] = 1.0 / det[valid]
    s = origin - v0
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inverse * np.einsum("ij,j->i", q, direction)
    t = inverse * np.einsum("ij,ij->i", edge2, q)
    hit = valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > t_min)
    if not np.any(hit):
        return np.inf
    return float(t[hit].min())


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = np.unique(np.round(np.asarray(points, dtype=np.float64), 8), axis=0)
    if len(pts) <= 2:
        return pts
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    pts = pts[order]

    def build(seq):
        hull = []
        for p in seq:
            while len(hull) >= 2:
                a, b = hull[-2], hull[-1]
                if (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]) <= 0.0:
                    hull.pop()
                else:
                    break
            hull.append(p)
        return hull

    lower = build(pts)
    upper = build(pts[::-1])
    return np.vstack(lower[:-1] + upper[:-1])
