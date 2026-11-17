"""Physical plausibility: no mesh intersection, and the asset rests on a surface.

The benchmark paper checks both with trimesh. python-fcl is not required here.
An asset edge that crosses a scene triangle counts as an intersection when the
crossing is more than 1 cm from either endpoint. That 1 cm contact tolerance is
the same one the paper uses for the ``on`` relation. A crossing of a nearly
horizontal triangle within 3 cm of the asset bottom is resting contact, not an
intersection. The asset rests on a surface when the bottom vertices sit within
2 cm of a scene surface below them.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import trimesh

CONTACT_CLEARANCE_M = 0.01
FLOOR_NORMAL_Z = 0.7
FLOOR_BAND_M = 0.03
SUPPORT_TOLERANCE_M = 0.02
SUPPORT_BAND_M = 0.02
_CELL = 0.25


def _first_hit(origin: np.ndarray, direction: np.ndarray, triangles: np.ndarray, t_min: float):
    if len(triangles) == 0:
        return np.inf, -1
    v0 = triangles[:, 0, :]
    edge1 = triangles[:, 1, :] - v0
    edge2 = triangles[:, 2, :] - v0
    h = np.cross(direction, edge2)
    det = np.einsum("ij,ij->i", edge1, h)
    valid = np.abs(det) > 1e-8
    inverse = np.zeros_like(det)
    inverse[valid] = 1.0 / det[valid]
    s = origin - v0
    u = inverse * np.einsum("ij,ij->i", s, h)
    q = np.cross(s, edge1)
    v = inverse * np.einsum("ij,j->i", q, direction)
    t = inverse * np.einsum("ij,ij->i", edge2, q)
    ok = valid & (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0) & (t > t_min)
    if not np.any(ok):
        return np.inf, -1
    t = np.where(ok, t, np.inf)
    local = int(np.argmin(t))
    return float(t[local]), local


def _index_triangles(triangles: np.ndarray):
    buckets = defaultdict(list)
    tri_min = triangles.min(1)
    tri_max = triangles.max(1)
    lo = np.floor(tri_min / _CELL).astype(np.int32)
    hi = np.floor(tri_max / _CELL).astype(np.int32)
    for index, (start, end) in enumerate(zip(lo, hi)):
        for x in range(int(start[0]), int(end[0]) + 1):
            for y in range(int(start[1]), int(end[1]) + 1):
                for z in range(int(start[2]), int(end[2]) + 1):
                    buckets[(x, y, z)].append(index)
    return buckets, tri_min, tri_max


def _candidate_ids(buckets, tri_min, tri_max, start, end):
    low = np.minimum(start, end)
    high = np.maximum(start, end)
    lo = np.floor(low / _CELL).astype(np.int32)
    hi = np.floor(high / _CELL).astype(np.int32)
    ids = []
    for x in range(int(lo[0]), int(hi[0]) + 1):
        for y in range(int(lo[1]), int(hi[1]) + 1):
            for z in range(int(lo[2]), int(hi[2]) + 1):
                found = buckets.get((x, y, z))
                if found:
                    ids.extend(found)
    if not ids:
        return np.empty(0, dtype=np.int64)
    ids = np.unique(np.asarray(ids, dtype=np.int64))
    overlap = np.all(tri_max[ids] >= low - 1e-6, axis=1) & np.all(tri_min[ids] <= high + 1e-6, axis=1)
    return ids[overlap]


def _intersects(mesh: trimesh.Trimesh, triangles: np.ndarray, normals: np.ndarray) -> bool:
    if len(triangles) == 0 or len(mesh.faces) == 0:
        return False
    buckets, tri_min, tri_max = _index_triangles(triangles)
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    asset_bottom = float(verts[:, 2].min())
    for a, b in np.asarray(mesh.edges_unique):
        start = verts[int(a)]
        end = verts[int(b)]
        delta = end - start
        length = float(np.linalg.norm(delta))
        if length < 1e-4:
            continue
        ids = _candidate_ids(buckets, tri_min, tri_max, start, end)
        if len(ids) == 0:
            continue
        direction = delta / length
        walked = 0.0
        for _ in range(4):
            t, local = _first_hit(start + direction * walked, direction, triangles[ids], 1e-3)
            if not np.isfinite(t):
                break
            along = walked + t
            if along >= length - 1e-3:
                break
            hit = start + direction * along
            clearance = min(along, length - along)
            normal = normals[ids[local]]
            on_floor = abs(float(normal[2])) > FLOOR_NORMAL_Z and hit[2] <= asset_bottom + FLOOR_BAND_M
            if not on_floor and clearance > CONTACT_CLEARANCE_M:
                return True
            walked = along + 1e-3
            if walked >= length:
                break
    return False


def _rests(mesh: trimesh.Trimesh, triangles: np.ndarray) -> bool:
    if len(triangles) == 0:
        return False
    verts = np.asarray(mesh.vertices, dtype=np.float64)
    bottom = float(verts[:, 2].min())
    samples = verts[verts[:, 2] <= bottom + SUPPORT_BAND_M]
    if len(samples) == 0:
        return False
    if len(samples) > 80:
        samples = samples[:: max(1, len(samples) // 80)]
    lift = 0.002
    direction = np.array([0.0, 0.0, -1.0])
    gaps = []
    for point in samples:
        t, _ = _first_hit(point + np.array([0.0, 0.0, lift]), direction, triangles, 1e-5)
        if np.isfinite(t):
            gaps.append(t - lift)
    if not gaps:
        return False
    return float(np.median(gaps)) <= SUPPORT_TOLERANCE_M


def physically_plausible(scene_mesh: trimesh.Trimesh, asset_mesh: trimesh.Trimesh, asset_box=None,
                         tri_min: np.ndarray | None = None, tri_max: np.ndarray | None = None) -> bool:
    del asset_box
    bounds = np.asarray(asset_mesh.bounds, dtype=np.float64)
    low = bounds[0] - np.array([0.02, 0.02, 0.25])
    high = bounds[1] + np.array([0.02, 0.02, 0.02])
    triangles = np.asarray(scene_mesh.triangles, dtype=np.float64)
    if tri_min is None or tri_max is None:
        tri_min = triangles.min(1)
        tri_max = triangles.max(1)
    local = triangles[np.all(tri_max >= low, axis=1) & np.all(tri_min <= high, axis=1)]
    if len(local) == 0:
        return False
    normals = np.cross(local[:, 1] - local[:, 0], local[:, 2] - local[:, 0])
    normals /= np.clip(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12, None)
    if _intersects(asset_mesh, local, normals):
        return False
    return _rests(asset_mesh, local)
