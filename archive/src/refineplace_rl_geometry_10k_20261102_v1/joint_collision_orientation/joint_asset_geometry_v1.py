"""Opt-in, sampled geometry signals; the original support/crossing query is intact.

The scene is allowed to be open.  Its triangles are never assigned an inside,
an SDF sign, a physical penetration depth, or an observed-free-space label.
Existing closed components of an *asset* provide an asset-local even/odd voxel
interior and a tolerance-adjusted intrusion proxy for observed scene samples.
This is a sampled proxy, not a solid collision certificate.  Open assets have
no retained closed components have no signed target. Partial-asset zero costs
are unknown; positive intrusion is partial observed evidence. The original
crossing diagnostics remain available. Directional features are unsigned.

No candidate search, pose change, support relaxation, semantic front, mesh
shrinkage, or annotation access is performed here.  All lengths are metres.
"""
from collections import OrderedDict
import hashlib
import math
import time

import numpy as np
from scipy.ndimage import distance_transform_edt, map_coordinates
from scipy.spatial import cKDTree

from asset_mesh_geometry_v6 import AssetMeshGeometry as OriginalAssetMeshGeometry
from asset_mesh_geometry_v6 import vertical_triangle_heights
from mesh_component_geometry_v1 import closed_component_subset


DIRECTIONAL_FEATURE_NAMES = (
    'collision_cost', 'local_scene_evidence_mask',
    'closed_asset_interior_proxy_mask', 'proximity_pos_object_x',
    'proximity_neg_object_x', 'proximity_pos_object_y',
    'proximity_neg_object_y', 'sin_yaw', 'cos_yaw',
    'horizontal_shape_anisotropy',
)
DIRECTIONAL_FEATURE_DIM = len(DIRECTIONAL_FEATURE_NAMES)
_ASSET_CACHE = OrderedDict()
_ASSET_CACHE_LIMIT = 32


def _closed_edge_topology(triangles):
    """Exact-coordinate welding and two incident faces per undirected edge.

    This checks boundary/nonmanifold edges, not self-intersections or material
    density. Even/odd occupancy remains explicitly a geometry proxy.
    """
    _, indices = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = indices.reshape(-1, 3)
    edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return bool(len(counts) and np.all(counts == 2))


def _surface_samples(triangles, count=4096):
    """Deterministic area-weighted points lying on actual triangles."""
    # Canonical vertex order makes the sampling independent of mesh winding.
    order = np.lexsort((triangles[:, :, 2], triangles[:, :, 1], triangles[:, :, 0]), axis=1)
    triangles = triangles[np.arange(len(triangles))[:, None], order]
    area = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                   triangles[:, 2]-triangles[:, 0]), axis=1)
    cdf = np.cumsum(area) / area.sum()
    index = np.searchsorted(cdf, (np.arange(count)+.5)/count)
    # Irrational sequences avoid a fixed strip of barycentric positions.
    u = np.mod((np.arange(count)+.5)*0.6180339887498949, 1.)
    v = np.mod((np.arange(count)+.5)*0.4142135623730951, 1.)
    root = np.sqrt(u)
    weights = np.column_stack((1-root, root*(1-v), root*v))
    result = np.einsum('ni,nij->nj', weights, triangles[index])
    # Vertices retain thin tips and small components that area sampling misses.
    vertices = np.unique(triangles.reshape(-1, 3), axis=0)
    if len(vertices) > count:
        vertices = vertices[np.linspace(0, len(vertices)-1, count, dtype=int)]
    return np.concatenate((result, vertices))


def _prepare_asset(triangles, grid_side, surface_sample_count):
    digest = hashlib.sha256(np.ascontiguousarray(triangles, dtype='<f8').tobytes()).hexdigest()
    key = (digest, int(grid_side), int(surface_sample_count))
    if key in _ASSET_CACHE:
        _ASSET_CACHE.move_to_end(key)
        return _ASSET_CACHE[key], True
    started = time.perf_counter()
    lo, hi = triangles.min((0, 1)), triangles.max((0, 1))
    extent = hi-lo
    closed = _closed_edge_topology(triangles)
    proxy_triangles, recovery = closed_component_subset(triangles)
    points = _surface_samples(triangles, surface_sample_count)
    # A single metric spacing makes resolution and its conservative allowance
    # explicit; the grid is bounded by (grid_side+5)^3 cells.
    spacing = float(extent.max()/(grid_side-1))
    origin = lo-2*spacing
    shape = tuple((np.ceil(extent/spacing).astype(int)+5).tolist())
    depth = np.zeros(shape, dtype=np.float32)
    parity_valid_fraction = 0.
    if len(proxy_triangles):
        x = origin[0]+np.arange(shape[0])*spacing
        y = origin[1]+np.arange(shape[1])*spacing
        z = origin[2]+np.arange(shape[2])*spacing
        xy = np.stack(np.meshgrid(x, y, indexing='ij'), -1).reshape(-1, 2)
        occupied = np.zeros(shape, dtype=bool)
        valid = 0
        # Projected-triangle broad phase avoids grid_rays x all_asset_faces.
        # Bounds only select triangles; the final heights use original triangles.
        normal = np.cross(proxy_triangles[:, 1]-proxy_triangles[:, 0],
                          proxy_triangles[:, 2]-proxy_triangles[:, 0])
        ray_triangles = proxy_triangles[np.abs(normal[:, 2]) > 1e-14]
        projected_centroids = ray_triangles[:, :, :2].mean(1)
        projected_radius = np.linalg.norm(ray_triangles[:, :, :2]-projected_centroids[:, None], axis=2).max(1)
        projected_tree = cKDTree(projected_centroids)
        projected_min = ray_triangles[:, :, :2].min(1)
        projected_max = ray_triangles[:, :, :2].max(1)
        ray_candidates = projected_tree.query_ball_point(xy, float(projected_radius.max(initial=0.))+1e-9)
        for k, candidates in enumerate(ray_candidates):
            candidates = np.asarray(candidates, int)
            eligible = ((projected_min[candidates] <= xy[k]+1e-9) &
                        (projected_max[candidates] >= xy[k]-1e-9)).all(1)
            heights, _ = vertical_triangle_heights(ray_triangles[candidates[eligible]], xy[k:k+1])[0]
            # One surface crossing can be reported by adjacent triangles.
            heights = np.sort(heights)
            if len(heights):
                heights = heights[np.r_[True, np.diff(heights) > 1e-8]]
            if len(heights) % 2:
                continue  # Ambiguous columns do not contribute inside evidence.
            valid += 1
            inside = np.zeros(len(z), bool)
            for low, high in heights.reshape(-1, 2):
                inside |= (z > low+1e-9) & (z < high-1e-9)
            occupied[k//shape[1], k % shape[1]] = inside
        parity_valid_fraction = valid/len(xy)
        # Distance to exterior grid centres can overestimate a surface distance.
        # Subtract a whole voxel diagonal before the separate scan tolerance.
        # Thin or unresolved components therefore lose evidence, not acquire it.
        depth = np.maximum(distance_transform_edt(occupied)*spacing-
                           math.sqrt(3)*spacing, 0.).astype(np.float32)
    vertices = np.unique(triangles.reshape(-1, 3), axis=0)
    anchors = [(lo+hi)/2]
    for axis in range(3):
        anchors.extend((vertices[np.argmin(vertices[:, axis])],
                        vertices[np.argmax(vertices[:, axis])]))
    result = dict(closed=closed, recovery=recovery, depth=depth, origin=origin, spacing=spacing,
                  parity_valid_fraction=float(parity_valid_fraction),
                  resolved=bool(len(proxy_triangles) and np.any(depth > 0)),
                  points=points, tree=cKDTree(points), anchors=np.asarray(anchors),
                  lo=lo, hi=hi, extent=extent, digest=digest,
                  preparation_seconds=float(time.perf_counter()-started))
    for array in (depth, origin, points, result['anchors'], lo, hi, extent):
        array.flags.writeable = False
    _ASSET_CACHE[key] = result
    if len(_ASSET_CACHE) > _ASSET_CACHE_LIMIT:
        _ASSET_CACHE.popitem(last=False)
    return result, False


def _closest_points_on_triangles(triangles, point):
    """Exact Euclidean closest point to each triangle; no spatial dependency."""
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    ab, ac = b-a, c-a
    ap = point-a
    d1, d2 = np.einsum('ij,ij->i', ab, ap), np.einsum('ij,ij->i', ac, ap)
    bp = point-b
    d3, d4 = np.einsum('ij,ij->i', ab, bp), np.einsum('ij,ij->i', ac, bp)
    cp = point-c
    d5, d6 = np.einsum('ij,ij->i', ab, cp), np.einsum('ij,ij->i', ac, cp)
    va, vb, vc = d3*d6-d5*d4, d5*d2-d1*d6, d1*d4-d3*d2
    denom = np.maximum(va+vb+vc, np.finfo(float).tiny)
    out = a+(vb/denom)[:, None]*ab+(vc/denom)[:, None]*ac
    masks = [
        ((d1 <= 0) & (d2 <= 0), a),
        ((d3 >= 0) & (d4 <= d3), b),
        ((d6 >= 0) & (d5 <= d6), c),
        ((vc <= 0) & (d1 >= 0) & (d3 <= 0),
         a+(d1/np.maximum(d1-d3, np.finfo(float).tiny))[:, None]*ab),
        ((vb <= 0) & (d2 >= 0) & (d6 <= 0),
         a+(d2/np.maximum(d2-d6, np.finfo(float).tiny))[:, None]*ac),
        ((va <= 0) & (d4-d3 >= 0) & (d5-d6 >= 0),
         b+((d4-d3)/np.maximum((d4-d3)+(d5-d6), np.finfo(float).tiny))[:, None]*(c-b)),
    ]
    for mask, values in masks:
        out[mask] = values[mask]
    return out


class AssetMeshGeometry(OriginalAssetMeshGeometry):
    """Drop-in core: old stats survive; new fields are appended to the dict."""
    directional_feature_dim = DIRECTIONAL_FEATURE_DIM
    directional_feature_names = DIRECTIONAL_FEATURE_NAMES

    def __init__(self, *args, collision_tolerance=.02, collision_cost_scale=.03,
                 directional_radius=.15, interior_grid_side=48,
                 max_local_scene_triangles=2048, asset_surface_sample_count=4096,
                 **kwargs):
        values = [collision_tolerance, collision_cost_scale, directional_radius]
        if not np.isfinite(values).all() or collision_tolerance < 0 or min(values[1:]) <= 0:
            raise ValueError('finite nonnegative tolerance and positive geometry scales required')
        if not isinstance(interior_grid_side, int) or not 16 <= interior_grid_side <= 96:
            raise ValueError('interior_grid_side must be an integer in [16,96]')
        if not isinstance(max_local_scene_triangles, int) or max_local_scene_triangles < 1:
            raise ValueError('positive max_local_scene_triangles required')
        if not isinstance(asset_surface_sample_count, int) or asset_surface_sample_count < 128:
            raise ValueError('asset_surface_sample_count must be integer >=128')
        super().__init__(*args, **kwargs)
        self.collision_tolerance = float(collision_tolerance)
        self.collision_cost_scale = float(collision_cost_scale)
        self.directional_radius = float(directional_radius)
        self.max_local_scene_triangles = max_local_scene_triangles
        self.joint_asset, self.joint_asset_cache_hit = _prepare_asset(
            self.asset, interior_grid_side, asset_surface_sample_count)

    def _local_scene_samples(self, center, yaw):
        """Bounded local evidence from real triangles, including large triangles.

        Closest points to actual-asset anchors prevent a large wall triangle
        from being missed just because its original vertices are far away.
        Triangle caps are reported; they never change original exact crossings.
        """
        prepared = self.joint_asset
        cs, sn = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[cs, -sn, 0.], [sn, cs, 0.], [0., 0., 1.]])
        local_mid = (prepared['lo']+prepared['hi'])/2
        world_mid = local_mid @ rotation.T+center
        radius = np.linalg.norm(prepared['extent']/2)+self.directional_radius
        ids = np.asarray(self.tree.query_ball_point(world_mid, radius+self.max_scene_radius), int)
        if not len(ids):
            return np.zeros((0, 3)), 0, True
        # Exclude exactly the original support-band triangle classification.
        ids = ids[~self._contact_scene_mask(self.scene[ids], center)]
        if not len(ids):
            return np.zeros((0, 3)), 0, True
        triangles = (self.scene[ids]-center) @ rotation
        lo = prepared['lo']-self.directional_radius
        hi = prepared['hi']+self.directional_radius
        keep = ((triangles.max(1) >= lo) & (triangles.min(1) <= hi)).all(1)
        triangles, ids = triangles[keep], ids[keep]
        before = len(triangles)
        if before > self.max_local_scene_triangles:
            # Deterministic AABB lower distance; no candidate pose enumeration.
            gap = np.maximum(np.maximum(triangles.min(1)-prepared['hi'],
                                        prepared['lo']-triangles.max(1)), 0.)
            order = np.lexsort((ids, np.linalg.norm(gap, axis=1)))
            triangles = triangles[order[:self.max_local_scene_triangles]]
        if not len(triangles):
            return np.zeros((0, 3)), before, True
        samples = np.concatenate([_closest_points_on_triangles(triangles, anchor)
                                  for anchor in prepared['anchors']]+[triangles.mean(1)])
        keep = ((samples >= lo) & (samples <= hi)).all(1)
        return samples[keep], before, before <= self.max_local_scene_triangles

    def joint_signals(self, center, yaw):
        """Additional feature computation only; never modifies the input pose."""
        center = np.asarray(center, float)
        if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(yaw):
            raise ValueError('finite center[3] and yaw required')
        asset = self.joint_asset
        recovery = asset['recovery']
        samples, triangle_count, complete = self._local_scene_samples(center, yaw)
        local_mask = bool(len(samples))
        field_mask = bool(asset['resolved'])
        depths = np.zeros(len(samples))
        proximity = np.zeros(4)
        if len(samples):
            if field_mask:
                coordinates = ((samples-asset['origin'])/asset['spacing']).T
                depths = map_coordinates(asset['depth'], coordinates, order=1,
                                         mode='constant', cval=0., prefilter=False)
            distances = np.asarray(asset['tree'].query(samples, k=1)[0])
            # These axes are geometric canonical axes, never a semantic front.
            offsets = samples-(asset['lo']+asset['hi'])/2
            side_masks = (offsets[:, 0] >= 0, offsets[:, 0] < 0,
                          offsets[:, 1] >= 0, offsets[:, 1] < 0)
            for i, selected in enumerate(side_masks):
                if selected.any():
                    proximity[i] = max(0., 1.-float(distances[selected].min())/self.directional_radius)
        maximum = float(depths.max()) if len(depths) else 0.
        excess = max(0., maximum-self.collision_tolerance)
        # A capped zero is missing evidence. Positive sampled intrusion remains
        # valid as a lower-bound proxy even when other scene triangles were cut.
        field_complete = asset['parity_valid_fraction'] == 1.
        full_asset = recovery['full_surface_coverage']
        partial_asset = recovery['closed_component_count'] > 0 and not full_asset
        cost_known = local_mask and field_mask and ((complete and field_complete and full_asset) or excess > 0.)
        cost = float(np.clip(excess/self.collision_cost_scale, 0., 1.)) if cost_known else 0.
        extent = asset['extent']
        anisotropy = float((extent[0]-extent[1])/max(extent[0]+extent[1], 1e-12))
        vector = np.asarray([cost, float(local_mask), float(field_mask), *proximity,
                             math.sin(yaw), math.cos(yaw), anisotropy], dtype=np.float32)
        return dict(
            collision_cost=cost, collision_cost_known=bool(cost_known),
            local_scene_evidence_mask=bool(local_mask),
            closed_asset_interior_proxy_mask=bool(field_mask),
            asset_closed_edge_topology=bool(asset['closed']),
            directional_collision_vector=vector,
            directional_feature_names=DIRECTIONAL_FEATURE_NAMES,
            directional_feature_frame='canonical_object_axes_no_semantic_front',
            directional_proximity_provenance='unsigned_scene_triangle_samples_to_actual_asset_surface_samples',
            collision_cost_provenance='observed_scene_samples_inside_existing_closed_asset_components_even_odd_voxel_proxy',
            collision_cost_unknown_reason=None if cost_known else (
                'no_local_noncontact_scene_samples' if not local_mask else
                'asset_open_or_interior_unresolved' if not field_mask else
                'partial_asset_without_positive_intrusion' if partial_asset else
                'ambiguous_asset_parity_without_positive_intrusion' if not field_complete else
                'local_triangle_cap_without_positive_intrusion'),
            collision_proxy_partial_asset=bool(partial_asset),
            collision_proxy_full_asset=bool(full_asset),
            collision_proxy_covered_surface_area_fraction=recovery['closed_surface_area_fraction'],
            collision_proxy_covered_face_fraction=recovery['closed_face_fraction'],
            collision_proxy_closed_component_count=recovery['closed_component_count'],
            collision_proxy_component_count=recovery['component_count'],
            collision_proxy_original_face_count=recovery['original_face_count'],
            collision_proxy_deduplicated_face_count=recovery['deduplicated_face_count'],
            collision_proxy_duplicate_faces_removed=recovery['duplicates_removed'],
            collision_proxy_component_rule=recovery['component_rule'],
            collision_proxy_holes_filled=False,
            collision_proxy_max_interior_distance_m=maximum,
            collision_proxy_excess_m=excess,
            collision_tolerance_m=self.collision_tolerance,
            collision_cost_scale_m=self.collision_cost_scale,
            collision_proxy_grid_spacing_m=asset['spacing'],
            collision_proxy_grid_shape=tuple(asset['depth'].shape),
            collision_proxy_grid_allowance_m=math.sqrt(3)*asset['spacing'],
            collision_proxy_parity_valid_fraction=asset['parity_valid_fraction'],
            collision_proxy_field_complete=bool(field_complete),
            collision_proxy_scene_sample_count=int(len(samples)),
            collision_proxy_local_triangle_count=int(triangle_count),
            collision_proxy_triangle_cap=self.max_local_scene_triangles,
            collision_proxy_sampling_complete=bool(complete),
            collision_proxy_is_lower_bound_from_selected_samples=not bool(complete and full_asset),
            collision_proxy_asset_preparation_seconds=asset['preparation_seconds'],
            collision_proxy_asset_preparation_cache_hit=bool(self.joint_asset_cache_hit),
            collision_proxy_is_physical_penetration_depth=False,
            collision_proxy_certifies_clearance=False,
            collision_proxy_asset_hash=asset['digest'],
            collision_pose_repair=False,
            support_rule_preserved=True,
        )

    def query(self, center, yaw):
        stats = super().query(center, yaw)
        extra = self.joint_signals(center, yaw)
        if set(stats).intersection(extra):
            raise RuntimeError('joint extension attempted to overwrite an original geometry field')
        stats.update(extra)
        return stats


def install_on_adapter_module(adapter_module):
    """Opt in before constructing adapters, including in each spawned worker.

    Original MeshSceneAdapterV6.features remains 26-dimensional; the appended
    directional array is returned separately in its stats dictionary.
    """
    existing = adapter_module.AssetMeshGeometry
    if existing not in (OriginalAssetMeshGeometry, AssetMeshGeometry):
        raise ValueError('unexpected adapter core; refusing to patch another treatment')
    adapter_module.AssetMeshGeometry = AssetMeshGeometry
    return DIRECTIONAL_FEATURE_DIM
