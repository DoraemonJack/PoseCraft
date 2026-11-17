"""Actual triangle-surface collision and sampled actual-asset support evidence.

No box is substituted for the asset. Nonwatertight scene surfaces do not define
inside/outside: no intersection is NOT certified solid clearance. Penetration
depth, stability and unobserved free space are deliberately not inferred.
"""
import hashlib
from collections import OrderedDict
import numpy as np
from scipy.spatial import cKDTree


_BOTTOM_SAMPLE_CACHE = OrderedDict()
_BOTTOM_SAMPLE_CACHE_SIZE = 128


def _bottom_samples(asset, contact_tolerance, grid_side, geometry_digest):
    """Adaptive equal-area quadrature, cached by actual geometry and rule.

    Only triangles reaching the bottom band need ray tests: a retained ray's
    lowest hit is in that band, so triangles wholly above it cannot change it.
    Cached arrays are immutable; callers always receive an independent copy.
    """
    key = (geometry_digest, float(contact_tolerance), int(grid_side), 'midpoint_lower_envelope_min16_max128_v1')
    if key not in _BOTTOM_SAMPLE_CACHE:
        lo, hi = asset.min((0, 1)), asset.max((0, 1))
        low_triangles = asset[asset[:, :, 2].min(1) <= lo[2]+contact_tolerance+1e-9]
        levels = sorted(set([grid_side]+[side for side in (32, 64, 128) if side > grid_side]))
        samples = np.zeros((0, 3), float)
        for side in levels:
            uv = (np.arange(side)+.5)/side
            xy = np.stack(np.meshgrid(lo[0]+uv*(hi[0]-lo[0]), lo[1]+uv*(hi[1]-lo[1])), -1).reshape(-1, 2)
            hits = vertical_triangle_heights(low_triangles, xy)
            indices = [i for i, (z, _) in enumerate(hits) if len(z) and z.min() <= lo[2]+contact_tolerance+1e-9]
            samples = np.array([[*xy[i], hits[i][0].min()] for i in indices], float).reshape(-1, 3)
            if len(samples) >= 16:
                break
        samples.flags.writeable = False
        _BOTTOM_SAMPLE_CACHE[key] = (samples, side)
        if len(_BOTTOM_SAMPLE_CACHE) > _BOTTOM_SAMPLE_CACHE_SIZE:
            _BOTTOM_SAMPLE_CACHE.popitem(last=False)
    _BOTTOM_SAMPLE_CACHE.move_to_end(key)
    cached, side = _BOTTOM_SAMPLE_CACHE[key]
    result = cached.copy()
    result.flags.writeable = False
    return result, side


def _fcl_bvh(triangles):
    import fcl
    model = fcl.BVHModel()
    vertices = np.ascontiguousarray(triangles.reshape(-1, 3))
    faces = np.arange(len(vertices), dtype=np.int32).reshape(-1, 3)
    model.beginModel(len(vertices), len(faces))
    model.addSubModel(vertices, faces)
    model.endModel()
    return model


def _mesh(vertices, faces, name):
    vertices, faces = np.asarray(vertices, float), np.asarray(faces)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise ValueError(name + ' vertices must be finite [V,3]')
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.dtype.kind not in 'iu':
        raise ValueError(name + ' faces must be integer [F,3]')
    if np.any(faces < 0) or np.any(faces >= len(vertices)):
        raise ValueError(name + ' face index outside vertices')
    triangles = vertices[faces].copy()
    if len(triangles):
        area2 = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0]), axis=1)
        triangles = triangles[area2 > 1e-14]
    return triangles


def triangle_surface_overlap(triangle, triangles, epsilon=1e-9):
    """Closed-boundary triangle intersection by SAT, including coplanar pairs.

    Epsilon is a numerical length tolerance, not a collision/contact allowance.
    Triangles must be nondegenerate. Includes both normals, nine edge cross
    products and in-plane edge normals (needed for coplanar triangles).
    """
    a, b = np.asarray(triangle, float), np.asarray(triangles, float)
    if a.shape != (3, 3) or b.ndim != 3 or b.shape[1:] != (3, 3):
        raise ValueError('expected triangle[3,3], triangles[N,3,3]')
    if not np.isfinite(a).all() or not np.isfinite(b).all() or not np.isfinite(epsilon) or epsilon < 0:
        raise ValueError('finite triangles and nonnegative epsilon required')
    ea, eb = np.roll(a, -1, axis=0)-a, np.roll(b, -1, axis=1)-b
    na, nb = np.cross(ea[0], ea[1]), np.cross(eb[:, 0], eb[:, 1])
    axes = [np.broadcast_to(na, (len(b), 3)), nb]
    axes += [np.cross(ea[i], eb[:, j]) for i in range(3) for j in range(3)]
    axes += [np.broadcast_to(np.cross(na, e), (len(b), 3)) for e in ea]
    axes += [np.cross(nb, eb[:, j]) for j in range(3)]
    keep = np.ones(len(b), bool)
    for axis in axes:
        norm = np.linalg.norm(axis, axis=1)
        unit = axis / np.maximum(norm[:, None], np.finfo(float).tiny)
        pa = a @ unit.T
        pb = np.einsum('nij,nj->ni', b, unit)
        separated = (pa.max(0) < pb.min(1)-epsilon) | (pb.max(1) < pa.min(0)-epsilon)
        keep &= ~(separated & (norm > 1e-14))
    return keep


def vertical_triangle_heights(triangles, xy, epsilon=1e-9):
    """Return per-ray surface Z hits and |normal.Z|; vertical faces give no hit.

    Scene normals are unoriented because reconstructed mesh winding need not
    consistently face upward. This measures surface inclination, not solidity.
    """
    tri, xy = np.asarray(triangles, float), np.asarray(xy, float)
    if tri.ndim != 3 or tri.shape[1:] != (3, 3) or xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError('expected triangles[N,3,3] and xy[M,2]')
    v0, e1, e2 = tri[:, 0], tri[:, 1]-tri[:, 0], tri[:, 2]-tri[:, 0]
    normal = np.cross(e1, e2)
    denom = e1[:, 0]*e2[:, 1]-e1[:, 1]*e2[:, 0]
    nonvertical = np.abs(denom) > 1e-14
    v0, e1, e2, normal, denom = [x[nonvertical] for x in (v0, e1, e2, normal, denom)]
    normal_z = np.abs(normal[:, 2]) / np.maximum(np.linalg.norm(normal, axis=1), np.finfo(float).tiny)
    result = []
    for point in xy:
        d = point-v0[:, :2]
        u = (d[:, 0]*e2[:, 1]-d[:, 1]*e2[:, 0])/denom
        v = (e1[:, 0]*d[:, 1]-e1[:, 1]*d[:, 0])/denom
        inside = (u >= -epsilon) & (v >= -epsilon) & (u+v <= 1+epsilon)
        result.append((v0[inside, 2]+u[inside]*e1[inside, 2]+v[inside]*e2[inside, 2], normal_z[inside]))
    return result


class AssetMeshGeometry:
    """Per-scene/per-asset geometry; canonical asset is bottom-centered metres.

    Constructor never scales, rotates axes, simplifies or shrinks the mesh.
    The loader must establish the canonical asset frame and metric size.
    """
    def __init__(self, scene_vertices, scene_faces, asset_vertices, asset_faces,
                 *, contact_tolerance=.02, support_search=.15,
                 min_support_normal_z=.7, grid_side=16, collision_backend='sat',
                 max_collision_contacts=4096, prepared_scene=None):
        if not np.isfinite([contact_tolerance, support_search, min_support_normal_z]).all() or not 0 < contact_tolerance <= support_search or not 0 < min_support_normal_z <= 1:
            raise ValueError('invalid contact/search/normal thresholds')
        if not isinstance(grid_side, (int, np.integer)) or not 2 <= grid_side <= 128:
            raise ValueError('grid_side must be integer in [2,128]')
        if collision_backend not in ('sat', 'fcl'):
            raise ValueError('collision_backend must be sat or fcl')
        if not isinstance(max_collision_contacts, (int, np.integer)) or max_collision_contacts < 1:
            raise ValueError('max_collision_contacts must be positive integer')
        self.collision_backend = collision_backend
        self.max_collision_contacts = int(max_collision_contacts)
        shared = self.prepare_scene(scene_vertices, scene_faces, collision_backend) if prepared_scene is None else prepared_scene
        if shared['collision_backend'] != collision_backend:
            raise ValueError('prepared scene backend differs')
        self.scene = shared['triangles']
        self.asset = _mesh(asset_vertices, asset_faces, 'asset')
        if not len(self.asset):
            raise ValueError('asset must contain nondegenerate triangles')
        if abs(self.asset[:, :, 2].min()) > 1e-6:
            raise ValueError('asset must already have bottom Z=0 in metres')
        self.contact_tolerance, self.support_search = float(contact_tolerance), float(support_search)
        self.min_support_normal_z = float(min_support_normal_z)
        self.scene_centroids, self.tree = shared['centroids'], shared['tree']
        self.scene_min, self.scene_max = shared['minimum'], shared['maximum']
        self.scene_radius, self.max_scene_radius = shared['radius'], shared['max_radius']
        self._scene_bvh = shared.get('bvh')
        self._scene_object = shared.get('collision_object')
        self._asset_bvh = _fcl_bvh(self.asset) if collision_backend == 'fcl' else None
        # Midpoint quadrature rays over the real footprint; holes are excluded by
        # ray-mesh intersection and each retained ray uses actual lowest Z.
        geometry_hash = hashlib.sha256()
        for source, dtype in ((asset_vertices, '<f8'), (asset_faces, '<i8')):
            array = np.ascontiguousarray(source, dtype=dtype)
            geometry_hash.update(np.asarray(array.shape, dtype='<i8').tobytes())
            geometry_hash.update(array.tobytes())
        self.support_samples, self.support_grid_side = _bottom_samples(self.asset, contact_tolerance, grid_side, geometry_hash.hexdigest())
        self.support_sampling_insufficient = len(self.support_samples) < 16

    @staticmethod
    def prepare_scene(scene_vertices, scene_faces, collision_backend='sat'):
        """Prepare once per scene; share returned data among all its asset cores.

        The caller owns cache lifetime. No module-global scene cache is retained.
        """
        if collision_backend not in ('sat', 'fcl'):
            raise ValueError('collision_backend must be sat or fcl')
        triangles = _mesh(scene_vertices, scene_faces, 'scene')
        centroids = triangles.mean(1)
        radius = np.linalg.norm(triangles-centroids[:, None], axis=2).max(1) if len(triangles) else np.zeros(0)
        result = dict(triangles=triangles, centroids=centroids, tree=cKDTree(centroids),
                      minimum=triangles.min(1) if len(triangles) else np.zeros((0, 3)),
                      maximum=triangles.max(1) if len(triangles) else np.zeros((0, 3)),
                      radius=radius, max_radius=float(radius.max()) if len(radius) else 0.,
                      collision_backend=collision_backend)
        if collision_backend == 'fcl':
            import fcl
            result['bvh'] = _fcl_bvh(triangles) if len(triangles) else None
            result['collision_object'] = fcl.CollisionObject(result['bvh']) if len(triangles) else None
        return result

    def _fcl_pairs(self, center, yaw):
        import fcl
        if not len(self.scene):
            return [], True
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        obj = fcl.CollisionObject(self._asset_bvh, fcl.Transform(rotation, np.asarray(center, float)))
        request = fcl.CollisionRequest(enable_contact=True, num_max_contacts=self.max_collision_contacts)
        result = fcl.CollisionResult()
        fcl.collide(self._scene_object, obj, request, result)
        pairs = set()
        for contact in result.contacts:
            scene_face, asset_face = int(contact.b1), int(contact.b2)
            if not 0 <= scene_face < len(self.scene) or not 0 <= asset_face < len(self.asset):
                raise RuntimeError('FCL returned invalid scene-first mesh triangle IDs')
            pairs.add((scene_face, asset_face))
        return sorted(pairs), len(result.contacts) < self.max_collision_contacts

    def _contact_scene_mask(self, triangles, center):
        if not len(triangles):
            return np.zeros(0, bool)
        normal = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
        nz = abs(normal[:, 2])/np.linalg.norm(normal, axis=1)
        in_band = np.max(abs(triangles[:, :, 2]-center[2]), axis=1) <= self.contact_tolerance+1e-9
        return in_band & (nz >= self.min_support_normal_z)

    def posed_asset(self, center, yaw):
        center = np.asarray(center, float)
        if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(yaw):
            raise ValueError('finite center[3] and yaw required')
        c, s = np.cos(yaw), np.sin(yaw)
        rotation = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.]])
        return self.asset @ rotation.T+center, self.support_samples @ rotation.T+center

    def query(self, center, yaw):
        asset, samples = self.posed_asset(center, yaw)
        center = np.asarray(center, float)
        raw_pairs, contact_pairs = 0, 0
        collision_complete = True
        if self.collision_backend == 'fcl':
            pairs, collision_complete = self._fcl_pairs(center, yaw)
            raw_pairs = len(pairs)
            contact_pairs = int(self._contact_scene_mask(self.scene[[pair[0] for pair in pairs]], center).sum()) if pairs else 0
        for triangle in asset if self.collision_backend == 'sat' else ():
            centroid = triangle.mean(0)
            radius = float(np.linalg.norm(triangle-centroid, axis=1).max())
            candidates = np.asarray(self.tree.query_ball_point(centroid, radius+self.max_scene_radius+1e-9), int)
            if not len(candidates):
                continue
            lo, hi = triangle.min(0), triangle.max(0)
            candidates = candidates[((self.scene_max[candidates] >= lo-1e-9) &
                                     (self.scene_min[candidates] <= hi+1e-9)).all(1)]
            if not len(candidates):
                continue
            scene = self.scene[candidates]
            overlap = triangle_surface_overlap(triangle, scene)
            intersecting = scene[overlap]
            raw_pairs += len(intersecting)
            if len(intersecting):
                # Explicit conservative contact-band classification, not eroded
                # geometry or a claim of measured physical penetration depth.
                contact_pairs += int(self._contact_scene_mask(intersecting, center).sum())
        known, contact = np.zeros(len(samples), bool), np.zeros(len(samples), bool)
        distances = np.full(len(samples), np.inf)
        if len(samples):
            lo, hi = samples.min(0), samples.max(0)
            ids = np.asarray(self.tree.query_ball_point((lo+hi)/2, np.linalg.norm((hi-lo)/2)+self.support_search+self.max_scene_radius), int)
            for i, (z, nz) in enumerate(vertical_triangle_heights(self.scene[ids], samples[:, :2])):
                gaps = samples[i, 2]-z
                eligible = (nz >= self.min_support_normal_z) & (gaps >= -self.contact_tolerance-1e-9) & (gaps <= self.support_search+1e-9)
                if eligible.any():
                    known[i] = True
                    distances[i] = abs(gaps[eligible]).min()
                    contact[i] = distances[i] <= self.contact_tolerance+1e-9
        fraction = lambda mask: float(mask.mean()) if len(mask) else 0.
        noncontact = raw_pairs-contact_pairs
        return dict(mesh_surface_intersections=raw_pairs,
                    collision_contact_band_pairs=contact_pairs, collision_noncontact_pairs=noncontact,
                    mesh_clearance='refute' if noncontact else ('support' if len(self.scene) and collision_complete else 'unknown'),
                    collision_backend=self.collision_backend, collision_complete=collision_complete,
                    collision_counts_are_lower_bounds=not collision_complete,
                    support_contact_fraction=fraction(contact), support_known_fraction=fraction(known),
                    support_unknown_fraction=fraction(~known) if len(known) else 1.,
                    support_gap_fraction=fraction(known & ~contact),
                    support_sample_count=len(samples), support_samples_world=samples,
                    support_grid_side=self.support_grid_side,
                    support_sampling_insufficient=self.support_sampling_insufficient,
                    support_distances_m=distances,
                    collision_evidence='actual_asset_triangles_vs_actual_scene_triangles_closed_surface_intersection',
                    contact_band_rule='entire_intersecting_near_horizontal_scene_triangle_within_bottom_z_tolerance',
                    support_evidence='quadrature_rays_actual_asset_lower_envelope_vs_scene_triangle_heights',
                    scene_normal_convention='absolute_normal_z_unoriented',
                    scene_geometry_available=bool(len(self.scene)),
                    containment='not_checked_fully_embedded_geometry_without_surface_crossing_can_be_missed',
                    watertight_scene_required_for_solid_containment=False,
                    clearance_scope='triangle_surface_crossings_only_not_solid_containment', penetration_depth_m=None,
                    stability='unknown', observed_free_space=False,
                    contact_tolerance_m=self.contact_tolerance, support_search_m=self.support_search,
                    query_count=1)
