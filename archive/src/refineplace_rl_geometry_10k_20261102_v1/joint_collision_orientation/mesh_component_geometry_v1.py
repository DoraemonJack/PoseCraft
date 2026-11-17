"""Existing closed-surface subset extraction; no hole filling or mesh repair.

Exact duplicate removal and manifold-edge partitioning affect a proxy copy
only. Original asset triangles remain the support/collision source of truth.
All retained closed components must be queried together with even/odd parity:
independently filling each component would fill nested cavities.
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components


def closed_component_subset(triangles, *, deduplicate=True,
                            split_nonmanifold=True, include_components=False):
    triangles = np.asarray(triangles, float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.isfinite(triangles).all():
        raise ValueError('finite triangles[F,3,3] required')
    area2 = np.linalg.norm(np.cross(triangles[:, 1]-triangles[:, 0],
                                    triangles[:, 2]-triangles[:, 0]), axis=1)
    triangles, area2 = triangles[area2 > 1e-14], area2[area2 > 1e-14]
    original_count = len(triangles)
    if not original_count:
        return triangles.copy(), dict(face_count=0, original_face_count=0,
            deduplicated_face_count=0, duplicates_removed=0, component_count=0,
            closed_component_count=0, closed_face_fraction=0.,
            closed_surface_area_fraction=0., global_closed_edge_topology=False,
            full_surface_coverage=False, boundary_edges=0, nonmanifold_edges=0,
            components=[])
    _, inverse = np.unique(triangles.reshape(-1, 3), axis=0, return_inverse=True)
    faces = inverse.reshape(-1, 3)
    if deduplicate:
        # Faces with the same three exact points represent the same surface,
        # regardless of winding. Keep the first; never merge nearby vertices.
        _, unique = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
        unique.sort()
        triangles, area2, faces = triangles[unique], area2[unique], faces[unique]
    directed_edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    _, edge_id, edge_count = np.unique(np.sort(directed_edges, axis=1), axis=0,
                                       return_inverse=True, return_counts=True)
    owners = np.tile(np.arange(len(triangles)), 3)
    order = np.argsort(edge_id, kind='stable')
    same = edge_id[order[1:]] == edge_id[order[:-1]]
    if split_nonmanifold:
        same &= edge_count[edge_id[order[1:]]] == 2
    left, right = owners[order[:-1][same]], owners[order[1:][same]]
    graph = coo_matrix((np.ones(len(left), dtype=np.int8), (left, right)),
                       shape=(len(triangles), len(triangles))).tocsr()
    component_count, labels = connected_components(graph, directed=False)
    # Recount incidence INSIDE each extracted component. An attached open sheet
    # may give the original body edge three incidences; the body itself has two.
    component_edges, incidence = np.unique(np.column_stack((labels[owners], edge_id)),
                                            axis=0, return_counts=True)
    bad = np.bincount(component_edges[incidence != 2, 0], minlength=component_count)
    is_closed = bad == 0
    selected = is_closed[labels]
    face_counts = np.bincount(labels, minlength=component_count)
    areas = np.bincount(labels, weights=area2/2, minlength=component_count)
    report = dict(face_count=len(triangles), original_face_count=original_count,
        deduplicated_face_count=len(triangles), duplicates_removed=original_count-len(triangles),
        component_count=int(component_count), closed_component_count=int(is_closed.sum()),
        closed_face_fraction=float(selected.mean()),
        closed_surface_area_fraction=float(areas[is_closed].sum()/areas.sum()),
        global_closed_edge_topology=bool(np.all(edge_count == 2)),
        full_surface_coverage=bool(selected.all()),
        boundary_edges=int(np.count_nonzero(edge_count == 1)),
        nonmanifold_edges=int(np.count_nonzero(edge_count > 2)),
        component_rule='degree_two_edges' if split_nonmanifold else 'all_shared_edges',
        exact_duplicate_faces_removed=bool(deduplicate),
        interpretation='existing_closed_surfaces_only; no_hole_filling; parity_over_all_retained_components')
    if include_components:
        lo, hi = np.full((component_count, 3), np.inf), np.full((component_count, 3), -np.inf)
        np.minimum.at(lo, labels, triangles.min(1))
        np.maximum.at(hi, labels, triangles.max(1))
        volumes = np.bincount(labels, weights=np.einsum('ij,ij->i', triangles[:, 0],
                                np.cross(triangles[:, 1], triangles[:, 2]))/6., minlength=component_count)
        boundaries = np.bincount(component_edges[incidence == 1, 0], minlength=component_count)
        nonmanifold = np.bincount(component_edges[incidence > 2, 0], minlength=component_count)
        report['components'] = [dict(face_count=int(face_counts[i]), surface_area_m2=float(areas[i]),
            boundary_edges=int(boundaries[i]), nonmanifold_edges=int(nonmanifold[i]),
            closed_edge_topology=bool(is_closed[i]), signed_volume_m3=float(volumes[i]),
            bounds_min_m=lo[i].tolist(), bounds_max_m=hi[i].tolist())
            for i in sorted(range(component_count), key=lambda k: (-face_counts[k], lo[k].tolist()))]
    return triangles[selected].copy(), report
