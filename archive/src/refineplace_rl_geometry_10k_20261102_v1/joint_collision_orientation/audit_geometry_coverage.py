"""Replay at most 64 recorded TRAIN poses through the opt-in geometry backend.

No policy, labels(), placement arrays, checkpoint, reward quality or annotation
is loaded. Only observation inputs and the recorded center/yaw are consumed.
Disconnected-component statistics diagnose possible future geometry changes;
this audit does not alter the collision implementation or its targets.
"""
import argparse
from collections import Counter, OrderedDict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def recorded_poses(path, limit):
    if not 1 <= limit <= 64:
        raise ValueError('This bounded audit permits 1 through 64 poses')
    result = []
    with Path(path).open(encoding='utf-8') as handle:
        for line in handle:
            row = json.loads(line)
            if len(row['task_ids']) != len(row['poses']):
                raise ValueError('Recorded task IDs and poses do not align')
            for key, pose in zip(row['task_ids'], row['poses']):
                # Deliberately ignore paired_quality and every other pose field.
                center = np.asarray(pose['center'], float)
                yaw = float(pose['yaw'])
                if center.shape != (3,) or not np.isfinite(center).all() or not np.isfinite(yaw):
                    raise ValueError('Invalid recorded center/yaw')
                result.append(dict(scene=str(key[0]), group_id=int(key[1]),
                                   center=center.tolist(), yaw=yaw))
                if len(result) == limit:
                    return result
    raise ValueError(f'Only {len(result)} recorded poses; requested {limit}')


def locked_prefix(manifest, limit):
    result = []
    for row in manifest:
        if len(row['positions']) != len(row['group_ids']):
            raise ValueError('Locked positions and group IDs do not align')
        for position, group_id in zip(row['positions'], row['group_ids']):
            result.append(dict(scene=row['scene'], position=int(position), group_id=int(group_id)))
            if len(result) == limit:
                return result
    raise ValueError('Locked manifest contains fewer tasks than the audit limit')


def component_topology(vertices, faces):
    """Original raw topology plus recovery diagnostics; never edits the asset."""
    from mesh_component_geometry_v1 import closed_component_subset
    triangles = np.asarray(vertices, float)[np.asarray(faces, int)]
    _, original = closed_component_subset(triangles, deduplicate=False,
                                          split_nonmanifold=False, include_components=True)
    _, recovery = closed_component_subset(triangles, deduplicate=True,
                                          split_nonmanifold=True, include_components=False)
    original['proxy_recovery'] = recovery
    return original


def coverage_causes(stats):
    flags = []
    if not stats['asset_closed_edge_topology']:
        flags.append('open_or_nonmanifold_asset_topology')
    elif not stats['closed_asset_interior_proxy_mask']:
        flags.append('closed_asset_field_unresolved')
    if not stats['local_scene_evidence_mask']:
        flags.append('no_local_noncontact_scene_samples')
    if not stats['collision_proxy_sampling_complete']:
        flags.append('local_scene_triangle_cap')
    if not stats['collision_proxy_field_complete']:
        flags.append('asset_parity_incomplete')
    if stats.get('collision_proxy_partial_asset', False):
        flags.append('partial_closed_component_asset')
    elif stats.get('collision_proxy_full_asset', False):
        flags.append('full_closed_component_asset')
    if stats['collision_cost_known']:
        flags.append('continuous_positive' if stats['collision_cost'] > 0 else 'continuous_zero')
    elif stats['collision_noncontact_pairs'] > 0:
        flags.append('exact_surface_crossing_fallback')
    else:
        flags.append('unknown_without_crossing')
    return flags


STAT_FIELDS = (
    'collision_cost', 'collision_cost_known', 'collision_cost_unknown_reason',
    'asset_closed_edge_topology', 'closed_asset_interior_proxy_mask',
    'local_scene_evidence_mask', 'collision_proxy_sampling_complete',
    'collision_proxy_field_complete', 'collision_proxy_parity_valid_fraction',
    'collision_proxy_grid_spacing_m', 'collision_proxy_grid_shape',
    'collision_proxy_grid_allowance_m', 'collision_proxy_asset_preparation_seconds',
    'collision_proxy_asset_preparation_cache_hit', 'collision_proxy_scene_sample_count',
    'collision_proxy_local_triangle_count', 'collision_proxy_triangle_cap',
    'collision_proxy_max_interior_distance_m', 'collision_proxy_excess_m',
    'collision_tolerance_m', 'collision_cost_scale_m', 'collision_noncontact_pairs',
    'collision_complete', 'directional_collision_vector', 'support_contact_fraction',
    'support_known_fraction', 'support_gap_fraction', 'scene_geometry_available',
    'collision_proxy_partial_asset', 'collision_proxy_full_asset',
    'collision_proxy_covered_surface_area_fraction', 'collision_proxy_covered_face_fraction',
    'collision_proxy_closed_component_count', 'collision_proxy_component_count',
    'collision_proxy_duplicate_faces_removed',
)


def run(args):
    # Import the admitted reader and treatment backend only at run time so this
    # script's pure audit helpers remain locally testable without server data.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    sys.path.insert(0, str(HERE))
    import train_joint as training
    d = training.d
    from cpu_pool_backend_v1 import observation_scene_class
    from geometry_backend import CollisionGeometryBackend
    from constrained_objective import collision_target
    started = time.perf_counter()
    steps_path = args.smoke_dir/'steps.jsonl'
    manifest_path = args.smoke_dir/'locked_training_tasks.json'
    protocol = d.read(args.smoke_dir/'protocol_before_training.json')
    if sha(manifest_path) != protocol['training_manifest_sha256']:
        raise ValueError('Smoke copied task manifest does not match its locked protocol')
    if sha(d.LOCK) != protocol['official_split_lock_sha256']:
        raise ValueError('Official split lock does not match the smoke protocol')
    poses = recorded_poses(steps_path, args.limit)
    prefix = locked_prefix(d.read(manifest_path), args.limit)
    if [(r['scene'], r['group_id']) for r in poses] != [(r['scene'], r['group_id']) for r in prefix]:
        raise ValueError('Recorded pose order differs from the first locked tasks')
    lock = d.read(d.LOCK)
    library = d.paired.load_mesh_manifest('/path/to/research300/placeit_mesh_v6/mesh_manifest_v6.json')
    scene_groups = OrderedDict()
    for index, record in enumerate(prefix):
        d.guard(lock, record['scene'])
        scene_groups.setdefault(record['scene'], []).append((index, record))
    results = [None]*args.limit
    asset_topologies = {}
    observation = observation_scene_class(d)

    def forbidden_labels(*_args, **_kwargs):
        raise AssertionError('Coverage audit must not read labels')

    with CollisionGeometryBackend(workers=12) as backend:
        for scene, records in scene_groups.items():
            source = observation(scene, lock, library)
            source.labels = forbidden_labels
            try:
                rows = source.inputs([record['position'] for _, record in records])
                assert source.labels_arrays is None and not source.label_cache
                tasks = []
                for (index, expected), row in zip(records, rows):
                    assert (row['scene'], int(row['group_id'])) == (expected['scene'], expected['group_id'])
                    uid = row['asset_uuid']
                    if uid not in asset_topologies:
                        tick = time.perf_counter()
                        vertices, faces = library.asset(uid, row['dims'])
                        asset_topologies[uid] = component_topology(vertices, faces)
                        asset_topologies[uid]['audit_seconds'] = time.perf_counter()-tick
                    tasks.append(dict(scene=scene, split='train', asset_uuid=uid,
                                      dims=np.asarray(row['dims'], float).tolist(),
                                      center=poses[index]['center'], yaw=poses[index]['yaw']))
                tick = time.perf_counter()
                values = backend.query_many(tasks)
                for (index, expected), task, value in zip(records, tasks, values):
                    vector, stats, query_count = value
                    assert np.asarray(vector).shape == (26,) and query_count == 1
                    cost, known, mode = collision_target(stats)
                    results[index] = dict(index=index, group_id=expected['group_id'],
                        observation_position=expected['position'], **task,
                        causes=coverage_causes(stats), target_cost=cost,
                        target_known=bool(known), target_mode=mode,
                        stats={key: stats[key] for key in STAT_FIELDS})
                print(json.dumps(dict(event='coverage_scene_complete', scene=scene,
                                      tasks=len(tasks), query_seconds=time.perf_counter()-tick)), flush=True)
                assert source.labels_arrays is None and not source.label_cache
            finally:
                source.close()
    assert all(row is not None for row in results)
    causes, modes, reasons = Counter(), Counter(), Counter()
    assets = OrderedDict()
    for row in results:
        causes.update(row['causes'])
        modes[row['target_mode']] += 1
        if not row['stats']['collision_cost_known']:
            reasons[str(row['stats']['collision_cost_unknown_reason'])] += 1
        uid = row['asset_uuid']
        group = assets.setdefault(uid, dict(task_count=0, dims=row['dims'], causes=Counter(),
            modes=Counter(), unknown_reasons=Counter(), known_costs=[], topology=asset_topologies[uid],
            example_indices=[], grid_spacings_m=set(), preparation_seconds=[]))
        group['task_count'] += 1
        group['causes'].update(row['causes'])
        group['modes'][row['target_mode']] += 1
        group['example_indices'].append(row['index'])
        group['grid_spacings_m'].add(row['stats']['collision_proxy_grid_spacing_m'])
        group['preparation_seconds'].append(row['stats']['collision_proxy_asset_preparation_seconds'])
        if row['stats']['collision_cost_known']:
            group['known_costs'].append(row['stats']['collision_cost'])
        else:
            group['unknown_reasons'][str(row['stats']['collision_cost_unknown_reason'])] += 1
    for group in assets.values():
        group['grid_spacings_m'] = sorted(group['grid_spacings_m'])
        group['preparation_seconds_max'] = max(group.pop('preparation_seconds'))
    summary = dict(tasks=len(results), assets=len(assets), causes=dict(causes),
                   unknown_reasons=dict(reasons), target_modes=dict(modes),
                   geometry_query_count=len(results), labels_called=0,
                   annotation_arrays_opened=0, policy_inference_calls=0,
                   closed_component_recovery_candidates=[uid for uid, group in assets.items()
                       if not group['topology'].get('global_closed_edge_topology', False)
                       and group['topology']['closed_component_count'] > 0],
                   elapsed_seconds=time.perf_counter()-started)
    report = dict(summary=summary, assets=assets, records=results,
                  evidence_scope='first_locked_train_tasks_at_recorded_center_yaw; observation_only',
                  geometry_unchanged=True, independent_test_accessed=False,
                  steps_sha256=sha(steps_path), task_manifest_sha256=sha(manifest_path),
                  split_lock_sha256=sha(d.LOCK), audit_sha256=sha(__file__),
                  geometry_sha256=sha(HERE/'joint_asset_geometry_v1.py'),
                  component_geometry_sha256=sha(HERE/'mesh_component_geometry_v1.py'))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    d.write(args.output, report)
    print(json.dumps(d.clean(summary), allow_nan=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-dir', type=Path, default=HERE/'smoke_train')
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--output', type=Path, default=HERE/'geometry_coverage_audit.json')
    run(parser.parse_args())
