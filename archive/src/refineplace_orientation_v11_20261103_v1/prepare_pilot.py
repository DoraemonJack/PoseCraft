"""Lock a scene-efficient training-only pilot before reading labels/results.

This metadata preparation does not import the model, read placement targets,
or run inference. Whole scenes retain their original facing/nonfacing mixture;
only the final scene is truncated by a reproducible identity hash ordering.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


DEFAULT_ROOT = Path('/path/to/research300')
DEFAULT_LOCK = Path('/path/to/research100/official_split_training_audit_20261102_v1/official563_training_split_lock.json')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity_hash(seed, kind, *identity):
    return hashlib.sha256(':'.join(map(str, ('orientation-v11', seed, kind, *identity))).encode()).hexdigest()


def room(scene):
    return scene.split('_', 1)[0]


def build_manifest(catalog, split_lock, validation_manifest, cache_root, *, count=10000, seed=419):
    if count <= 0:
        raise ValueError('Positive task count required')
    catalog, split_lock, validation_manifest, cache_root = map(Path, (catalog, split_lock, validation_manifest, cache_root))
    lock = json.loads(split_lock.read_text(encoding='utf-8'))
    validation = json.loads(validation_manifest.read_text(encoding='utf-8'))
    train_scenes = set(lock['train_scenes'])
    validation_tasks = validation['tasks']
    if len(validation_tasks) != 150:
        raise ValueError('Expected fixed 150 validation tasks')
    if not all(t['scene'] in lock['validation_scenes'] for t in validation_tasks):
        raise ValueError('Validation metadata not in official validation split')
    excluded_rooms = {room(s) for s in lock['validation_scenes'] + lock['test_scenes']}
    if {room(s) for s in train_scenes} & excluded_rooms:
        raise ValueError('Training room identity overlaps evaluation partitions')
    grouped = defaultdict(list)
    all_keys = set()
    full_stats = Counter()
    with catalog.open(encoding='utf-8') as handle:
        for line in handle:
            z = json.loads(line)
            scene, group_id = z['scene'], int(z['group_id'])
            if scene not in train_scenes or room(scene) in excluded_rooms:
                raise ValueError('Catalog row outside training split')
            key = (scene, group_id)
            if key in all_keys:
                raise ValueError('Duplicate catalog task identity')
            all_keys.add(key)
            facing = any(q['type'] == 'facing' for q in z['queries'])
            active = any(q['type'] == 'facing' and q['relation_mask'] for q in z['queries'])
            task = {k: z[k] for k in ('scene', 'asset_scene', 'entry', 'prompt_sha256')}
            task.update(group_id=group_id, facing_requested=bool(facing), facing_reference_active=bool(active))
            grouped[scene].append(task)
            full_stats.update(tasks=1, facing_tasks=int(facing), facing_reference_active_tasks=int(active))
    if len(all_keys) != 100505 or set(grouped) != train_scenes or len(train_scenes) != 563:
        raise ValueError('Expected full official 100505-task/563-scene training catalog')
    if count > len(all_keys):
        raise ValueError('Requested pilot exceeds catalog')
    selected, summaries = [], []
    for scene in sorted(grouped, key=lambda s: identity_hash(seed, 'scene', s)):
        remaining = count - len(selected)
        if remaining == 0:
            break
        ordered = sorted(grouped[scene], key=lambda t: identity_hash(seed, 'task', scene, t['group_id']))
        chosen = ordered[:remaining]
        registry_path = cache_root / scene / 'task_registry.json'
        registry = json.loads(registry_path.read_text(encoding='utf-8'))
        registry_by_id = {int(r['group_id']): r for r in registry}
        if len(registry_by_id) != len(registry):
            raise ValueError('Duplicate feature-cache task identity')
        for task in chosen:
            original = registry_by_id[task['group_id']]
            if not all(task[k] == original[k] for k in ('scene', 'asset_scene', 'entry', 'prompt_sha256')):
                raise ValueError('Catalog and feature-cache identities disagree')
        selected.extend(chosen)
        summaries.append(dict(scene=scene, source_tasks=len(ordered), selected_tasks=len(chosen),
                              complete_scene=len(chosen) == len(ordered),
                              facing_tasks=sum(t['facing_requested'] for t in chosen),
                              registry_sha256=sha(registry_path)))
    assert len(selected) == count and len({(t['scene'], t['group_id']) for t in selected}) == count
    selected_rooms = {room(t['scene']) for t in selected}
    assert not selected_rooms & {room(t['scene']) for t in validation_tasks}
    return dict(version='RefinePlace_orientation_v11_pilot', created_utc=datetime.now(timezone.utc).isoformat(),
                seed=seed, tasks_total=count, scenes_total=len(summaries), tasks=selected, scenes=summaries,
                catalog_sha256=sha(catalog), split_lock_sha256=sha(split_lock),
                validation_manifest_sha256=sha(validation_manifest),
                sources=dict(catalog=str(catalog), split_lock=str(split_lock), validation_manifest=str(validation_manifest),
                             cache_root=str(cache_root)),
                facing_tasks=sum(t['facing_requested'] for t in selected),
                nonfacing_tasks=sum(not t['facing_requested'] for t in selected),
                facing_reference_active_tasks=sum(t['facing_reference_active'] for t in selected),
                full_catalog_counts=dict(full_stats),
                selection='Seeded scene-identity hash; complete scenes in that order; final scene cut by seeded task-identity hash',
                mixture='Preserve original within-scene facing/nonfacing mixture; no outcome-based or class-balanced selection',
                scope='Training subset, not a representative estimate of full-dataset performance',
                validation_tasks=150, validation_train_room_overlap=0, placement_labels_read=False,
                evaluation_results_read=False, test_task_metadata_read=False,
                script_sha256=sha(__file__))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True, help='New manifest JSON path; refuses overwrite')
    p.add_argument('--catalog', type=Path, default=DEFAULT_ROOT/'refineplace_reference_multiframe_assoc_20261102_v1/reference_catalog_compact_v4_inferred_pilot.jsonl')
    p.add_argument('--split-lock', type=Path, default=DEFAULT_LOCK)
    p.add_argument('--validation-manifest', type=Path, default=DEFAULT_ROOT/'refineplace_sync20_inferred_official150_20261102_v1/fixed_validation150.json')
    p.add_argument('--cache-root', type=Path, default=DEFAULT_ROOT/'refineplace_v01_official563_feature_cache_20261102_v1')
    p.add_argument('--count', type=int, default=10000)
    p.add_argument('--seed', type=int, default=419)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(f'Preserve the locked manifest: {a.output}')
    result = build_manifest(a.catalog, a.split_lock, a.validation_manifest, a.cache_root, count=a.count, seed=a.seed)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x', encoding='utf-8') as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
        handle.write('\n')
    print(json.dumps({k: result[k] for k in ('tasks_total', 'scenes_total', 'facing_tasks', 'nonfacing_tasks',
                                           'facing_reference_active_tasks', 'validation_train_room_overlap')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
