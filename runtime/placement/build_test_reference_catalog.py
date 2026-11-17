"""Build TEST reference observations with frozen training-time algorithms/weights.

Reads prompt parser output, RGB-D detections, raw geometry, and frozen text only.
No test H5 or annotation arrays are opened. Missing inference jobs are blocking;
actual empty/failed detections and unsupported language remain explicit masks.
"""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'references'
FRAMES = ROOT / 'receivers/test_grounded_frames/inputs'
FOLDS = Path('/path/to/research300/reference_grounder_crossfit_full_20261101_v1')
SPLIT_LOCK = Path('/path/to/research100/official_split_training_audit_20261102_v1/official563_training_split_lock.json')
MODULES = Path('/path/to/research100/strict_real30_image_modules_20261102_v1')
sys.path.insert(0, str(MODULES))
from train_reference_grounding_v1 import PromptRelationGrounder

FROZEN_WEIGHT_SHAS = [
    'd74cb059de5beea0f9cf8f00530d689117392e2a72bc9d1f36048b8a8d6c2b26',
    '03f317bfe49cd0c79ad21ebf7590cfdbf07efe892c0e303bc0f630f18dd95663',
    '4dee15d527660420bc3945c9d46e7a7adedc8100e2469ba4eb7a3316b16ed090',
]
SURFACES = ('floor', 'table', 'coffee table', 'desk', 'counter', 'shelves', 'night stand', 'dresser', 'ottoman')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8') as source:
        for line in source:
            yield json.loads(line)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1048576), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    tmp = Path(path).with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    tmp.replace(path)


def relation(raw):
    # Verbatim mappings from frozen pilot_grounder_real_logits_v1.rel.
    x = (raw or '').lower().strip()
    if 'between' in x or 'middle of' in x or 'space between' in x: return 'between'
    if any(y in x for y in ['facing', 'directed', 'pointing', 'oriented', 'angled', 'turned', 'orientation', 'looking']): return 'facing'
    if any(y in x for y in ['not visible', 'hidden from', 'out of sight']): return 'not_visible'
    if any(y in x for y in ['visible', 'seen from', 'sight of', 'in view']): return 'is_visible'
    if any(y in x for y in ['adjacent', 'abutting', 'alongside', 'beside', 'next to']): return 'adjacent'
    if any(y in x for y in ['near', 'close', 'distance', 'vicinity', 'reach', 'proximity', 'not far']): return 'near'
    if x in ['on', 'resting on', 'lying on', 'sitting on', 'placed on', 'support surface']: return 'on'
    if any(y in x for y in ['below', 'under', 'beneath', 'lower than']): return 'below'
    if any(y in x for y in ['above', 'higher than']): return 'above'
    return None


def detection_index():
    jobs = list(rows(OUT / 'cleaned_detection_jobs_v2.jsonl'))
    split = list(rows(OUT / 'split_between_candidates_v1.jsonl'))
    lookup = defaultdict(set)
    selected = set()
    for job in jobs:
        if job['cleanup_status'] == 'detector_candidate':
            lookup[job['scene'], job['phrase'].strip().lower(), job['kind']].add(job['detector_phrase'])
            selected.add((job['scene'], job['detector_phrase']))
    for job in split:
        lookup[job['scene'], job['parent_phrase'].strip().lower(), job['kind']].add(job['detector_phrase'])
        selected.add((job['scene'], job['detector_phrase']))
    grouped = defaultdict(list)
    counts = Counter()
    missing = []
    for scene, phrase in sorted(selected):
        frames = sorted((FRAMES / scene / 'color').glob('*.jpg'))
        assert len(frames) == 4, 'Expected frozen four-frame budget'
        for rgb in frames:
            stem = rgb.stem
            key = hashlib.sha256((scene + '|' + stem + '|' + phrase).encode()).hexdigest()[:24]
            path = OUT / (key + '.json')
            if not path.exists():
                missing.append(key)
                continue
            item = read(path)
            assert (item['scene'], item['frame'], item['detector_phrase']) == (scene, stem, phrase)
            assert item['status'] in ('detected', 'no_detection', 'failed')
            if item['status'] != 'failed':
                assert path.with_suffix('.npz').exists(), 'Completed detection missing numerical observations'
            counts[item['status']] += 1
            grouped[scene, phrase].append((item, path))
    assert not missing, f'{len(missing)} required detector jobs unfinished; never replace unprocessed jobs with masks'
    return lookup, grouped, dict(counts)


def associate(grouped, geometry):
    features = {}
    counts = Counter()
    rgb_cache = {}
    with (OUT / 'groups.jsonl').open('w', encoding='utf-8') as world_output, \
         (OUT / 'groups_policy_frame_v1.jsonl').open('w', encoding='utf-8') as policy_output, \
         (OUT / 'grounder_candidate_features_v1.jsonl').open('w', encoding='utf-8') as feature_output:
        for (scene, phrase), items in sorted(grouped.items()):
            clusters, frame_rows = [], []
            for record, path in sorted(items, key=lambda value: value[0]['frame']):
                frame = record['frame']
                frame_rows.append(dict(frame=frame, status=record['status'], record=path.name, record_sha256=sha(path)))
                if record['status'] != 'detected': continue
                npz_path = path.with_suffix('.npz')
                used, observations = set(), []
                with np.load(npz_path, allow_pickle=False) as data:
                    for index, score in enumerate(data['scores']):
                        prefix = f'projection_{index}_'
                        if prefix + 'geometry_observed' not in data or not bool(data[prefix + 'geometry_observed']):
                            counts['projection_without_geometry'] += 1
                            continue
                        center = np.asarray(data[prefix + 'center_world'], float)
                        extent = np.asarray(data[prefix + 'observed_extent_world'], float)
                        assert center.shape == extent.shape == (3,) and np.isfinite(center).all() and np.isfinite(extent).all()
                        observations.append((float(score), index, center, extent))
                for score, index, center, extent in sorted(observations, key=lambda value: (-value[0], value[1])):
                    choices = [(float(np.linalg.norm(center - np.mean(item['centers'], axis=0))), key)
                               for key, item in enumerate(clusters) if key not in used]
                    distance, key = min(choices, default=(float('inf'), -1))
                    if distance > .20:
                        key = len(clusters)
                        clusters.append(dict(centers=[], extents=[], observations=[]))
                    used.add(key)
                    cluster = clusters[key]
                    cluster['centers'].append(center.tolist())
                    cluster['extents'].append(extent.tolist())
                    cluster['observations'].append(dict(frame=frame, detection_index=index, dino_score=score, source_npz=npz_path.name))
            rotation = np.asarray(geometry['alignments'][scene], float).reshape(4, 4)[:3, :3]
            translation = np.asarray(geometry['scenes'][scene]['observation_mean_translation_m'], float)
            assert rotation.shape == (3, 3) and translation.shape == (3,)
            assert np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
            world_candidates, policy_candidates, candidate_features = [], [], []
            for index, cluster in enumerate(clusters):
                center = np.mean(cluster['centers'], axis=0)
                extent = np.median(cluster['extents'], axis=0)
                xyz = (center - translation) @ rotation.T
                extents = np.abs(rotation) @ extent
                world = dict(id=index, center_world=center.tolist(), extent_median_world=extent.tolist(),
                             observations=cluster['observations'], frame_count=len(cluster['observations']))
                policy = dict(id=index, reference_xyz=xyz.tolist(), reference_extents=extents.tolist(),
                              observations=cluster['observations'], frame_count=len(cluster['observations']))
                best = max(cluster['observations'], key=lambda item: item['dino_score'])
                candidate = dict(id=index, reference_xyz=xyz.tolist(), reference_extents=extents.tolist(), observation=best, status='missing')
                try:
                    with np.load(OUT / best['source_npz'], allow_pickle=False) as data:
                        prefix = f"projection_{best['detection_index']}_"
                        points = np.asarray(data[prefix + 'points_world'], float)
                        uv = np.asarray(data[prefix + 'color_uv'], int)
                    assert len(points) >= 3 and points.shape == (len(uv), 3) and uv.shape[1] == 2 and np.isfinite(points).all()
                    rgb_key = (scene, best['frame'])
                    if rgb_key not in rgb_cache:
                        if len(rgb_cache) > 16: rgb_cache.clear()
                        rgb_cache[rgb_key] = np.asarray(Image.open(FRAMES / scene / 'color' / (best['frame'] + '.jpg')).convert('RGB'))
                    rgb = rgb_cache[rgb_key]
                    assert (uv >= 0).all() and (uv[:, 0] < rgb.shape[1]).all() and (uv[:, 1] < rgb.shape[0]).all()
                    color = rgb[uv[:, 1], uv[:, 0]].astype(float) / 127.5 - 1
                    # Match the existing grounder feature contract: raw-world,
                    # mean centered, without applying axis alignment rotation.
                    centered = points - translation
                    vector = np.r_[centered.mean(0), np.ptp(centered, axis=0), np.log1p(len(centered)), color.mean(0), color.std(0)]
                    assert vector.shape == (13,) and np.isfinite(vector).all()
                    candidate.update(status='available', grounder_features=vector.tolist())
                    counts['features_available'] += 1
                except Exception as exc:
                    candidate['error'] = type(exc).__name__ + ': ' + str(exc)[:200]
                    counts['features_failed'] += 1
                world_candidates.append(world)
                policy_candidates.append(policy)
                candidate_features.append(candidate)
            for output, value in [(world_output, dict(scene=scene, detector_phrase=phrase, frames=frame_rows, candidates=world_candidates)),
                                  (policy_output, dict(scene=scene, detector_phrase=phrase, frames=frame_rows, candidates=policy_candidates, coordinate_frame='axis_aligned_scene_mean_centered')),
                                  (feature_output, dict(scene=scene, detector_phrase=phrase, candidates=candidate_features))]:
                output.write(json.dumps(value, ensure_ascii=False) + '\n')
            features[scene, phrase] = [item for item in candidate_features if item['status'] == 'available']
            counts['groups'] += 1
            counts['clusters'] += len(clusters)
    return features, dict(counts)


def models_and_splits():
    models, splits, vocabulary, identities = [], [], None, []
    lock = read(SPLIT_LOCK)
    train_rooms = {scene[:9] for scene in lock['train_scenes']}
    test_rooms = {scene[:9] for scene in lock['test_scenes']}
    for index in range(3):
        root = FOLDS / f'fold{index}'
        checkpoint_path = root / 'last.pt'
        assert sha(checkpoint_path) == FROZEN_WEIGHT_SHAS[index]
        split = read(root / 'split.json')
        assert set(split['train_rooms']) <= train_rooms and not (set(split['train_rooms']) & test_rooms)
        assert set(scene[:9] for scene in split['normalization_scenes']) <= train_rooms
        state = torch.load(checkpoint_path, map_location='cpu', mmap=True, weights_only=False)
        if vocabulary is None: vocabulary = state['relation_types']
        assert vocabulary == state['relation_types']
        model = PromptRelationGrounder(len(vocabulary), hidden=128)
        model.load_state_dict(state['model'], strict=True)
        models.append(model.eval().requires_grad_(False))
        splits.append(split)
        identities.append(dict(fold=index, checkpoint_sha256=FROZEN_WEIGHT_SHAS[index], split_sha256=sha(root / 'split.json'),
                               epoch=state['epoch'], train_rooms=len(split['train_rooms']), test_overlap=0))
    return models, splits, vocabulary, identities


def run_logits(model, text, candidates, queries, vocabulary):
    assert candidates and queries
    with torch.inference_mode():
        output = model(torch.as_tensor(text, dtype=torch.float32)[None],
                       torch.tensor([item['grounder_features'] for item in candidates], dtype=torch.float32)[None],
                       torch.tensor([[vocabulary.index(query['type']) for query in queries]], dtype=torch.long),
                       torch.ones((1, len(candidates)), dtype=torch.bool),
                       torch.ones((1, len(queries)), dtype=torch.bool))[0].numpy()
    assert output.shape == (len(queries), len(candidates) + 1) and np.isfinite(output).all()
    return output


def score(features, lookup, models, splits, vocabulary):
    tasks = list(rows(OUT / 'mapped_test_tasks.jsonl'))
    assert len(tasks) == 3500 and [task['original_entry_id'] for task in tasks] == list(range(3500))
    text_features = np.load(ROOT / 'text_features.npy', allow_pickle=False)
    assert text_features.shape == (3500, 768) and np.isfinite(text_features).all()
    counts = Counter()
    path = OUT / 'reference_catalog_compact_v4_test.jsonl'
    temporary = path.with_suffix('.partial')
    with temporary.open('w', encoding='utf-8') as output:
        for task in tasks:
            scene, base = task['scene'], task['scene'][:9]
            eligible = [index for index, split in enumerate(splits) if base not in split['train_rooms']]
            assert len(eligible) == 3
            fold = int.from_bytes(hashlib.sha256(f'1729:{base}'.encode()).digest()[:8], 'big') % 3
            text = text_features[task['original_entry_id']]
            candidates, query_specs, unresolved, inferred = {}, [], [], []
            for kind in ('explicit', 'inferred'):
                for query in task[kind]:
                    phrase = query['phrase'].strip().lower()
                    phrases = sorted(lookup.get((scene, phrase, kind), set()))
                    details = dict(kind=kind, phrase=phrase, relation=query.get('relation'),
                                   detectors=[dict(detector_phrase=value, scene=scene, candidate_count=len(features.get((scene, value), []))) for value in phrases])
                    if kind == 'inferred':
                        inferred.append(details)
                        continue
                    relation_type = relation(query.get('relation'))
                    if relation_type not in vocabulary:
                        unresolved.append(dict(details, state='unsupported_relation'))
                        continue
                    keys = []
                    for detector in phrases:
                        for candidate in features.get((scene, detector), []):
                            key = detector + '#' + str(candidate['id'])
                            candidates[key] = candidate
                            keys.append(key)
                    if keys:
                        query_specs.append(dict(type=relation_type, phrase=phrase, candidate_keys=sorted(set(keys))))
                    else:
                        unresolved.append(dict(details, state='no_available_candidate' if phrases else 'needs_semantic_review_or_generic'))
            keys = sorted(candidates, key=lambda key: (-candidates[key]['observation']['dino_score'], key))[:32]
            indices = {key: index for index, key in enumerate(keys)}
            for query in query_specs:
                query['indices'] = [indices[key] for key in query.pop('candidate_keys') if key in indices]
            query_specs = [query for query in query_specs if query['indices']]
            selected = [candidates[key] for key in keys]
            compact_candidates = [dict(id=key, reference_xyz=value['reference_xyz'], reference_extents=value['reference_extents'], observation=value['observation']) for key, value in zip(keys, selected)]
            compact_queries = []
            if query_specs and selected:
                raw = run_logits(models[fold], text, selected, query_specs, vocabulary)
                for query, values in zip(query_specs, raw):
                    allowed = set(query['indices'])
                    effective = [float(value) if index in allowed else -10000. for index, value in enumerate(values[:-1])] + [float(values[-1])]
                    null_top = max(range(len(effective)), key=lambda index: effective[index]) == len(keys)
                    compact_queries.append(dict(type=query['type'], phrase=query['phrase'], candidate_indices=sorted(allowed),
                                                actual_grounder_logits=values.tolist(), effective_logits=effective,
                                                relation_mask=bool(allowed and not null_top), null_is_top=null_top,
                                                reference_kind='explicit'))
            # Frozen v4 policy permits only floor/stable-surface hypotheses.
            # As in its keyed pilot merge, the last supported inferred query wins.
            inferred_selected = None
            for query in inferred:
                phrase = query['phrase']
                if phrase not in ('floor', 'the floor', 'stable surface'): continue
                extras = []
                for detector in query['detectors']:
                    extras.extend(features.get((scene, detector['detector_phrase']), []))
                if phrase == 'stable surface':
                    for surface in SURFACES:
                        extras.extend(dict(item, source_detector_phrase=surface) for item in features.get((scene, surface), []))
                extras = sorted(extras, key=lambda item: -item['observation']['dino_score'])[:32]
                if extras and 'on' in vocabulary:
                    values = run_logits(models[fold], text, extras, [dict(type='on')], vocabulary)[0]
                    inferred_selected = (query, extras, values)
            if inferred_selected is not None:
                query, extras, values = inferred_selected
                old_count = len(compact_candidates)
                for explicit in compact_queries:
                    for name in ('actual_grounder_logits', 'effective_logits'):
                        logits = explicit[name]
                        explicit[name] = logits[:-1] + [-10000.] * len(extras) + logits[-1:]
                for candidate in extras:
                    compact_candidates.append(dict(id='inferred_' + candidate.get('source_detector_phrase', 'floor') + '#' + str(candidate['id']),
                                                   reference_xyz=candidate['reference_xyz'], reference_extents=candidate['reference_extents'], observation=candidate['observation']))
                padded = [-10000.] * old_count + values.tolist()
                compact_queries.append(dict(type='on', phrase=query['phrase'], candidate_indices=list(range(old_count, old_count + len(extras))),
                                            actual_grounder_logits=padded, effective_logits=padded, relation_mask=True,
                                            null_is_top=bool(values[-1] >= max(values[:-1])), reference_kind='inferred',
                                            hypothesis='possible_support_surface', source_fold=fold))
                counts['added_inferred_tasks'] += 1
            active = any(query['relation_mask'] for query in compact_queries)
            record = dict(scene=scene, group_id=task['group_id'], original_entry_id=task['original_entry_id'], asset_scene=task['asset_scene'], entry=task['entry'],
                          prompt_sha256=task['prompt_sha256'], grounder_fold=fold, candidates=compact_candidates,
                          reference_mask=[True] * len(compact_candidates), queries=compact_queries, unresolved=unresolved,
                          inferred_pending=inferred, status='active' if active else 'masked')
            for query in record['queries']:
                assert len(query['effective_logits']) == len(record['candidates']) + 1
            output.write(json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n')
            counts['tasks'] += 1
            counts[record['status']] += 1
            counts['explicit_queries'] += sum(query.get('reference_kind') == 'explicit' for query in compact_queries)
            counts['inferred_queries'] += sum(query.get('reference_kind') == 'inferred' for query in compact_queries)
            counts['tasks_with_unresolved'] += bool(unresolved)
            if counts['tasks'] % 100 == 0:
                save(OUT / 'catalog_progress.json', dict(status='scoring', counts=counts))
    assert counts['tasks'] == 3500
    temporary.replace(path)
    return path, dict(counts)


def main():
    torch.set_num_threads(2)
    started = time.monotonic()
    geometry_path = ROOT / 'geometry/mesh_manifest_v6.json'
    geometry = read(geometry_path)
    lookup, groups, detection_counts = detection_index()
    assert detection_counts.get('failed', 0) == 0, 'Detector internal errors retained; resolve before catalog admission'
    features, association_counts = associate(groups, geometry)
    assert association_counts.get('features_failed', 0) == 0, 'Candidate feature errors retained; do not silently mask internal errors'
    models, splits, vocabulary, identities = models_and_splits()
    catalog, counts = score(features, lookup, models, splits, vocabulary)
    manifest = dict(status='catalog_built_pending_model_smoke', version='compact_reference_catalog_v4_test', split='test', data_path=str(catalog), data_sha256=sha(catalog),
                    task_count=3500, counts=counts, detection_counts=detection_counts, association_counts=association_counts,
                    grounder_identities=identities, official_split_lock_sha256=sha(SPLIT_LOCK),
                    task_manifest_sha256=sha(ROOT / 'tasks_manifest.json'), text_features_sha256=sha(ROOT / 'text_features.npy'),
                    geometry_manifest_sha256=sha(geometry_path), parsed_prompts_sha256=sha(OUT / 'deepseek_reference_results.jsonl'),
                    frame_source_identity_sha256=sha(ROOT / 'receivers/test_grounded_frames/frame_source_identity_before_models.json'),
                    detector_source_sha256=sha(ROOT / 'reference_detector.py'), builder_source_sha256=sha(__file__),
                    coordinate_frame='axis_aligned_scene_mean_centered',
                    grounder_feature_frame='raw world minus official float32 observation mean, no axis rotation',
                    association_distance_gate_m=.20, explicit_candidate_limit=32, inferred_candidate_limit=32,
                    inferred_policy='frozen v4 floor/the floor/stable surface only; separate inferred residual; no test calibration',
                    fold_policy='sha256(1729:base_room) first8 bytes big-endian modulo3; all folds train-only',
                    python_executable=sys.executable, torch_version=torch.__version__,
                    placement_labels_read=False, ground_truth_relationships_read=False, ground_truth_instances_used=False,
                    no_unprocessed_detection_mask_substitution=True, elapsed_seconds=time.monotonic() - started)
    save(OUT / 'reference_catalog_compact_v4_test_pending.json', manifest)
    save(OUT / 'catalog_progress.json', dict(status='catalog_built_pending_model_smoke', counts=counts, elapsed_seconds=manifest['elapsed_seconds']))
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
