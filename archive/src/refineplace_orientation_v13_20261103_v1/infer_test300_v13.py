"""Frozen V1.3 inference for official test entry IDs 0..299, with action traces.

--limit N runs the first N identities as a smoke test; only N=300 produces
predictions.json and declares the 300-case diagnostic subset complete.
The official numeric yaw definition is preserved: no extra 90-degree offset.

Uses the same sealed observation inputs as the original full-test export.
No placement/relationship ground truth is opened, no official evaluator is
modified, and the full-test baseline is read-only. This subset is exposed test
diagnostics and must not be presented as an untouched validation partition.
The existing Linux dependency tree/absolute mesh manifests must be available;
--data-root relocates the prepared input directory, not its declared files.
"""
from __future__ import annotations

import argparse
import copy
from collections import Counter
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import time


HERE = Path(__file__).resolve().parent
DEFAULT_DATA = Path('/path/to/research300/refineplace_v1_epoch2_fulltest_20261103_v1')
BASELINE_SHA256 = '97038720bcf7aae9ca29513108a500c91119fc856f96311a3bca4743c616dab4'
MAX_TASKS = 300
V11_PREDICTION_SHA256 = 'c511211848fd0d89ca6185b4789603679c7659643600e8dba1c97c535aa73c74'
V11_CHECKPOINT_SHA256 = '480bb902bbbeb174a36da395fdf531c1845386cc66d653478c2584081e41c774'
DEFAULT_V11_PREDICTIONS = Path('/path/to/research300/refineplace_orientation_v11_20261103_v1/paired10k/upgraded_test300/predictions.json')


def selected_entry_ids(limit):
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= MAX_TASKS:
        raise ValueError('limit must be an integer in [1, 300]')
    return tuple(range(limit))


def prediction_filename(limit):
    selected_entry_ids(limit)
    return 'predictions.json' if limit == MAX_TASKS else f'predictions_smoke_{limit}.json'


def nonfinite_paths(value, prefix=''):
    if isinstance(value, dict):
        return [p for k, v in value.items() for p in nonfinite_paths(v, f'{prefix}.{k}' if prefix else str(k))]
    if isinstance(value, (list, tuple)):
        return [p for i, v in enumerate(value) for p in nonfinite_paths(v, f'{prefix}[{i}]')]
    return [prefix] if isinstance(value, float) and not math.isfinite(value) else []


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    pending = path.with_name(path.name + '.pending')
    pending.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(pending, path)


def diagnostic_json(value):
    if isinstance(value, dict):
        return {k: diagnostic_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [diagnostic_json(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_runtime(data_root):
    path = Path(data_root) / 'infer_fulltest.py'
    if not path.is_file():
        raise FileNotFoundError('Prepared original inference runtime required: ' + str(path))
    spec = importlib.util.spec_from_file_location('_orientation_frozen_test_runtime', path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    return runtime


def check_baseline(path, entry_ids):
    if path is None:
        return None
    if sha(path) != BASELINE_SHA256:
        raise ValueError('Baseline must be the preserved original 3500-entry export')
    rows = read(path)
    by_id = {int(r['entry_id']): r for r in rows}
    if len(rows) != 3500 or set(by_id) != set(range(3500)):
        raise ValueError('Baseline entry identities differ from the original complete export')
    return {i: by_id[i] for i in entry_ids}


def check_v11_predictions(path, entry_ids):
    if sha(path) != V11_PREDICTION_SHA256:
        raise ValueError('V11 baseline must be the pinned upgraded 300-task export')
    rows = read(path)
    if (not isinstance(rows, list) or len(rows) != MAX_TASKS
            or any(type(r.get('entry_id')) is not int for r in rows)
            or [r['entry_id'] for r in rows] != list(range(MAX_TASKS))):
        raise ValueError('V11 prediction identities must be the original ordered 0..299')
    return {i: rows[i] for i in entry_ids}


def assert_v11_prediction_identity(actual, original):
    for key in ('entry_id', 'dataset_idx', 'tx', 'ty', 'tz', 'rotation'):
        if actual.get(key) != original.get(key):
            raise ValueError(f'Frozen V11 output differs from original at entry {original.get("entry_id")}, field {key}')
    if not all(math.isfinite(actual[k]) for k in ('tx', 'ty', 'tz', 'rotation')):
        raise ValueError('Nonfinite frozen V11 prediction')


def snapshot_baseline(result):
    fields = ('center', 'yaw', 'path', 'initial_center', 'initial_yaw', 'initial_candidate', 'final_candidate')
    saved = {key: copy.deepcopy(result[key]) for key in fields}
    if len(saved['center']) != 3 or not all(math.isfinite(v) for v in saved['center']):
        raise ValueError('Baseline requires a finite 3D center')
    if not math.isfinite(saved['yaw']) or not isinstance(saved['path'], list) or len(saved['path']) > 2:
        raise ValueError('Baseline must be the finite original two-step rollout')
    return saved


def assert_baseline_preserved(before, after):
    for key in ('center', 'initial_center', 'initial_yaw', 'initial_candidate', 'final_candidate'):
        if after[key] != before[key]:
            raise ValueError('V13 refinement changed frozen baseline field: ' + key)
    prefix, refinement = before['path'], after['refinement_path']
    if (after['baseline_path'] != prefix or after['path'][:len(prefix)] != prefix
            or after['path'][len(prefix):] != refinement):
        raise ValueError('V13 refinement changed the original V11 rollout trace')
    if not isinstance(refinement, list) or len(refinement) > 2:
        raise ValueError('At most two in-place refinement steps are allowed')
    previous_yaw = before['yaw']
    for i, step in enumerate(refinement):
        if step.get('center') != before['center'] or step.get('previous_center') != before['center']:
            raise ValueError('A refinement step changed the fixed placement center')
        if step.get('mode') not in ('rotate', 'stop'):
            raise ValueError('Refinement may only rotate or stop')
        if not all(math.isfinite(step[k]) for k in ('yaw', 'previous_yaw')):
            raise ValueError('Nonfinite refinement yaw')
        circular = lambda value: math.atan2(math.sin(value), math.cos(value))
        if abs(circular(step['previous_yaw'] - previous_yaw)) > 1e-7:
            raise ValueError('Refinement yaw trace is discontinuous')
        delta = circular(step['yaw'] - step['previous_yaw'])
        allowed = (0.,) if step['mode'] == 'stop' else (-math.pi/4, math.pi/4)
        if min(abs(delta - target) for target in allowed) > 1e-7:
            raise ValueError('Refinement yaw must be stop=0 or rotate=+/-45 degrees')
        if step['mode'] == 'stop' and i != len(refinement)-1:
            raise ValueError('Refinement cannot continue after stop')
        previous_yaw = step['yaw']
    if not math.isfinite(after['yaw']) or abs(math.atan2(math.sin(after['yaw']-previous_yaw), math.cos(after['yaw']-previous_yaw))) > 1e-7:
        raise ValueError('Final yaw does not match the refinement trace')


def main(a):
    entry_ids = selected_entry_ids(a.limit)
    expected_ids = set(entry_ids)
    m = load_runtime(a.data_root)
    np, torch, d = m.np, m.torch, m.d
    torch.set_num_threads(2)
    os.environ['REFINEPLACE_WRAPPED_FIELDS'] = str(a.data_root / 'wrapped_fields/manifest.json')
    baseline = check_baseline(a.baseline_predictions, entry_ids)
    v11_predictions = check_v11_predictions(a.v11_baseline_predictions, entry_ids)
    manifest_path = a.data_root / 'tasks_manifest.json'
    manifest = read(manifest_path)
    all_tasks = manifest['tasks']
    assert manifest['split'] == 'test' and len(all_tasks) == 3500 and manifest['scene_count'] == 134
    by_id = {int(t['original_entry_id']): t for t in all_tasks}
    assert len(by_id) == 3500 and set(by_id) == set(range(3500))
    tasks = [by_id[i] for i in entry_ids]
    scenes = {t['scene'] for t in tasks}
    lock = read(d.LOCK)
    assert manifest['split_lock_sha256'] == sha(d.LOCK)
    assert {t['scene'] for t in all_tasks} == set(lock['test_scenes'])
    assert not {s.split('_')[0] for s in lock['test_scenes']} & {s.split('_')[0] for s in lock['train_scenes']}
    cp = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    architecture = cp['architecture']
    assert architecture == 'RefinePlaceV13', 'Only a V1.3 checkpoint is accepted'
    assert isinstance(cp.get('joint_policy'), dict), 'Checkpoint lacks V1.3 joint_policy state'
    assert cp['normalization_sha256'] == sha(d.F / 'normalization.json')
    assert cp['official_split_lock_sha256'] == sha(d.LOCK)
    assert cp['protocol_sha256'] == sha(a.training_protocol)
    checkpoint_epoch, parent_epoch = cp.get('epoch'), cp.get('parent_epoch')
    textmeta = read(a.data_root / 'text_features.json')
    assert textmeta['manifest_sha256'] == sha(manifest_path)
    assert textmeta['array_sha256'] == sha(a.data_root / 'text_features.npy')
    texts = np.load(a.data_root / 'text_features.npy', mmap_mode='r', allow_pickle=False)
    assert texts.shape == (3500, 768)
    refmeta = read(a.references_summary)
    assert refmeta['status'] == 'complete' and refmeta['data_sha256'] == sha(a.references)
    assert refmeta['task_manifest_sha256'] == sha(manifest_path)
    refs = {}
    with a.references.open(encoding='utf-8') as handle:
        for line in handle:
            r = json.loads(line)
            key = r['scene'], r['group_id']
            assert key not in refs
            refs[key] = r
    assert len(refs) == 3500 and {(t['scene'], t['group_id']) for t in all_tasks} == set(refs)
    for task in all_tasks:
        r = refs[task['scene'], task['group_id']]
        assert all(r[k] == task[k] for k in ('scene', 'group_id', 'asset_scene', 'entry', 'prompt_sha256'))
    grounded = read(a.grounded_root / 'index.json')
    assert grounded['status'] == 'complete' and {r['scene'] for r in grounded['records']} == set(lock['test_scenes'])
    assert grounded['split_lock_sha256'] == sha(d.LOCK)
    grounded_audit = read(a.grounded_root.parent / 'admission_audit.json')
    assert grounded_audit['status'] == 'passed' and grounded_audit['data_index_sha256'] == sha(a.grounded_root / 'index.json')
    library = m.load_mesh_manifest(a.data_root / 'geometry/mesh_manifest_v6.json')
    geometry_audit = read(a.data_root / 'geometry/geometry_admission_audit.json')
    assert geometry_audit['status'] == 'passed' and geometry_audit['manifest_sha256'] == library.manifest_sha256
    asset_ids = {t['asset_uuid'] for t in tasks}
    library.validate_required(scenes, asset_ids)
    wrapped = read(a.data_root / 'wrapped_fields/manifest.json')
    assert wrapped['no_annotation_reads'] and wrapped['original_asset_and_anchor_unchanged']
    assert sha(wrapped['wrapped_manifest_path']) == wrapped['wrapped_manifest_sha256']
    embedding_hashes = {}
    for uid in sorted(asset_ids):
        field = wrapped['assets'][uid]
        assert field['asset_uuid'] == uid and field['coordinate_frame'] == 'canonical_aligned_asset_bottom_centered'
        for kind in ('source', 'wrapped', 'field'):
            assert sha(field[kind + '_path']) == field[kind + '_sha256'], (uid, kind)
        ep = d.b.B / 'placeit3d_reproduction_v1/placeit3d/data/pointbert_embeddings' / (uid + '.pt')
        embedding_hashes[uid] = sha(ep)
    norm = read(d.F / 'normalization.json')
    mean, std = np.asarray(norm['mean']), np.asarray(norm['std'])
    types = read('/path/to/research300/relation_grounding_train_holdout_20261101_v2/run/protocol.json')['relation_types']
    # Import the isolated implementation after the original runtime has installed
    # its dependency paths. Assert that Python did not reuse another version.
    sys.path.insert(0, str(HERE))
    import model_v13
    import refiner_v13
    assert Path(model_v13.__file__).resolve() == (HERE/'model_v13.py').resolve()
    assert Path(refiner_v13.__file__).resolve() == (HERE/'refiner_v13.py').resolve()
    assert Path(m.rollout.__file__).name == 'joint_rollout_v1.py', 'Preserve the original three-mode V11 rollout'
    import model_v11
    dependency_files = [HERE/'model_v13.py', HERE/'refiner_v13.py',
                        Path(model_v11.__file__), m.V1/'model_v10.py',
                        Path(m.rollout.__file__), Path(m.rollout.__file__).parent/'orientation_policy_v1.py']
    dependency_hashes = {str(p.resolve()): sha(p) for p in dependency_files}
    protocol = dict(version='orientation_v13_test300_frozen_forward' if a.limit == MAX_TASKS else 'orientation_v13_smoke_forward',
        split='test', tasks=a.limit, diagnostic_subset_size=MAX_TASKS,
        run_kind='complete_test300' if a.limit == MAX_TASKS else 'smoke_subset',
        original_entry_ids=list(entry_ids), scenes=len(scenes), architecture=architecture,
        checkpoint_epoch=checkpoint_epoch, checkpoint_parent_epoch=parent_epoch,
        checkpoint_sha256=sha(a.checkpoint), training_protocol_sha256=sha(a.training_protocol),
        manifest_sha256=sha(manifest_path), h5_sha256=manifest['h5_sha256'],
        h5_entry_paths_sha256=manifest['h5_entry_paths_sha256'], normalization_sha256=sha(d.F/'normalization.json'),
        reference_catalog_sha256=sha(a.references), reference_summary_sha256=sha(a.references_summary),
        receiver_index_sha256=sha(a.grounded_root/'index.json'), geometry_manifest_sha256=library.manifest_sha256,
        wrapped_fields_sha256=sha(a.data_root/'wrapped_fields/manifest.json'), source_sha256=sha(__file__),
        pointbert_embedding_sha256=embedding_hashes,
        original_runtime_sha256=sha(m.__file__), v10_model_sha256=sha(m.V1/'model_v10.py'),
        region_code_sha256=sha(m.V1/'support_regions.py'), rollout_code_sha256=sha(m.rollout.__file__), refiner_code_sha256=sha(refiner_v13.__file__),
        v13_model_sha256=sha(model_v13.__file__), dependency_sha256=dependency_hashes,
        original_rollout_code_sha256=sha(m.rollout.__file__),
        v11_model_sha256=sha(model_v11.__file__),
        baseline_predictions_sha256=BASELINE_SHA256 if baseline is not None else None,
        deterministic=True, steps=2, baseline_rollout_steps=2, refinement_max_steps=2,
        refinement_action_degrees=[-45, 0, 45], baseline_architecture='RefinePlaceV11',
        v11_prediction_sha256=V11_PREDICTION_SHA256, v11_checkpoint_sha256=V11_CHECKPOINT_SHA256,
        frozen_v11_baseline_required=True, centers_must_remain_unchanged=True,
        test_labels_read=False, ground_truth_relationships_read=False,
        instance_annotation_inputs=False, official_evaluator_modified=False,
        numeric_yaw_export='degrees(yaw) modulo 360, unchanged; no added front-axis offset',
        scope=f'First ordered {a.limit} of fixed 300 official TEST diagnostic tasks; exposed comparison, not untouched validation')
    if a.preflight_only:
        print(json.dumps(dict(status='input_preflight_passed', **protocol), ensure_ascii=False), flush=True)
        return
    torch.cuda.set_device(0)
    model = model_v13.model_from_checkpoint(d, types, len(m.FEATURE_NAMES), cp).cuda().eval().requires_grad_(False)
    if not hasattr(model, 'baseline'):
        raise TypeError('V1.3 must expose its original frozen V1.1 baseline')
    if any(p.requires_grad for p in model.baseline.parameters()):
        raise ValueError('V1.1 baseline parameters must remain frozen during inference')
    del cp
    before = d.tiny.digest(model.state_dict())
    m.install_geometry()
    a.output.mkdir(parents=True, exist_ok=a.resume)
    sealed = a.output/'protocol_before_forward.json'
    if sealed.exists():
        assert read(sealed) == protocol, 'Resume changed inputs/checkpoint/source'
    else:
        assert not a.resume, 'Resume requires a sealed protocol'
        write(sealed, protocol)
    if baseline is not None:
        subset = [baseline[i] for i in entry_ids]
        path = a.output/f'baseline_predictions_{a.limit}.json'
        if path.exists():
            assert read(path) == subset
        else:
            write(path, subset)
    done = {}
    journal = a.output/'predictions_in_progress.jsonl'
    if journal.exists():
        assert a.resume
        for line in journal.read_text(encoding='utf-8').splitlines():
            r = json.loads(line)
            i = int(r['original_entry_id'])
            assert i in expected_ids and i not in done and r['scene'] == by_id[i]['scene']
            assert r['asset_uuid'] == by_id[i]['asset_uuid']
            assert 'initial_yaw' in r and 'path' in r
            assert r.get('center_unchanged') is True and r.get('baseline_first_two_steps_unchanged') is True
            assert r.get('frozen_v11_prediction_identical') is True
            assert_v11_prediction_identity(r['frozen_v11_prediction'], v11_predictions[i])
            done[i] = r
    start, assets = time.monotonic(), {}
    for scene in sorted(scenes):
        pending = [t for t in tasks if t['scene'] == scene and t['original_entry_id'] not in done]
        if not pending:
            continue
        path, entry = library._file('scenes', scene)
        with np.load(path, allow_pickle=False) as z:
            xyz, rgb, world, faces = (z[k].copy() for k in ('xyz', 'rgb', 'vertices', 'faces'))
        assert xyz.shape == rgb.shape == world.shape
        translation = np.asarray(entry['observation_mean_translation_m'], float)
        assert np.linalg.norm(world-xyz-translation, axis=1).max() < 1e-5
        A = np.asarray(library.manifest['alignments'][scene], float)
        aligned = (xyz.astype(np.float64) @ A[:3, :3].T).astype(np.float32)
        offset = A[:3, :3] @ translation + A[:3, 3]
        assert np.max(np.linalg.norm((world @ A[:3, :3].T+A[:3, 3])-(aligned+offset), axis=1)) < 2e-5
        g = m.load_grounded_surfaces(a.grounded_root, scene)
        assert g['metadata']['mesh_sha256'] == entry['sha256']
        assert g['metadata']['coordinate_frame'] == 'original_scene_vertex_ids' and g['metadata']['mesh_vertices'] == len(xyz)
        receiver = m.prepare_scene(aligned, scene, g, pool_size=8192)
        original = m.candidate_ids(xyz, scene)
        geometry_cache, asset_priors = {}, {}
        adapter = d.paired.MeshSceneAdapterV6(aligned, faces, library)
        for task in pending:
            i, uid = task['original_entry_id'], task['asset_uuid']
            dims = np.asarray(task['asset_size'], np.float32)
            if uid not in assets:
                assets[uid] = torch.load(d.b.B/'placeit3d_reproduction_v1/placeit3d/data/pointbert_embeddings'/(uid+'.pt'), map_location='cpu', weights_only=True).reshape(-1).cuda()
            if uid not in asset_priors:
                vertices, afaces = library.asset(uid, dims)
                asset_priors[uid] = m.prepare_asset(receiver, dims, vertices, afaces)
            raw = m.raw_features(rgb, original, dims, task['prompt'])
            features = m.build_task_candidates(receiver, asset_priors[uid], original, task, (xyz, rgb, None, None, None), raw, A, mean, std, geometry_cache)
            row = dict(task, split='test', dims=dims, text=texts[i].copy(), **features)
            f, text, asset, ctx, decision, inferred = d.make_context([row], assets, refs, 2, types)
            model.reference_head.inferred_mask = inferred
            rf = torch.as_tensor(row['region_features'][None], device='cuda', dtype=torch.float32)
            active = bool(ctx['reference_mask'].any() and (ctx['relation_mask'] | inferred).any())
            queries = 0
            class Service:
                def query_many(self, items):
                    nonlocal queries
                    result = []
                    for q in items:
                        assert q['scene'] == scene and q['split'] == 'test' and q['asset_uuid'] == uid
                        adapter.set_asset(uid, dims)
                        result.append(adapter.features(q['center'], q['dims'], q['yaw']))
                        queries += 1
                    return result
            obs = [{k: row[k] for k in ('scene', 'split', 'asset_uuid', 'dims', 'centers')}]
            service = Service()
            with torch.inference_mode(), model.reference_batch(**ctx), model.region_batch(rf):
                tr = m.rollout.rollout_batch(model.baseline, f, text, asset, ctx['candidate_mask'], obs, service, steps=2,
                                            deterministic=True, apply_offsets=True, anti_cycle=True, mode_epsilon=0.)[0]
                baseline_view = snapshot_baseline(diagnostic_json(d.clean({key: tr[key] for key in
                    ('center', 'yaw', 'path', 'initial_center', 'initial_yaw', 'initial_candidate', 'final_candidate')})))
                tr = refiner_v13.refine_after_baseline_v13(model, tr, obs[0], service,
                                                         max_steps=2, deterministic=True)
                assert_baseline_preserved(baseline_view, diagnostic_json(d.clean({key: tr[key] for key in
                    ('center', 'yaw', 'path', 'baseline_path', 'refinement_path', 'initial_center', 'initial_yaw', 'initial_candidate', 'final_candidate')})))
            baseline_position = np.asarray(baseline_view['center']) + offset + np.array([0., 0., float(dims[2])/2])
            frozen_prediction = dict(entry_id=i, dataset_idx=-1, tx=float(baseline_position[0]),
                ty=float(baseline_position[1]), tz=float(baseline_position[2]),
                rotation=math.degrees(baseline_view['yaw']) % 360)
            assert_v11_prediction_identity(frozen_prediction, v11_predictions[i])
            position = np.asarray(tr['center']) + offset + np.array([0., 0., float(dims[2])/2])
            prediction = dict(entry_id=i, dataset_idx=-1, tx=float(position[0]), ty=float(position[1]), tz=float(position[2]), rotation=math.degrees(float(tr['yaw'])) % 360)
            assert all(math.isfinite(prediction[k]) for k in ('tx','ty','tz','rotation')), 'Nonfinite predicted pose'
            record = dict(original_entry_id=i, scene=scene, asset_scene=task['asset_scene'], entry=task['entry'],
                asset_uuid=uid, prompt_sha256=task['prompt_sha256'], center=d.clean(tr['center']), yaw=float(tr['yaw']),
                initial_center=d.clean(tr['initial_center']), initial_yaw=float(tr['initial_yaw']),
                initial_yaw_degrees=math.degrees(float(tr['initial_yaw'])) % 360,
                final_yaw_degrees=prediction['rotation'], path=d.clean(tr['path']),
                reference_active=active, region_report=row['region_report'], geometry_queries=queries,
                initial_candidate=int(tr['initial_candidate']), final_candidate=int(tr['final_candidate']),
                stopped=bool(tr['stopped']), switches=int(tr['switches']), prediction=prediction,
                baseline_center=baseline_view['center'], baseline_yaw=baseline_view['yaw'],
                baseline_path=baseline_view['path'], frozen_v11_prediction=frozen_prediction,
                refinement_path=d.clean(tr['refinement_path']), refinement_steps=len(tr['refinement_path']),
                center_unchanged=True, baseline_first_two_steps_unchanged=True, frozen_v11_prediction_identical=True)
            record = d.clean(record)
            replaced_paths = nonfinite_paths(record)
            record = diagnostic_json(record)
            record['nonfinite_diagnostic_paths_replaced_with_null'] = replaced_paths
            with journal.open('a', encoding='utf-8') as handle:
                handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
                handle.flush()
            done[i] = record
            write(a.output/'progress.json', dict(status='inference', completed=len(done), total=a.limit, scene=scene,
                reference_active=sum(r['reference_active'] for r in done.values()), elapsed_this_process=time.monotonic()-start))
            if len(done) % 10 == 0:
                print(json.dumps(dict(completed=len(done), total=a.limit, architecture=architecture)), flush=True)
        del adapter, receiver, asset_priors, geometry_cache, world, faces, xyz, rgb, aligned
        gc.collect()
    assert set(done) == expected_ids and before == d.tiny.digest(model.state_dict())
    output = a.output/prediction_filename(a.limit)
    write(output, [done[i]['prediction'] for i in entry_ids])
    frozen_output = a.output/f'frozen_v11_predictions_{a.limit}.json'
    write(frozen_output, [done[i]['frozen_v11_prediction'] for i in entry_ids])
    if a.limit == MAX_TASKS and sha(frozen_output) != V11_PREDICTION_SHA256:
        raise ValueError('Full frozen V11 prediction bytes differ from the pinned historical export')
    active = sum(r['reference_active'] for r in done.values())
    summary = dict(protocol, status='complete' if a.limit == MAX_TASKS else 'smoke_complete',
        inference_complete=True, official_test300_complete=(a.limit == MAX_TASKS),
        prediction_filename=output.name, weights_unchanged=True,
        frozen_v11_prediction_identical=all(r['frozen_v11_prediction_identical'] for r in done.values()),
        center_unchanged=all(r['center_unchanged'] for r in done.values()),
        baseline_first_two_steps_unchanged=all(r['baseline_first_two_steps_unchanged'] for r in done.values()),
        frozen_v11_prediction_filename=frozen_output.name, frozen_v11_prediction_sha256=sha(frozen_output),
        refinement_action_modes=dict(Counter(p['mode'] for r in done.values() for p in r['refinement_path'])),
        refinement_changed_yaw_count=sum(abs(math.atan2(math.sin(r['yaw']-r['baseline_yaw']), math.cos(r['yaw']-r['baseline_yaw']))) > 1e-7 for r in done.values()),
        nonfinite_diagnostic_values_replaced=sum(len(r.get('nonfinite_diagnostic_paths_replaced_with_null', [])) for r in done.values()),
        prediction_sha256=sha(output), reference_active=active, reference_masked=a.limit-active,
        initial_yaw_histogram=dict(Counter(str(round(r['initial_yaw_degrees'], 6)) for r in done.values())),
        final_yaw_histogram=dict(Counter(str(round(r['final_yaw_degrees'], 6)) for r in done.values())),
        action_modes=dict(Counter(p['mode'] for r in done.values() for p in r['path'])),
        elapsed_this_process=time.monotonic()-start, protocol_sha256=sha(sealed))
    if baseline is not None:
        summary['poses_changed_from_original_epoch2'] = sum(any(abs(done[i]['prediction'][k]-baseline[i][k]) > 1e-7 for k in ('tx', 'ty', 'tz', 'rotation')) for i in entry_ids)
        summary['baseline_comparison_is_prediction_only'] = True
    write(a.output/'summary.json', summary)
    write(a.output/'progress.json', dict(status=summary['status'], completed=len(done), total=a.limit,
        official_test300_complete=summary['official_test300_complete'], prediction_filename=output.name,
        reference_active=active, elapsed_this_process=time.monotonic()-start))
    print(json.dumps({k: summary[k] for k in ('status', 'tasks', 'architecture', 'reference_active', 'action_modes', 'prediction_sha256')}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint', 'training-protocol', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--references', type=Path)
    parser.add_argument('--references-summary', type=Path)
    parser.add_argument('--grounded-root', type=Path)
    parser.add_argument('--baseline-predictions', type=Path)
    parser.add_argument('--v11-baseline-predictions', type=Path, default=DEFAULT_V11_PREDICTIONS,
                        help='Read-only pinned V11 upgraded 300-entry prediction export for exact baseline identity checks')
    parser.add_argument('--limit', type=int, default=MAX_TASKS, help='First N of the fixed 0..299 identities; N<300 is a smoke run, not a completed 300-case inference')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--preflight-only', action='store_true', help='Validate complete input identities on CPU, without inference or output writes')
    args = parser.parse_args()
    try:
        selected_entry_ids(args.limit)
    except ValueError as exc:
        parser.error(str(exc))
    args.references = args.references or args.data_root/'references/reference_catalog_compact_v4_test.jsonl'
    args.references_summary = args.references_summary or args.data_root/'references/reference_catalog_compact_v4_test_manifest.json'
    args.grounded_root = args.grounded_root or args.data_root/'receivers/test_grounded_support/grounded_surfaces'
    main(args)
