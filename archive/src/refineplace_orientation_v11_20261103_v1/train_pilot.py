"""Matched train-only continuation pilot; original V1.0 and test exports stay frozen."""
import argparse
from collections import Counter
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V10 = Path('/path/to/research300/refineplace_v1_0_20261103_v1')
OLD = Path('/path/to/research300/refineplace_rl_geometry_10k_20261102_v1/joint_collision_orientation')
MODULES = Path('/path/to/research300/refineplace_v01_official563_modules_20261102_v1')
sys.path[:0] = [str(HERE), str(V10), str(OLD), str(MODULES)]
import train_official563_joint_v7_sync_full as d
from train_v10 import load_references, atomic_checkpoint, CAT
from model_v10 import model_from_checkpoint as baseline_model
from model_v11 import model_from_checkpoint as upgraded_model
from orientation_objectives import supervised_full8, dense_paired_pose_quality
from orientation_policy_v1 import collision_auxiliary_loss
from constrained_objective import constrained_rl_losses, update_multiplier
from paired_orientation_v1 import paired_pose_quality, make_positive_pose_pairs
import joint_rollout_v1 as rollout
from wrapped_geometry_backend_v10 import CollisionGeometryBackend
from support_regions import FEATURE_NAMES
from v1_scene import V10Scene


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + '.pending')
    tmp.write_text(json.dumps(d.clean(value), ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(tmp, path)


def main(a):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    lock = d.read(d.LOCK)
    manifest = d.read(a.manifest)
    assert manifest['split_lock_sha256'] == d.sha(d.LOCK)
    assert manifest['catalog_sha256'] == d.sha(CAT)
    tasks = manifest['tasks'][:a.max_tasks or None]
    assert tasks and len({(t['scene'], int(t['group_id'])) for t in tasks}) == len(tasks)
    for t in tasks:
        d.guard(lock, t['scene'])
    refs = load_references()
    types = d.read('/path/to/research300/relation_grounding_train_holdout_20261101_v2/run/protocol.json')['relation_types']
    cp = torch.load(a.parent, map_location='cpu', weights_only=False)
    assert cp['architecture'] == 'RefinePlaceV10'
    assert cp['normalization_sha256'] == d.sha(d.F / 'normalization.json')
    assert cp['official_split_lock_sha256'] == d.sha(d.LOCK)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    constructor = upgraded_model if a.arm == 'upgraded' else baseline_model
    model = constructor(d, types, len(FEATURE_NAMES), cp).cuda().train()
    del cp
    model.requires_grad_(True)
    for name, p in model.named_parameters():
        if '.base.values.2.' in name:
            p.requires_grad_(False)
    # Reset the exploration stream after architecture-dependent initialization.
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    random.seed(a.seed)
    parameters = [p for p in model.parameters() if p.requires_grad]
    new = [p for n, p in model.named_parameters() if p.requires_grad and n.startswith(('bound_pose.', 'bound_local.'))]
    new_ids = {id(p) for p in new}
    existing = [p for p in parameters if id(p) not in new_ids]
    groups = [dict(params=existing, lr=a.learning_rate)]
    if new:
        groups.append(dict(params=new, lr=a.new_learning_rate))
    opt = torch.optim.AdamW(groups, weight_decay=0.)
    a.output.mkdir(parents=True, exist_ok=False)
    protocol = dict(version='RefinePlace_orientation_v1.1_pilot', arm=a.arm,
        tasks=len(tasks), epochs=1, seed=a.seed, global_batch_size=a.batch_size,
        geometry_workers=a.workers, steps=2, rl_weight=.5, collision_auxiliary_weight=.2,
        angular_reward_weight=.25 if a.arm=='upgraded' else 0., angular_reward_max_distance_m=.5 if a.arm=='upgraded' else None,
        learning_rate=a.learning_rate, new_learning_rate=a.new_learning_rate,
        parent_checkpoint=str(a.parent), parent_sha256=d.sha(a.parent), parent_epoch=2,
        manifest_sha256=d.sha(a.manifest), official_split_lock_sha256=d.sha(d.LOCK),
        normalization_sha256=d.sha(d.F / 'normalization.json'),
        initial_parameter_digest=d.tiny.digest(model.state_dict()),
        reference_catalog_sha256=manifest['catalog_sha256'],
        grounded_root=str(a.grounded_root), grounded_index_sha256=d.sha(a.grounded_root / 'index.json'),
        validation_manifest_sha256=manifest['validation_manifest_sha256'],
        test_labels_read=False, test_predictions_modified=False,
        supervision='official full-eight angle mask and paired reward' if a.arm=='upgraded' else 'unchanged v1.0 objectives',
        semantic_front='not externally verified; learned asset-conditioned numeric heading',
        source_hashes={p.name:d.sha(p) for p in HERE.glob('*.py')})
    write(a.output / 'protocol_before_training.json', protocol)
    library = d.paired.load_mesh_manifest('/path/to/research300/placeit_mesh_v6/mesh_manifest_v6.json')
    assets = {}
    multiplier = 1.
    seen = []
    logs = []
    changes = Counter()
    start = time.monotonic()
    scenes = list(dict.fromkeys(t['scene'] for t in tasks))

    def save(filename):
        state = dict(joint_policy=model.state_dict(), optimizer=opt.state_dict(),
            architecture='RefinePlaceV11' if a.arm=='upgraded' else 'RefinePlaceV10',
            region_dim=len(FEATURE_NAMES), epoch=1, parent_epoch=2, updates=len(logs),
            tasks_completed=len(seen), seen_task_ids=seen, collision_multiplier=multiplier,
            normalization_sha256=protocol['normalization_sha256'],
            official_split_lock_sha256=protocol['official_split_lock_sha256'],
            protocol_sha256=d.sha(a.output / 'protocol_before_training.json'),
            rng=dict(cpu=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all(),
                     numpy=np.random.get_state(), python=random.getstate()))
        atomic_checkpoint(a.output / filename, state)

    with CollisionGeometryBackend(workers=a.workers) as service:
        for scene in scenes:
            source = V10Scene(scene, lock, library, driver=d, grounded_root=a.grounded_root)
            selected = [t for t in tasks if t['scene'] == scene]
            pair_cache = {}
            try:
                for offset in range(0, len(selected), a.batch_size):
                    tick = time.monotonic()
                    batch = selected[offset:offset + a.batch_size]
                    rows = source.inputs([source.bygi[int(t['group_id'])] for t in batch])
                    keys = [(scene, int(row['group_id'])) for row in rows]
                    for task, row in zip(batch, rows):
                        assert all(task[k] == row[k] for k in ('scene','group_id','asset_scene','entry','prompt_sha256'))
                        uid = row['asset_uuid']
                        if uid not in assets:
                            assets[uid] = torch.load(d.b.B / 'placeit3d_reproduction_v1/placeit3d/data/pointbert_embeddings' / (uid+'.pt'), map_location='cpu', weights_only=True).reshape(-1).cuda()
                    features, text, asset, ctx, decisions, inferred = d.make_context(rows, assets, refs, 2, types)
                    model.reference_head.inferred_mask = inferred
                    regions = torch.as_tensor(np.stack([r['region_features'] for r in rows]), device='cuda', dtype=torch.float32)
                    queries = 0
                    geometry_seconds = 0.
                    class Service:
                        def query_many(self, requests):
                            nonlocal queries, geometry_seconds
                            assert all(q['scene'] == scene and q['split'] == 'train' for q in requests)
                            now = time.monotonic()
                            result = service.query_many(requests)
                            geometry_seconds += time.monotonic() - now
                            queries += len(requests)
                            return result
                    obs = [{k:r[k] for k in ('scene','split','asset_uuid','dims','centers')} for r in rows]
                    opt.zero_grad(set_to_none=True)
                    with model.reference_batch(**ctx), model.region_batch(regions):
                        tr = rollout.rollout_batch(model, features, text, asset, ctx['candidate_mask'], obs,
                            Service(), steps=2, deterministic=False, apply_offsets=True, anti_cycle=True, mode_epsilon=.2)
                    # Labels enter here, after the annotation-free rollout.
                    labels = source.labels(rows)
                    qualities = []
                    angular = []
                    for task, key, result in zip(batch, keys, tr):
                        an = labels[key]['annotation']
                        if an not in pair_cache:
                            z = source.labels_arrays
                            lo, hi = map(int, z['annotation_offsets'][an:an+2])
                            pair_cache[an] = make_positive_pose_pairs(z['scene_xyz'], np.asarray(z['valid_ids'][lo:hi]), np.asarray(z['valid_mask8'][lo:hi]))
                        if a.arm == 'upgraded':
                            q = dense_paired_pose_quality(result['center'], result['yaw'], pair_cache[an], facing_active=bool(task['facing_requested']))
                        else:
                            q = paired_pose_quality(result['center'], result['yaw'], pair_cache[an])
                        qualities.append(float(q['quality']))
                        angular.append(q)
                    scores = torch.stack([r['scores'] for r in tr])
                    offsets = torch.stack([r['supervised_offsets'] for r in tr])
                    if a.arm == 'upgraded':
                        sup, diagnostics = supervised_full8(scores, offsets, labels, keys, candidate_mask=ctx['candidate_mask'])
                    else:
                        sup, _ = d.supervised_safe(scores, offsets, labels, keys)
                        diagnostics = {}
                    rl = constrained_rl_losses(tr, qualities, collision_multiplier=multiplier, rl_weight=.5)
                    aux = collision_auxiliary_loss(tr)
                    loss = (sup + rl['rl_loss'] + .2 * aux).mean()
                    assert torch.isfinite(loss)
                    loss.backward()
                    for n, p in model.named_parameters():
                        if p.grad is not None and bool(p.grad.abs().sum() > 0):
                            changes[n.split('.')[0]] += 1
                    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True))
                    opt.step()
                    used_multiplier = multiplier
                    multiplier = update_multiplier(multiplier, rl['collision_costs'], rl['collision_known'])
                    seen.extend([[s,g] for s,g in keys])
                    h = dict(update=len(logs)+1, scene=scene, tasks=len(batch), completed=len(seen),
                        task_ids=keys, loss=float(loss.detach()), supervised_loss=float(sup.mean().detach()),
                        rl_loss=float(rl['rl_loss'].mean().detach()), collision_aux_loss=float(aux.mean().detach()),
                        task_quality=float(np.mean(qualities)), yaw_diagnostics=d.clean(diagnostics),
                        angular_reward_diagnostics=d.clean(angular),
                        collision_multiplier_used=used_multiplier, collision_multiplier_next=multiplier,
                        active_reference=int((ctx['reference_mask'].any(-1)&(ctx['relation_mask']|inferred).any(-1)).sum()),
                        facing_tasks=sum(bool(t['facing_requested']) for t in batch),
                        geometry_queries=queries, geometry_seconds=geometry_seconds, batch_seconds=time.monotonic()-tick,
                        gradient_norm=norm,
                        action_modes=dict(Counter(p['mode'] for r in tr for p in r['path'])),
                        initial_yaws=[float(r['initial_yaw']) for r in tr], final_yaws=[float(r['yaw']) for r in tr])
                    logs.append(h)
                    with (a.output / 'steps.jsonl').open('a', encoding='utf-8') as fp:
                        fp.write(json.dumps(d.clean(h), allow_nan=False) + '\n')
                    write(a.output / 'progress.json', dict(status='training', arm=a.arm, completed=len(seen), total=len(tasks),
                        updates=len(logs), elapsed_seconds=time.monotonic()-start,
                        last={k:v for k,v in h.items() if k not in ('yaw_diagnostics','angular_reward_diagnostics','task_ids')}))
                    print(json.dumps(dict(arm=a.arm, completed=len(seen), total=len(tasks), loss=h['loss'], seconds=h['batch_seconds'])), flush=True)
                    if len(logs) % a.checkpoint_every == 0:
                        save('last.pt')
                    if (a.output / 'stop_after_current_batch').exists():
                        save('last.pt')
                        write(a.output/'status.json', dict(status='stopped', completed=len(seen), total=len(tasks)))
                        return
            finally:
                source.close()
                del source
                gc.collect()
    assert len(seen) == len(tasks) and len({tuple(k) for k in seen}) == len(tasks)
    save('final.pt')
    summary = dict(status='training_complete', arm=a.arm, tasks=len(seen), updates=len(logs),
        elapsed_seconds=time.monotonic()-start, nonzero_gradient_groups=dict(changes),
        final_checkpoint_sha256=d.sha(a.output/'final.pt'), parent_checkpoint_unchanged=d.sha(a.parent)==protocol['parent_sha256'],
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), independent_test_used=False)
    write(a.output/'summary.json', summary)
    write(a.output/'status.json', summary)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('output','manifest','parent','grounded-root'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--arm', choices=('baseline','upgraded'), required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=419)
    p.add_argument('--learning-rate', type=float, default=1e-5)
    p.add_argument('--new-learning-rate', type=float, default=1e-4)
    p.add_argument('--checkpoint-every', type=int, default=100)
    p.add_argument('--max-tasks', type=int, default=0, help='Engineering smoke only; 0 uses the full sealed pilot')
    main(p.parse_args())
