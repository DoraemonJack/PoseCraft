"""Single combined treatment vs the preserved high_geometry 10k control."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
M = Path('/path/to/research300/refineplace_v01_official563_modules_20261102_v1')
sys.path.insert(0, str(M))
import train_official563_joint_v7_sync_full as d
from cpu_pool_backend_v1 import observation_scene_class
sys.path.insert(0, str(HERE))
from geometry_backend import CollisionGeometryBackend
from orientation_policy_v1 import DirectionalReferencePosePolicy, collision_auxiliary_loss, DIRECTIONAL_FEATURE_NAMES
from paired_orientation_v1 import make_positive_pose_pairs, paired_pose_quality, AXIS_CONVENTION
from constrained_objective import constrained_rl_losses, update_multiplier
import joint_rollout_v1 as rollout

ROOT = Path('/path/to/research300/refineplace_rl_geometry_10k_20261102_v1')
CONTROL = ROOT / 'high_geometry'
CAT = Path('/path/to/research300/refineplace_reference_multiframe_assoc_20261102_v1/reference_catalog_compact_v4_inferred_pilot.jsonl')


def build_original(types, state=None):
    prior = nn.Sequential(nn.Linear(141, 192), nn.SiLU(), nn.Linear(192, 128), nn.SiLU(), nn.Linear(128, 1))
    base = d.paired.BatchedJointModel(47, prior.state_dict(), 768, .25)
    inner = d.paired.ReferenceJointPolicy(base, d.InferredReferenceHead(len(types), inferred_weight=.3))
    model = d.joint.ReferencePosePolicy(inner, len(types), base.actor_body[0].in_features,
                                       preserve_candidate_logmass=False)
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model


def source_pairs(source, annotation):
    z = source.labels_arrays
    lo, hi = map(int, z['annotation_offsets'][annotation:annotation + 2])
    return make_positive_pose_pairs(z['scene_xyz'], np.asarray(z['valid_ids'][lo:hi]),
                                    np.asarray(z['valid_mask8'][lo:hi]))


def run(a):
    torch.set_num_threads(2)
    torch.manual_seed(419); np.random.seed(419); random.seed(419)
    a.output.mkdir(parents=True, exist_ok=False)
    control_protocol = d.read(CONTROL / 'protocol_before_training.json')
    assert control_protocol['rl_weight'] == .5 and control_protocol['geometry_reward_active']
    assert control_protocol['unique_train_tasks'] == 10000 and control_protocol['passes'] == 1
    parent = Path(control_protocol['parent_checkpoint'])
    assert d.sha(parent) == control_protocol['parent_sha256']
    cp = torch.load(parent, map_location='cpu', weights_only=False)
    assert cp['normalization_sha256'] == d.sha(d.F / 'normalization.json')
    assert cp['official_split_lock_sha256'] == d.sha(d.LOCK)
    lock = d.read(d.LOCK)
    manifest_path = CONTROL / 'locked_training_tasks.json'
    manifest = d.read(manifest_path)
    assert sum(len(x['positions']) for x in manifest) == 10000
    for item in manifest:
        d.guard(lock, item['scene'])
        groups = np.load(d.F / item['scene'] / 'group_ids.npy', mmap_mode='r')
        assert np.array_equal(groups[item['positions']], item['group_ids'])
    shutil.copyfile(manifest_path, a.output / 'locked_training_tasks.json')
    total_updates = sum((len(x['positions']) + 31) // 32 for x in manifest)
    assert d.sha(CAT) == control_protocol['reference_catalog_sha256']
    scenes = {x['scene'] for x in manifest}
    refs = {}
    with CAT.open(encoding='utf-8') as fp:
        for line in fp:
            row = json.loads(line)
            if row['scene'] in scenes:
                refs[row['scene'], row['group_id']] = row
    types = d.read('/path/to/research300/relation_grounding_train_holdout_20261101_v2/run/protocol.json')['relation_types']
    original = build_original(types, cp['joint_policy'])
    assert d.tiny.digest(original.state_dict()) == control_protocol['initial_parameter_sha256']
    del cp
    model = DirectionalReferencePosePolicy(original).cuda()
    model.requires_grad_(True)
    for name, parameter in model.named_parameters():
        if '.base.values.2.' in name:
            parameter.requires_grad_(False)
    initial = d.tiny.digest(model.state_dict())
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5, weight_decay=0.)
    library = d.paired.load_mesh_manifest('/path/to/research300/placeit_mesh_v6/mesh_manifest_v6.json')
    assert library.geometry_config == control_protocol['geometry_config']
    files = {p.name: d.sha(p) for p in HERE.glob('*.py')}
    protocol = dict(version='RefinePlace_v01_joint_collision_orientation', arm='joint',
        source_sha256=d.sha(d.__file__), treatment_sources=files,
        control_checkpoint=str(CONTROL / 'final.pt'), control_checkpoint_sha256=d.sha(CONTROL / 'final.pt'),
        control_protocol_sha256=d.sha(CONTROL / 'protocol_before_training.json'),
        parent_checkpoint=str(parent), parent_sha256=d.sha(parent),
        original_initial_parameter_sha256=control_protocol['initial_parameter_sha256'],
        initial_parameter_sha256=initial, normalization_sha256=d.sha(d.F / 'normalization.json'),
        official_split_lock_sha256=d.sha(d.LOCK), training_manifest_sha256=d.sha(manifest_path),
        reference_catalog_sha256=d.sha(CAT), train_scenes=[x['scene'] for x in manifest],
        unique_train_tasks=10000, passes=1, batch_size=32, total_updates=total_updates,
        seed=419, learning_rate=1e-5, optimizer='fresh AdamW', rl_weight=.5, steps=2,
        original_support_rule_unchanged=True, original_support_bonus_weight=.1,
        geometry_config=library.geometry_config, geometry_role='observed collision evidence, no exhaustive search',
        orientation_reward='exp(-min_joint_pair[distance_squared/.1^2 + 1-cos(delta_yaw)])',
        axis_convention=AXIS_CONVENTION, universal_semantic_front_assumed=False,
        yaw_reference_trainable=True, same_position_yaw_switch_enabled=True,
        directional_features=list(DIRECTIONAL_FEATURE_NAMES), collision_auxiliary_weight=.2,
        collision_multiplier_initial=1., collision_budget=.05, multiplier_step=.05,
        collision_objective='separate task/cost critics, Lagrangian actor advantage; expected evidence cost only',
        unknown_geometry='continuous target masked where unknown; exact crossing/nonintersection proxy fallback if available; missing/incomplete evidence masked; no physical clearance claim',
        validation_manifest='/path/to/research300/refineplace_sync20_inferred_official150_20261102_v1/fixed_validation150.json',
        comparison='preserved high_geometry control vs all changes together; no component ablations',
        phase='engineering_smoke' if a.smoke_batches else '10k_training',
        independent_test_accessed=False)
    d.write(a.output / 'protocol_before_training.json', protocol)
    started = time.monotonic(); assets = {}; history = []; multiplier = 1.; modes = Counter()
    coverage = Counter(); changed_gradients = Counter(); stop = False
    with CollisionGeometryBackend(workers=12) as service:
        model.train()
        for item in manifest:
            source = observation_scene_class(d)(item['scene'], lock, library)
            pairs = {}
            for offset in range(0, len(item['positions']), 32):
                tick = time.monotonic()
                rows = source.inputs(item['positions'][offset:offset + 32])
                keys = [(r['scene'], r['group_id']) for r in rows]
                for row in rows:
                    uid = row['asset_uuid']
                    if uid not in assets:
                        assets[uid] = torch.load(d.b.B / 'placeit3d_reproduction_v1/placeit3d/data/pointbert_embeddings' / (uid + '.pt'),
                                                map_location='cpu', weights_only=True).reshape(-1).cuda()
                f, t, av, ctx, decisions, inf = d.make_context(rows, assets, refs, 1, types)
                model.reference_head.inferred_mask = inf
                optimizer.zero_grad(set_to_none=True)
                obs = [{k: r[k] for k in ['scene', 'split', 'asset_uuid', 'dims', 'centers']} for r in rows]
                with model.reference_batch(**ctx):
                    tr = rollout.rollout_batch(model, f, t, av, ctx['candidate_mask'], obs, service,
                        steps=2, deterministic=False, apply_offsets=True, anti_cycle=True, mode_epsilon=.2)
                # Only after observation-only decisions are finished do labels enter reward/loss.
                labels = source.labels(rows); quality = []; witnesses = []
                for key, result in zip(keys, tr):
                    annotation = labels[key]['annotation']
                    if annotation not in pairs:
                        pairs[annotation] = source_pairs(source, annotation)
                    detail = paired_pose_quality(result['center'], result['yaw'], pairs[annotation])
                    quality.append(detail['quality']); witnesses.append(detail)
                sup, _ = d.supervised_safe(torch.stack([r['scores'] for r in tr]),
                                           torch.stack([r['supervised_offsets'] for r in tr]), labels, keys)
                rl = constrained_rl_losses(tr, quality, collision_multiplier=multiplier, rl_weight=.5)
                aux = collision_auxiliary_loss(tr)
                loss = (sup + rl['rl_loss'] + .2 * aux).mean()
                assert torch.isfinite(loss)
                loss.backward()
                active = bool((ctx['reference_mask'].any(-1) & ctx['relation_mask'].any(-1)).any())
                if not active:
                    for name, parameter in model.named_parameters():
                        if name.startswith(('inner.local_reference.', 'inner.yaw_reference.')) or 'reference_head.' in name:
                            parameter.grad = None
                for name, parameter in model.named_parameters():
                    if parameter.grad is not None:
                        assert torch.isfinite(parameter.grad).all(), name
                        if parameter.grad.abs().sum() > 0:
                            changed_gradients[name.split('.')[0]] += 1
                            if name.startswith('inner.yaw_reference.'):
                                changed_gradients['yaw_reference'] += 1
                            if '.base.values.1.' in name:
                                changed_gradients['collision_value_head'] += 1
                gradient = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1., error_if_nonfinite=True))
                optimizer.step()
                multiplier_before = multiplier
                multiplier = update_multiplier(multiplier, rl['collision_costs'], rl['collision_known'])
                modes.update(rl['collision_modes'])
                coverage.update(tasks=len(rows), reference_active=int((ctx['reference_mask'].any(-1) & ctx['relation_mask'].any(-1)).sum()),
                    continuous_collision=sum(bool(r['geometry'].get('collision_cost_known', False)) for r in tr),
                    partial_asset_interior=sum(bool(r['geometry'].get('collision_proxy_partial_asset', False)) for r in tr),
                    cost_evidence=sum(rl['collision_known']), pose_reward_evidence=sum(w['has_positive'] for w in witnesses))
                row = dict(update=len(history) + 1, total_updates=total_updates, scene=item['scene'], task_ids=keys,
                    loss=float(loss.detach()), supervised_loss=float(sup.mean().detach()), rl_loss=float(rl['rl_loss'].mean().detach()),
                    collision_aux_loss=float(aux.mean().detach()), paired_pose_quality_mean=float(np.mean(quality)),
                    collision_multiplier_used=multiplier_before, collision_multiplier_next=multiplier,
                    collision_cost_mean=float(np.mean(rl['collision_costs'])), collision_modes=dict(Counter(rl['collision_modes'])),
                    gradient_norm=gradient, batch_seconds=time.monotonic() - tick,
                    poses=[dict(center=r['center'], yaw=r['yaw'], paired_quality=w, collision_mode=m,
                        collision_cost=c, actions=[p['mode'] for p in r['path']])
                        for r, w, m, c in zip(tr, witnesses, rl['collision_modes'], rl['collision_costs'])])
                row = d.clean(row); history.append(row)
                with (a.output / 'steps.jsonl').open('a', encoding='utf-8') as fp:
                    fp.write(json.dumps(row, allow_nan=False) + '\n')
                status = dict(status='training', phase=protocol['phase'], updates=len(history), total_updates=total_updates,
                    elapsed_seconds=time.monotonic() - started, coverage=dict(coverage), collision_modes=dict(modes),
                    last={k: v for k, v in row.items() if k not in ('poses', 'task_ids')})
                d.write(a.output / 'progress.json', status)
                print(json.dumps(status), flush=True)
                if a.smoke_batches and len(history) >= a.smoke_batches:
                    stop = True; break
            source.close()
            if stop: break
    state = dict(joint_policy=model.state_dict(), optimizer=optimizer.state_dict(), epoch=1, updates=len(history),
        collision_multiplier=multiplier, normalization_sha256=d.sha(d.F / 'normalization.json'),
        official_split_lock_sha256=d.sha(d.LOCK), protocol_sha256=d.sha(a.output / 'protocol_before_training.json'),
        architecture='DirectionalReferencePosePolicy')
    torch.save(state, a.output / 'final.pt')
    assert initial != d.tiny.digest(model.state_dict())
    assert d.sha(parent) == protocol['parent_sha256']
    assert d.sha(CONTROL / 'final.pt') == protocol['control_checkpoint_sha256']
    d.write(a.output / 'summary.json', dict(status='smoke_complete' if a.smoke_batches else 'training_complete',
        updates=len(history), task_exposures=coverage['tasks'], wall_seconds=time.monotonic() - started,
        coverage=dict(coverage), collision_modes=dict(modes), nonzero_gradient_groups=dict(changed_gradients),
        final_multiplier=multiplier, parent_checkpoint_unchanged=True, control_checkpoint_unchanged=True,
        checkpoint_sha256=d.sha(a.output / 'final.pt'), official_evaluation_pending=not bool(a.smoke_batches)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-batches', type=int, default=0)
    run(parser.parse_args())
