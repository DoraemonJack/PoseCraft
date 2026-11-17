"""V7 batched joint candidate and full-circle yaw placement training on train/dev only.

Geometry uses actual asset triangles against the reconstructed scene mesh.
Annotations enter supervised loss, terminal training reward and dev metrics only.
V6 adds frozen official PointBERT asset features alongside training-only distance ranking targets and bounded world-XY offsets.
Offsets are optional supervised pre-refinement, without physical support verification.
"""
import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch import nn
from torch.nn import functional as F

from mesh_scene_adapter_v6 import (MeshSceneAdapterV6 as SceneGeometry, costs_with_reliability,
    load_mesh_manifest, load_scene_mesh)
from pose_targets_v4 import (decode_valid_rotations, match_candidate_pose_targets, joint_valid_set_loss,
    pose_quality_and_metrics, load_axis_alignment, raw_centered_to_aligned_centered, validate_pure_yaw_alignment)
from joint_objectives_v2 import discounted_returns
from distance_targets_v5 import build_distance_targets, distance_auxiliary_loss


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):
            hasher.update(block)
    return hasher.hexdigest()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def jsonable(value):
    if isinstance(value, dict):
        return {k: jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def load_text_cache(path, entries, selected_ids, source_sha256=None, model_signature=None):
    """Validate original group IDs and prompt identity before using embeddings."""
    with np.load(path, allow_pickle=False) as cache:
        ids = cache['group_ids'].astype(int)
        vectors = cache['embeddings'].copy()
        hashes = cache['prompt_sha256'].copy()
        metadata = json.loads(str(cache['metadata_json'].item()))
        if len(np.unique(ids)) != len(ids) or vectors.shape != (len(ids), 768):
            raise ValueError("Malformed text cache: " + str(path))
        if vectors.dtype != np.float32 or not np.isfinite(vectors).all():
            raise ValueError("Nonfinite or incorrectly typed text cache")
        if metadata.get('frozen') is not True or metadata.get('dimension') != 768:
            raise ValueError("Expected frozen 768-dimensional contextual text features")
        if source_sha256 is not None and metadata.get('source_sha256') != source_sha256:
            raise ValueError("Text cache source checksum mismatch")
        if model_signature is not None and metadata.get('model_signature') != model_signature:
            raise ValueError("Text cache encoder identity mismatch")
        mapping = {}
        for row, gi in enumerate(ids):
            if not 0 <= gi < len(entries):
                raise ValueError("Cache group ID outside original groups.json")
            entry = entries[gi]
            expected = hashlib.sha256(entry['prompt'].encode('utf-8')).hexdigest()
            if str(hashes[row]) != expected:
                raise ValueError("Prompt hash mismatch at group " + str(gi))
            for key in ('scene', 'asset_scene', 'entry'):
                if str(cache[key][row]) != str(entry[key]):
                    raise ValueError("Cache group identity mismatch: " + key)
            mapping[int(gi)] = vectors[row]
        if not set(map(int, selected_ids)).issubset(mapping):
            raise ValueError("Missing selected group IDs in " + str(path))
    return mapping



def asset_uuid(entry):
    """Match the official dataset's asset_id.split('_')[-1] join."""
    key = entry['asset_scene'].split('_')[-1]
    if not key or '/' in key or chr(92) in key:
        raise ValueError('Invalid asset UUID: ' + key)
    return key


def load_asset_features(directory, entries):
    """Load frozen pretrained vectors, preserving actual dimension and checksums.

    Official PointBERT files hold one max-pooled vector, normally [1,768].
    Missing files, malformed shapes and inconsistent dimensions fail explicitly.
    """
    directory = Path(directory)
    features, checksums = {}, {}
    dimension = None
    for key in sorted({asset_uuid(entry) for entry in entries}):
        path = directory / (key + '.pt')
        if not path.is_file():
            raise FileNotFoundError('Missing pretrained PointBERT feature: ' + str(path))
        value = torch.load(path, map_location='cpu', weights_only=True)
        if not isinstance(value, torch.Tensor):
            raise ValueError('Expected PointBERT tensor: ' + str(path))
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1 or not value.numel() or not torch.isfinite(value).all():
            raise ValueError('Expected one finite pooled asset vector: ' + str(path))
        if dimension is None:
            dimension = value.numel()
        if value.numel() != dimension:
            raise ValueError('Inconsistent PointBERT feature dimension: ' + str(path))
        features[key] = value.detach().float().contiguous()
        checksums[key] = digest(path)
    if not features:
        raise ValueError('No asset features selected')
    return features, dimension, checksums


from batched_policy_v7 import BatchedJointModel as JointModel, batched_supervised_losses
from training_batch_v7 import collate_records
from training_cache_v7 import load_scene_cache, file_identity
from mesh_workers_v7 import MeshGeometryService
from batched_rollout_v7 import rollout_batch, batched_rl_losses
from concurrent.futures import ThreadPoolExecutor

def make_features(raw, mean, std, device):
    normalized = np.clip((raw[:, :len(mean)] - mean) / std, -10, 10).astype(np.float32)
    pooled = np.concatenate((normalized, np.broadcast_to(normalized.mean(0), normalized.shape),
                             np.broadcast_to(normalized.max(0), normalized.shape)), 1)
    return torch.as_tensor(pooled, device=device)


def load_scene(base, scene, split):
    if split not in ('train', 'val'):
        raise ValueError("Only train/val scene reads are permitted")
    raw = torch.load(base / 'scans_full_refer_v1' / split / (scene + '_refer.pth'),
                     map_location='cpu', weights_only=False)
    xyz = np.asarray(raw[0], dtype=np.float32)
    return xyz, None  # geometry is constructed only after mandatory alignment


def restart_choices(scores, centers, current_pose, visited_candidates, visited_poses,
                    anti_cycle=True, radius=.05, predicted_offsets=None, visited_centers=None):
    """Choose distinct spatial poses in score order; yaw changes use local move."""
    flat = scores.reshape(-1)
    if not anti_cycle:
        top = torch.topk(flat, min(64, flat.numel())).indices
        return top[top != current_pose]
    choices, accepted, accepted_centers = [], set(), []
    centers = np.asarray(centers)
    if visited_centers is None:
        visited_centers = [centers[c] for c in visited_candidates]
    blocked_centers = [np.asarray(c) for c in visited_centers]
    for pose in torch.argsort(flat, descending=True).tolist():
        candidate = pose // 8
        if candidate in visited_candidates or candidate in accepted or pose in visited_poses:
            continue
        proposed = offset_center(centers, pose, predicted_offsets)
        blocked = blocked_centers + accepted_centers
        if blocked and np.any(np.linalg.norm(np.asarray(blocked) - proposed, axis=1) < radius):
            continue
        choices.append(pose)
        accepted.add(candidate)
        accepted_centers.append(proposed)
        if len(choices) == 64:
            break
    return torch.as_tensor(choices, dtype=torch.long, device=scores.device)


def offset_center(centers, pose, offsets=None):
    center = np.array(centers[pose // 8], dtype=float, copy=True)
    if offsets is not None:
        center[:2] += offsets[pose // 8, pose % 8].detach().cpu().numpy()
    return center


def rollout(model, features, text, asset, centers, dimensions, geometry, steps=3, deterministic=False,
            apply_offsets=False, anti_cycle=True, offset_radius=.15):
    """Execute without any annotation/target input; expose trajectory for auditing."""
    device = features.device
    scores, fused, scene_values = model.candidates(features, text, asset)
    predicted_offsets = model.offsets(fused, offset_radius) if apply_offsets else None
    candidate_dist = torch.distributions.Categorical(logits=scores.reshape(-1))
    selected = scores.reshape(-1).argmax() if deterministic else candidate_dist.sample()
    initial_pose = current_pose = int(selected.item())
    initial_j = current_j = initial_pose // 8
    center = offset_center(centers, current_pose, predicted_offsets)
    initial_center = center.copy()
    anchor_center = np.array(centers[current_j], dtype=float, copy=True)
    visited_candidates, visited_poses = {current_j}, {current_pose}
    visited_centers = [initial_center.copy()]
    repeat_count = 0
    initial_yaw = (current_pose % 8) * math.pi/4
    dims = np.asarray(dimensions, dtype=float)
    yaw = (current_pose % 8) * math.pi/4
    vector, stats, queries = geometry.features(center, dims, yaw)
    initial_stats = stats
    initial_queries = queries
    spatial_queries = 0
    path, logprobs, values, entropies, step_costs = [], [], [], [], []
    switches, stopped = 0, False
    for step in range(steps):
        relative = (center - initial_center) / dims
        local_state = torch.cat((fused[current_j], torch.as_tensor(vector, device=device),
                                 torch.as_tensor(relative, dtype=features.dtype, device=device),
                                 torch.tensor([math.sin(yaw), math.cos(yaw),
                                               (steps-step)/steps], device=device)))
        mode_logits, means, local_values = model.local(local_state)
        flat_scores = scores.reshape(-1)
        choices = restart_choices(scores, centers, current_pose, visited_candidates, visited_poses, anti_cycle,
                                  predicted_offsets=predicted_offsets, visited_centers=visited_centers)
        if not len(choices):
            mode_logits = mode_logits.clone()
            mode_logits[2] = -torch.inf
        mode_dist = torch.distributions.Categorical(logits=mode_logits)
        mode = mode_logits.argmax() if deterministic else mode_dist.sample()
        kind = int(mode.item())
        logprob = mode_dist.log_prob(mode)
        entropy = mode_dist.entropy()
        previous_center, previous_yaw = center.copy(), yaw
        meta = {}
        cost = 0.
        if kind == 0:
            normal = torch.distributions.Normal(means, model.log_std.exp().clamp(.1, 1.5))
            latent = means if deterministic else normal.sample()
            logprob = logprob + normal.log_prob(latent).sum()
            action = torch.tanh(latent).detach().cpu().numpy()
            local_xy = action[:2] * .05
            norm = np.linalg.norm(local_xy)
            if norm > .05:
                local_xy *= .05 / norm
            c, s = math.cos(yaw), math.sin(yaw)
            world_xy = np.array([[c, -s], [s, c]]) @ local_xy
            proposed = center + np.r_[world_xy, 0.]
            proposed_yaw = float((yaw + action[2]*math.pi/12 + math.pi) % (2*math.pi)-math.pi)
            proposed, snap = geometry.snap_height(proposed, dims, proposed_yaw)
            queries += int(snap.get('query_count', 1))
            spatial_queries += int(snap.get('spatial_queries', 0))
            center, yaw, stats, guard = geometry.project_move(center, yaw, proposed, proposed_yaw, dims)
            queries += int(guard['query_count'])
            meta = {'snap': snap, 'guard': guard, 'latent_action': action.tolist()}
            vector, stats, count = geometry.features(center, dims, yaw)
            queries += count
            cost = .002
        elif kind == 2:
            switch_dist = torch.distributions.Categorical(logits=flat_scores[choices])
            switch_index = flat_scores[choices].argmax() if deterministic else switch_dist.sample()
            logprob = logprob + switch_dist.log_prob(switch_index)
            entropy = entropy + switch_dist.entropy()
            current_pose = int(choices[int(switch_index.item())].item())
            current_j = current_pose // 8
            repeat_count += int(current_j in visited_candidates)
            visited_candidates.add(current_j)
            visited_poses.add(current_pose)
            center = offset_center(centers, current_pose, predicted_offsets)
            yaw = (current_pose % 8) * math.pi/4
            vector, stats, count = geometry.features(center, dims, yaw)
            queries += count
            switches += 1
            cost = .01
            meta = {'candidate_restart': current_j, 'yaw_bin_restart': current_pose % 8, 'anchor_center': np.asarray(centers[current_j]).tolist(), 'applied_offset_xy_m': (center[:2]-np.asarray(centers[current_j])[:2]).tolist()}
        else:
            stopped = True
        if kind in (0, 2):
            visited_centers.append(center.copy())
        values.append(local_values)
        logprobs.append(logprob)
        entropies.append(entropy)
        step_costs.append(cost)
        path.append({'step': step + 1, 'mode': ('move', 'stop', 'switch')[kind],
                     'candidate': current_j, 'previous_center': previous_center.tolist(),
                     'previous_yaw': previous_yaw, 'center': center.tolist(), 'yaw': yaw,
                     'geometry': jsonable(stats), 'meta': jsonable(meta), 'step_cost': cost,
                     'cumulative_queries': queries})
        if stopped:
            break
    return {'scores': scores, 'candidate_logprob': candidate_dist.log_prob(selected),
            'candidate_entropy': candidate_dist.entropy(), 'scene_values': scene_values,
            'residual_logprobs': logprobs, 'residual_values': values, 'entropies': entropies,
            'step_costs': step_costs, 'initial_candidate': initial_j, 'final_candidate': current_j,
            'anchor_center': anchor_center, 'predicted_offsets': predicted_offsets, 'repeat_count': repeat_count,
            'visited_candidates': sorted(visited_candidates), 'visited_poses': sorted(visited_poses),
            'visited_centers': [c.tolist() for c in visited_centers],
            'apply_offsets': apply_offsets, 'anti_cycle': anti_cycle,
            'initial_center': initial_center, 'initial_yaw': initial_yaw, 'center': center, 'yaw': yaw,
            'initial_geometry': initial_stats, 'geometry': stats, 'path': path,
            'steps': len(path), 'switches': switches, 'stopped': stopped, 'queries': queries,
            'initial_queries': initial_queries, 'snap_spatial_queries': spatial_queries}


def undiscounted_terminal_utility(result, quality, geometry_weight):
    """Logged terminal utility uses the same reliable geometry costs as RL."""
    costs, reliability = costs_with_reliability(result['geometry'])
    return float(quality - geometry_weight*float(np.dot(costs,reliability)) - sum(result['step_costs']))


def constrained_loss(result, quality, positive, geometry_weight=0., rl_weight=.05):
    costs, reliability = costs_with_reliability(result['geometry'])
    geometric_cost = float(np.dot(costs, reliability)) * geometry_weight
    rewards = discounted_returns([-c for c in result['step_costs'][:-1]] +
                                  [quality-geometric_cost-result['step_costs'][-1]])
    target = result['scene_values'].new_tensor(rewards[0])
    advantage = target - result['scene_values'][0]
    rl = -result['candidate_logprob']*advantage.detach() + .5*advantage.square()
    local = torch.stack(result['residual_values'])[:, 0]
    advantages = local.new_tensor(rewards)-local
    rl = rl -(torch.stack(result['residual_logprobs'])*advantages.detach()).sum() + .5*advantages.square().mean()
    entropy = result['candidate_entropy']+torch.stack(result['entropies']).mean()
    return joint_valid_set_loss(result['scores'], positive)+rl_weight*(rl-.005*entropy), costs


def alignment_for(scene, args):
    if args.alignment_manifest:
        manifest = read_json(args.alignment_manifest)
        manifest = manifest.get('alignments',manifest)
        if scene not in manifest:
            raise ValueError('Missing axis alignment for '+scene)
        return validate_pure_yaw_alignment(manifest[scene])
    if args.alignment_root:
        for path in (args.alignment_root/scene/(scene+'.txt'), args.alignment_root/(scene+'.txt')):
            if path.is_file():
                return load_axis_alignment(path)
    raise ValueError('Missing mandatory axis alignment for '+scene)


def resolve_candidate_indices(arrays, rows, xyz, entry=None):
    centers = arrays['centers'][rows]
    if 'candidate_indices' in arrays:
        indices = np.asarray(arrays['candidate_indices'][rows])
        if indices.dtype.kind not in 'iu' or np.any(indices<0) or np.any(indices>=len(xyz)):
            raise ValueError('Invalid candidate indices')
        distances = np.linalg.norm(xyz[indices]-centers, axis=1)
    elif entry is not None and 'point_indices' in entry:
        indices = np.asarray(entry['point_indices'],dtype=np.int64)
        if len(indices)!=len(rows) or np.any(indices<0) or np.any(indices>=len(xyz)):
            raise ValueError('Invalid group point_indices')
        distances = np.linalg.norm(xyz[indices]-centers,axis=1)
    else:
        raise ValueError('Original candidate indices required; coordinate matching cannot establish point identity')
    if np.any(distances>=1e-5):
        raise ValueError('Candidates do not match original scene points within 1e-5m')
    return indices.astype(np.int64)


def aligned_task(raw, centers, raw_xyz, alignment, cache=None):
    cache = {} if cache is None else cache
    key = hashlib.sha256(np.asarray(centers).tobytes()).hexdigest()
    if key not in cache:
        if '_xyz' not in cache:
            cache['_xyz']=raw_centered_to_aligned_centered(raw_xyz,alignment).astype(np.float32)
            cache['_tree']=cKDTree(cache['_xyz'])
        xyz=cache['_xyz']
        aligned=raw_centered_to_aligned_centered(centers,alignment).astype(np.float32)
        _,near=cache['_tree'].query(aligned,k=min(16,len(xyz)))
        local=xyz[near]-aligned[:,None]
        base=np.concatenate(((aligned-xyz.min(0))/np.maximum(xyz.max(0)-xyz.min(0),.1),local.mean(1),local.std(1)),1)
        cache[key]=(aligned,base)
    centers,base=cache[key]
    result=raw[:,:47].copy()
    result[:,:3]=base[:,:3]
    result[:,6:12]=base[:,3:]
    return result,centers


def annotations(group, xyz):
    valid = group['pointcloud_ids'][()]
    if np.any(valid<0) or np.any(valid>=len(xyz)):
        raise ValueError('Annotation point index outside scene')
    rots, masks = group['rotations'][()], group['valid_rotations'][()]
    return valid, rots, masks, decode_valid_rotations(rots, masks)


def summarize(rows):
    if not rows:
        raise ValueError('No evaluable development tasks')
    summary = {'dev_tasks':len(rows), 'candidate_recall':float(np.mean([r['candidate_has_positive'] for r in rows])),
               'no_positive_tasks':sum(not r['candidate_has_positive'] for r in rows),
               'scope':'Position proximity and existential position+yaw pairs (<=22.5deg); nearest-conditioned metrics separately retained. Annotation proxies, not official mesh/language/physical success'}
    for prefix in ('anchor','initial','final'):
        metrics = [r[prefix] for r in rows]
        summary[prefix] = {'distance_mean_m':float(np.mean([m['nearest_distance_m'] for m in metrics])),
                          'yaw_error_mean_rad':float(np.mean([m['nearest_angle_rad'] for m in metrics])),
                          'quality':float(np.mean([m['quality'] for m in metrics]))}
        for threshold in (.05,.1):
            summary[prefix]['position_within_'+str(threshold)] = float(np.mean([m['nearest_distance_m']<=threshold for m in metrics]))
            summary[prefix]['position_yaw_proxy_within_'+str(threshold)] = float(np.mean([
                m['nearest_distance_m']<=threshold and m['nearest_angle_rad']<=math.pi/8+1e-8 for m in metrics]))
            summary[prefix]['any_valid_pose_within_'+str(threshold)] = float(np.mean([
                m.get('angle_compatible_distance_m') is not None and m['angle_compatible_distance_m']<=threshold for m in metrics]))
        geometry = [r[prefix+'_geometry'] for r in rows]
        summary[prefix]['surface_contact_mean'] = float(np.mean([g['plane_contact_fraction'] for g in geometry]))
        summary[prefix]['surface_known_mean'] = float(np.mean([g['plane_valid_fraction'] for g in geometry]))
        summary[prefix]['mesh_collision_cost_mean'] = float(np.mean([g['penetration_cost'] for g in geometry]))
        for key in ('mesh_surface_intersections','collision_noncontact_pairs','collision_contact_band_pairs',
                    'support_contact_fraction','support_known_fraction','support_gap_fraction','support_unknown_fraction'):
            summary[prefix][key+'_mean'] = float(np.mean([g[key] for g in geometry]))
        summary[prefix]['mesh_query_complete_fraction'] = float(np.mean([g['collision_complete'] for g in geometry]))
        summary[prefix]['mesh_noncontact_crossing_fraction'] = float(np.mean([g['collision_noncontact_pairs']>0 for g in geometry]))
    summary['initial_exact_position_yaw_hit'] = float(np.mean([r['initial_exact_pose_hit'] for r in rows]))
    summary['stop_fraction'] = float(np.mean([r['stopped'] for r in rows]))
    summary['mean_steps'] = float(np.mean([r['steps'] for r in rows]))
    summary['mean_repeat_count'] = float(np.mean([r['repeat_count'] for r in rows]))
    summary['apply_offsets'] = rows[0]['apply_offsets']
    summary['anti_cycle'] = rows[0]['anti_cycle']
    summary['mean_switches'] = float(np.mean([r['switches'] for r in rows]))
    return summary


def evaluate(model, base, val, selected, entries, texts, mean, std, device, steps, h5path,
             rows_path=None, args=None, asset_features=None, asset_library=None):
    was_training=model.training
    model.eval()
    rows=[]
    with np.load(val/'data.npz',allow_pickle=False) as z:
        arrays={key:z[key] for key in z.files if key in ('features','group_ids','centers','candidate_indices')}
    scene_cache=None
    try:
        with h5py.File(h5path,'r') as annotation, torch.no_grad():
            for gi in selected:
                entry=entries[gi]; scene=entry['scene']
                if scene_cache is None or scene_cache[0]!=scene:
                    raw_xyz,_=load_scene(base,scene,'val')
                    A=alignment_for(scene,args)
                    xyz=raw_centered_to_aligned_centered(raw_xyz,A).astype(np.float32)
                    mesh_xyz,faces=load_scene_mesh(base,scene,'val',asset_library,reference_xyz=raw_xyz)
                    if mesh_xyz.shape != xyz.shape or not np.allclose(mesh_xyz,xyz,atol=1e-4,rtol=0):
                        raise ValueError('Scene mesh/reference alignment mismatch: '+scene)
                    scene_cache=(scene,raw_xyz,xyz,SceneGeometry(mesh_xyz,faces,asset_library),A,{})
                _,raw_xyz,xyz,geometry,A,feature_cache=scene_cache
                ids=np.flatnonzero(arrays['group_ids']==gi)
                candidate_ids=resolve_candidate_indices(arrays,ids,raw_xyz,entry)
                raw,centers=aligned_task(arrays['features'][ids],arrays['centers'][ids],raw_xyz,A,feature_cache)
                geometry.set_asset(asset_uuid(entry),raw[0,12:15])
                result=rollout(model,make_features(raw,mean,std,device),torch.as_tensor(texts[gi],device=device),
                               asset_features[asset_uuid(entry)].to(device),centers,raw[0,12:15],geometry,steps,deterministic=True, apply_offsets=args.apply_offsets, anti_cycle=args.anti_cycle, offset_radius=args.offset_radius)
                valid,rots,masks,mask8=annotations(annotation[entry['asset_scene']][entry['entry']],xyz)
                positive=match_candidate_pose_targets(candidate_ids,valid,rots,masks)
                initial_bin = int(round(result['initial_yaw']/(math.pi/4))) % 8
                rows.append({'group_id':int(gi),'scene':scene,'asset_scene':entry['asset_scene'], 'entry':entry['entry'],
                    'prompt_sha256':hashlib.sha256(entry['prompt'].encode()).hexdigest(),
                    'coordinate_frame':'axis_aligned_scene_mean_centered',
                    'dimensions':raw[0,12:15].tolist(), 'asset_uuid':asset_uuid(entry),
                    'initial_original_point_index':int(candidate_ids[result['initial_candidate']]),
                    'final_anchor_original_point_index':int(candidate_ids[result['final_candidate']]),
                    'initial_exact_pose_hit':bool(positive[result['initial_candidate'],initial_bin]),
                    'candidate_has_positive':bool(positive.any()),
                    'anchor':pose_quality_and_metrics(result['anchor_center'],result['initial_yaw'],xyz[valid],mask8),
                    'anchor_center':result['anchor_center'].tolist(),
                    'anchor_geometry':jsonable(geometry.features(result['anchor_center'],raw[0,12:15],result['initial_yaw'])[1]),
                    'apply_offsets':result['apply_offsets'],'anti_cycle':result['anti_cycle'],'repeat_count':result['repeat_count'],
                    'initial':pose_quality_and_metrics(result['initial_center'],result['initial_yaw'],xyz[valid],mask8),
                    'final':pose_quality_and_metrics(result['center'],result['yaw'],xyz[valid],mask8),
                    'initial_yaw':result['initial_yaw'],'final_yaw':result['yaw'],
                    'initial_center':result['initial_center'].tolist(),'final_center':result['center'].tolist(),
                    'initial_candidate':result['initial_candidate'],'final_candidate':result['final_candidate'],
                    'initial_geometry':jsonable(result['initial_geometry']),'final_geometry':jsonable(result['geometry']),
                    'steps':result['steps'],'switches':result['switches'],'stopped':result['stopped'],
                    'queries':result['queries'],'path':result['path']})
    finally:
        model.train(was_training)
    if rows_path:
        temporary=rows_path.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(jsonable(row))+'\n' for row in rows),encoding='utf-8')
        os.replace(temporary,rows_path)
    return summarize(rows)


def save_checkpoint(path, model, optimizer, mean, std, multipliers, protocol,
                    epoch, next_scene, order, epoch_tasks, epoch_rewards, epoch_stats=None):
    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
             'mean': mean, 'std': std, 'multipliers': multipliers,
             'protocol': protocol, 'epoch': epoch, 'next_scene': next_scene,
             'scene_order': order, 'epoch_tasks': epoch_tasks, 'epoch_rewards': epoch_rewards,
             'epoch_stats':dict(epoch_stats or {}),
             'python_rng': random.getstate(), 'numpy_rng': np.random.get_state(),
             'torch_rng': torch.get_rng_state(),
             'cuda_rng': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}
    temporary = path.with_suffix('.tmp')
    torch.save(state, temporary)
    os.replace(temporary, path)


def validate_prepared_cache(args, paths, mean, std):
    index=read_json(args.training_cache/'index.json')
    if index.get('test_reads')!=0 or not all(p.name in index.get('scenes',{}) for p in paths):
        raise ValueError('Selected train scenes missing from train-only prepared cache')
    if not (args.max_scenes or args.max_tasks_per_scene or args.max_dev_tasks):
        if not index.get('complete') or index.get('scene_count')!=563 or index.get('task_count')!=100505:
            raise ValueError('Full training requires complete 563-scene/100505-task prepared cache')
    expected=dict(distance_temperature=args.distance_temperature,offset_radius=args.offset_radius,vertical_tolerance=args.vertical_tolerance)
    for path in paths:
        entry=index['scenes'][path.name];mp=args.training_cache/path.name/'manifest.json'
        if digest(mp)!=entry['manifest_sha256']:raise ValueError('Prepared scene manifest changed: '+path.name)
        meta=read_json(mp);deps=meta['dependencies']
        if meta['fingerprint']!=entry['fingerprint'] or meta['scene']!=path.name or meta['split']!='train':raise ValueError('Cache identity mismatch')
        if any(deps[k]!=v for k,v in expected.items()) or not np.array_equal(deps['mean'],mean) or not np.array_equal(deps['std'],std):
            raise ValueError('Cache supervision/normalization parameters differ')
        for source in deps['sources']:
            if file_identity(source['path'])['sha256']!=source['sha256']:raise ValueError('Cache source changed: '+source['path'])
    return index


def train_batch_v7(model,optimizer,records,asset_features,device,service,scene,args,epoch):
    batch=collate_records(records,asset_features,device)
    rl=None;utilities=None
    if epoch<=args.warmup_epochs:
        scores,fused,_=model.candidates_batch(batch['features'],batch['texts'],batch['assets'],batch['candidate_mask'])
        offsets=model.offsets_batch(fused,batch['candidate_mask'],args.offset_radius)
    else:
        observations=[dict(scene=scene,split='train',asset_uuid=r['asset_uuid'],dims=r['dims'],centers=r['centers']) for r in records]
        results=rollout_batch(model,batch['features'],batch['texts'],batch['assets'],batch['candidate_mask'],observations,service,
            steps=args.steps,deterministic=False,apply_offsets=args.apply_offsets,anti_cycle=args.anti_cycle,offset_radius=args.offset_radius)
        count=batch['candidate_mask'].shape[1]
        scores=torch.stack([F.pad(r['scores'],(0,0,0,count-len(r['scores'])),value=-torch.inf) for r in results])
        offsets=torch.stack([F.pad(r['supervised_offsets'],(0,0,0,0,0,count-len(r['scores']))) for r in results])
        qualities=[pose_quality_and_metrics(r['center'],r['yaw'],record['valid_xyz'],record['valid_mask8'])['quality'] for r,record in zip(results,records)]
        rl=batched_rl_losses(results,qualities,args.geometry_weight,args.rl_weight)
        utilities=rl['utilities'].tolist()
    components=batched_supervised_losses(scores,offsets,batch['positive'],batch['rank_prob'],batch['offset_xy'],batch['offset_mask'],batch['candidate_mask'],
        distance_weight=args.distance_weight,offset_weight=args.offset_weight,offset_radius=args.offset_radius)
    losses=components['loss']+(rl['rl_loss'] if rl is not None else 0)
    loss=losses.mean()
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite V7 batch objective')
    loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step();optimizer.zero_grad(set_to_none=True)
    logs=torch.stack([components['rank_loss'].sum(),components['offset_loss'].sum()]).detach().cpu().tolist()
    return dict(rank_loss_sum=logs[0],offset_loss_sum=logs[1],utilities=utilities)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(2)
    base = args.root/'rl_position_adjustment_v1'
    shards = base/'placeit_full_train_shards_v1'
    val = base/'placeit_train32_geometry_v1'
    cache = args.text_cache or base/'joint_text_cache_v2'
    output = base/args.output
    output.mkdir(parents=True, exist_ok=True)
    if (output/'protocol.json').exists() and args.resume is None:
        raise FileExistsError('Run already exists; choose a fresh output or explicitly resume its checkpoint')
    paths = sorted(q for q in shards.iterdir() if q.is_dir() and (q/'data.npz').is_file()
                   and (q/'groups.json').is_file())
    cache_manifest = read_json(cache/'manifest.json')
    vm = read_json(val/'manifest.json')
    smoke = bool(args.max_scenes or args.max_tasks_per_scene or args.max_dev_tasks)
    if cache_manifest.get('complete') is not True or cache_manifest.get('test_reads') != 0:
        raise ValueError("Completed train/dev-only text cache required")
    if not smoke:
        if (not cache_manifest.get('full_dataset_complete') or len(paths) != 563
                or cache_manifest.get('train_scene_count') != 563
                or cache_manifest.get('train_groups') != 100505
                or cache_manifest.get('dev_groups') != 297):
            raise ValueError("Full run requires 563 scenes / 100505 train / 297 dev text groups")
    if args.max_scenes:
        paths = paths[:args.max_scenes]
    development_scenes = set(vm['scene_sets']['val'])
    development_rooms = {scene[:9] for scene in development_scenes}
    training_rooms = {p.name[:9] for p in paths}
    if not paths or training_rooms & development_rooms:
        raise ValueError("Empty training set or train/development room overlap")
    entries = read_json(val/'groups.json')
    selected = [gi for gi, entry in enumerate(entries) if entry['scene'] in development_scenes]
    if not smoke and (len(selected) != 297 or len(development_rooms) != 20):
        raise ValueError("Full development scope must be 20 rooms / 297 tasks")
    if args.max_dev_tasks:
        selected = selected[:args.max_dev_tasks]
    if smoke:
        with np.load(cache/'dev.npz', allow_pickle=False) as z:
            available = set(map(int, z['group_ids']))
        selected = [gi for gi in selected if gi in available]
    if not selected:
        raise ValueError("No development groups in selected/cache intersection")
    dev_hash = digest(val/'groups.json')
    if cache_manifest['dev']['source_sha256'] != dev_hash:
        raise ValueError("Development text manifest source mismatch")
    dev_texts = load_text_cache(cache/'dev.npz', entries, selected, dev_hash,
                                cache_manifest['model_signature'])
    init_path = base/'streaming_ranker_full_train_v1/epoch2.pt'
    pretrained = torch.load(init_path, map_location='cpu', weights_only=False)
    mean, std = np.asarray(pretrained['mean']), np.asarray(pretrained['std'])
    if len(mean) != 47 or not np.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Expected original 47-feature ranker normalization")
    device = torch.device(args.device)
    asset_entries = list(entries[gi] for gi in selected)
    for shard in paths:
        asset_entries.extend(read_json(shard/'groups.json'))
    asset_features, asset_dimension, asset_checksums = load_asset_features(args.asset_features,asset_entries)
    asset_library = load_mesh_manifest(args.mesh_manifest)
    # The loader verifies manifest file checksums; demand every selected scene/asset
    # before initializing the model, including supervised warmup runs.
    required_scene_ids = sorted(set(p.name for p in paths) | {entries[gi]['scene'] for gi in selected})
    required_asset_ids = sorted({asset_uuid(entry) for entry in asset_entries})
    asset_library.validate_required(required_scene_ids,required_asset_ids)
    prepared_index = validate_prepared_cache(args, paths, mean, std)
    model = JointModel(len(mean), pretrained['model'], asset_dimension,args.prior_weight).to(device)
    asset_features = {k:v.to(device) for k,v in asset_features.items()}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    multipliers = np.zeros(2, dtype=np.float64)
    sources = [Path(__file__), Path(__file__).with_name('mesh_scene_adapter_v6.py'), Path(__file__).with_name('asset_mesh_geometry_v6.py'), Path(__file__).with_name('distance_targets_v5.py'), Path(__file__).with_name('joint_objectives_v2.py'),
               Path(__file__).with_name('contact_geometry_v4.py'), Path(__file__).with_name('directional_geometry_v2.py'), Path(__file__).with_name('pose_targets_v4.py'),
               Path(__file__).with_name('build_joint_text_cache_v2.py')]
    sources.extend(Path(__file__).with_name(n) for n in ['train_joint_pose_v6.py','batched_policy_v7.py','batched_rollout_v7.py','training_batch_v7.py','training_cache_v7.py','mesh_workers_v7.py'])
    train_h5 = args.root/'placeit3d_reproduction_v1/downloads/data/placeit3d_train.h5'
    val_h5 = args.root/'placeit3d_reproduction_v1/downloads/data/placeit3d_val.h5'
    protocol = {'label_sha256':{'train':digest(train_h5),'val':digest(val_h5)}, 'candidate_cache_sha256':{'train':{p.name:{name:digest(p/name) for name in ('groups.json','data.npz')} for p in paths}, 'dev':{name:digest(val/name) for name in ('groups.json','data.npz')}}, 'version': 'joint_pose_v7', 'mesh_manifest_sha256':asset_library.manifest_sha256, 'asset_features_sha256':asset_checksums, 'asset_feature_dimension':asset_dimension, 'asset_feature_source':'Official frozen max-pooled PointBERT embeddings; learned 128 projection and joint fusion; no asset encoder finetuning', 'warmup_epochs': args.warmup_epochs, 'distance_weight':args.distance_weight, 'offset_weight':args.offset_weight, 'distance_temperature':args.distance_temperature, 'offset_radius':args.offset_radius, 'vertical_tolerance':args.vertical_tolerance, 'apply_offsets':args.apply_offsets, 'anti_cycle':args.anti_cycle, 'rl_weight':args.rl_weight, 'geometry_weight':args.geometry_weight, 'alignment_sha256':{p.name: digest(p) for p in [args.alignment_manifest] if p is not None}, 'alignments': {scene:alignment_for(scene,args).tolist() for scene in sorted(set([p.name for p in paths]) | development_scenes)}, 'test_reads': 0,
                'train_scenes': len(paths), 'train_scene_ids': [p.name for p in paths],
                'development_rooms': sorted(development_rooms), 'dev_tasks': len(selected),
                'smoke': smoke, 'arguments': {k: str(v) if isinstance(v, Path) else v
                                             for k, v in vars(args).items()},
                'cache_manifest_sha256': digest(cache/'manifest.json'),
                'pretrained_ranker_sha256': digest(init_path),
                'source_sha256': {p.name: digest(p) for p in sources},
                'text': 'Frozen cached contextual RoBERTa 768; trainable 128 projection and cross fusion',
                'reward': 'Joint position+yaw pose quality minus reliability-weighted terminal mesh geometry cost and move/restart costs; aligned annotation rotation bins', 'reward_mean_definition':'Mean undiscounted terminal utility: quality - geometry_weight * dot(mesh_costs,reliability) - sum(step_costs); discounted loss unchanged', 'training_schedule':'Configured warmup epochs supervised only; later epochs supervised plus RL; offsets are supervised pre-refinement without support verification', 'joint_policy':'P(candidate,yaw8) with fused candidate and sin/cos angle; restart samples joint pose', 'prior_weight':args.prior_weight,
                'policy_reduction': 'Sum local score-function terms over the trajectory; mean across tasks',
                'geometry': {'mesh_manifest_sha256':asset_library.manifest_sha256,
                             'geometry_weight':args.geometry_weight, 'scene_alignment_method':'Indexwise raw mesh minus original float32 raw-centered refer XYZ must be a common median translation with max Euclidean residual <=1e-5m; mesh coordinates then use that reference XYZ rotated by manifest alignment; per-scene audit retained in dev geometry stats', 'actor_feature_labels':list(SceneGeometry.feature_labels),
                             'collision':'Actual asset/scene triangle-surface intersections; contact-band pairs recorded separately',
                             'support':'Sampled actual-bottom-mesh contact against reconstructed scene triangles',
                             'unknown_evidence':'No certified solid containment, free-space, COM/friction or physical stability'},
                'constraints': {'collision_budget': .01, 'support_cost_budget': .2,
                                'dual_lr': 0., 'dual_clip': [0., 0.]},
                'query_unit': 'Pose feature/guard evaluations plus height-estimation calls; not tree-search count',
                'limits': ['Triangle surface intersections and sampled actual-bottom contact are evidence; reconstructed mesh is not certified solid free space', 'Fixed frozen text encoder; no explicit relation grounding',
                           'Ground-truth object box dimensions; bottom-face centers',
                           'Actual asset/scene triangle collision and sampled bottom contact; no COM/friction or physical stability model',
                           'Legacy point RGB statistics retained; no image encoder/multiframe branch or Ground3D',
                           'No full semantic or physical success claim',
                           'Candidate restart switch, not semantic region discovery',
                           'Development previously exposed; train/dev only; no test reads',
                           'Terminal placement constraints only; no manipulation-trajectory collision checks'],
                'evaluation': 'Deterministic mean/argmax actor; raw anchor and optional supervised XY-corrected initial and final poses recorded; recall/exact hits use original vertices'}
    protocol['training_cache_sha256'] = digest(args.training_cache/'index.json')
    protocol['training_cache_fingerprints'] = {p.name:prepared_index['scenes'][p.name]['fingerprint'] for p in paths}
    protocol['execution'] = {'batching':'True independent-task GPU batches with masked candidates', 'geometry_workers':args.geometry_workers, 'geometry_processes':'persistent spawn CPU workers', 'target_cache':'Exact V6 preprocessing and supervision cached once; no label changes', 'sampling':'Batched random draws change RNG order, not action distribution or objective', 'mesh_query_reuse':'One actual query per new pose; V6 logical query budget also retained'}
    start_epoch, resume_order, next_scene, epoch_tasks, epoch_rewards = 1, None, 0, 0, []
    epoch_stats = {'no_positive_tasks':0,'supervised_positive_tasks':0,'distance_target_tasks':0,'offset_target_tasks':0,'rank_loss_sum':0.,'offset_loss_sum':0.}
    if args.resume:
        restored = torch.load(args.resume, map_location=device, weights_only=False)
        previous = restored['protocol']
        for key in ('training_cache_sha256','training_cache_fingerprints','mesh_manifest_sha256', 'asset_features_sha256', 'asset_feature_dimension', 'train_scene_ids', 'cache_manifest_sha256', 'pretrained_ranker_sha256', 'source_sha256', 'alignments', 'alignment_sha256', 'warmup_epochs', 'rl_weight', 'geometry_weight', 'prior_weight', 'distance_weight', 'offset_weight', 'distance_temperature', 'offset_radius', 'vertical_tolerance', 'apply_offsets', 'anti_cycle', 'label_sha256', 'candidate_cache_sha256'):
            if previous[key] != protocol[key]:
                raise ValueError("Resume provenance mismatch: " + key)
        for key in ('steps', 'batch', 'max_scenes', 'max_tasks_per_scene',
                    'max_dev_tasks', 'lr', 'seed'):
            if previous['arguments'][key] != protocol['arguments'][key]:
                raise ValueError('Resume training configuration mismatch: ' + key)
        if not np.array_equal(restored['mean'], mean) or not np.array_equal(restored['std'], std):
            raise ValueError("Resume normalization differs from pretrained ranker")
        model.load_state_dict(restored['model'])
        optimizer.load_state_dict(restored['optimizer'])
        multipliers = np.asarray(restored['multipliers'], dtype=float)
        start_epoch = int(restored['epoch'])
        next_scene = int(restored['next_scene'])
        resume_order = restored['scene_order']
        epoch_tasks, epoch_rewards = int(restored['epoch_tasks']), restored['epoch_rewards']
        epoch_stats = restored['epoch_stats']
        random.setstate(restored['python_rng'])
        np.random.set_state(restored['numpy_rng'])
        torch.set_rng_state(restored['torch_rng'].cpu())
        if restored.get('cuda_rng') is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all([rng.cpu() for rng in restored['cuda_rng']])
    source_dir = output/'source'
    source_dir.mkdir(exist_ok=True)
    for source in sources:
        (source_dir/source.name).write_bytes(source.read_bytes())
    atomic_json(output/'protocol.json', protocol)
    train_h5 = args.root/'placeit3d_reproduction_v1/downloads/data/placeit3d_train.h5'
    val_h5 = args.root/'placeit3d_reproduction_v1/downloads/data/placeit3d_val.h5'
    if args.geometry_workers < 1:
        raise ValueError('At least one CPU geometry worker required')
    optimizer.zero_grad(set_to_none=True)
    with MeshGeometryService(args.root,args.mesh_manifest,workers=args.geometry_workers) as service, ThreadPoolExecutor(max_workers=1) as prefetch:
        for epoch in range(start_epoch,args.epochs+1):
            model.train();started=time.time()
            if epoch==start_epoch and resume_order is not None:
                order=list(map(int,resume_order));offset=next_scene
            else:
                order=np.random.permutation(len(paths)).tolist();offset=0;epoch_tasks=0;epoch_rewards=[]
                epoch_stats=dict(no_positive_tasks=0,supervised_positive_tasks=0,distance_target_tasks=0,offset_target_tasks=0,rank_loss_sum=0.,offset_loss_sum=0.)
            def load_cached(index):
                scene=paths[index].name;entry=prepared_index['scenes'][scene]
                return load_scene_cache(args.training_cache/scene,expected_fingerprint=entry['fingerprint'])
            future=prefetch.submit(load_cached,order[offset]) if offset<len(order) else None
            for order_index in range(offset,len(order)):
                path=paths[order[order_index]];scene_data=future.result()
                future=prefetch.submit(load_cached,order[order_index+1]) if order_index+1<len(order) else None
                scene_entries=read_json(path/'groups.json')
                group_order=np.random.permutation(len(scene_entries)).tolist()
                if args.max_tasks_per_scene:group_order=group_order[:args.max_tasks_per_scene]
                scene_count=skipped=no_positive=0;pending=[];batch_seconds=0.
                def update_batch():
                    nonlocal batch_seconds
                    if not pending:return
                    t=time.time()
                    result=train_batch_v7(model,optimizer,pending,asset_features,device,service,path.name,args,epoch)
                    epoch_stats['rank_loss_sum']+=result['rank_loss_sum'];epoch_stats['offset_loss_sum']+=result['offset_loss_sum']
                    if result['utilities'] is not None:epoch_rewards.extend(result['utilities'])
                    batch_seconds+=time.time()-t;pending.clear()
                for gi in group_order:
                    if gi not in scene_data:
                        skipped+=1;continue
                    record=scene_data[gi]
                    has_positive=bool(record['positive'].any());has_distance=bool(record['has_target']);has_offset=bool(record['offset_mask'].any())
                    no_positive+=int(not has_positive)
                    epoch_stats['supervised_positive_tasks' if has_positive else 'no_positive_tasks']+=1
                    epoch_stats['distance_target_tasks']+=int(has_distance);epoch_stats['offset_target_tasks']+=int(has_offset)
                    epoch_tasks+=1;scene_count+=1
                    if has_positive or (has_distance and args.distance_weight>0) or (has_offset and args.offset_weight>0) or epoch>args.warmup_epochs:
                        pending.append(record)
                    if len(pending)>=args.batch:update_batch()
                update_batch()
                progress=dict(epoch=epoch,scene_done=order_index+1,scene_total=len(paths),scene=path.name,scene_tasks=scene_count,scene_skipped=skipped,
                    scene_no_positive_tasks=no_positive,scene_candidate_recall=1-no_positive/max(scene_count,1),epoch_tasks=epoch_tasks,
                    rank_loss_mean=epoch_stats['rank_loss_sum']/max(epoch_tasks,1),offset_loss_mean=epoch_stats['offset_loss_sum']/max(epoch_tasks,1),
                    epoch_supervision=dict(epoch_stats),apply_offsets=args.apply_offsets,anti_cycle=args.anti_cycle,
                    phase='supervised_warmup' if epoch<=args.warmup_epochs else 'supervised_plus_rl',
                    reward_mean=float(np.mean(epoch_rewards)) if epoch_rewards else None,multipliers=multipliers.tolist(),
                    elapsed_s=round(time.time()-started,2),scene_batch_seconds=round(batch_seconds,3),true_gpu_batch=args.batch,geometry_workers=args.geometry_workers)
                atomic_json(output/'progress.json',progress)
                if order_index==offset or (order_index+1)%10==0:print(json.dumps(progress),flush=True)
                if (order_index+1)%args.checkpoint_scenes==0:
                    save_checkpoint(output/'partial.pt',model,optimizer,mean,std,multipliers,protocol,epoch,order_index+1,order,epoch_tasks,epoch_rewards,epoch_stats)
                del scene_data
            save_checkpoint(output/'partial.pt',model,optimizer,mean,std,multipliers,protocol,epoch,len(order),order,epoch_tasks,epoch_rewards,epoch_stats)
            metrics=evaluate(model,base,val,selected,entries,dev_texts,mean,std,device,args.steps if epoch>args.warmup_epochs else 0,
                val_h5,output/f'dev_rows_epoch{epoch}.jsonl',args=args,asset_features=asset_features,asset_library=asset_library)
            metrics.update(epoch=epoch,phase='supervised_anchor_and_offset_eval' if epoch<=args.warmup_epochs else 'supervised_plus_rl',
                train_tasks=epoch_tasks,train_supervision=dict(epoch_stats),rank_loss_mean=epoch_stats['rank_loss_sum']/max(epoch_tasks,1),
                offset_loss_mean=epoch_stats['offset_loss_sum']/max(epoch_tasks,1),train_reward_mean=float(np.mean(epoch_rewards)) if epoch_rewards else None,
                multipliers=multipliers.tolist(),seconds=time.time()-started)
            atomic_json(output/f'epoch{epoch}_metrics.json',metrics);print(json.dumps(metrics),flush=True)
            save_checkpoint(output/f'epoch{epoch}.pt',model,optimizer,mean,std,multipliers,protocol,epoch+1,0,None,0,[],
                dict(no_positive_tasks=0,supervised_positive_tasks=0,distance_target_tasks=0,offset_target_tasks=0,rank_loss_sum=0.,offset_loss_sum=0.))
            resume_order=None;next_scene=0

    return model


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--root', type=Path, required=True)
    result.add_argument('--output', default='joint_pose_full_train_v7')
    result.add_argument('--device', default='cuda:1')
    result.add_argument('--epochs', type=int, default=3)
    result.add_argument('--max-scenes', type=int, default=0)
    result.add_argument('--max-tasks-per-scene', type=int, default=0)
    result.add_argument('--max-dev-tasks', type=int, default=0)
    result.add_argument('--steps', type=int, default=3)
    result.add_argument('--batch', type=int, default=16)
    result.add_argument('--seed', type=int, default=41)
    result.add_argument('--lr', type=float, default=1e-4)
    result.add_argument('--text-cache', type=Path)
    result.add_argument('--asset-features', type=Path, required=True)
    result.add_argument('--mesh-manifest', type=Path, required=True)
    result.add_argument('--resume', type=Path)
    result.add_argument('--training-cache',type=Path,required=True)
    result.add_argument('--geometry-workers',type=int,default=2)
    result.add_argument('--alignment-manifest',type=Path)
    result.add_argument('--alignment-root',type=Path)
    result.add_argument('--warmup-epochs',type=int,default=2)
    result.add_argument('--distance-weight',type=float,default=.5)
    result.add_argument('--offset-weight',type=float,default=1.)
    result.add_argument('--distance-temperature',type=float,default=.1)
    result.add_argument('--offset-radius',type=float,default=.15)
    result.add_argument('--vertical-tolerance',type=float,default=.01)
    result.add_argument('--apply-offsets',action='store_true')
    result.add_argument('--anti-cycle',dest='anti_cycle',action='store_true',default=True)
    result.add_argument('--no-anti-cycle',dest='anti_cycle',action='store_false')
    result.add_argument('--rl-weight',type=float,default=.05)
    result.add_argument('--geometry-weight',type=float,default=0.)
    result.add_argument('--prior-weight',type=float,default=.25)
    result.add_argument('--checkpoint-scenes', type=int, default=20)
    return result


if __name__ == '__main__':
    cli = parser()
    args = cli.parse_args()
    if (min(args.epochs, args.steps, args.batch, args.checkpoint_scenes) < 1
            or min(args.max_scenes, args.max_tasks_per_scene, args.max_dev_tasks) < 0
            or not math.isfinite(args.lr) or args.lr <= 0):
        cli.error('Require positive epochs/steps/batch/checkpoint-scenes/lr and nonnegative smoke limits')
    if (args.warmup_epochs<1 or not 0<=args.prior_weight<=1 or min(args.rl_weight,args.geometry_weight)<0
            or not np.isfinite([args.rl_weight,args.geometry_weight,args.prior_weight]).all()):
        cli.error('Require warmup>=1, prior weight in [0,1], nonnegative RL/geometry weights')
    if (min(args.distance_weight,args.offset_weight,args.vertical_tolerance)<0 or min(args.distance_temperature,args.offset_radius)<=0
            or not np.isfinite([args.distance_weight,args.offset_weight,args.vertical_tolerance,args.distance_temperature,args.offset_radius]).all()):
        cli.error('Require finite nonnegative auxiliary weights/tolerance and positive temperature/radius')
    if not args.alignment_manifest and not args.alignment_root:
        cli.error('Mandatory --alignment-manifest or --alignment-root')
    main(args)

