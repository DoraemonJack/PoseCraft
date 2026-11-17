"""V1.0 observation proposals, rebuilt features, and deferred training labels.

No labels or GT instance types enter candidate generation. The original cached
candidate feature arrays provide task metadata and normalization identity only;
new 47D point features and all pooled141D features are reconstructed afresh.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import sys
import numpy as np
import torch

from support_regions import (prepare_scene,prepare_asset,propose,load_grounded_surfaces,
                             save_prepared_scene,load_prepared_scene,FEATURE_NAMES)

ROOT=Path('/path/to/research300/refineplace_v1_0_20261103_v1')
LEGACY=Path('/path/to/home/codex_3d_research_20261028/rl_position_adjustment_v1')
_HASH_CACHE={}
_NORM_CACHE={}


def sha(path):
    path=Path(path);stat=path.stat();key=(str(path),stat.st_size,stat.st_mtime_ns)
    if key not in _HASH_CACHE:
        h=hashlib.sha256()
        with path.open('rb') as fp:
            for block in iter(lambda:fp.read(8*1024*1024),b''):h.update(block)
        _HASH_CACHE[key]=h.hexdigest()
    return _HASH_CACHE[key]


def join_canonical_masks(candidate_ids, valid_ids, masks):
    candidate_ids=np.asarray(candidate_ids,np.int64);valid_ids=np.asarray(valid_ids,np.int64);masks=np.asarray(masks,bool)
    if masks.shape!=(len(valid_ids),8):raise ValueError('Canonical valid mask shape mismatch')
    out=np.zeros((len(candidate_ids),8),bool)
    if not len(valid_ids):return out
    unique,inverse=np.unique(valid_ids,return_inverse=True)
    union=np.zeros((len(unique),8),bool);np.logical_or.at(union,inverse,masks)
    p=np.searchsorted(unique,candidate_ids);inside=p<len(unique)
    rows=np.flatnonzero(inside);rows=rows[unique[p[rows]]==candidate_ids[rows]]
    out[rows]=union[p[rows]]
    return out


def build_task_candidates(prepared, asset, baseline_ids, task, payload,
                          original_raw, alignment, mean, std, geometry_cache=None,
                          *, budget=256, global_quota=32, receiver_preferences=None):
    """Shared train/validation observation-only proposal and feature path.

    ``payload`` contains original scene XYZ/RGB in its first two slots; its
    remaining three legacy slots may be None. ``original_raw`` is the source
    task's 47D raw feature matrix, in the same order as ``baseline_ids``. The
    normalization must be the training-only checkpoint normalization. No
    labels, valid pose IDs, or instance/type arrays are accepted here.
    """
    sys.path.insert(0,str(LEGACY))
    from novel_candidate_features_v1 import reconstruct
    from train_joint_pose_v7 import make_features
    ids,region,report=propose(prepared,asset,baseline_ids,task,budget=budget,
        global_quota=global_quota,receiver_preferences=receiver_preferences)
    raw,centers=reconstruct(payload,ids,baseline_ids,original_raw,alignment,
                            {} if geometry_cache is None else geometry_cache)
    features=make_features(raw,mean,std,torch.device('cpu')).numpy()
    if features.shape!=(budget,141) or not np.isfinite(features).all():
        raise ValueError('Invalid reconstructed candidate feature tensor')
    return dict(candidate_ids=ids,centers=centers,features=features,
                region_features=region,region_report=report,
                candidate_generation='RefinePlace_v1.0_grounded_size_yaw_regions')


class V10Scene:
    def __init__(self, scene, lock, library, *, driver=None, grounded_root=None,
                 cache_root=None, budget=256, global_quota=32, pool_size=8192):
        if driver is None:
            sys.path.insert(0,'/path/to/research300/refineplace_v01_official563_modules_20261102_v1')
            import train_official563_joint_v7_sync_full as driver
        from cpu_pool_backend_v1 import observation_scene_class
        sys.path.insert(0,str(LEGACY))
        from novel_candidate_features_v1 import reconstruct
        from train_joint_pose_v7 import make_features
        self.d=driver;self.s=scene;self.lock=lock;self.library=library
        self.source=observation_scene_class(driver)(scene,lock,library)
        self.a=self.source.a;self.m=self.source.m;self.tm=self.source.tm
        self.registry=self.source.registry;self.bygi=self.source.bygi
        self.budget=budget;self.global_quota=global_quota
        self.reconstruct=reconstruct;self.make_features=make_features
        self.geometry_cache={};self.asset_priors={};self.label_cache={};self.trees=self.source.trees
        self.original_features_validated=False;self.feature_reconstruction_max_error=None
        source_path=LEGACY/'scans_full_refer_v1/train'/(scene+'_refer.pth')
        if sha(source_path)!=self.m['source_identity']['geometry_package']:
            raise ValueError('Scene observation package checksum mismatch')
        payload=torch.load(source_path,map_location='cpu',weights_only=False)
        # Legacy container also carries instance arrays. They are discarded;
        # only observed XYZ/RGB are selected and passed to the feature builder.
        self.payload=(np.asarray(payload[0],np.float32).copy(),np.asarray(payload[1],np.float32).copy(),None,None,None)
        del payload
        self.alignment=np.asarray(library.manifest['alignments'][scene],float)
        self.xyz=(self.payload[0].astype(np.float64)@self.alignment[:3,:3].T).astype(np.float32)
        rawpath=LEGACY/'placeit_full_train_shards_v1'/scene/'data.npz'
        if sha(rawpath)!=self.m['source_identity']['raw_npz']:raise ValueError('Raw task-feature source changed')
        with np.load(rawpath,allow_pickle=False) as z:
            self.raw=np.asarray(z['features'][:,:47],np.float32).copy();self.raw_group_ids=np.asarray(z['group_ids']).copy()
        self.raw_rows={int(gi):np.flatnonzero(self.raw_group_ids==gi) for gi in np.unique(self.raw_group_ids)}
        normpath=driver.F/'normalization.json';normsha=sha(normpath)
        if normsha!=self.m['normalization_sha256']:raise ValueError('Train-only normalization changed')
        if normsha not in _NORM_CACHE:
            norm=json.loads(normpath.read_text());_NORM_CACHE[normsha]=(np.asarray(norm['mean']),np.asarray(norm['std']))
        self.mean,self.std=_NORM_CACHE[normsha]
        self.grounded_root=Path(grounded_root or ROOT/'grounded_support/grounded_surfaces')
        grounded=load_grounded_surfaces(self.grounded_root,scene)
        gm=grounded.get('metadata',{})
        if gm:
            if gm.get('coordinate_frame')!='original_scene_vertex_ids':raise ValueError('Grounded mask frame mismatch')
            if gm.get('mesh_sha256')!=library.manifest['scenes'][scene]['sha256']:raise ValueError('Grounded mask mesh identity mismatch')
            if gm.get('mesh_vertices')!=len(self.xyz):raise ValueError('Grounded mask mesh vertex count mismatch')
        grounded_npz=self.grounded_root/(scene+'.npz');grounded_json=self.grounded_root/(scene+'.json')
        fingerprint=dict(source_mesh_identity=self.m['source_identity']['geometry_package'],
            alignment=self.alignment.tolist(),
            grounded_npz=sha(grounded_npz) if grounded_npz.exists() else None,
            grounded_json=sha(grounded_json) if grounded_json.exists() else None,
            region_code=sha(Path(__file__).with_name('support_regions.py')),pool_size=pool_size)
        self.scene_fingerprint=hashlib.sha256(json.dumps(fingerprint,sort_keys=True).encode()).hexdigest()
        self.cache_root=Path(cache_root or ROOT/'support_cache')/scene
        self.cache_root.mkdir(parents=True,exist_ok=True)
        scene_cache=self.cache_root/('scene_'+self.scene_fingerprint[:16]+'.npz')
        if scene_cache.exists():self.prepared=load_prepared_scene(scene_cache,self.xyz,self.scene_fingerprint)
        else:
            self.prepared=prepare_scene(self.xyz,scene,grounded,pool_size=pool_size)
            save_prepared_scene(scene_cache,self.prepared,self.scene_fingerprint)

    @property
    def labels_arrays(self):return self.source.labels_arrays

    def _asset(self, uid, dims):
        if uid not in self.asset_priors:
            v,f=self.library.asset(uid,dims)
            identity=dict(scene=self.scene_fingerprint,asset=self.library.manifest['assets'][uid]['sha256'],dims=np.asarray(dims).tolist())
            fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
            path=self.cache_root/(uid+'_'+fingerprint[:16]+'.npz')
            if path.exists():
                with np.load(path,allow_pickle=False) as z:
                    if str(z['fingerprint'].item())!=fingerprint:raise ValueError('Asset proposal cache mismatch')
                    asset=dict(features=np.asarray(z['features']).copy(),dims=np.asarray(dims),
                        scene=self.s,contact_geometry=bool(z['contact_geometry'].item()),footprint_samples=int(z['footprint_samples'].item()))
            else:
                asset=prepare_asset(self.prepared,dims,v,f)
                tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
                with tmp.open('wb') as handle:np.savez_compressed(handle,features=asset['features'],
                    fingerprint=np.array(fingerprint),contact_geometry=np.array(asset['contact_geometry']),
                    footprint_samples=np.array(asset['footprint_samples']))
                os.replace(tmp,path)
            self.asset_priors[uid]=asset
        return self.asset_priors[uid]

    def inputs(self, positions):
        rows=self.source.inputs(positions)
        for row in rows:
            original=np.asarray(row['candidate_ids'],np.int64);raw=self.raw[self.raw_rows[int(row['group_id'])]]
            if not self.original_features_validated:
                rebuilt,center=self.reconstruct(self.payload,original,original,raw,self.alignment,self.geometry_cache)
                feature=self.make_features(rebuilt,self.mean,self.std,torch.device('cpu')).numpy()
                error=float(np.max(np.abs(feature-row['features'])))
                if error>2e-4 or not np.allclose(center,row['centers'],atol=1e-5):
                    raise ValueError('Original feature reconstruction identity failed: '+str(error))
                self.feature_reconstruction_max_error=error;self.original_features_validated=True
            task={k:row[k] for k in ('scene','asset_scene','entry')}
            generated=build_task_candidates(self.prepared,self._asset(row['asset_uuid'],row['dims']),
                original,task,self.payload,raw,self.alignment,self.mean,self.std,self.geometry_cache,
                budget=self.budget,global_quota=self.global_quota)
            row.update(**generated,original_candidate_ids=original,
                       feature_reconstruction_max_error=self.feature_reconstruction_max_error)
        return rows

    def labels(self, rows):
        # Authoritative task/known-yaw supervision is opened only at this call,
        # after the observation-only rollout in the training driver.
        from distance_targets_v5 import build_distance_targets
        old=self.source.labels(rows);z=self.source.labels_arrays;out={}
        for row in rows:
            key=(self.s,row['group_id']);base=old[key];annotation=base['annotation']
            lo,hi=map(int,z['annotation_offsets'][annotation:annotation+2])
            valid_ids=np.asarray(z['valid_ids'][lo:hi],np.int64);valid_masks=np.asarray(z['valid_mask8'][lo:hi],bool)
            centers=np.asarray(row['centers']);ids=np.asarray(row['candidate_ids'],np.int64)
            identity=hashlib.sha256(ids.tobytes()).hexdigest()
            cached=self.label_cache.get(key)
            if cached is not None and cached['candidate_identity']==identity:out[key]=cached;continue
            positive=join_canonical_masks(ids,valid_ids,valid_masks)
            targets=build_distance_targets(centers,np.asarray(z['scene_xyz'][valid_ids]),valid_masks)
            label=dict(known=base['known'],positive=positive,rank=targets['rank_prob'],
                       offset_xy=targets['offset_xy_m'],offset_mask=targets['offset_mask'],
                       annotation=annotation,candidate_identity=identity)
            if np.any(positive & ~np.asarray(base['known'])[None]):raise ValueError('Positive yaw is undeclared')
            self.label_cache[key]=label;out[key]=label
        return out

    def close(self):
        self.source.close();self.raw=None;self.payload=None;self.xyz=None;self.prepared=None
        self.asset_priors.clear();self.geometry_cache.clear();self.label_cache.clear()
