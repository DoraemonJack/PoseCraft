"""Actual-mesh policy adapter; its 26 feature labels replace V4 box features."""
import hashlib
import json
import math
from pathlib import Path
from collections import OrderedDict
import numpy as np
from asset_mesh_geometry_v6 import AssetMeshGeometry


def _sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda:handle.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


class AssetLibrary:
    def __init__(self,path):
        self.path=Path(path)
        self.manifest=json.loads(self.path.read_text(encoding='utf-8'))
        self.manifest_sha256=_sha(self.path)
        if self.manifest.get('test_reads') != 0:raise ValueError('Train/dev-only mesh manifest required')
        self.assets={};self._verified=set();self.scene_alignment_audits={}
        train=set(self.manifest['train_scenes']);dev=set(self.manifest['development_scenes'])
        if train & dev or {s[:9] for s in train} & {s[:9] for s in dev}:
            raise ValueError('Mesh train/development room overlap')
        self.geometry_config=dict(self.manifest['geometry_config'])
        if self.geometry_config.get('collision_backend') not in ('sat','fcl'):
            raise ValueError('Explicit mesh collision backend required')

    def _file(self,kind,key):
        if key not in self.manifest[kind]:raise ValueError('Missing mesh manifest '+kind+': '+key)
        entry=self.manifest[kind][key];path=Path(entry['path'])
        if not path.is_absolute() or not path.is_file():raise FileNotFoundError('Missing absolute mesh path: '+str(path))
        token=(kind,key)
        if token not in self._verified:
            if _sha(path) != entry['sha256']:raise ValueError('Mesh checksum mismatch: '+str(path))
            self._verified.add(token)
        return path,entry

    def validate_required(self,scenes,assets):
        allowed=set(self.manifest['train_scenes'])|set(self.manifest['development_scenes'])
        if not set(scenes).issubset(allowed):raise ValueError('Selected scene absent from train/dev manifest')
        for scene in scenes:
            if scene not in self.manifest['alignments']:raise ValueError('Missing mesh alignment: '+scene)
            self._file('scenes',scene)
        for uuid in assets:self._file('assets',uuid)

    def asset(self,uuid,dims):
        dims=np.asarray(dims,float)
        path,entry=self._file('assets',uuid)
        if uuid not in self.assets:
            with np.load(path,allow_pickle=False) as z:
                vertices=np.asarray(z['vertices'],float);faces=np.asarray(z['faces'])
            if vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices) or not np.isfinite(vertices).all():raise ValueError('Invalid asset vertices: '+uuid)
            if abs(vertices[:,2].min())>1e-6 or np.any(abs((vertices[:,:2].max(0)+vertices[:,:2].min(0))/2)>1e-4):raise ValueError('Asset mesh not bottom-centered: '+uuid)
            actual=np.ptp(vertices,axis=0)
            if not np.allclose(actual,entry['dimensions_m'],atol=1e-4,rtol=0):raise ValueError('Asset manifest dimensions mismatch: '+uuid)
            self.assets[uuid]=(vertices,faces,actual)
        vertices,faces,actual=self.assets[uuid]
        if dims.shape != (3,) or not np.isfinite(dims).all() or not np.allclose(actual,dims,atol=1e-4,rtol=0):raise ValueError('Asset task dimensions mismatch: '+uuid)
        return vertices,faces


def load_mesh_manifest(path):return AssetLibrary(path)


def load_scene_mesh(base,scene,split,library,reference_xyz=None):
    if split not in ('train','val'):raise ValueError('Only train/val mesh reads permitted')
    allowed=library.manifest['train_scenes' if split=='train' else 'development_scenes']
    if scene not in allowed:raise ValueError('Scene mesh split mismatch: '+scene)
    path,entry=library._file('scenes',scene)
    with np.load(path,allow_pickle=False) as z:
        vertices=np.asarray(z['vertices'],float);faces=np.asarray(z['faces'])
    if (vertices.ndim != 2 or vertices.shape[1:] != (3,) or not len(vertices) or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1:] != (3,) or faces.dtype.kind not in 'iu'
            or np.any(faces<0) or np.any(faces>=len(vertices))):raise ValueError('Invalid scene mesh: '+scene)
    if len(vertices)!=entry['vertex_count'] or len(faces)!=entry['face_count']:raise ValueError('Scene mesh count mismatch: '+scene)
    alignment=np.asarray(library.manifest['alignments'][scene],float)
    from pose_targets_v4 import validate_pure_yaw_alignment
    alignment=validate_pure_yaw_alignment(alignment)
    if reference_xyz is None:
        import torch
        reference_path=Path(base)/'scans_full_refer_v1'/split/(scene+'_refer.pth')
        raw=torch.load(reference_path,map_location='cpu',weights_only=False)
        reference_xyz=raw[0]
    reference=np.asarray(reference_xyz)
    if reference.dtype != np.float32 or reference.shape != vertices.shape or not np.isfinite(reference).all():
        raise ValueError('Expected finite float32 original reference vertices with identical shape: '+scene)
    # Preserve the official float32 centering, whose accumulated mean can differ
    # from a newly computed float64 mean. Verify original vertex identity first:
    # every indexed vertex must differ by one common translation, never a warp.
    delta=vertices-reference.astype(np.float64)
    translation=np.median(delta,axis=0)
    maximum_residual=float(np.linalg.norm(delta-translation,axis=1).max())
    if maximum_residual>1e-5:
        raise ValueError('Mesh/reference vertex order or geometry differs beyond common translation: '+scene+' residual='+str(maximum_residual))
    audit=dict(scene=scene,method='verified_indexwise_common_translation_then_original_float32_reference_rotation',
               raw_minus_reference_translation_m=translation.tolist(),max_translation_residual_m=maximum_residual,
               translation_minus_float64_vertex_mean_m=(translation-vertices.mean(0)).tolist(),
               vertex_count=len(vertices),residual_tolerance_m=1e-5)
    library.scene_alignment_audits[scene]=audit
    library.last_scene_alignment_audit=audit
    return (reference.astype(np.float64) @ alignment[:3,:3].T).astype(np.float32),faces


def costs_with_reliability(stats):
    known=stats['support_known_fraction']
    # A detected crossing refutes clearance even if contact enumeration capped.
    collision=float(stats['collision_noncontact_pairs']>0)
    reliable=float(stats['scene_geometry_available'] and (stats['collision_complete'] or collision>0))
    return (np.array([collision,stats['support_gap_fraction']/known if known else 0.],np.float32),
            np.array([reliable,known],np.float32))


class MeshSceneAdapterV6:
    feature_labels=('mesh_contact_fraction','mesh_known_fraction','mesh_gap_fraction','mesh_unknown_fraction',
        'log_surface_pairs','log_noncontact_pairs','log_contact_band_pairs','collision_reliability','support_reliability',
        'size_x_m','size_y_m','size_z_m','sin_yaw','cos_yaw','center_x_m','center_y_m','center_z_m',
        'log_bottom_samples','contact_tolerance_m','support_search_m','collision_complete','scene_mesh_available',
        'has_noncontact_crossing','contact_fraction_of_known','gap_fraction_of_known','actual_mesh_evidence')
    feature_dim=len(feature_labels)
    def __init__(self,xyz,faces,library):
        self.library=library;self.xyz=np.asarray(xyz,float);self.faces=np.asarray(faces)
        self.alignment_audit=dict(getattr(library,'last_scene_alignment_audit',{}))
        self.config=dict(library.geometry_config)
        backend=self.config.get('collision_backend','sat')
        self.prepared=AssetMeshGeometry.prepare_scene(self.xyz,self.faces,backend)
        self._cores=OrderedDict();self._pose_cache=OrderedDict();self.uuid=None;self.dims=None
    def set_asset(self,uuid,dims):
        vertices,faces=self.library.asset(uuid,dims)
        self.uuid=uuid;self.dims=np.asarray(dims,float).copy()
        if uuid not in self._cores:
            self._cores[uuid]=AssetMeshGeometry(self.xyz,self.faces,vertices,faces,prepared_scene=self.prepared,**self.config)
            if len(self._cores)>32:self._cores.popitem(last=False)
        self._cores.move_to_end(uuid)
        self.core=self._cores[uuid]
    def query(self,center,dims,yaw):
        if self.uuid is None:raise ValueError('set_asset required before mesh query')
        if not np.allclose(dims,self.dims,atol=1e-4,rtol=0):raise ValueError('Query dimensions differ from selected actual asset')
        center=np.asarray(center,float)
        key=(self.uuid,tuple(center),float(yaw))
        if key not in self._pose_cache:
            stats=self.core.query(center,yaw)
            costs,reliability=costs_with_reliability(stats)
            stats.update(plane_contact_fraction=stats['support_contact_fraction'],plane_valid_fraction=stats['support_known_fraction'],
                         plane_gap_fraction=stats['support_gap_fraction'],penetration_cost=float(costs[0]),
                         penetration_cost_alias='binary_noncontact_triangle_crossing_not_depth',
                         collision_reliability=float(reliability[0]),support_reliability=float(reliability[1]),
                         evidence='actual_asset_and_scene_triangle_mesh',feature_labels=self.feature_labels,
                         scene_alignment_audit=self.alignment_audit)
            self._pose_cache[key]=stats
            if len(self._pose_cache)>512:self._pose_cache.popitem(last=False)
        self._pose_cache.move_to_end(key)
        return self._pose_cache[key]
    def features(self,center,dims,yaw):
        s=self.query(center,dims,yaw);known=s['support_known_fraction']
        values=[s['support_contact_fraction'],known,s['support_gap_fraction'],s['support_unknown_fraction'],
                math.log1p(s['mesh_surface_intersections']),math.log1p(s['collision_noncontact_pairs']),math.log1p(s['collision_contact_band_pairs']),
                s['collision_reliability'],s['support_reliability'],*np.asarray(dims,float),math.sin(yaw),math.cos(yaw),*np.asarray(center,float),
                math.log1p(s['support_sample_count']),s['contact_tolerance_m'],s['support_search_m'],float(s['collision_complete']),float(s['scene_geometry_available']),
                float(s['collision_noncontact_pairs']>0),s['support_contact_fraction']/known if known else 0.,s['support_gap_fraction']/known if known else 0.,1.]
        return np.asarray(values,np.float32),s,1
    def snap_height(self,center,dims,yaw,max_change=.03):
        return np.asarray(center,float).copy(),dict(snapped=False,reason='actual_asset_bottom_frame_preserved',height_change=0.,query_count=0,spatial_queries=0)
    def project_move(self,oldcenter,oldyaw,proposedcenter,proposedyaw,dims):
        stats=self.query(proposedcenter,dims,proposedyaw)
        return np.asarray(proposedcenter,float).copy(),float(proposedyaw),stats,dict(accepted_fraction=1.,query_count=1,heuristic_guard=False,mesh_constraints='soft_reward')
