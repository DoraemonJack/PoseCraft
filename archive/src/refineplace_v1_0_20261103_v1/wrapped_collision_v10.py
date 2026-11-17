"""V1.0: wrapped collision geometry with bitwise-original support samples.

All poses remain in the original canonical bottom-anchor frame. The wrap is a
collision proxy, not new task dimensions, a recovered material model, or a
replacement for the original official-evaluation asset. Fields are prepared
once and checksum-verified before each process's first use.
"""
from collections import OrderedDict
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy.spatial import cKDTree

HERE=Path(__file__).resolve().parent
PREVIOUS=HERE.parent/'refineplace_rl_geometry_10k_20261102_v1'/'joint_collision_orientation'
if str(PREVIOUS) not in sys.path:
    sys.path.insert(0,str(PREVIOUS))
for candidate in (Path('/path/to/home/codex_3d_research_20261028/rl_position_adjustment_v1'),HERE.parent/'rl_placement_v1'):
    if candidate.exists() and str(candidate) not in sys.path:sys.path.insert(0,str(candidate))
from joint_asset_geometry_v1 import (AssetMeshGeometry as PreviousGeometry,
    DIRECTIONAL_FEATURE_NAMES, DIRECTIONAL_FEATURE_DIM, _prepare_asset)
from asset_mesh_geometry_v6 import AssetMeshGeometry as OriginalGeometry, _mesh, _fcl_bvh

DEFAULT_WRAP=Path('/path/to/research300/asset_alpha_wrap3d_20261103_v1/wrapped_asset_manifest.json')
DEFAULT_FIELDS=Path('/path/to/research300/refineplace_v1_0_20261103_v1/wrapped_fields/manifest.json')
_INDEX={}
_CACHE=OrderedDict()


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as fp:
        for block in iter(lambda:fp.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def geometry_digest(vertices,faces):
    h=hashlib.sha256()
    for data,dtype in ((vertices,'<f8'),(faces,'<i8')):
        a=np.ascontiguousarray(data,dtype=dtype)
        h.update(np.asarray(a.shape,dtype='<i8').tobytes());h.update(a.tobytes())
    return h.hexdigest()


def _index(path):
    path=Path(path).resolve()
    if path not in _INDEX:
        data=json.loads(path.read_text(encoding='utf-8'))
        if data.get('version')!='RefinePlace_v1_wrapped_fields' or len(data['assets'])!=data['expected_assets']:
            raise ValueError('Incomplete/invalid wrapped field manifest')
        source=Path(data['wrapped_manifest_path'])
        if sha(source)!=data['wrapped_manifest_sha256']:
            raise ValueError('Wrapped asset manifest changed after preparation')
        lookup={r['original_geometry_digest']:r for r in data['assets'].values()}
        if len(lookup)!=len(data['assets']):raise ValueError('Ambiguous source geometry digests')
        _INDEX[path]=(data,lookup)
    return _INDEX[path]


def _load_proxy(original_vertices,original_faces,field_manifest=None):
    path=Path(field_manifest or os.environ.get('REFINEPLACE_WRAPPED_FIELDS',str(DEFAULT_FIELDS)))
    metadata,index=_index(path)
    key=geometry_digest(original_vertices,original_faces)
    if key not in index:
        raise ValueError('No checksum-admitted Alpha Wrap proxy for original asset geometry '+key)
    row=index[key]
    cache_key=(str(path.resolve()),row['wrapped_sha256'],row['field_sha256'])
    if cache_key in _CACHE:
        _CACHE.move_to_end(cache_key)
        return _CACHE[cache_key],True
    if sha(row['wrapped_path'])!=row['wrapped_sha256'] or sha(row['field_path'])!=row['field_sha256']:
        raise ValueError('Alpha Wrap proxy/field checksum mismatch: '+row['asset_uuid'])
    with np.load(row['wrapped_path'],allow_pickle=False) as z:
        vertices=np.asarray(z['vertices'],float);faces=np.asarray(z['faces'],np.int64)
    triangles=_mesh(vertices,faces,'wrapped_asset')
    with np.load(row['field_path'],allow_pickle=False) as z:
        field={k:np.asarray(z[k]).copy() for k in z.files}
    prepared=dict(row['field_metadata'])
    prepared.update(field)
    prepared['spacing']=float(prepared['spacing'])
    prepared['tree']=cKDTree(prepared['points'])
    for value in field.values():value.flags.writeable=False
    proxy=dict(vertices=vertices,faces=faces,triangles=triangles,prepared=prepared,row=row,bvh=None)
    _CACHE[cache_key]=proxy
    if len(_CACHE)>32:_CACHE.popitem(last=False)
    return proxy,False


class WrappedAssetMeshGeometry(PreviousGeometry):
    """Adapter-compatible core. Original support, wrapped collision only."""
    def __init__(self,scene_vertices,scene_faces,asset_vertices,asset_faces,*,
                 collision_tolerance=.02,collision_cost_scale=.03,directional_radius=.15,
                 interior_grid_side=64,max_local_scene_triangles=2048,
                 asset_surface_sample_count=4096,field_manifest=None,
                 wrapped_vertices=None,wrapped_faces=None,**kwargs):
        values=[collision_tolerance,collision_cost_scale,directional_radius]
        if not np.isfinite(values).all() or collision_tolerance<0 or min(values[1:])<=0:
            raise ValueError('Invalid collision tolerance/scale/radius')
        if not isinstance(max_local_scene_triangles,int) or max_local_scene_triangles<1:
            raise ValueError('Invalid local triangle budget')
        # This establishes exact original lower-envelope samples, contact band
        # and bottom anchor before any collision proxy is substituted.
        OriginalGeometry.__init__(self,scene_vertices,scene_faces,asset_vertices,asset_faces,**kwargs)
        self.original_asset_triangles=self.asset
        self.original_dimensions=np.ptp(np.asarray(asset_vertices,float),axis=0)
        self.original_geometry_digest=geometry_digest(asset_vertices,asset_faces)
        if wrapped_vertices is None and wrapped_faces is None:
            proxy,hit=_load_proxy(asset_vertices,asset_faces,field_manifest)
            self.asset=proxy['triangles']
            self.joint_asset=proxy['prepared']
            self.joint_asset_cache_hit=hit
            self.proxy_metadata=dict(proxy['row'])
            if self.collision_backend=='fcl':
                if proxy['bvh'] is None:proxy['bvh']=_fcl_bvh(self.asset)
                self._asset_bvh=proxy['bvh']
        elif wrapped_vertices is not None and wrapped_faces is not None:
            # Explicit fixture path: never used to bypass production admission.
            self.asset=_mesh(wrapped_vertices,wrapped_faces,'wrapped_fixture')
            self.joint_asset,self.joint_asset_cache_hit=_prepare_asset(
                self.asset,interior_grid_side,asset_surface_sample_count)
            if self.collision_backend=='fcl':self._asset_bvh=_fcl_bvh(self.asset)
            self.proxy_metadata={'asset_uuid':'explicit_fixture','parameters':{},'wrapped_sha256':None}
        else:
            raise ValueError('Both wrapped vertices and faces required')
        if not self.joint_asset['closed'] or not self.joint_asset['recovery']['full_surface_coverage']:
            raise ValueError('Admitted Alpha Wrap must have full closed-edge topology')
        self.collision_tolerance=float(collision_tolerance)
        self.collision_cost_scale=float(collision_cost_scale)
        self.directional_radius=float(directional_radius)
        self.max_local_scene_triangles=max_local_scene_triangles

    def query(self,center,yaw):
        # posed_asset now rotates the wrap and original support samples around
        # the same unmodified source anchor. No support ray changes occur.
        stats=OriginalGeometry.query(self,center,yaw)
        extra=self.joint_signals(center,yaw)
        if set(stats)&set(extra):raise RuntimeError('Unexpected original/new stats overlap')
        stats.update(extra)
        stats.update(
            collision_evidence='alpha_wrapped_asset_triangles_vs_observed_scene_triangles',
            collision_cost_provenance='observed_scene_samples_inside_closed_alpha_wrap_voxel_proxy',
            directional_proximity_provenance='unsigned_scene_samples_to_alpha_wrap_surface_samples',
            collision_proxy_kind='MeshLab_CGAL_Alpha_Wrap',
            collision_proxy_asset_uuid=self.proxy_metadata['asset_uuid'],
            collision_proxy_file_sha256=self.proxy_metadata.get('wrapped_sha256'),
            collision_proxy_parameters=self.proxy_metadata.get('parameters',{}),
            collision_proxy_original_geometry_digest=self.original_geometry_digest,
            collision_proxy_original_dimensions_m=self.original_dimensions.tolist(),
            collision_proxy_original_anchor_preserved=True,
            collision_proxy_recentering_applied=False,
            collision_proxy_holes_filled=True,
            collision_proxy_geometry_semantics='closed_envelope_not_recovered_material_truth',
            support_evidence='unchanged_original_asset_lower_envelope_vs_observed_scene',
            support_rule_preserved=True)
        return stats


def install_geometry(adapter_module=None):
    if adapter_module is None:
        import mesh_scene_adapter_v6 as adapter_module
    adapter_module.AssetMeshGeometry=WrappedAssetMeshGeometry
    return adapter_module


AssetMeshGeometry=WrappedAssetMeshGeometry
