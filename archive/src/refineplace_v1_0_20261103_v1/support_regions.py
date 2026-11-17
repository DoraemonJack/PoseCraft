"""Observation-only, size/yaw-aware support-region proposal priors.

Grounded SAM masks identify possible receivers. Metric scene points establish
near-horizontal observed surface patches. Raster holes are never filled, and
the priors never become hard physical labels. Original scene point identities
are retained so downstream feature/annotation joins remain exact.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

FRAME = 'axis_aligned_scene_mean_centered'
FEATURE_NAMES = (
    'horizontal_normal_abs_z', 'local_plane_confidence',
    'observed_bottom_coverage', 'minimum_bottom_edge_margin_scaled',
    'observed_body_surface_fraction', 'grounded_receiver_confidence',
    'grounded_receiver_available', 'region_observed_area_scaled',
    'region_width_over_rotated_asset_width', 'region_depth_over_rotated_asset_depth',
    'sin_yaw', 'cos_yaw', 'original_mesh_contact_samples_available',
    'receiver_preference_weight', 'global_backup_candidate',
)


def save_prepared_scene(path, scene, fingerprint):
    """Own-format numerical cache; never pickle scene or annotation objects."""
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    arrays={k:scene[k] for k in ('pool_ids','normal','planar','assignment','occupancy_keys',
                                'occupancy_origin','occupancy_shape')}
    metadata=dict(fingerprint=fingerprint,scene=scene['scene'],frame=scene['frame'],
                  occupancy_cell=scene['occupancy_cell'],report=scene['report'],regions=[])
    for i,r in enumerate(scene['regions']):
        m={k:r[k] for k in ('cell','height','area','confidence','maximum_confidence','grounded_fraction','label_id','label')}
        for k in ('origin','mask','margin','extent','pool_rows'):
            arrays[f'r{i}_{k}']=r[k]
        metadata['regions'].append(m)
    arrays['metadata_json']=np.array(json.dumps(metadata))
    tmp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    with tmp.open('wb') as fp:np.savez_compressed(fp,**arrays)
    os.replace(tmp,path)


def load_prepared_scene(path, xyz, fingerprint):
    with np.load(path,allow_pickle=False) as z:
        meta=json.loads(str(z['metadata_json'].item()))
        if meta['fingerprint']!=fingerprint:raise ValueError('Scene proposal cache fingerprint mismatch')
        out={k:np.asarray(z[k]).copy() for k in ('pool_ids','normal','planar','assignment','occupancy_keys','occupancy_origin','occupancy_shape')}
        regions=[]
        for i,m in enumerate(meta['regions']):
            r=dict(m)
            for k in ('origin','mask','margin','extent','pool_rows'):r[k]=np.asarray(z[f'r{i}_{k}']).copy()
            regions.append(r)
    out.update(xyz=np.asarray(xyz,np.float64),scene=meta['scene'],frame=meta['frame'],
               occupancy_cell=meta['occupancy_cell'],report=meta['report'],regions=regions)
    return out


def seed_for(key):
    return int.from_bytes(hashlib.sha256(str(key).encode()).digest()[:8], 'little')


def load_grounded_surfaces(root, scene):
    """Load detector associations only; never fall back to GT instance types."""
    root = Path(root)
    npz, meta = root/(scene+'.npz'), root/(scene+'.json')
    if not npz.exists():
        return dict(scene=scene, point_ids=np.empty(0,np.int64), label_ids=np.empty(0,np.int16),
                    confidence=np.empty(0,np.float32), labels=[], available=False,
                    source='grounding_dino_sam2', missing_reason='no_predicted_receiver_file')
    header = json.loads(meta.read_text(encoding='utf-8')) if meta.exists() else {}
    if header.get('scene', scene) != scene:
        raise ValueError('Grounded receiver scene identity mismatch')
    if header.get('npz_sha256') and hashlib.sha256(npz.read_bytes()).hexdigest()!=header['npz_sha256']:
        raise ValueError('Grounded receiver association checksum mismatch')
    with np.load(npz, allow_pickle=False) as z:
        out = {k:np.asarray(z[k]).copy() for k in ('point_ids','label_ids','confidence')}
        if 'observation_counts' in z:
            out['observation_counts'] = np.asarray(z['observation_counts']).copy()
    out.update(scene=scene, labels=header.get('labels', []), available=bool(len(out['point_ids'])),
               source=header.get('source','grounding_dino_sam2'), metadata=header)
    return out


def _occupancy_query(keys, points, origin, cell, shape):
    q = np.floor((points-origin)/cell).astype(np.int64)
    good = np.all((q>=0)&(q<shape), axis=-1)
    linear = (q[...,0]*shape[1]+q[...,1])*shape[2]+q[...,2]
    p = np.searchsorted(keys, linear)
    return good & (p<len(keys)) & (keys[np.minimum(p,len(keys)-1)] == linear)


def prepare_scene(xyz, scene, grounded=None, *, pool_size=8192, cell_size=.04,
                  max_regions=32, normal_k=16):
    """Prepare reusable geometry without any object or placement annotation."""
    xyz=np.asarray(xyz,np.float64)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or len(xyz)<normal_k or not np.isfinite(xyz).all():
        raise ValueError('Finite metric aligned scene XYZ required')
    rng=np.random.default_rng(seed_for(scene+':support_regions_v1'))
    n=len(xyz); tree=cKDTree(xyz)
    candidate_pool=rng.choice(n,min(pool_size,n),replace=False)
    semantic=np.zeros(n,np.float32); semantic_label=np.full(n,-1,np.int16)
    label_names=[]
    if grounded is not None:
        if grounded.get('scene')!=scene or grounded.get('source') not in ('grounding_dino_sam2','GroundingDINO_SAM2'):
            raise ValueError('Predicted receiver provenance mismatch')
        ids=np.asarray(grounded['point_ids']); confidence=np.asarray(grounded['confidence'],float)
        labels=np.asarray(grounded['label_ids'])
        if (ids.ndim!=1 or ids.dtype.kind not in 'iu' or ids.shape!=confidence.shape or ids.shape!=labels.shape
                or np.any(ids<0) or np.any(ids>=n) or not np.isfinite(confidence).all()
                or np.any(confidence<0) or np.any(confidence>1)):
            raise ValueError('Invalid image-derived receiver point identities/confidence')
        # Resolve overlapping masks explicitly. NumPy repeated-index assignment
        # does not define an iteration order, so assign each point only once.
        order=np.lexsort((np.arange(len(ids)),confidence,ids))
        ordered_ids=ids[order]
        last=np.r_[ordered_ids[:-1]!=ordered_ids[1:],True] if len(order) else np.empty(0,bool)
        winners=order[last]
        semantic[ids[winners]]=confidence[winners]; semantic_label[ids[winners]]=labels[winners]
        label_names=list(grounded.get('labels',[]))
        predicted=np.unique(ids)
        if len(predicted):
            extra=rng.choice(predicted,min(pool_size//2,len(predicted)),replace=False)
            candidate_pool=np.unique(np.r_[candidate_pool,extra])
    _, nn=tree.query(xyz[candidate_pool],k=normal_k,workers=1)
    neighborhood=xyz[nn]-xyz[nn].mean(1,keepdims=True)
    values,vectors=np.linalg.eigh(np.einsum('nki,nkj->nij',neighborhood,neighborhood)/normal_k)
    normal=np.abs(vectors[:,2,0])
    planar=np.clip(1-12*values[:,0]/np.maximum(values.sum(-1),1e-10),0,1)
    horizontal=(normal>=.85)&(planar>=.7)
    # Candidate height modes are observations; broad surface bounds are never
    # filled as a convex hull. Each occupied raster island remains separate.
    z=xyz[candidate_pool,2]
    zbase=float(xyz[:,2].min())-.05
    zbin=np.floor((z-zbase)/.05).astype(int)
    hist=np.bincount(zbin[horizontal], minlength=int(zbin.max())+1)
    smoothed=ndimage.gaussian_filter1d(hist.astype(float),.7)
    peaks=np.flatnonzero((smoothed==ndimage.maximum_filter1d(smoothed,3)) & (smoothed>=3))
    peaks=sorted(peaks,key=lambda b:-smoothed[b])[:max_regions]
    regions=[]; assignment=np.full(len(candidate_pool),-1,np.int32)
    for peak in peaks:
        seedmask=horizontal & (np.abs(zbin-peak)<=1)
        if seedmask.sum()<4: continue
        height=float(np.median(z[seedmask])); band=.035
        point_ids=np.flatnonzero(np.abs(xyz[:,2]-height)<=band)
        if len(point_ids)<16: continue
        xy=xyz[point_ids,:2]; origin=xy.min(0)-cell_size
        shape=np.ceil((xy.max(0)-origin)/cell_size).astype(int)+2
        if np.any(shape>1024): continue
        cells=np.floor((xy-origin)/cell_size).astype(int)
        mask=np.zeros(tuple(shape),bool);mask[cells[:,0],cells[:,1]]=True
        components,count=ndimage.label(mask,np.ones((3,3),np.int8))
        counts=np.bincount(components.ravel())
        for label in np.argsort(counts[1:])[::-1]+1:
            if counts[label]<6: break
            component_mask=components==label
            q=np.floor((xyz[candidate_pool,:2]-origin)/cell_size).astype(int)
            inbounds=np.all((q>=0)&(q<shape),axis=1)
            selected=np.flatnonzero(inbounds & (np.abs(z-height)<=band) & horizontal)
            selected=selected[component_mask[q[selected,0],q[selected,1]]]
            if len(selected)<2: continue
            # Keep the first stronger height hypothesis when bands overlap.
            selected=selected[assignment[selected]<0]
            if not len(selected): continue
            rids=point_ids[components[cells[:,0],cells[:,1]]==label]
            extent=np.ptp(xyz[rids,:2],axis=0)
            finite_sem=semantic[rids]; best_label=-1
            if finite_sem.max(initial=0)>0:
                weighted=np.bincount(semantic_label[rids][finite_sem>0],weights=finite_sem[finite_sem>0])
                best_label=int(weighted.argmax())
            region=dict(origin=origin,cell=cell_size,mask=component_mask,
                        margin=ndimage.distance_transform_edt(component_mask)*cell_size,
                        height=height,extent=extent,area=float(counts[label]*cell_size**2),
                        confidence=float(finite_sem.mean()), maximum_confidence=float(finite_sem.max(initial=0)),
                        grounded_fraction=float((finite_sem>0).mean()), label_id=best_label,
                        label=label_names[best_label] if 0<=best_label<len(label_names) else 'unobserved_receiver',
                        point_ids=rids, pool_rows=selected)
            assignment[selected]=len(regions); regions.append(region)
            if len(regions)>=max_regions: break
        if len(regions)>=max_regions: break
    occupancy_cell=.04; occupancy_origin=xyz.min(0)-occupancy_cell
    voxel=np.floor((xyz-occupancy_origin)/occupancy_cell).astype(np.int64)
    occupancy_shape=voxel.max(0)+2
    keys=np.unique((voxel[:,0]*occupancy_shape[1]+voxel[:,1])*occupancy_shape[2]+voxel[:,2])
    return dict(scene=scene,frame=FRAME,xyz=xyz,pool_ids=candidate_pool,
                normal=normal.astype(np.float32),planar=planar.astype(np.float32),regions=regions,
                assignment=assignment,point_semantic_confidence=semantic,
                occupancy_keys=keys,occupancy_origin=occupancy_origin,occupancy_cell=occupancy_cell,
                occupancy_shape=occupancy_shape,
                report=dict(scene=scene,scene_points=n,pool_points=len(candidate_pool),regions=len(regions),
                    grounded_points=int(np.count_nonzero(semantic)),grounded_available=bool(np.any(semantic)),
                    geometry_role='observed_soft_prior_not_hard_feasibility',convex_hull_filled=False,
                    labels_read=False,instance_types_read=False))


def _footprint(dims,vertices=None,faces=None,count=49):
    if vertices is not None and faces is not None:
        v=np.asarray(vertices,float); f=np.asarray(faces,int)
        tri=v[f]; cross=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0]); area=np.linalg.norm(cross,axis=1)
        low=(tri[:,:,2].max(1)<=v[:,2].min()+.025)&(np.abs(cross[:,2])>=.5*area)&(area>1e-12)
        if low.any():
            rng=np.random.default_rng(419); eligible=np.flatnonzero(low)
            ids=rng.choice(eligible,count,replace=True,p=area[eligible]/area[eligible].sum())
            uv=rng.random((count,2));uv[uv.sum(1)>1]=1-uv[uv.sum(1)>1]
            t=tri[ids];p=t[:,0]+uv[:,:1]*(t[:,1]-t[:,0])+uv[:,1:]*(t[:,2]-t[:,0])
            return p[:,:2],True
    axis=np.linspace(-.5,.5,7)
    x,y=np.meshgrid(axis*dims[0],axis*dims[1],indexing='ij')
    return np.column_stack((x.ravel(),y.ravel())),False


def prepare_asset(scene,dims,vertices=None,faces=None):
    """Cache pose-wise weak geometry once per scene/asset, not once per task."""
    dims=np.asarray(dims,float)
    if dims.shape!=(3,) or not np.isfinite(dims).all() or np.any(dims<=0):
        raise ValueError('Positive metric dimensions required')
    points=scene['xyz'][scene['pool_ids']]; n=len(points)
    features=np.zeros((n,8,len(FEATURE_NAMES)),np.float32)
    features[:,:,0]=scene['normal'][:,None];features[:,:,1]=scene['planar'][:,None]
    bottom,contact_geometry=_footprint(dims,vertices,faces)
    axis=np.linspace(-.45,.45,3); bx,by=np.meshgrid(axis*dims[0],axis*dims[1],indexing='ij')
    body=np.column_stack((bx.ravel(),by.ravel()))
    for b in range(8):
        angle=b*np.pi/4;c,s=np.cos(angle),np.sin(angle);rotation=np.array([[c,-s],[s,c]])
        rotated=bottom@rotation.T; body_xy=body@rotation.T
        features[:,b,10]=s;features[:,b,11]=c;features[:,b,12]=contact_geometry
        for region in scene['regions']:
            ids=region['pool_rows']; query=points[ids,None,:2]+rotated[None]
            q=np.floor((query-region['origin'])/region['cell']).astype(np.int64)
            shape=np.array(region['mask'].shape);good=np.all((q>=0)&(q<shape),axis=-1)
            qc=np.clip(q,0,shape-1)
            supported=good & region['mask'][qc[...,0],qc[...,1]]
            margin=np.where(good,region['margin'][qc[...,0],qc[...,1]],0.)
            features[ids,b,2]=supported.mean(1)
            features[ids,b,3]=np.clip(margin.min(1)/max(min(dims[:2]),.05),0,2)
            features[ids,b,5]=region['maximum_confidence']*region['grounded_fraction']
            features[ids,b,6]=region['grounded_fraction']>0
            features[ids,b,7]=np.clip(region['area']/max(dims[0]*dims[1],.01),0,10)/10
            projected=np.abs(rotation)@dims[:2]
            features[ids,b,8:10]=np.minimum(region['extent']/np.maximum(projected,1e-6),3)/3
        # Sparse height-aware occupied-surface observations. Absence is not
        # certified clearance, and this is never used as a binary exclusion.
        qs=[]
        for h in (.15,.5,.85):
            q=np.broadcast_to(points[:,None,:],(n,len(body_xy),3)).copy()
            q[:,:,:2]+=body_xy[None];q[:,:,2]+=dims[2]*h;qs.append(q)
        query=np.concatenate(qs,axis=1)
        hit=_occupancy_query(scene['occupancy_keys'],query,scene['occupancy_origin'],scene['occupancy_cell'],scene['occupancy_shape'])
        features[:,b,4]=hit.mean(1)
    return dict(features=features,dims=dims,contact_geometry=contact_geometry,
                scene=scene['scene'],footprint_samples=len(bottom))


def propose(scene,asset,baseline_ids,task,*,budget=256,global_quota=32,receiver_preferences=None):
    """Keep a mandatory global backup; never remove unknown regions outright."""
    if task['scene']!=scene['scene'] or asset['scene']!=scene['scene']:
        raise ValueError('Task/scene mismatch')
    xyz=scene['xyz']; original=np.asarray(baseline_ids,np.int64)
    if (budget<global_quota or budget>len(xyz) or global_quota<1 or original.ndim!=1
            or np.any(original<0) or np.any(original>=len(xyz))):
        raise ValueError('Invalid candidate budget/identities')
    seed=seed_for((task['scene'],task.get('asset_scene'),task.get('entry'),'support_v1'))
    rng=np.random.default_rng(seed);features=asset['features'].copy()
    prefs=receiver_preferences or {}
    for region in scene['regions']:
        weight=float(prefs.get(region['label'],1.))
        if not np.isfinite(weight) or not 0<=weight<=2: raise ValueError('Preference weight must be in [0,2]')
        features[region['pool_rows'],:,13]=weight/2
    # Preserve the original observation-based sampler as global backup.
    available=np.unique(original)
    if len(available)<global_quota:available=np.arange(len(xyz))
    chosen=rng.choice(available,global_quota,replace=False).tolist();origins=['global_backup']*len(chosen)
    pool=scene['pool_ids']; score=(features[:,:,0]+features[:,:,1]+2*features[:,:,2]
        +.5*features[:,:,3]-features[:,:,4]+.5*features[:,:,5]+.2*features[:,:,13]).max(1)
    # Round-robin region queues protect small but relevant receivers from a
    # large floor patch taking every regional slot.
    queues=[]
    for region in scene['regions']:
        ids=region['pool_rows'];priority=score[ids]+.05*rng.random(len(ids))
        queues.append(ids[np.argsort(-priority,kind='stable')].tolist())
    chosen_set=set(chosen); cursor=[0]*len(queues);seen_cells=set();cell=np.r_[np.clip(asset['dims'][:2]*.25,.04,.15),.06]
    while len(chosen)<budget and any(cursor[i]<len(q) for i,q in enumerate(queues)):
        for i,q in enumerate(queues):
            while cursor[i]<len(q):
                row=q[cursor[i]];cursor[i]+=1;pid=int(pool[row]);key=tuple(np.floor(xyz[pid]/cell).astype(int))
                if pid in chosen_set or key in seen_cells:continue
                chosen.append(pid);chosen_set.add(pid);seen_cells.add(key);origins.append('regional');break
            if len(chosen)==budget:break
    for row in np.argsort(-score,kind='stable'):
        if len(chosen)==budget:break
        pid=int(pool[row])
        if pid not in chosen_set:chosen.append(pid);chosen_set.add(pid);origins.append('geometry_fallback')
    if len(chosen)<budget:
        remainder=np.setdiff1d(np.arange(len(xyz)),np.asarray(chosen))
        extra=rng.choice(remainder,budget-len(chosen),replace=False).tolist();chosen+=extra;origins+=['uniform_fallback']*len(extra)
    ids=np.asarray(chosen,np.int64);selected=np.zeros((budget,8,len(FEATURE_NAMES)),np.float32)
    positions={int(pid):i for i,pid in enumerate(pool)}
    for j,pid in enumerate(ids):
        if int(pid) in positions:selected[j]=features[positions[int(pid)]]
        else:
            selected[j,:,10]=np.sin(np.arange(8)*np.pi/4);selected[j,:,11]=np.cos(np.arange(8)*np.pi/4)
        selected[j,:,14]=origins[j]=='global_backup'
    if len(np.unique(ids))!=budget or not np.isfinite(selected).all():raise ValueError('Invalid proposed candidate set')
    report=dict(scene['report'],budget=budget,global_backup=global_quota,
        origin_counts={k:origins.count(k) for k in sorted(set(origins))},
        new_point_count=int((~np.isin(ids,original)).sum()),contact_samples_from_mesh=asset['contact_geometry'],
        feature_names=list(FEATURE_NAMES),hard_geometry_filter=False)
    return ids,selected,report
