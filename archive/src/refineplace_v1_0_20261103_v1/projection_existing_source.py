"""Per-detection RGB-D projection boundary; no GT or identity fusion.

Inputs are already predicted masks, RGB/depth intrinsics and camera-to-world
pose. Color/depth extrinsics must be supplied or explicitly left approximate.
Geometry-ready and policy-ready are different states.
"""
import numpy as np


def project_instances(masks, depth_m, depth_intrinsics, color_intrinsics,
                      world_from_depth, *, color_from_depth=None,
                      policy_from_world=None, calibration_verified=False):
    masks=np.asarray(masks);depth=np.asarray(depth_m,dtype=float)
    if masks.ndim!=3 or masks.dtype!=bool or depth.ndim!=2:
        raise ValueError('Boolean masks [detections,H,W] and metric depth [H,W] required')
    kd=np.asarray(depth_intrinsics,float);kc=np.asarray(color_intrinsics,float)
    pose=np.asarray(world_from_depth,float)
    for k in (kd,kc):
        if k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0:
            raise ValueError('Finite camera intrinsics required')
    def transform(t):
        t=np.asarray(t,float)
        if t.shape!=(4,4) or not np.isfinite(t).all() or not np.allclose(t[3],[0,0,0,1]):
            raise ValueError('Finite homogeneous transform required')
        if not np.allclose(t[:3,:3].T@t[:3,:3],np.eye(3),atol=1e-5) or not np.isclose(np.linalg.det(t[:3,:3]),1,atol=1e-5):
            raise ValueError('Rigid proper rotation required')
        return t
    pose=transform(pose);extrinsic=transform(color_from_depth) if color_from_depth is not None else np.eye(4)
    target=transform(policy_from_world) if policy_from_world is not None else None
    if calibration_verified and color_from_depth is None:
        raise ValueError('Verified calibration requires explicit color/depth extrinsics')
    yy,xx=np.nonzero(np.isfinite(depth)&(depth>.05)&(depth<10));z=depth[yy,xx]
    camera=np.c_[(xx-kd[0,2])*z/kd[0,0],(yy-kd[1,2])*z/kd[1,1],z]
    color=camera@extrinsic[:3,:3].T+extrinsic[:3,3]
    front=color[:,2]>0;uv=np.zeros((len(color),2),dtype=int)
    uv[front,0]=np.rint(color[front,0]/color[front,2]*kc[0,0]+kc[0,2]).astype(int)
    uv[front,1]=np.rint(color[front,1]/color[front,2]*kc[1,1]+kc[1,2]).astype(int)
    inside=front&(uv[:,0]>=0)&(uv[:,0]<masks.shape[2])&(uv[:,1]>=0)&(uv[:,1]<masks.shape[1])
    world=camera@pose[:3,:3].T+pose[:3,3]
    out=[]
    for i,mask in enumerate(masks):
        selected=np.zeros(len(camera),bool);selected[inside]=mask[uv[inside,1],uv[inside,0]]
        raw=world[selected];policy=(raw@target[:3,:3].T+target[:3,3]) if target is not None else None
        sufficient=len(raw)>=3;ready=bool(sufficient and calibration_verified and target is not None)
        out.append(dict(detection_index=i,points_world=raw,points_policy=policy,
            center_world=raw.mean(0) if sufficient else np.zeros(3),
            observed_extent_world=np.ptp(raw,axis=0) if sufficient else np.zeros(3),
            depth_uv=np.c_[xx[selected],yy[selected]],color_uv=uv[selected],
            geometry_observed=sufficient,policy_mask=ready,
            calibration_state='verified_explicit_extrinsics' if calibration_verified else 'approximate_unverified',
            policy_frame_state='supplied_transform' if target is not None else 'missing_transform'))
    return out
