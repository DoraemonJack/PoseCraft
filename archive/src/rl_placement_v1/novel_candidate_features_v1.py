"""Reconstruct V7's 47 input features for new observed scene-point candidates.

This reconstructs the existing preprocessor, not support or collision labels.
The caller must validate original-candidate features and rerun V7's pooling on
the entire proposed candidate set; new candidates cannot use cached logits.
"""
import numpy as np
from train_joint_pose_v7 import aligned_task


def reconstruct(payload,new_point_ids,original_point_ids,original_raw,alignment,cache=None):
    if len(payload)!=5:raise ValueError('Expected five-array scene payload')
    xyz=np.asarray(payload[0]);rgb=np.asarray(payload[1]);raw=np.asarray(original_raw)
    original=np.asarray(original_point_ids);ids=np.asarray(new_point_ids)
    if xyz.ndim!=2 or xyz.shape[1]!=3 or rgb.shape!=xyz.shape or not np.isfinite(xyz).all() or not np.isfinite(rgb).all():
        raise ValueError('Finite observed XYZ/RGB required')
    for x in (original,ids):
        if x.ndim!=1 or x.dtype.kind not in 'iu' or np.any(x<0) or np.any(x>=len(xyz)) or len(np.unique(x))!=len(x):
            raise ValueError('Unique in-range integer point identities required')
    if len(ids)==0 or raw.ndim!=2 or raw.shape[0]!=len(original) or raw.shape[1]<47 or not np.isfinite(raw[:,:47]).all():
        raise ValueError('Original 47-dimensional features required')
    if not np.allclose(raw[:,12:47],raw[0,12:47],atol=1e-6,rtol=0):
        raise ValueError('Task dimensions/hash features are not constant')
    if not np.allclose(raw[:,3:6],rgb[original],atol=1e-6,rtol=0):
        raise ValueError('Original point identities/RGB differ from scene')
    generated=np.broadcast_to(raw[0,:47],(len(ids),47)).copy().astype(np.float32)
    generated[:,3:6]=rgb[ids]
    # normalized XYZ/local offsets are overwritten by aligned_task using the
    # same aligned-scene normalization and 16-neighbor query as original V7.
    generated[:,:3]=0.;generated[:,6:12]=0.
    return aligned_task(generated,xyz[ids],xyz,alignment,cache)
