"""RefinePlace v1.0: language/asset-conditioned support-region pose fusion."""
from contextlib import contextmanager
import sys
from pathlib import Path
import torch
from torch import nn

PREVIOUS=Path('/path/to/research300/refineplace_rl_geometry_10k_20261102_v1/joint_collision_orientation')
if PREVIOUS.is_dir():
    sys.path.insert(0,str(PREVIOUS))
else:
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'refineplace_rl_geometry_10k_20261102_v1/joint_collision_orientation'))
from orientation_policy_v1 import DirectionalReferencePosePolicy


class RefinePlaceV10(DirectionalReferencePosePolicy):
    def __init__(self, inner, region_dim, hidden=96):
        super().__init__(inner)
        self.region_dim=int(region_dim)
        self.region_encoder=nn.Sequential(nn.Linear(self.region_dim,hidden),nn.LayerNorm(hidden),nn.SiLU())
        self.region_text=nn.Sequential(nn.Linear(768,hidden),nn.SiLU())
        self.region_asset=nn.Sequential(nn.Linear(768,hidden),nn.SiLU())
        self.region_candidate=nn.Linear(128,hidden)
        self.region_condition=nn.Sequential(nn.Linear(2*hidden,2*hidden),nn.SiLU(),nn.Linear(2*hidden,2*hidden))
        self.region_fusion=nn.Sequential(nn.Linear(3*hidden,hidden),nn.SiLU(),nn.Linear(hidden,hidden),nn.SiLU())
        self.region_pose_score=nn.Linear(hidden,1)
        self.region_to_fused=nn.Linear(hidden,128)
        self.region_to_value=nn.Linear(hidden,3)
        for layer in (self.region_pose_score,self.region_to_fused,self.region_to_value):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)
        self._regions=None

    @contextmanager
    def region_batch(self, region_features):
        if self._regions is not None:raise RuntimeError('nested region context')
        if region_features.ndim!=4 or region_features.shape[-2:]!=(8,self.region_dim):
            raise ValueError('region_features must be [B,C,8,D]')
        if not torch.isfinite(region_features).all():raise ValueError('finite region features required')
        self._regions=region_features.detach()
        try:yield self
        finally:self._regions=None

    def candidates_batch(self,features,texts,assets,candidate_mask):
        scores,fused,values=super().candidates_batch(features,texts,assets,candidate_mask)
        if self._regions is None:raise RuntimeError('v1.0 requires explicit region observations (zero fallback permitted)')
        regions=self._regions
        if regions.shape[:2]!=features.shape[:2]:raise ValueError('candidate/region identity mismatch')
        r=self.region_encoder(regions)
        c=self.region_candidate(fused)[:,:,None,:].expand_as(r)
        conditioning=self.region_condition(torch.cat((self.region_text(texts.detach()),self.region_asset(assets.detach())),-1))
        scale,bias=conditioning.chunk(2,-1)
        r=r*(1.+scale[:,None,None,:].tanh())+bias[:,None,None,:]
        h=self.region_fusion(torch.cat((r,c,r*c),-1))
        available=(regions.abs().sum(-1)>0)&candidate_mask[:,:,None]
        delta=self.region_pose_score(h).squeeze(-1)*available
        pooled=(h*available[...,None]).sum(2)/available.sum(2,keepdim=True).clamp_min(1)
        fused=fused+self.region_to_fused(pooled)*available.any(2)[...,None]
        scene=(pooled*available.any(2)[...,None]).sum(1)/available.any(2).sum(1,keepdim=True).clamp_min(1)
        values=values+self.region_to_value(scene)*available.any((1,2))[:,None]
        return (scores+delta).masked_fill(~candidate_mask[...,None],-torch.inf),fused,values


def model_from_checkpoint(d, types, region_dim, checkpoint=None):
    prior=nn.Sequential(nn.Linear(141,192),nn.SiLU(),nn.Linear(192,128),nn.SiLU(),nn.Linear(128,1))
    base=d.paired.BatchedJointModel(47,prior.state_dict(),768,.25)
    inner=d.paired.ReferenceJointPolicy(base,d.InferredReferenceHead(len(types),inferred_weight=.3))
    original=d.joint.ReferencePosePolicy(inner,len(types),base.actor_body[0].in_features,preserve_candidate_logmass=False)
    model=RefinePlaceV10(original,region_dim)
    if checkpoint is not None:
        state=checkpoint['joint_policy']
        if checkpoint.get('architecture')=='RefinePlaceV10':
            model.load_state_dict(state,strict=True)
        else:
            # Strictly restore every existing module before adding the region branch.
            previous=DirectionalReferencePosePolicy(original)
            previous.load_state_dict(state,strict=True)
            absent,extra=model.load_state_dict(previous.state_dict(),strict=False)
            assert not extra and all(k.startswith('region_') for k in absent)
    return model
