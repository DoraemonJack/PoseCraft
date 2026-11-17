"""Padding-safe V7 batching with exactly V6 parameters and taskwise objectives.

Targets must be validated on CPU before transfer. The hot path performs no
Tensor.item()/bool() conversions or host checks; callers check finite batch loss.
"""
import math
import torch
from torch.nn import functional as F
from train_joint_pose_v6 import JointModel


class BatchedJointModel(JointModel):
    """No constructor override: initialization and state_dict are exactly V6."""
    def candidates_batch(self,features,texts,assets,candidate_mask):
        if features.ndim!=3 or texts.shape!=(features.shape[0],768) or assets.shape!=(features.shape[0],self.asset_dimension):
            raise ValueError('Expected features[B,C,D], texts[B,768], assets[B,A]')
        if candidate_mask.shape!=features.shape[:2] or candidate_mask.dtype!=torch.bool or features.shape[1]<1:
            raise ValueError('Expected boolean candidate_mask[B,C] with C>=1')
        mask=candidate_mask.unsqueeze(-1)
        features=features.masked_fill(~mask,0.)
        text=self.text(texts)
        geometry=self.geometry(features)
        repeated=text[:,None,:].expand_as(geometry)
        fused=self.fusion(torch.cat((geometry,repeated,geometry*repeated),-1))
        shape=self.asset_projection(assets.detach())[:,None,:].expand_as(fused)
        fused=self.asset_fusion(torch.cat((fused,shape,fused*shape),-1)).masked_fill(~mask,0.)
        angles=torch.arange(8,device=features.device,dtype=features.dtype)*(math.pi/4)
        angle=self.angle(torch.stack((angles.sin(),angles.cos()),-1))
        joint=torch.cat((fused[:,:,None,:].expand(-1,-1,8,-1),
                         angle[None,None,:,:].expand(features.shape[0],features.shape[1],-1,-1)),-1)
        scores=(self.prior_weight*self.prior(features)+self.joint_head(joint).squeeze(-1)).masked_fill(~mask,-torch.inf)
        count=mask.sum(1)
        mean=fused.sum(1)/count.clamp_min(1)
        maximum=fused.masked_fill(~mask,-torch.inf).max(1).values
        maximum=torch.where(count>0,maximum,torch.zeros_like(maximum))
        values=self.scene_critic(torch.cat((text,mean,maximum),-1))
        return scores,fused,values

    def offsets_batch(self,fused,candidate_mask=None,radius=.15):
        if fused.ndim!=3:raise ValueError('Expected fused[B,C,128]')
        batch,candidates,width=fused.shape
        offsets=self.offsets(fused.reshape(batch*candidates,width),radius).reshape(batch,candidates,8,2)
        if candidate_mask is not None:
            if candidate_mask.shape!=fused.shape[:2] or candidate_mask.dtype!=torch.bool:raise ValueError('Expected boolean candidate mask')
            offsets=offsets.masked_fill(~candidate_mask[:,:,None,None],0.)
        return offsets

    def local_batch(self,state):
        hidden=self.actor_body(state)
        return self.mode(hidden),self.mean(hidden),torch.cat([value(hidden) for value in self.values],-1)


def batched_supervised_losses(logits,predicted_offset_xy_m,positive,rank_prob,offset_xy_m,offset_mask,candidate_mask,
                              *,distance_weight=.5,offset_weight=1.,offset_radius=.15):
    """Return per-task V6 exact-set + soft rank CE + normalized offset losses.

    Each rank softmax spans only its own C*8 valid poses. Empty targets produce
    differentiable zeros. Padded targets are ignored, never renormalized; input
    rank probabilities over valid poses must already have total mass 0 or 1.
    An all-padded task is supported as an empty task with zero losses.
    """
    if logits.ndim!=3 or logits.shape[-1]!=8 or logits.shape[1]<1:raise ValueError('Expected logits[B,C,8], C>=1')
    if predicted_offset_xy_m.shape!=(*logits.shape,2) or offset_xy_m.shape!=predicted_offset_xy_m.shape:
        raise ValueError('Expected predicted/target offsets[B,C,8,2]')
    if positive.shape!=logits.shape or rank_prob.shape!=logits.shape or offset_mask.shape!=logits.shape or candidate_mask.shape!=logits.shape[:2]:
        raise ValueError('Target/mask shapes differ from logits')
    if candidate_mask.dtype!=torch.bool or positive.dtype!=torch.bool or offset_mask.dtype!=torch.bool:raise ValueError('Masks must be boolean')
    if not math.isfinite(offset_radius) or offset_radius<=0:raise ValueError('Positive finite offset radius required')
    valid=candidate_mask[:,:,None].expand_as(logits)
    batch=logits.shape[0]
    flat=logits.masked_fill(~valid,-torch.inf).reshape(batch,-1)
    has_candidates=candidate_mask.any(-1)
    safe_flat=torch.where(has_candidates[:,None],flat,torch.zeros_like(flat))
    log_norm=torch.logsumexp(safe_flat,-1)
    positives=(positive.detach() & valid).reshape(batch,-1)
    has_positive=positives.any(-1)
    positive_logits=safe_flat.masked_fill(~positives,-torch.inf)
    safe_positive=torch.where(has_positive[:,None],positive_logits,torch.zeros_like(positive_logits))
    exact=torch.where(has_positive,log_norm-torch.logsumexp(safe_positive,-1),torch.zeros_like(log_norm))
    probs=rank_prob.detach().to(dtype=logits.dtype).masked_fill(~valid,0.)
    log_probs=F.log_softmax(safe_flat,-1).reshape_as(logits).masked_fill(~valid,0.)
    rank=-(probs*log_probs).sum((1,2))
    eligible=offset_mask.detach() & valid
    safe_predictions=predicted_offset_xy_m.masked_fill(~eligible[:,:,:,None],0.)
    safe_targets=offset_xy_m.detach().to(dtype=predicted_offset_xy_m.dtype).masked_fill(~eligible[:,:,:,None],0.)
    element=F.smooth_l1_loss(safe_predictions/offset_radius,safe_targets/offset_radius,reduction='none').mean(-1)
    offset=(element*eligible).sum((1,2))/eligible.sum((1,2)).clamp_min(1)
    return dict(loss=exact+distance_weight*rank+offset_weight*offset,exact_loss=exact,rank_loss=rank,
                offset_loss=offset,has_positive=has_positive,has_distance=probs.sum((1,2))>0,has_offset=eligible.any((1,2)))
