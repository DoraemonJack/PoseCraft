"""Position residual with explicit uncertain reference instances and pairs.

The head consumes frozen reference probabilities, not reference labels.  It
keeps up to K actual candidate instances and their pairwise geometry instead
of replacing a multi-modal distribution with a single potentially empty mean.
"""
import torch
from torch import nn


class ReferenceConditionedPositionHead(nn.Module):
    def __init__(self,relation_count,fused_dim=128,projection_dim=32,relation_dim=16,hidden_dim=64,top_k=6):
        super().__init__()
        if min(relation_count,fused_dim,projection_dim,relation_dim,hidden_dim,top_k)<1:raise ValueError('positive dimensions required')
        self.relation_count=relation_count;self.top_k=top_k
        self.fused=nn.Sequential(nn.Linear(fused_dim,projection_dim),nn.SiLU())
        self.relation=nn.Embedding(relation_count,relation_dim)
        self.single=nn.Sequential(nn.Linear(projection_dim+relation_dim+8,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim),nn.SiLU())
        self.pair=nn.Sequential(nn.Linear(projection_dim+relation_dim+11,hidden_dim),nn.SiLU(),nn.Linear(hidden_dim,hidden_dim),nn.SiLU())
        self.merge=nn.Sequential(nn.Linear(hidden_dim*2+2,hidden_dim),nn.SiLU())
        self.attention=nn.Linear(hidden_dim,1);self.output=nn.Linear(hidden_dim,1)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def forward(self,fused,candidate_xyz,reference_xyz,reference_extents,reference_logits,
                reference_mask,relation_types,relation_mask,candidate_mask):
        if fused.ndim!=3 or candidate_xyz.shape!=(*fused.shape[:2],3):raise ValueError('candidate shape')
        b,c,_=fused.shape
        if reference_xyz.ndim!=3 or reference_xyz.shape[0]!=b or reference_xyz.shape[-1]!=3:raise ValueError('reference shape')
        r=reference_xyz.shape[1]
        if relation_types.ndim!=2 or relation_types.shape[0]!=b:raise ValueError('relation shape')
        q=relation_types.shape[1]
        if (reference_extents.shape!=(b,r,3) or reference_logits.shape!=(b,q,r+1) or reference_mask.shape!=(b,r)
                or relation_mask.shape!=(b,q) or candidate_mask.shape!=(b,c)):raise ValueError('shape mismatch')
        if any(m.dtype!=torch.bool for m in (reference_mask,relation_mask,candidate_mask)):raise ValueError('boolean masks required')
        if relation_types.dtype not in (torch.int32,torch.int64):raise ValueError('integer relation codes required')
        if any(not torch.isfinite(x).all() for x in (fused,candidate_xyz,reference_xyz,reference_extents,reference_logits)):
            raise ValueError('finite features required')
        if (reference_extents<0).any() or ((relation_types[relation_mask]<0)|(relation_types[relation_mask]>=self.relation_count)).any():
            raise ValueError('invalid extent or relation')
        if q==0 or r==0:return fused.new_zeros((b,c))
        valid=torch.cat((reference_mask,torch.ones((b,1),dtype=torch.bool,device=fused.device)),dim=-1)
        probs=torch.softmax(reference_logits.masked_fill(~valid[:,None,:],-torch.inf),dim=-1)[...,:r]
        k=min(self.top_k,r);weights,ids=probs.topk(k,dim=-1)
        idx=ids[...,None].expand(b,q,k,3)
        xyz=reference_xyz[:,None,:,:].expand(b,q,r,3).gather(2,idx)
        ext=reference_extents[:,None,:,:].expand(b,q,r,3).gather(2,idx)
        mass=weights.sum(-1);normalized=weights/mass.clamp_min(1e-8)[...,None]
        delta=xyz[:,None,:,:,:]-candidate_xyz[:,:,None,None,:]
        dist=torch.log1p(torch.linalg.vector_norm(delta,dim=-1,keepdim=True))
        extent=ext[:,None,:,:,:].expand(b,c,q,k,3)
        prob=weights[:,None,:,:,None].expand(b,c,q,k,1)
        geometry=torch.cat((delta,dist,extent,prob),dim=-1)
        f=self.fused(fused)[:,:,None,None,:].expand(b,c,q,k,-1)
        codes=relation_types.masked_fill(~relation_mask,0)
        rel=self.relation(codes)[:,None,:,None,:].expand(b,c,q,k,-1)
        single=self.single(torch.cat((f,rel,geometry),dim=-1))
        single=(single*normalized[:,None,:,:,None]).sum(-2)
        pairs=torch.triu_indices(k,k,offset=1,device=fused.device)
        if pairs.shape[1]:
            i,j=pairs[0],pairs[1];a=xyz[:,:,i,:];bb=xyz[:,:,j,:]
            # Symmetric features: midpoint, squared direction components,
            # endpoint-distance sum/difference and distances to the segment.
            mid=(a+bb)*.5;direction=bb-a;length=torch.linalg.vector_norm(direction,dim=-1,keepdim=True)
            relative=mid[:,None,:,:,:]-candidate_xyz[:,:,None,None,:]
            da=torch.linalg.vector_norm(a[:,None,:,:,:]-candidate_xyz[:,:,None,None,:],dim=-1,keepdim=True)
            db=torch.linalg.vector_norm(bb[:,None,:,:,:]-candidate_xyz[:,:,None,None,:],dim=-1,keepdim=True)
            direction2=direction.square()/(length.square()+1e-8)
            pairgeom=torch.cat((relative,direction2[:,None,:,:,:].expand(b,c,q,-1,3),
                torch.log1p(length)[:,None,:,:,:].expand(b,c,q,-1,1),torch.log1p(da+db),
                torch.log1p((da-db).abs()),
                ext[:,:,i,:].mean(-1)[:,None,:,:,None].expand(b,c,q,-1,1)+ext[:,:,j,:].mean(-1)[:,None,:,:,None].expand(b,c,q,-1,1),
                (weights[:,:,i]*weights[:,:,j])[:,None,:,:,None].expand(b,c,q,-1,1)),dim=-1)
            n=pairs.shape[1];fp=self.fused(fused)[:,:,None,None,:].expand(b,c,q,n,-1)
            rp=self.relation(codes)[:,None,:,None,:].expand(b,c,q,n,-1)
            pair_weight=weights[:,:,i]*weights[:,:,j];pair_mass=pair_weight.sum(-1)
            pn=pair_weight/pair_mass.clamp_min(1e-8)[...,None]
            ph=self.pair(torch.cat((fp,rp,pairgeom),dim=-1));ph=(ph*pn[:,None,:,:,None]).sum(-2)
        else:
            pair_mass=torch.zeros_like(mass);ph=torch.zeros_like(single)
        hidden=self.merge(torch.cat((single,ph,mass[:,None,:,None].expand(b,c,q,1),pair_mass[:,None,:,None].expand(b,c,q,1)),dim=-1))
        active=relation_mask & (mass>1e-8)
        attention=self.attention(hidden).squeeze(-1).masked_fill(~active[:,None,:],-1e9)
        aw=torch.softmax(attention,dim=-1)*active[:,None,:];aw=aw/aw.sum(-1,keepdim=True).clamp_min(1e-8)
        pooled=(aw[...,None]*hidden).sum(-2);observed_mass=(aw*mass[:,None,:]).sum(-1)
        return (self.output(pooled).squeeze(-1)*observed_mass).masked_fill(~candidate_mask,0.)
