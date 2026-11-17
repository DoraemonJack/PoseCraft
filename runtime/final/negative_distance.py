import torch,numpy as np
MARGIN_M=.5
LOGIT_STRENGTH=2.0
def wrap_negative(base,catalog,centers,candidate_mask):
 anchors=[]
 for query in catalog.get('negative_queries',[]):
  if not query.get('relation_mask'):continue
  inds=query['candidate_indices'];logits=query['effective_logits'];idx=max(inds,key=lambda i:logits[i]);c=catalog['candidates'][idx];anchors.append((np.asarray(c['reference_xyz'][:2],dtype=np.float32),np.asarray(c['reference_extents'][:2],dtype=np.float32)))
 if not anchors:return base
 class NegativeDistanceScorer:
  def __getattr__(self,name):return getattr(base,name)
  def candidates_batch(self,features,texts,assets,mask):
   scores,fused,values=base.candidates_batch(features,texts,assets,mask)
   pos=np.asarray(centers,dtype=np.float32);positions=torch.nonzero(mask[0],as_tuple=False).flatten();assert len(positions)==len(pos), (len(positions),len(pos))
   d=[]
   for center,extent in anchors:
    gap=np.maximum(np.abs(pos[:,:2]-center)-extent/2,0);dist=np.linalg.norm(gap,axis=1);d.append(np.clip((MARGIN_M-dist)/MARGIN_M,0,1))
   penalty=torch.as_tensor(np.max(d,axis=0),device=scores.device,dtype=scores.dtype)*LOGIT_STRENGTH
   scores=scores.clone();scores[0,positions,:]-=penalty[:,None]
   return scores,fused,values
 return NegativeDistanceScorer()
