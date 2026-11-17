"""Warm-started semantic refinement with immutable V13 physical-risk estimates."""
import sys,copy
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'refineplace_orientation_v13_20261103_v1'))
from model_v13 import RefinePlaceV13,make_v11,select_action

class RefinePlaceV19(RefinePlaceV13):
    def __init__(self,baseline,relation_count):
        super().__init__(baseline,relation_count)
        self.requires_grad_(False)
        for name in ('value_state','value_action','value_relation','value_body','task_value','mode_teacher'):
            setattr(self,'semantic_'+name,copy.deepcopy(getattr(self,name)).requires_grad_(True))
    def warm_semantic(self):
        for name in ('value_state','value_action','value_relation','value_body','task_value','mode_teacher'):
            getattr(self,'semantic_'+name).load_state_dict(getattr(self,name).state_dict())
    def train(self,mode=True):
        super().train(False)
        for name,module in self.named_children():
            if name.startswith('semantic_'):module.train(mode)
        return self
    def action_values(self,state,centers,yaws,task_indices,action_delta=None):
        original=super().action_values(state,centers,yaws,task_indices,action_delta)
        b=self.baseline;delta=original['action_delta'];n=len(state)
        ctx={k:v.index_select(0,task_indices) for k,v in b._bound_context.items() if k not in ('candidate_xyz','candidate_mask')}
        ctx['candidate_mask']=torch.ones(n,1,device=state.device,dtype=torch.bool)
        next_yaw=yaws[:,None]+delta;current=yaws[:,None].expand_as(delta)
        encoded=torch.stack([current.cos(),current.sin(),next_yaw.cos(),next_yaw.sin(),delta.cos(),delta.sin(),delta/(torch.pi/4),delta.abs()/(torch.pi/4)],-1)
        relation=b._bound_forward(self.semantic_value_relation,state[:,None],b._bound_assets.index_select(0,task_indices),centers[:,None],next_yaw[:,None],ctx,task_indices)[:,0]
        latent=self.semantic_value_body(self.semantic_value_state(state)[:,None]+self.semantic_value_action(encoded)+relation)
        raw=self.semantic_task_value(latent).squeeze(-1)
        # Zero at warm start; hard cap |logit change| <=0.5. Geometry heads
        # retain their ORIGINAL latent trunk, so tuning cannot change risk.
        logits=original['task_q_logits']+.5*torch.tanh((raw-original['task_q_logits'])/.5)
        stop=original['stop_index'];rows=torch.arange(n,device=state.device)
        return dict(original,task_q=logits.sigmoid(),task_q_logits=logits,
            mode_logits=self.semantic_mode_teacher(latent[rows,stop]),
            original_task_q=original['task_q'],original_task_logits=original['task_q_logits'],
            original_mode_logits=original['mode_logits'])

def model_from_checkpoint(d,types,region_dim,checkpoint):
    if checkpoint['architecture'] not in ('RefinePlaceV13','RefinePlaceV19'):raise ValueError('Expected V13 or V19')
    model=RefinePlaceV19(make_v11(d,types,region_dim,None),len(types))
    missing,extra=model.load_state_dict(checkpoint['joint_policy'],strict=False)
    if extra or any(not k.startswith('semantic_') for k in missing):raise ValueError('Frozen parent mismatch')
    if checkpoint['architecture']=='RefinePlaceV13':model.warm_semantic()
    elif missing:raise ValueError('Missing tuned semantic layers')
    return model
