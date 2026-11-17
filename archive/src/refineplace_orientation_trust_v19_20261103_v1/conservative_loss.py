"""Preserve valid headings, improve invalid headings, retain parent action prior."""
import torch
from torch.nn import functional as F

def targets(valid):
    b=len(valid);a=torch.arange(8,device=valid.device,dtype=torch.float32)*torch.pi/4
    close=(1+(a[:,None]-a[None,:]).cos()[None].expand(b,-1,-1).masked_fill(~valid[:,None,:],-torch.inf).amax(-1))/2
    reward=.5*valid.float()+.5*close
    endpoint=(torch.arange(8,device=valid.device)[:,None]+torch.tensor([-1,0,1],device=valid.device)[None])%8
    desired=reward[:,endpoint].reshape(-1,3)
    optimal=(desired-desired.max(-1,keepdim=True).values).abs()<1e-6
    correct=valid.reshape(-1)
    optimal[correct]=False;optimal[correct,1]=True
    return desired,optimal,correct

def balanced(values,correct):
    b=values.numel()//8;v=values.reshape(b,8);c=correct.reshape(b,8)
    pos=(v*c).sum(-1)/c.sum(-1).clamp_min(1)
    neg=(v*~c).sum(-1)/(~c).sum(-1).clamp_min(1)
    return ((pos+neg)/(1+(~c).any(-1).float())).mean()

def loss_fn(output,valid):
    desired,optimal,correct=targets(valid)
    logits=output['task_q_logits'];q=output['task_q'];teacher=output['original_task_logits'].detach()
    probability=optimal.float()/optimal.sum(-1,keepdim=True)
    action=balanced(-(probability*(logits/.5).log_softmax(-1)).sum(-1),correct)
    adv=F.smooth_l1_loss(q-q[:,1,None],desired-desired[:,1,None])
    preservation=(.08+q[:,[0,2]]-q[:,1,None]).relu().mean(-1)
    preserve=preservation[correct].mean()
    best=(q*probability).sum(-1)
    repair=(.08+q[:,1]-best).relu()[~correct].mean() if (~correct).any() else q.sum()*0
    kl=F.kl_div((logits/.5).log_softmax(-1),(teacher/.5).softmax(-1),reduction='batchmean')
    absolute=F.smooth_l1_loss(q,desired)
    mode=balanced(F.cross_entropy(output['mode_logits'],(~correct).long(),reduction='none'),correct)
    loss=action+adv+.5*preserve+.5*repair+.2*kl+.1*absolute+.25*mode
    return loss,dict(action=action,advantage=adv,preservation=preserve,repair=repair,distillation=kl,absolute=absolute,mode=mode)
