"""TRAIN-only repair replay and first-order gradient protection, not a safety proof."""
import torch
from torch.nn import functional as F
from conservative_loss import targets
from model_v19 import select_action

def repair_replay(values,valid):
    b=len(valid);actions=select_action(values)['action_index'].reshape(b,8)
    cur=torch.arange(8,device=valid.device)[None].expand(b,-1).clone();active=torch.ones_like(valid)
    visits=[]
    for _ in range(2):
        visits.append((cur.clone(),active.clone()))
        a=actions.gather(1,cur);active &= a!=1;cur=torch.where(active,(cur+a-1)%8,cur)
    repaired=(~valid)&valid.gather(1,cur)
    protect=torch.zeros(b*8,device=valid.device,dtype=torch.bool)
    group=torch.arange(b,device=valid.device)[:,None]*8
    for node,alive in visits:protect[(group+node)[repaired&alive]]=True
    return actions.reshape(-1).detach(),protect.detach()

def losses(output,valid,teacher_action,protect):
    logits=output['task_q_logits'];q=output['task_q']
    desired,_,correct=targets(valid)
    keep=F.softplus(.2+logits[:,[0,2]]-logits[:,1,None])[correct].mean()
    maintenance=F.cross_entropy(logits[protect]/.5,teacher_action[protect]) if protect.any() else logits.sum()*0
    absolute=F.smooth_l1_loss(q,desired)
    advantage=F.smooth_l1_loss(q-q[:,1,None],desired-desired[:,1,None])
    kl=F.kl_div((logits/.5).log_softmax(-1),(output['original_task_logits'].detach()/.5).softmax(-1),reduction='batchmean')
    objective=keep+.25*maintenance+.5*absolute+.5*advantage+.05*kl
    return objective,maintenance,dict(keep=keep,repair_maintenance=maintenance,absolute=absolute,advantage=advantage,distillation=kl)

def project_gradient(gradient,constraint_gradient):
    dot=sum((g*r).sum() for g,r in zip(gradient,constraint_gradient))
    norm=sum(r.square().sum() for r in constraint_gradient)
    conflict=bool(dot<0 and norm>1e-20)
    if conflict:
        gradient=[g-(dot/norm)*r for g,r in zip(gradient,constraint_gradient)]
    after=sum((g*r).sum() for g,r in zip(gradient,constraint_gradient))
    return gradient,dict(conflict=conflict,dot_before=float(dot.detach()),dot_after=float(after.detach()))
