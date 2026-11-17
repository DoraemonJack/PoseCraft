"""Warm-start V13 semantic values with bounded changes and preservation learning.
No changes to frozen candidate positions, V11 actions, or V13 physical risk.
"""
import sys,json,time,math
from pathlib import Path
import numpy as np
import torch
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;V13=ROOT/'refineplace_orientation_v13_20261103_v1'
sys.path[:0]=[str(HERE),str(ROOT/'refineplace_orientation_v15_20261103_v1'),str(V13)]
from train_head_development import split_rows
from train_paired_v13 import PairedTrainer,write
from build_pair_cache_v13 import load_cache,sha
from model_v19 import model_from_checkpoint,select_action,RefinePlaceV13
from projected_loss import losses,repair_replay,project_gradient
def trajectory(v,truth,config=None):
    choice=select_action(v)
    b=len(truth);dec=choice['action_index'].reshape(b,8)
    cur=torch.arange(8,device=truth.device)[None].expand(b,-1).clone();active=torch.ones_like(truth)
    for _ in range(2):
        a=dec.gather(1,cur);active &= a!=1;cur=torch.where(active,(cur+a-1)%8,cur)
    end=truth.gather(1,cur)
    counts=dict(correct=int(truth.sum()),incorrect=int((~truth).sum()),preserved=int((truth&end).sum()),damaged=int((truth&~end).sum()),
        repaired=int((~truth&end).sum()),unresolved=int((~truth&~end).sum()),final_success=int(end.sum()))
    return counts,end

def main():
    import train_official563_joint_v7_sync_full as d
    from train_v10 import load_references,atomic_checkpoint
    from support_regions import FEATURE_NAMES
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(622)
    out=HERE/'projected_run';out.mkdir(exist_ok=False)
    parent=V13/'refine1000_v1/final.pt'
    assert sha(parent)==json.loads((V13/'refine1000_v1/summary.json').read_text())['final_checkpoint_sha256']
    train,dev=split_rows(load_cache(V13/'paired_yaw1000_v1'))
    types=d.read('/path/to/research300/relation_grounding_train_holdout_20261101_v2/run/protocol.json')['relation_types']
    model=model_from_checkpoint(d,types,len(FEATURE_NAMES),torch.load(parent,map_location='cpu',weights_only=False)).cuda()
    helper=PairedTrainer(d,model,load_references(),types)
    protocol=dict(architecture='RefinePlaceV19',version='v19_repair_protected_gradient_finetune',epochs=12,seed=622,learning_rate=.005,optimizer='SGD_no_momentum_no_weight_decay',
        train_tasks=819,head_dev_tasks=181,parent_checkpoint_sha256=sha(parent),
        normalization_sha256=d.sha(d.F/'normalization.json'),official_split_lock_sha256=d.sha(d.LOCK),
        source_sha256=sha(__file__),model_sha256=sha(HERE/'model_v19.py'),loss_sha256=sha(HERE/'projected_loss.py'),
        checkpoints_to_compare=[1,2,4,8,12],maximum_logit_change=.5,unchanged_selector=True,
        selection='damage <=90% of V13, repair >=90%, net final success >V13; maximize repaired-2*damaged',
        supervision='Preserve annotated valid heading; protect actions on frozen V13 successful two-step repair paths; project conflicting gradients',
        frozen='Original V13 and V11 including all geometry risk estimates; only cloned semantic modules update',
        test_labels_read=False,parent_has_seen_head_development=True,new_rl_training=False,continuous_micro_adjustment_enabled=False)
    write(out/'protocol_before_training.json',protocol)
    frozen=d.tiny.digest({k:v for k,v in model.state_dict().items() if not k.startswith('semantic_')})
    def extract(part):
        batches=[]
        for offset in range(0,len(part),16):
            rr=part[offset:offset+16];b=len(rr)
            for r in rr:
                uid=r['asset_uuid']
                if uid not in helper.assets:
                    helper.assets[uid]=torch.load(d.b.B/'placeit3d_reproduction_v1/placeit3d/data/pointbert_embeddings'/(uid+'.pt'),map_location='cpu',weights_only=True).reshape(-1).cuda()
            f,t,a,ctx,_,inferred=d.make_context(rr,helper.assets,helper.refs,2,types)
            ctx['candidate_mask']=torch.ones(b,1,device='cuda',dtype=torch.bool);model.reference_head.inferred_mask=inferred
            region=torch.as_tensor(np.stack([r['region_features'] for r in rr]),device='cuda',dtype=torch.float32)
            y=torch.as_tensor(np.stack([r['current_yaws'] for r in rr]),device='cuda',dtype=torch.float32)
            g=torch.as_tensor(np.stack([r['geom_vector'] for r in rr]),device='cuda',dtype=torch.float32)
            with torch.no_grad(),model.reference_batch(**ctx),model.region_batch(region):
                _,fused,_=model.baseline.candidates_batch(f,t,a,ctx['candidate_mask'])
                assets=model.baseline._bound_assets.detach().clone()
                state=torch.cat([fused[:,0,None].expand(b,8,-1),g,torch.zeros(b,8,3,device='cuda'),y.sin()[...,None],y.cos()[...,None],torch.ones(b,8,1,device='cuda')],-1).reshape(b*8,160)
                values=RefinePlaceV13.action_values(model,state,ctx['candidate_xyz'][:,0,None].expand(b,8,3).reshape(-1,3),y.reshape(-1),torch.arange(b,device='cuda').repeat_interleave(8))
            teacher_action,protect=repair_replay(values,torch.as_tensor(np.stack([r['valid_mask8'][0] for r in rr]),device='cuda',dtype=torch.bool))
            batches.append(dict(teacher_action=teacher_action,protect=protect,state=state.detach(),fused=fused[:,0].detach(),assets=assets,ctx=ctx,inferred=inferred,
                centers=ctx['candidate_xyz'][:,0],yaws=y,values={k:v.detach() for k,v in values.items()},
                valid=torch.as_tensor(np.stack([r['valid_mask8'][0] for r in rr]),device='cuda',dtype=torch.bool),
                ids=[(r['scene'],int(r['group_id'])) for r in rr]))
        return batches
    started=time.monotonic();model.eval();training=extract(train);development=extract(dev)
    def forward(b):
        model.reference_head.inferred_mask=b['inferred']
        n=len(b['valid'])
        with model.reference_batch(**b['ctx']):
            model.baseline._bound_assets=b['assets']
            output=model.action_values(b['state'],b['centers'][:,None].expand(n,8,3).reshape(-1,3),b['yaws'].reshape(-1),torch.arange(n,device='cuda').repeat_interleave(8))
        # Physical evidence heads have an immutable independent path.
        for k in ('collision_cost','collision_known_logits'):
            assert torch.equal(output[k],b['values'][k]),k
        assert float((output['task_q_logits']-output['original_task_logits']).detach().abs().max())<=.500001
        return output
    base_values={k:torch.cat([b['values'][k] for b in development]) for k in development[0]['values']}
    truth=torch.cat([b['valid'] for b in development]);base,base_end=trajectory(base_values,truth)
    with torch.no_grad():
        initial=forward(development[0]);assert torch.equal(initial['task_q'],development[0]['values']['task_q'])
    params=[v for k,v in model.named_parameters() if k.startswith('semantic_')]
    opt=torch.optim.SGD(params,lr=.005,momentum=0.,weight_decay=0.)
    best=None;best_state=None;grid=[];reports=[]
    def save(name,epoch):
        assert frozen==d.tiny.digest({k:v for k,v in model.state_dict().items() if not k.startswith('semantic_')})
        atomic_checkpoint(out/name,dict(architecture='RefinePlaceV19',joint_policy=model.state_dict(),epoch=epoch,
            normalization_sha256=protocol['normalization_sha256'],official_split_lock_sha256=protocol['official_split_lock_sha256'],protocol_sha256=sha(out/'protocol_before_training.json')))
    for epoch in range(1,13):
        model.train();total=0.;components={};conflicts=0;minimum_after=0.
        for bi in np.random.default_rng(622+epoch).permutation(len(training)):
            b=training[bi];result=forward(b);loss,maintenance,parts=losses(result,b['valid'],b['teacher_action'],b['protect'])
            opt.zero_grad(set_to_none=True)
            protected_grad=torch.autograd.grad(maintenance,params,retain_graph=True,allow_unused=True)
            objective_grad=torch.autograd.grad(loss,params,allow_unused=True)
            protected_grad=[torch.zeros_like(p) if g is None else g for p,g in zip(params,protected_grad)]
            objective_grad=[torch.zeros_like(p) if g is None else g for p,g in zip(params,objective_grad)]
            gradients,record=project_gradient(objective_grad,protected_grad)
            conflicts+=int(record['conflict']);minimum_after=min(minimum_after,record['dot_after'])
            for p,g in zip(params,gradients):p.grad=g
            assert all(p.grad is None for k,p in model.named_parameters() if not k.startswith('semantic_'))
            torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
            total+=float(loss.detach())*len(b['valid'])
            for k,v in parts.items():components[k]=components.get(k,0.)+float(v.detach())*len(b['valid'])
        reports.append(dict(epoch=epoch,loss=total/819,conflict_batches=conflicts,minimum_projected_dot=minimum_after,components={k:v/819 for k,v in components.items()}))
        print(json.dumps(reports[-1]),flush=True);write(out/'progress.json',dict(status='training',**reports[-1]))
        if epoch not in protocol['checkpoints_to_compare']:continue
        model.eval()
        with torch.no_grad():outputs=[forward(b) for b in development]
        merged={k:torch.cat([v[k] for v in outputs]) for k in outputs[0]}
        metrics,end=trajectory(merged,truth)
        metrics.update(epoch=epoch,admitted=metrics['damaged']<=.9*base['damaged'] and metrics['repaired']>=.9*base['repaired'] and metrics['final_success']>base['final_success'],
            vs_v13_success_to_failure=int((base_end&~end).sum()),vs_v13_failure_to_success=int((~base_end&end).sum()))
        grid.append(metrics);save('epoch_'+str(epoch)+'.pt',epoch)
        np.savez_compressed(out/('dev_epoch_'+str(epoch)+'.npz'),valid=truth.cpu().numpy(),final=end.cpu().numpy(),task_q=merged['task_q'].cpu().numpy(),baseline=base_end.cpu().numpy())
        if metrics['admitted'] and (best is None or (metrics['repaired']-2*metrics['damaged'],metrics['final_success'])>(best['repaired']-2*best['damaged'],best['final_success'])):
            best=metrics;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if k.startswith('semantic_')}
        print(json.dumps(dict(development=metrics)),flush=True)
    if best:
        model.load_state_dict(best_state,strict=False);save('final.pt',best['epoch'])
    summary=dict(status='admitted' if best else 'not_admitted',base_v13=base,selected=best,candidates=grid,epoch_reports=reports,
        frozen_v13_unchanged=True,risk_predictions_bitwise_unchanged=True,warm_start_predictions_exact=True,
        checkpoint_sha256=sha(out/'final.pt') if best else None,elapsed_seconds=time.monotonic()-started,
        scope='181 original TRAIN facing tasks /1448 starting states; V13 parent has seen them. Head development only, not official success.',test_labels_read=False)
    write(out/'summary.json',summary);write(out/'progress.json',dict(status=summary['status']));print(json.dumps(summary),flush=True)

if __name__=='__main__':main()
