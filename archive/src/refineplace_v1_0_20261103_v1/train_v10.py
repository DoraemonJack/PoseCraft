"""Full official training, synchronized two GPU updates, eight complete passes.

Reference/image encoders are frozen. New candidates use observations only;
annotation arrays are read after the sampled policy rollout for supervision.
"""
import argparse
from collections import Counter
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist

HERE=Path(__file__).resolve().parent
OLD=Path('/path/to/research300/refineplace_rl_geometry_10k_20261102_v1/joint_collision_orientation')
MODULES=Path('/path/to/research300/refineplace_v01_official563_modules_20261102_v1')
sys.path[:0]=[str(HERE),str(OLD),str(MODULES)]
import train_official563_joint_v7_sync_full as d
from model_v10 import model_from_checkpoint
from orientation_policy_v1 import collision_auxiliary_loss
from constrained_objective import constrained_rl_losses
from paired_orientation_v1 import paired_pose_quality,make_positive_pose_pairs
import joint_rollout_v1 as rollout
from wrapped_geometry_backend_v10 import CollisionGeometryBackend
from support_regions import FEATURE_NAMES
from v1_scene import V10Scene

CAT=Path('/path/to/research300/refineplace_reference_multiframe_assoc_20261102_v1/reference_catalog_compact_v4_inferred_pilot.jsonl')
PARENT=OLD/'train/final.pt'


def load_references():
    refs={}
    with CAT.open(encoding='utf-8') as fp:
        for line in fp:
            row=json.loads(line);key=row['scene'],row['group_id']
            if key in refs:raise ValueError('Duplicate reference task')
            refs[key]=row
    if len(refs)!=100505:raise ValueError('Full reference catalog must contain 100505 task records')
    return refs


def synchronize_gradients(model):
    parameters=[p for p in model.parameters() if p.requires_grad]
    present=torch.tensor([p.grad is not None for p in parameters],device='cpu',dtype=torch.int32)
    dist.all_reduce(present,op=dist.ReduceOp.SUM)
    flat=torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).cpu() for p in parameters])
    dist.all_reduce(flat,op=dist.ReduceOp.SUM)
    cursor=0
    for flag,p in zip(present.cpu().tolist(),parameters):
        n=p.numel();p.grad=flat[cursor:cursor+n].reshape_as(p).to(p.device).clone() if flag else None;cursor+=n


def atomic_checkpoint(path,state):
    tmp=path.with_name(path.name+'.pending')
    torch.save(state,tmp);os.replace(tmp,path)


def run(a):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.cuda.set_device(0)
    dist.init_process_group('gloo',timeout=timedelta(minutes=10))
    rank=dist.get_rank();world=dist.get_world_size()
    assert world==2 and torch.cuda.is_available()
    torch.manual_seed(419);np.random.seed(419);random.seed(419)
    lock=d.read(d.LOCK);admit=d.read(d.ADMIT);index=d.read(d.F/'index.json')
    assert admit['training_admission_passed'] and index['full_official_train563']
    assert len(lock['train_scenes'])==563 and index['tasks']==100505
    assert d.sha(d.LOCK)==admit['official_split_lock_sha256']
    assert d.sha(d.F/'normalization.json')==admit['normalization_sha256']
    grounding=d.read(a.grounded_root/'index.json')
    # The grounding builder may validly return no detection, but every scene
    # must have an explicit observation-only processed record before admission.
    records=grounding.get('records',{})
    if isinstance(records,list):records={r['scene']:r for r in records}
    if not a.smoke_batches:
        assert set(lock['train_scenes']).issubset(records),'Grounded scene coverage incomplete'
        assert grounding['status']=='complete' and grounding['no_GT_labels']
        assert grounding['split_lock_sha256']==d.sha(d.LOCK)
        ground_audit=d.read(a.grounded_root.parent/'admission_audit.json')
        assert ground_audit['status']=='passed' and ground_audit['data_index_sha256']==d.sha(a.grounded_root/'index.json')
    fields=d.read(HERE/'wrapped_fields/manifest.json')
    refs=load_references()
    types=d.read('/path/to/research300/relation_grounding_train_holdout_20261101_v2/run/protocol.json')['relation_types']
    cp=torch.load(a.parent,map_location='cpu',weights_only=False)
    assert cp['normalization_sha256']==d.sha(d.F/'normalization.json')
    assert cp['official_split_lock_sha256']==d.sha(d.LOCK)
    model=model_from_checkpoint(d,types,len(FEATURE_NAMES),cp).cuda().train()
    del cp
    model.requires_grad_(True)
    for name,p in model.named_parameters():
        if '.base.values.2.' in name:p.requires_grad_(False)
    initial=d.tiny.digest(model.state_dict())
    # Both ranks initialize identical modules; exploration streams then differ.
    torch.manual_seed(419+rank);np.random.seed(419+rank);random.seed(419+rank)
    new=[p for n,p in model.named_parameters() if p.requires_grad and n.startswith('region_')]
    old=[p for n,p in model.named_parameters() if p.requires_grad and not n.startswith('region_')]
    opt=torch.optim.AdamW([dict(params=old,lr=a.learning_rate),dict(params=new,lr=a.region_learning_rate)],weight_decay=0.)
    if rank==0:a.output.mkdir(parents=True,exist_ok=a.resume is not None)
    dist.barrier()
    out=a.output/f'rank{rank}';out.mkdir(exist_ok=a.resume is not None)
    counts=d.read(d.F/'normalization.json')['scene_counts']
    scenes=sorted(lock['train_scenes'])
    if a.smoke_batches:
        scenes=[s for s in (a.smoke_scene or ['scene0190_00']) if s in records]
        assert scenes,'No prepared training scene for smoke'
    per_epoch=sum(math.ceil(counts[s]['tasks']/a.batch_size) for s in scenes)
    protocol=dict(version='RefinePlace_v1.0',epochs=a.epochs,official_train_tasks=100505,official_train_scenes=563,
        expected_total_task_exposures=100505*a.epochs,global_batch_size=a.batch_size,world_size=world,
        geometry_workers_per_rank=a.workers,distributed_backend='Gloo CPU gradient SUM after dual-GPU forward/backward, sample-weighted global loss',
        steps=2,rl_weight=.5,collision_auxiliary_weight=.2,support_bonus_weight=.1,
        learning_rate=a.learning_rate,region_learning_rate=a.region_learning_rate,
        initialization='continue train-only joint collision/orientation checkpoint; new region modules initialized zero-output',
        parent_checkpoint=str(a.parent),parent_sha256=d.sha(a.parent),initial_parameter_sha256=initial,
        normalization_sha256=d.sha(d.F/'normalization.json'),official_split_lock_sha256=d.sha(d.LOCK),
        source_cache_index_sha256=d.sha(d.F/'index.json'),reference_catalog_sha256=d.sha(CAT),
        grounded_index_sha256=d.sha(a.grounded_root/'index.json'),grounded_root=str(a.grounded_root),
        wrapped_fields_sha256=d.sha(HERE/'wrapped_fields/manifest.json'),
        region_features=list(FEATURE_NAMES),candidates_per_task=256,global_backup_candidates=32,
        support_rule='original mesh unchanged',collision_geometry='independent Alpha Wrap proxy; observation evidence',
        generic_frozen='GroundingDINO,SAM2,RoBERTa/PointBERT cached encodings,LLM parses',
        trainable='joint candidate-position-yaw fusion,reference adapters,region fusion,offsets,local actor,task/cost critics',
        validation_manifest='/path/to/research300/refineplace_sync20_inferred_official150_20261102_v1/fixed_validation150.json',
        validation_schedule='one frozen checkpoint after every epoch; official150, asynchronous sequential queue',
        independent_test_opened=False,smoke_batches=a.smoke_batches,updates_per_epoch=per_epoch,
        source_sha256=d.sha(__file__),modules={p.name:d.sha(p) for p in HERE.glob('*.py')})
    if a.resume:
        assert a.resume.resolve()==a.output.resolve()
        previous=d.read(a.output/'protocol_before_training.json')
        for key in ('parent_sha256','normalization_sha256','official_split_lock_sha256','reference_catalog_sha256',
                    'grounded_index_sha256','wrapped_fields_sha256','source_sha256','epochs','global_batch_size',
                    'learning_rate','region_learning_rate','modules'):
            assert previous[key]==protocol[key],f'Resume protocol mismatch: {key}'
        protocol=previous
    elif rank==0:d.write(a.output/'protocol_before_training.json',protocol)
    dist.barrier()
    assert d.read(a.output/'protocol_before_training.json')['initial_parameter_sha256']==initial
    library=d.paired.load_mesh_manifest('/path/to/research300/placeit_mesh_v6/mesh_manifest_v6.json')
    assets={};history=[];seen=set();multiplier=1.;start_epoch=1;start_scene=0;start_offset=0;update=0
    if a.resume:
        state=torch.load(a.resume/f'rank{rank}/last.pt',map_location='cpu',weights_only=False)
        model.load_state_dict(state['joint_policy'],strict=True);opt.load_state_dict(state['optimizer'])
        multiplier=float(state['collision_multiplier']);start_epoch,start_scene,start_offset=state['next_cursor'];update=int(state['updates'])
        seen={tuple(x) for x in state['seen_this_epoch']}
        torch.set_rng_state(state['rng']['cpu']);torch.cuda.set_rng_state_all(state['rng']['cuda'])
        np.random.set_state(state['rng']['numpy']);random.setstate(state['rng']['python'])
        resume_identity=(update,start_epoch,start_scene,start_offset,multiplier,d.tiny.digest(model.state_dict()))
        resumed=[None]*world;dist.all_gather_object(resumed,resume_identity)
        assert len(set(resumed))==1,'Rank checkpoint writes were interrupted at different updates; use a common complete epoch'
    wall_start=time.monotonic();coverage=Counter();modes=Counter();gradient_groups=Counter();stop=False

    def save(epoch,cursor,keys,filename):
        state=dict(joint_policy=model.state_dict(),optimizer=opt.state_dict(),architecture='RefinePlaceV10',
            region_dim=len(FEATURE_NAMES),epoch=epoch,updates=update,next_cursor=cursor,
            seen_this_epoch=[list(k) for k in sorted(keys)],collision_multiplier=multiplier,
            normalization_sha256=d.sha(d.F/'normalization.json'),official_split_lock_sha256=d.sha(d.LOCK),
            protocol_sha256=d.sha(a.output/'protocol_before_training.json'),
            rng=dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate()))
        atomic_checkpoint(out/filename,state)

    with CollisionGeometryBackend(workers=a.workers) as service:
      for epoch in range(start_epoch,(1 if a.smoke_batches else a.epochs)+1):
        epoch_start=time.monotonic()
        if epoch!=start_epoch:seen=set()
        order=sorted(scenes,key=lambda s:hashlib.sha256(f'v1:{epoch}:{s}'.encode()).hexdigest())
        for si,scene in enumerate(order):
          if epoch==start_epoch and si<start_scene:continue
          d.guard(lock,scene);setup=time.monotonic()
          source=V10Scene(scene,lock,library,driver=d,grounded_root=a.grounded_root,budget=256)
          positions=sorted(range(len(source.registry)),key=lambda p:hashlib.sha256(f'v1:{epoch}:{scene}:{p}'.encode()).hexdigest())
          if a.smoke_batches:positions=positions[:a.batch_size*a.smoke_batches]
          pair_cache={};setup_seconds=time.monotonic()-setup
          for offset in range(0,len(positions),a.batch_size):
            if epoch==start_epoch and si==start_scene and offset<start_offset:continue
            tick=time.monotonic();batch=positions[offset:offset+a.batch_size];local=batch[rank::world]
            rows=source.inputs(local if local else batch[:1]);keys=[(r['scene'],r['group_id']) for r in rows]
            actual=keys if local else [];assert not set(actual)&seen;seen.update(actual)
            for row in rows:
                uid=row['asset_uuid']
                if uid not in assets:
                    assets[uid]=torch.load(d.b.B/'placeit3d_reproduction_v1/placeit3d/data/pointbert_embeddings'/(uid+'.pt'),map_location='cpu',weights_only=True).reshape(-1).cuda()
            features,text,asset,ctx,decisions,inferred=d.make_context(rows,assets,refs,epoch,types)
            regions=torch.as_tensor(np.stack([r['region_features'] for r in rows]),device='cuda',dtype=torch.float32)
            model.reference_head.inferred_mask=inferred
            opt.zero_grad(set_to_none=True);queries=0;geometry_seconds=0.
            class MeasuredService:
                def query_many(self,tasks):
                    nonlocal queries,geometry_seconds
                    for q in tasks:assert q['scene']==scene and q['split']=='train'
                    t=time.monotonic();result=service.query_many(tasks);geometry_seconds+=time.monotonic()-t;queries+=len(tasks);return result
            obs=[{k:r[k] for k in ('scene','split','asset_uuid','dims','centers')} for r in rows]
            with model.reference_batch(**ctx),model.region_batch(regions):
                tr=rollout.rollout_batch(model,features,text,asset,ctx['candidate_mask'],obs,MeasuredService(),
                    steps=2,deterministic=False,apply_offsets=True,anti_cycle=True,mode_epsilon=.2)
            labels=source.labels(rows);quality=[]
            for key,result in zip(keys,tr):
                an=labels[key]['annotation']
                if an not in pair_cache:
                    z=source.labels_arrays;lo,hi=map(int,z['annotation_offsets'][an:an+2])
                    pair_cache[an]=make_positive_pose_pairs(z['scene_xyz'],np.asarray(z['valid_ids'][lo:hi]),np.asarray(z['valid_mask8'][lo:hi]))
                quality.append(paired_pose_quality(result['center'],result['yaw'],pair_cache[an])['quality'])
            scores=torch.stack([r['scores'] for r in tr]);offsets=torch.stack([r['supervised_offsets'] for r in tr])
            sup,_=d.supervised_safe(scores,offsets,labels,keys)
            rl=constrained_rl_losses(tr,quality,collision_multiplier=multiplier,rl_weight=.5)
            aux=collision_auxiliary_loss(tr)
            loss=(sup+rl['rl_loss']+.2*aux).sum()*(1. if local else 0.)/len(batch)
            assert torch.isfinite(loss);loss.backward()
            active=bool((ctx['reference_mask'].any(-1)&(ctx['relation_mask']|inferred).any(-1)).any())
            if not active:
                for n,p in model.named_parameters():
                    if n.startswith(('inner.local_reference.','inner.yaw_reference.')) or 'reference_head.' in n:p.grad=None
            synchronize_gradients(model)
            for n,p in model.named_parameters():
                if p.grad is not None and bool(p.grad.abs().sum()>0):gradient_groups[n.split('.')[0]]+=1
            norm=float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True))
            opt.step()
            known=np.asarray(rl['collision_known'],bool);cost=np.asarray(rl['collision_costs'])
            stats=torch.tensor([float(cost[known].sum()) if local else 0.,int(known.sum()) if local else 0.],device='cpu',dtype=torch.float64)
            dist.all_reduce(stats);cost_sum,known_sum=stats.cpu().tolist()
            before=multiplier
            if known_sum:multiplier=float(np.clip(multiplier+.05*(cost_sum/known_sum-.05),.1,10.))
            update+=1
            if local:
                coverage.update(tasks=len(local),active_reference=int((ctx['reference_mask'].any(-1)&(ctx['relation_mask']|inferred).any(-1)).sum()),
                    continuous_collision=sum(r['geometry'].get('collision_cost_known',False) for r in tr))
                modes.update(rl['collision_modes'])
            h=dict(epoch=epoch,update=update,scene=scene,tasks=len(local),task_ids=actual,
                loss=float(loss.detach()),supervised_loss=float(sup.mean().detach()),rl_loss=float(rl['rl_loss'].mean().detach()),
                collision_aux_loss=float(aux.mean().detach()),paired_pose_quality=float(np.mean(quality)),
                global_collision_cost=cost_sum/max(known_sum,1),collision_multiplier_used=before,collision_multiplier_next=multiplier,
                geometry_queries=queries,cpu_geometry_seconds=geometry_seconds,gradient_norm=norm,
                batch_seconds=time.monotonic()-tick,scene_setup_seconds=setup_seconds if offset==0 else 0.,
                region_reports=[r.get('region_report',{}) for r in rows],collision_modes=dict(Counter(rl['collision_modes'])))
            h=d.clean(h);history.append(h)
            with (out/'steps.jsonl').open('a',encoding='utf-8') as fp:fp.write(json.dumps(h)+'\n')
            d.write(out/'progress.json',dict(status='training',epoch=epoch,epochs=a.epochs,updates=update,
                local_tasks_this_epoch=len(seen),global_tasks_per_epoch=100505,coverage=dict(coverage),
                collision_modes=dict(modes),elapsed_seconds=time.monotonic()-wall_start,last=h))
            print(json.dumps({k:v for k,v in h.items() if k not in ('region_reports','task_ids')}),flush=True)
            cursor=[epoch,si,offset+a.batch_size]
            if cursor[2]>=len(positions):cursor=[epoch,si+1,0]
            if cursor[1]>=len(order):cursor=[epoch+1,0,0]
            flags=torch.tensor([int((a.output/'stop_after_current_batch').exists())],device='cpu')
            dist.all_reduce(flags,op=dist.ReduceOp.MAX)
            stop=bool(flags.item()) or bool(a.smoke_batches and len(history)>=a.smoke_batches)
            if update%a.checkpoint_every==0 or stop:
                save(epoch,cursor,seen if cursor[0]==epoch else set(),'last.pt')
            if stop:break
          source.close()
          if stop:break
        if a.smoke_batches:
            save(epoch,[epoch+1,0,0],set(),'smoke.pt');break
        if stop:break
        total=torch.tensor([len(seen)],device='cpu',dtype=torch.int64);dist.all_reduce(total)
        assert int(total.item())==100505,'An epoch must include every official training task exactly once'
        save(epoch,[epoch+1,0,0],set(),f'epoch{epoch}.pt')
        save(epoch,[epoch+1,0,0],set(),'last.pt')
        digest=d.tiny.digest(model.state_dict());digests=[None]*world;dist.all_gather_object(digests,digest)
        assert len(set(digests))==1,'GPU model parameters diverged'
        if rank==0:
            d.write(a.output/f'epoch{epoch}_ready.json',dict(epoch=epoch,tasks=100505,checkpoint=str(out/f'epoch{epoch}.pt'),
                checkpoint_sha256=d.sha(out/f'epoch{epoch}.pt'),training_protocol=str(a.output/'protocol_before_training.json'),
                parameter_digest=digest,synchronized=True,epoch_seconds=time.monotonic()-epoch_start))
        dist.barrier()
    digest=d.tiny.digest(model.state_dict());digests=[None]*world;dist.all_gather_object(digests,digest)
    assert len(set(digests))==1
    d.write(out/'summary.json',dict(status='smoke_complete' if a.smoke_batches else 'stopped' if stop else 'training_complete',
        updates=update,epochs_completed=epoch if not stop or a.smoke_batches else epoch-1,
        task_exposures=coverage['tasks'],coverage=dict(coverage),collision_modes=dict(modes),
        nonzero_gradient_groups=dict(gradient_groups),synchronized_parameter_digest=digest,
        wall_seconds=time.monotonic()-wall_start,mean_batch_seconds=float(np.mean([h['batch_seconds'] for h in history])),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),parent_checkpoint_unchanged=d.sha(a.parent)==protocol['parent_sha256']))
    dist.destroy_process_group()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True);p.add_argument('--grounded-root',type=Path,required=True)
    p.add_argument('--epochs',type=int,default=8);p.add_argument('--batch-size',type=int,default=64)
    p.add_argument('--workers',type=int,default=12);p.add_argument('--parent',type=Path,default=PARENT)
    p.add_argument('--learning-rate',type=float,default=1e-5);p.add_argument('--region-learning-rate',type=float,default=1e-4)
    p.add_argument('--checkpoint-every',type=int,default=100);p.add_argument('--smoke-batches',type=int,default=0)
    p.add_argument('--smoke-scene',action='append');p.add_argument('--resume',type=Path)
    run(p.parse_args())
