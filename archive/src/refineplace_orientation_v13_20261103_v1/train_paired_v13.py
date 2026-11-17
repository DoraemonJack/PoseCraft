"""Frozen V11 + paired STOP/rotation and one-step action-value training.

This fits heading reward proxies and separate geometric risk evidence. It is
not a reproduction of full-horizon Q-learning or an official-success critic.
"""
import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
V10 = Path('/path/to/research300/refineplace_v1_0_20261103_v1')
V11 = Path('/path/to/research300/refineplace_orientation_v11_20261103_v1')
OLD = Path('/path/to/research300/refineplace_rl_geometry_10k_20261102_v1/joint_collision_orientation')
MODULES = Path('/path/to/research300/refineplace_v01_official563_modules_20261102_v1')
sys.path[:0] = [str(HERE), str(V11), str(V10), str(OLD), str(MODULES)]
from build_pair_cache_v13 import load_cache
from model_v13 import model_from_checkpoint, select_action
from paired_value_losses import paired_supervision_loss


def write(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)


class PairedTrainer:
    def __init__(self, driver, model, refs, types):
        self.d, self.model, self.refs, self.types = driver, model, refs, types
        self.assets = {}

    def batch(self, rows):
        started = time.monotonic()
        model, d = self.model, self.d
        device = next(model.parameters()).device
        for row in rows:
            if row['split'] != 'train' or not row['metadata']['auxiliary_gt_position']:
                raise ValueError('Paired teachers must be sealed training GT positions')
            uid = row['asset_uuid']
            if uid not in self.assets:
                self.assets[uid] = torch.load(d.b.B / 'placeit3d_reproduction_v1/placeit3d/data/pointbert_embeddings' / (uid+'.pt'),
                    map_location='cpu', weights_only=True).reshape(-1).to(device)
        features, text, asset, ctx, _, inferred = d.make_context(rows, self.assets, self.refs, 2, self.types)
        b = len(rows)
        ctx['candidate_mask'] = torch.ones((b,1), dtype=torch.bool, device=device)
        if ctx['candidate_xyz'].shape != (b,1,3):
            raise ValueError('One exact GT position per paired training group required')
        model.reference_head.inferred_mask = inferred
        region = torch.as_tensor(np.stack([r['region_features'] for r in rows]), device=device, dtype=torch.float32)
        yaws = torch.as_tensor(np.stack([r['current_yaws'] for r in rows]), device=device, dtype=torch.float32)
        expected = torch.arange(8, device=device, dtype=torch.float32) * math.pi/4
        if yaws.shape != (b,8) or not torch.allclose(yaws, expected[None].expand(b,-1), atol=1e-6, rtol=0):
            raise ValueError('Paired current yaw states must contain all eight ordered bins')
        vectors = torch.as_tensor(np.stack([r['geom_vector'] for r in rows]), device=device, dtype=torch.float32)
        valid = torch.as_tensor(np.stack([r['valid_mask8'][0] for r in rows]), device=device, dtype=torch.bool)
        cost = torch.as_tensor(np.stack([r['collision_cost'] for r in rows]), device=device, dtype=torch.float32)
        known = torch.as_tensor(np.stack([r['collision_known'] for r in rows]), device=device, dtype=torch.bool)
        with model.reference_batch(**ctx), model.region_batch(region):
            with torch.no_grad():
                _, fused, _ = model.baseline.candidates_batch(features, text, asset, ctx['candidate_mask'])
            extra = torch.cat((vectors, torch.zeros((b,8,3),device=device), yaws.sin()[...,None],
                               yaws.cos()[...,None], torch.ones((b,8,1),device=device)), -1)
            state = torch.cat((fused[:,0,None].expand(b,8,-1), extra), -1).reshape(b*8,-1)
            centers = ctx['candidate_xyz'][:,0,None].expand(b,8,3).reshape(b*8,3)
            ids = torch.arange(b,device=device,dtype=torch.long).repeat_interleave(8)
            outputs = model.action_values(state, centers, yaws.reshape(-1), ids)
            losses = paired_supervision_loss(outputs, valid, cost, known)
        with torch.no_grad():
            choice = select_action(outputs)
            selected = choice['action_index'].reshape(b,8)
            endpoints = (torch.arange(8,device=device)[None] + selected - 1) % 8
            end_valid = valid.gather(-1,endpoints)
            stopped = choice['stopped'].reshape(b,8)
            angles = torch.arange(8,device=device,dtype=torch.float32)*math.pi/4
            diff = angles[None,:,None] - angles[None,None,:]
            distance = torch.remainder(diff+math.pi,2*math.pi).sub(math.pi).abs().expand(b,-1,-1)
            error = distance.masked_fill(~valid[:,None,:],torch.inf).amin(-1)
            next_error = error.gather(-1,endpoints)
            next_known, next_cost = known.gather(-1,endpoints), cost.gather(-1,endpoints)
            paired_known = known & next_known
            risk_increase = paired_known & (next_cost > cost + .02)
            mode_stop = outputs['mode_logits'].argmax(-1).reshape(b,8) == 0
            record = dict(tasks=b, states=b*8, loss=float(losses['loss'].detach()),
                components={k:float(losses[k].detach()) for k in ('mode_loss','direction_loss','task_value_loss',
                    'advantage_loss','collision_loss','collision_known_loss')},
                correct_states=int(valid.sum()), incorrect_states=int((~valid).sum()),
                stop_on_correct=int((stopped & valid).sum()), correct_preserved=int((valid & end_valid).sum()),
                correct_broken=int((valid & ~end_valid).sum()), incorrect_fixed=int((~valid & end_valid).sum()),
                incorrect_improved=int((~valid & (next_error < error-1e-5)).sum()),
                incorrect_worsened=int((~valid & (next_error > error+1e-5)).sum()),
                selected_stop=int(stopped.sum()), selected_rotate=int((~stopped).sum()),
                mode_stop_correct=int((mode_stop & valid).sum()), mode_rotate_correct=int((~mode_stop & ~valid).sum()),
                angular_error_before_sum_deg=float(torch.rad2deg(error).sum()),
                angular_error_after_sum_deg=float(torch.rad2deg(next_error).sum()),
                known_risk_pairs=int(paired_known.sum()), known_risk_increases=int(risk_increase.sum()),
                selected_unknown_evidence=int(choice['uncertain'].sum()),
                observed_unknown_endpoint_evidence=int((~next_known).sum()),
                maximum_predicted_task_gain=float(choice['task_gain'].max()),
                active_reference_tasks=int((ctx['reference_mask'].any(-1) & (ctx['relation_mask'] | inferred).any(-1)).sum()),
                batch_seconds=time.monotonic()-started)
        return losses['loss'], record


def aggregate(records):
    counts = ('tasks','states','correct_states','incorrect_states','stop_on_correct','correct_preserved','correct_broken',
        'incorrect_fixed','incorrect_improved','incorrect_worsened','selected_stop','selected_rotate','mode_stop_correct',
        'mode_rotate_correct','known_risk_pairs','known_risk_increases','selected_unknown_evidence',
        'observed_unknown_endpoint_evidence','active_reference_tasks')
    total = {k:sum(r[k] for r in records) for k in counts}
    total['loss'] = sum(r['loss']*r['tasks'] for r in records)/total['tasks']
    total['components'] = {k:sum(r['components'][k]*r['tasks'] for r in records)/total['tasks'] for k in records[0]['components']}
    for field, numerator, denominator in (
        ('stop_recall','stop_on_correct','correct_states'),('valid_pose_preservation','correct_preserved','correct_states'),
        ('valid_pose_damage','correct_broken','correct_states'),('invalid_pose_one_step_repair','incorrect_fixed','incorrect_states'),
        ('invalid_pose_angular_improvement','incorrect_improved','incorrect_states'),
        ('mode_stop_recall','mode_stop_correct','correct_states'),('mode_rotate_recall','mode_rotate_correct','incorrect_states')):
        total[field] = total[numerator]/max(1,total[denominator])
    for name in ('before','after'):
        total['angular_error_'+name+'_mean_deg'] = sum(r['angular_error_'+name+'_sum_deg'] for r in records)/total['states']
    total['scope'] = 'Same training GT positions and all eight starts; one-step diagnostic, not held-out or official success'
    return total


def main(a):
    import train_official563_joint_v7_sync_full as d
    from train_v10 import load_references, atomic_checkpoint, CAT
    from support_regions import FEATURE_NAMES
    torch.set_num_threads(1); torch.set_num_interop_threads(1); torch.cuda.set_device(0)
    torch.manual_seed(a.seed); random.seed(a.seed); np.random.seed(a.seed)
    rows = load_cache(a.cache)
    if a.max_tasks:
        rows = rows[:a.max_tasks]
    lock = d.read(d.LOCK)
    for row in rows:
        d.guard(lock,row['scene'])
    metadata = d.read(a.cache/'metadata.json')
    # Both cache and original teacher cache retain the official split identity.
    assert metadata['split_lock_sha256'] == d.sha(d.LOCK)
    assert metadata['normalization_sha256'] == d.sha(d.F/'normalization.json')
    assert metadata['catalog_sha256'] == d.sha(CAT)
    cp = torch.load(a.parent,map_location='cpu',weights_only=False)
    assert cp['architecture']=='RefinePlaceV11'
    assert d.sha(a.parent)=='480bb902bbbeb174a36da395fdf531c1845386cc66d653478c2584081e41c774'
    assert cp['normalization_sha256']==d.sha(d.F/'normalization.json')
    assert cp['official_split_lock_sha256']==d.sha(d.LOCK)
    types = d.read('/path/to/research300/relation_grounding_train_holdout_20261101_v2/run/protocol.json')['relation_types']
    refs = load_references()
    model = model_from_checkpoint(d,types,len(FEATURE_NAMES),cp).cuda()
    del cp
    parameters = [p for p in model.parameters() if p.requires_grad]
    assert parameters and all(not p.requires_grad for p in model.baseline.parameters())
    optimizer = torch.optim.AdamW(parameters,lr=a.learning_rate,weight_decay=0.)
    baseline_digest = d.tiny.digest(model.baseline.state_dict())
    a.output.mkdir(parents=True,exist_ok=False)
    protocol = dict(version='RefinePlace_orientation_v13_paired_value_pilot', architecture='RefinePlaceV13',
        phase='paired_heading_and_one_step_action_value', tasks=len(rows), epochs=a.epochs, seed=a.seed,
        task_batch_size=a.batch_size, paired_states_per_task=8, actions_degrees=[-45,0,45],
        trainable_parameters=sum(p.numel() for p in parameters), learning_rate=a.learning_rate,
        baseline_frozen=True, baseline_digest=baseline_digest,
        parent_checkpoint=str(a.parent), parent_checkpoint_sha256=d.sha(a.parent),
        cache_data_sha256=metadata['data_sha256'], cache_metadata_sha256=d.sha(a.cache/'metadata.json'),
        normalization_sha256=d.sha(d.F/'normalization.json'), official_split_lock_sha256=d.sha(d.LOCK),
        reference_catalog_sha256=d.sha(CAT),
        task_value_target='At exact training position: 0.5*valid_yaw+0.5*(1+max_valid_cos_angle)/2',
        stop_rotate_balance='Per position, mean STOP loss and mean ROTATE loss each receive half weight',
        collision_targets='Actual queried pose cost; unknown has NaN target and false loss mask',
        selector=dict(task_margin=.02,uncertain_margin=.05,maximum_known_cost=.25,allowed_cost_increase=.02,known_threshold=.7),
        inference='Frozen original V11 two-step placement followed by at most two yaw-only value-selected steps',
        task_q_is_official_success_probability=False, full_horizon_q_learning=False,
        test_labels_read=False, validation_labels_read=False, original_full_training_resumed=False,
        source_hashes={p.name:d.sha(p) for p in HERE.glob('*.py')})
    write(a.output/'protocol_before_training.json',protocol)
    runner = PairedTrainer(d,model,refs,types)
    started = time.monotonic(); updates=0; tasks_seen=0; nonzero=Counter(); epoch_reports=[]
    def save(filename,epoch):
        assert d.tiny.digest(model.baseline.state_dict())==baseline_digest
        atomic_checkpoint(a.output/filename,dict(architecture='RefinePlaceV13',joint_policy=model.state_dict(),
            optimizer=optimizer.state_dict(),epoch=epoch,updates=updates,region_dim=len(FEATURE_NAMES),
            phase='paired_value',tasks_completed=tasks_seen,
            normalization_sha256=protocol['normalization_sha256'], official_split_lock_sha256=protocol['official_split_lock_sha256'],
            protocol_sha256=d.sha(a.output/'protocol_before_training.json'),baseline_digest=baseline_digest,
            parent_checkpoint_sha256=protocol['parent_checkpoint_sha256']))
    def diagnostic(label):
        model.eval(); records=[]
        with torch.no_grad():
            for offset in range(0,len(rows),a.batch_size):
                _,r=runner.batch(rows[offset:offset+a.batch_size]); records.append(r)
                write(a.output/'progress.json',dict(status=label,completed=min(offset+a.batch_size,len(rows)),total=len(rows),
                    elapsed_seconds=time.monotonic()-started))
        result=aggregate(records);write(a.output/(label+'.json'),result);return result
    before=diagnostic('before_training_diagnostic')
    for epoch in range(a.epochs):
        model.train(); order=np.random.default_rng(a.seed+epoch).permutation(len(rows)); reports=[]
        for offset in range(0,len(rows),a.batch_size):
            batch=[rows[i] for i in order[offset:offset+a.batch_size]]
            optimizer.zero_grad(set_to_none=True)
            loss,record=runner.batch(batch)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite paired training loss')
            loss.backward()
            if any(p.grad is not None for p in model.baseline.parameters()):
                raise RuntimeError('Frozen placement model received a gradient')
            for name,p in model.named_parameters():
                if p.grad is not None and bool(p.grad.abs().sum()>0):nonzero[name.split('.')[0]]+=1
            norm=float(torch.nn.utils.clip_grad_norm_(parameters,1.,error_if_nonfinite=True))
            optimizer.step();updates+=1;tasks_seen+=len(batch)
            record.update(epoch=epoch+1,update=updates,gradient_norm=norm,
                task_ids=[[r['scene'],int(r['group_id'])] for r in batch])
            reports.append(record)
            with (a.output/'steps.jsonl').open('a',encoding='utf-8') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            write(a.output/'progress.json',dict(status='training',epoch=epoch+1,epochs=a.epochs,
                completed=min(offset+a.batch_size,len(rows)),total=len(rows),updates=updates,
                elapsed_seconds=time.monotonic()-started,last={k:v for k,v in record.items() if k!='task_ids'}))
            if (a.output/'stop_after_current_batch').exists():
                save('last.pt',epoch);write(a.output/'progress.json',dict(status='stopped',updates=updates));return
        epoch_reports.append(aggregate(reports)); save('last.pt',epoch+1)
        print(json.dumps(dict(epoch=epoch+1,summary=epoch_reports[-1])),flush=True)
    after=diagnostic('after_training_diagnostic')
    save('final.pt',a.epochs)
    summary=dict(status='training_complete',architecture='RefinePlaceV13',tasks=len(rows),epochs=a.epochs,updates=updates,
        before=before,after=after,epoch_reports=epoch_reports,nonzero_gradient_groups=dict(nonzero),
        baseline_weights_unchanged=d.tiny.digest(model.baseline.state_dict())==baseline_digest,
        parent_checkpoint_unchanged=d.sha(a.parent)==protocol['parent_checkpoint_sha256'],
        final_checkpoint_sha256=d.sha(a.output/'final.pt'),elapsed_seconds=time.monotonic()-started,
        train_geometry_queries=0,geometry_source='verified precomputed same-pose cache',independent_test_used=False)
    write(a.output/'summary.json',summary)
    write(a.output/'progress.json',{k:v for k,v in summary.items() if k not in ('before','after','epoch_reports','nonzero_gradient_groups')})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('cache','parent','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--epochs',type=int,default=3)
    p.add_argument('--learning-rate',type=float,default=3e-4)
    p.add_argument('--seed',type=int,default=613)
    p.add_argument('--max-tasks',type=int,default=0)
    main(p.parse_args())
