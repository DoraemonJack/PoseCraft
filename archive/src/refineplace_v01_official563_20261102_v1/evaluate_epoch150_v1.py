"""Fixed 150 official-validation inference and unmodified container evaluation."""
import argparse, hashlib, json, math, os, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
from torch import nn

M=Path('/path/to/research300/refineplace_v01_official563_modules_20261102_v1');sys.path.insert(0,str(M))
import train_official563_joint_v4_cpu12_fresh as d
sys.path.insert(0,'/path/to/research100/native_val297_current_model_20261102_v1')
import run_native_val297_allmask_v1 as old
import prepare_official_val297 as prep
import train_joint_pose_v6 as v6
B=Path('/path/to/research100/native_val297_current_model_20261102_v1');D=B/'allmask_current'
V50=Path('/path/to/research100/native_val50_current_model_20261102_v1');R=V50/'real_image_observations/reference_predictions'
sha=d.sha;read=d.read;write=d.write;clean=d.clean

def main(a):
    torch.set_num_threads(2);a.output.mkdir(parents=True,exist_ok=False)
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False);assert cp['protocol_sha256']==sha(a.training_protocol)
    assert cp['normalization_sha256']==sha(d.F/'normalization.json') and cp['official_split_lock_sha256']==sha(d.LOCK)
    manifest=read(a.manifest);tasks=manifest['tasks'];assert len(tasks)==150
    if a.smoke:tasks=tasks[:2]
    if a.official_smoke:tasks=tasks[:32]
    lock=read(d.LOCK);assert all(t['scene'] in lock['validation_scenes'] and t['scene'] not in lock['train_scenes'] for t in tasks)
    types=read('/path/to/research300/relation_grounding_train_holdout_20261101_v2/run/protocol.json')['relation_types']
    norm=read(d.F/'normalization.json');mean,std=np.asarray(norm['mean']),np.asarray(norm['std']);oldnorm=read('/path/to/research100/prospective_train450_feature_cache_20261101_v1/normalization.json')
    geom=read(D/'geometry_provenance.json');seal=read(R/'sealed_reference_outputs.json');assert all(sha(R/n)==h for n,h in seal['files'].items())
    reg=read(R/'selected_reference_registry.json');small=read(V50/'locked_tasks50.json')['tasks'];bykey={(t['asset_scene'],t['entry']):i for i,t in enumerate(small)}
    prior=nn.Sequential(nn.Linear(141,192),nn.SiLU(),nn.Linear(192,128),nn.SiLU(),nn.Linear(128,1));base=d.paired.BatchedJointModel(47,prior.state_dict(),768,.25);inner=d.paired.ReferenceJointPolicy(base,d.paired.ReferenceConditionedPositionHead(len(types)));model=d.joint.ReferencePosePolicy(inner,len(types),base.actor_body[0].in_features,preserve_candidate_logmass=False).cuda();model.load_state_dict(cp['joint_policy'],strict=True);model.eval().requires_grad_(False)
    before=d.tiny.digest(model.state_dict());library=d.paired.load_mesh_manifest('/path/to/research300/placeit_mesh_v6/mesh_manifest_v6.json');scenes={};assets={};records=[];counts={'real_reference_available':0,'real_reference_active':0,'mask_fallback':0}
    for task in tasks:
        i=task['original_entry_id'];scene=task['scene'];uid=task['asset_uuid'];dims=np.asarray(task['asset_size'])
        with np.load(D/'observations'/f'{i:03d}.npz',allow_pickle=False) as z:cached={k:z[k].copy() for k in z.files}
        if scene not in scenes:
            payload=torch.load(d.b.P/'scans_full_refer_v1/val'/(scene+'_refer.pth'),map_location='cpu',weights_only=False);xyz=np.asarray(payload[0]).copy();rgb=np.asarray(payload[1]).copy();del payload
            g=geom[scene];A=np.asarray(g['A']);mp,me=library._file('scenes',scene);assert me['sha256']==g['mesh_sha256']
            with np.load(mp,allow_pickle=False) as z:world=z['vertices'].copy()
            assert np.max(np.linalg.norm(world-xyz-np.asarray(g['translation']),axis=1))<1e-5
            sx,sf=d.paired.load_scene_mesh(d.b.P,scene,'val',library,reference_xyz=xyz.astype(np.float32));scenes[scene]=dict(xyz=xyz,rgb=rgb,A=A,cache={},adapter=d.paired.MeshSceneAdapterV6(sx,sf,library))
        sd=scenes[scene];ids=cached['candidate_ids'];raw=old.raw_features(sd['rgb'],ids,dims,task['prompt']);raw,centers=v6.aligned_task(raw,sd['xyz'][ids],sd['xyz'],sd['A'],sd['cache']);assert np.array_equal(centers,cached['centers'])
        oldf=v6.make_features(raw,np.asarray(oldnorm['mean']),np.asarray(oldnorm['std']),'cpu').numpy();assert np.array_equal(oldf,cached['features'])
        features=v6.make_features(raw,mean,std,'cpu').numpy();row=dict(scene=scene,split='val',group_id=i,features=features,centers=centers,candidate_ids=ids,text=cached['text'],dims=dims,asset_uuid=uid,asset_scene=task['asset_scene'],entry=task['entry'],prompt_sha256=task['prompt_sha256']);refs={}
        k=bykey.get((task['asset_scene'],task['entry']))
        if k is not None:
            rr=reg[k];row['group_id']=k;identity={x:row[x] for x in ['scene','asset_scene','entry','group_id','prompt_sha256']};assert all(rr[x]==identity[x] for x in identity)
            assert d.candidate_context_fingerprint(identity,ids,centers)==rr['candidate_context_sha256'];path=Path(rr['selected_sidecar']);assert sha(path)==rr['selected_sidecar_sha256'];refs[scene,k]=[dict(frame=rr['selected_frame'],sidecar_path=str(path),sidecar_sha256=sha(path))];counts['real_reference_available']+=1
        if uid not in assets:assets[uid]=torch.load(d.b.B/'placeit3d_reproduction_v1/placeit3d/data/pointbert_embeddings'/(uid+'.pt'),map_location='cpu',weights_only=True).reshape(-1).cuda()
        f,t,av,ctx,_=d.make_context([row],assets,refs,1,types);active=bool(ctx['reference_mask'].any() and ctx['relation_mask'].any());counts['real_reference_active']+=active;counts['mask_fallback']+=not active
        class Service:
            def query_many(self,qs):
                out=[]
                for q in qs:
                    assert q['scene']==scene and q['split']=='val';ad=sd['adapter'];ad.set_asset(uid,dims);out.append(ad.features(q['center'],q['dims'],q['yaw']))
                return out
        obs=[{k:row[k] for k in ['scene','split','asset_uuid','dims','centers']}]
        with torch.inference_mode(),model.reference_batch(**ctx):tr=d.joint.rollout_batch(model,f,t,av,ctx['candidate_mask'],obs,Service(),steps=2,deterministic=True,apply_offsets=True,anti_cycle=True,mode_epsilon=0.)[0]
        records.append(dict(original_entry_id=i,scene=scene,center=clean(tr['center']),yaw=float(tr['yaw']),reference_active=active))
        write(a.output/'progress.json',dict(completed=len(records),total=150,counts=counts))
    assert before==d.tiny.digest(model.state_dict());write(a.output/'sealed_predictions.json',records)
    if a.smoke:
        write(a.output/'smoke_report.json',dict(tasks=len(records),counts=counts,weights_unchanged=True));return
    staged={x['scene']:x for x in read(B/'staged_meshes.json')};rawscans={};export=[]
    for task,pred in zip(tasks,records):
        scene=task['scene'];assert pred['original_entry_id']==task['original_entry_id'];g=geom[scene];A=np.asarray(g['A'],float).reshape(4,4);translation=np.asarray(g['translation']);offset=A[:3,:3]@translation+A[:3,3]
        if scene not in rawscans:rawscans[scene]=prep.vertices(staged[scene]['file'])
        with np.load(D/'observations'/('%03d.npz'%task['original_entry_id']),allow_pickle=False) as z:
            ids=z['candidate_ids'];centers=z['centers'];dims=z['dims'];assert np.allclose(dims,task['asset_size'],atol=1e-6);assert np.linalg.norm(rawscans[scene][ids]@A[:3,:3].T+A[:3,3]-(centers+offset),axis=1).max()<2e-5
        world=np.asarray(pred['center'])+offset+np.array([0.,0.,dims[2]/2]);export.append(dict(entry_id=task['original_entry_id'],dataset_idx=-1,tx=float(world[0]),ty=float(world[1]),tz=float(world[2]),rotation=math.degrees(pred['yaw'])%360))
    write(a.output/'predictions.json',export);(a.output/'tmp').mkdir();(a.output/'result').mkdir()
    oldjob=read(B/'official_job.json');cmd=oldjob['command'].copy();cmd[cmd.index('--name')+1]=f'refineplace-sync20-epoch{a.epoch}-val150'+('-smoke32' if a.official_smoke else '');cmd[cmd.index('--device')+1]='nvidia.com/gpu=1'
    for i in range(1,len(cmd)):
        if cmd[i-1]=='-v':
            src,dst,mode=cmd[i].rsplit(':',2);maps={'/tmp':a.output/'tmp','/app/mounted_scannet/scans':B/'scans','/app/val_input/predictions.json':a.output/'predictions.json','/app/val_output':a.output/'result'}
            if dst in maps:cmd[i]=str(maps[dst])+':'+dst+':'+mode
    job=subprocess.run(cmd,capture_output=True,text=True,timeout=90);job.check_returncode();cid=job.stdout.strip();docker=cmd[:3];wait=subprocess.run(docker+['wait',cid],capture_output=True,text=True,timeout=14400);wait.check_returncode();assert wait.stdout.strip()=='0';result=a.output/'result'/'val_results.json';assert result.exists();official=read(result)
    summary=dict(status='official_evaluation_complete',epoch=a.epoch,tasks=len(tasks),counts=counts,checkpoint_sha256=sha(a.checkpoint),manifest_sha256=sha(a.manifest),official_result_sha256=sha(result),official=official,warning='reference missing uses mask; partial validation, not untouched test')
    write(a.output/'report.json',summary)
    if not a.official_smoke:write(a.gate,summary)
    print(json.dumps(summary),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--training-protocol',type=Path,required=True);p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--gate',type=Path,required=True);p.add_argument('--epoch',type=int,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--official-smoke',action='store_true');main(p.parse_args())
