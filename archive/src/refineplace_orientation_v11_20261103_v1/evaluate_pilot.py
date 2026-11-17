"""Paired V1.0/V1.1 pilot inference on unchanged fixed validation150.

Inference-only by default. The unchanged official evaluator starts only when
--official is explicitly supplied. Original sources and test exports stay intact.
"""
import argparse,json,math,os,subprocess,sys,time,hashlib
from collections import Counter
from pathlib import Path
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
V10=Path('/path/to/research300/refineplace_v1_0_20261103_v1')
M=Path('/path/to/research300/refineplace_v01_official563_modules_20261102_v1')
PREV=Path('/path/to/research300/refineplace_rl_geometry_10k_20261102_v1/joint_collision_orientation')
sys.path[:0]=[str(HERE),str(V10),str(PREV),str(M),'/path/to/research100/native_val297_current_model_20261102_v1']
import train_official563_joint_v4_cpu12_fresh as legacy
import train_official563_joint_v7_sync_full as d
import run_native_val297_allmask_v1 as old
import prepare_official_val297 as prep
import train_joint_pose_v6 as v6
import joint_rollout_v1 as rollout
from model_v10 import model_from_checkpoint as baseline_model
from model_v11 import model_from_checkpoint as upgraded_model
from wrapped_geometry_backend_v10 import install_geometry
from support_regions import prepare_scene,prepare_asset,propose,load_grounded_surfaces,FEATURE_NAMES
from novel_candidate_features_v1 import reconstruct
from v1_scene import build_task_candidates

B=Path('/path/to/research100/native_val297_current_model_20261102_v1');D=B/'allmask_current'
V50=Path('/path/to/research100/native_val50_current_model_20261102_v1');R=V50/'real_image_observations/reference_predictions'
sha=d.sha;read=d.read;write=d.write;clean=d.clean


def main(a):
    torch.set_num_threads(2);torch.cuda.set_device(0)
    a.output.mkdir(parents=True,exist_ok=False)
    cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False)
    assert cp['architecture'] in ('RefinePlaceV10','RefinePlaceV11')
    assert cp['protocol_sha256']==sha(a.training_protocol)
    architecture=cp['architecture']
    checkpoint_epoch=cp.get('epoch');parent_epoch=cp.get('parent_epoch')
    constructor=upgraded_model if architecture=='RefinePlaceV11' else baseline_model
    checkpoint_sha256=sha(a.checkpoint)
    started=time.monotonic()
    assert cp['normalization_sha256']==sha(d.F/'normalization.json') and cp['official_split_lock_sha256']==sha(d.LOCK)
    tasks=read(a.manifest)['tasks'];assert len(tasks)==150
    assert len({int(t['original_entry_id']) for t in tasks})==150
    assert all(hashlib.sha256(t['prompt'].encode()).hexdigest()==t['prompt_sha256'] for t in tasks)
    grounded_index=read(a.grounded_root/'index.json');grounded_audit=read(a.grounded_root.parent/'admission_audit.json')
    assert grounded_index['partition']=='fixed_validation150' and grounded_index['task_manifest_sha256']==sha(a.manifest)
    assert grounded_audit['status']=='passed'
    assert grounded_audit['data_index_sha256']==sha(a.grounded_root/'index.json')
    grounded_records=grounded_index['records']
    grounded_scenes=set(grounded_records) if isinstance(grounded_records,dict) else {r['scene'] for r in grounded_records}
    assert {t['scene'] for t in tasks}.issubset(grounded_scenes)
    if a.smoke:tasks=tasks[:2]
    lock=read(d.LOCK)
    assert grounded_index['split_lock_sha256']==sha(d.LOCK)
    assert all(t['scene'] in lock['validation_scenes'] and t['scene'] not in lock['train_scenes'] for t in tasks)
    types=read('/path/to/research300/relation_grounding_train_holdout_20261101_v2/run/protocol.json')['relation_types']
    norm=read(d.F/'normalization.json');mean,std=np.asarray(norm['mean']),np.asarray(norm['std'])
    oldnorm=read('/path/to/research100/prospective_train450_feature_cache_20261101_v1/normalization.json')
    geom=read(D/'geometry_provenance.json');seal=read(R/'sealed_reference_outputs.json')
    assert all(sha(R/n)==h for n,h in seal['files'].items())
    reg=read(R/'selected_reference_registry.json');small=read(V50/'locked_tasks50.json')['tasks']
    bykey={(t['asset_scene'],t['entry']):i for i,t in enumerate(small)}
    model=constructor(d,types,len(FEATURE_NAMES),cp).cuda().eval().requires_grad_(False);del cp
    before=d.tiny.digest(model.state_dict());install_geometry()
    library=d.paired.load_mesh_manifest('/path/to/research300/placeit_mesh_v6/mesh_manifest_v6.json')
    sd=None;assets={};records=[];counts=dict(real_reference_available=0,real_reference_active=0,mask_fallback=0)
    # Group by scene without changing the official task identities.
    tasks=sorted(tasks,key=lambda t:(t['scene'],t['original_entry_id']))
    for task in tasks:
        i=task['original_entry_id'];scene=task['scene'];uid=task['asset_uuid'];dims=np.asarray(task['asset_size'])
        with np.load(D/'observations'/f'{i:03d}.npz',allow_pickle=False) as z:cached={k:z[k].copy() for k in z.files}
        if sd is None or sd['scene']!=scene:
            sd=None
            payload=torch.load(d.b.P/'scans_full_refer_v1/val'/(scene+'_refer.pth'),map_location='cpu',weights_only=False)
            xyz=np.asarray(payload[0],np.float32).copy();rgb=np.asarray(payload[1],np.float32).copy();del payload
            g=geom[scene];A=np.asarray(g['A']);mp,me=library._file('scenes',scene);assert me['sha256']==g['mesh_sha256']
            with np.load(mp,allow_pickle=False) as z:world=z['vertices'].copy()
            assert np.max(np.linalg.norm(world-xyz-np.asarray(g['translation']),axis=1))<1e-5
            aligned=(xyz.astype(np.float64)@A[:3,:3].T).astype(np.float32)
            grounded=load_grounded_surfaces(a.grounded_root,scene)
            assert grounded['metadata']['mesh_sha256']==me['sha256']
            assert grounded['metadata']['mesh_vertices']==len(xyz)
            assert grounded['metadata']['coordinate_frame']=='original_scene_vertex_ids'
            receiver=prepare_scene(aligned,scene,grounded,pool_size=8192)
            sx,sf=d.paired.load_scene_mesh(d.b.P,scene,'val',library,reference_xyz=xyz)
            sd=dict(scene=scene,xyz=xyz,rgb=rgb,A=A,cache={},payload=(xyz,rgb,None,None,None),
                    receiver=receiver,asset_priors={},adapter=d.paired.MeshSceneAdapterV6(sx,sf,library))
        oldids=cached['candidate_ids'];raw=old.raw_features(sd['rgb'],oldids,dims,task['prompt'])
        original_raw=raw.copy()
        raw,centers=v6.aligned_task(raw,sd['xyz'][oldids],sd['xyz'],sd['A'],sd['cache'])
        assert np.array_equal(centers,cached['centers'])
        oldf=v6.make_features(raw,np.asarray(oldnorm['mean']),np.asarray(oldnorm['std']),'cpu').numpy()
        assert np.array_equal(oldf,cached['features'])
        # Verify legacy image-reference sidecars against their original candidate context.
        # Reference positions themselves are observation outputs independent of a new pool.
        row=dict(scene=scene,split='val',group_id=i,features=v6.make_features(raw,mean,std,'cpu').numpy(),
                 centers=centers,candidate_ids=oldids,text=cached['text'],dims=dims,asset_uuid=uid,
                 asset_scene=task['asset_scene'],entry=task['entry'],prompt_sha256=task['prompt_sha256'])
        refs={};k=bykey.get((task['asset_scene'],task['entry']))
        if k is not None:
            rr=reg[k];row['group_id']=k;identity={x:row[x] for x in ('scene','asset_scene','entry','group_id','prompt_sha256')}
            assert all(rr[x]==identity[x] for x in identity)
            assert legacy.candidate_context_fingerprint(identity,oldids,centers)==rr['candidate_context_sha256']
            path=Path(rr['selected_sidecar']);assert sha(path)==rr['selected_sidecar_sha256']
            refs[scene,k]=[dict(frame=rr['selected_frame'],sidecar_path=str(path),sidecar_sha256=sha(path))]
            counts['real_reference_available']+=1
        if uid not in assets:
            assets[uid]=torch.load(d.b.B/'placeit3d_reproduction_v1/placeit3d/data/pointbert_embeddings'/(uid+'.pt'),map_location='cpu',weights_only=True).reshape(-1).cuda()
        _,text,asset,ctx,_=legacy.make_context([row],assets,refs,1,types)
        active=bool(ctx['reference_mask'].any() and ctx['relation_mask'].any())
        counts['real_reference_active']+=active;counts['mask_fallback']+=not active
        if uid not in sd['asset_priors']:
            vertices,faces=library.asset(uid,dims)
            sd['asset_priors'][uid]=prepare_asset(sd['receiver'],dims,vertices,faces)
        generated=build_task_candidates(sd['receiver'],sd['asset_priors'][uid],oldids,task,
                    sd['payload'],original_raw,sd['A'],mean,std,sd['cache'],budget=256,global_quota=32)
        ids,regions,region_report=(generated[k] for k in ('candidate_ids','region_features','region_report'))
        features,newcenters=generated['features'],generated['centers']
        row.update(candidate_ids=ids,centers=newcenters,features=features)
        ctx['candidate_xyz']=torch.as_tensor(newcenters[None],device='cuda',dtype=torch.float32)
        ctx['candidate_mask']=torch.ones((1,len(ids)),device='cuda',dtype=torch.bool)
        model.reference_head.inferred_mask=torch.zeros_like(ctx['relation_mask'])
        f=torch.as_tensor(features[None],device='cuda',dtype=torch.float32)
        rf=torch.as_tensor(regions[None],device='cuda',dtype=torch.float32)
        class Service:
            def query_many(self,qs):
                result=[]
                for q in qs:
                    assert q['scene']==scene and q['split']=='val'
                    ad=sd['adapter'];ad.set_asset(uid,dims);result.append(ad.features(q['center'],q['dims'],q['yaw']))
                return result
        obs=[{key:row[key] for key in ('scene','split','asset_uuid','dims','centers')}]
        with torch.inference_mode(),model.reference_batch(**ctx),model.region_batch(rf):
            tr=rollout.rollout_batch(model,f,text,asset,ctx['candidate_mask'],obs,Service(),steps=2,deterministic=True,
                                    apply_offsets=True,anti_cycle=True,mode_epsilon=0.)[0]
        record=dict(original_entry_id=i,scene=scene,asset_uuid=uid,prompt_sha256=task['prompt_sha256'],
                    initial_center=clean(tr['initial_center']),initial_yaw=float(tr['initial_yaw']),
                    center=clean(tr['center']),yaw=float(tr['yaw']),
                    initial_yaw_degrees=math.degrees(float(tr['initial_yaw']))%360,
                    final_yaw_degrees=math.degrees(float(tr['yaw']))%360,
                    path=clean(tr['path']),reference_active=active,region_report=region_report,
                    initial_candidate=int(tr['initial_candidate']),final_candidate=int(tr['final_candidate']),
                    stopped=bool(tr['stopped']),switches=int(tr['switches']),
                    actual_geometry_queries=int(tr['actual_geometry_queries']))
        records.append(record)
        with (a.output/'prediction_journal.jsonl').open('a',encoding='utf-8') as fp:
            fp.write(json.dumps(record,ensure_ascii=False,allow_nan=False)+'\n');fp.flush()
        write(a.output/'progress.json',dict(status='inference',architecture=architecture,
            completed=len(records),total=len(tasks),counts=counts,elapsed_seconds=time.monotonic()-started))
        print(json.dumps(dict(completed=len(records),total=len(tasks),scene=scene,
            original_entry_id=i,reference_active=active,initial_yaw=record['initial_yaw_degrees'],
            final_yaw=record['final_yaw_degrees'])),flush=True)
    assert before==d.tiny.digest(model.state_dict());write(a.output/'sealed_predictions.json',records)
    staged={x['scene']:x for x in read(B/'staged_meshes.json')};rawscans={};export=[]
    for task,pred in zip(tasks,records):
        scene=task['scene'];assert pred['original_entry_id']==task['original_entry_id']
        g=geom[scene];A=np.asarray(g['A'],float).reshape(4,4);translation=np.asarray(g['translation'])
        offset=A[:3,:3]@translation+A[:3,3]
        if scene not in rawscans:rawscans[scene]=prep.vertices(staged[scene]['file'])
        with np.load(D/'observations'/('%03d.npz'%task['original_entry_id']),allow_pickle=False) as z:
            ids=z['candidate_ids'];centers=z['centers'];dims=z['dims']
            assert np.allclose(dims,task['asset_size'],atol=1e-6)
            assert np.linalg.norm(rawscans[scene][ids]@A[:3,:3].T+A[:3,3]-(centers+offset),axis=1).max()<2e-5
        world=np.asarray(pred['center'])+offset+np.array([0.,0.,dims[2]/2])
        export.append(dict(entry_id=task['original_entry_id'],dataset_idx=-1,tx=float(world[0]),ty=float(world[1]),tz=float(world[2]),rotation=math.degrees(pred['yaw'])%360))
    write(a.output/'predictions.json',export)
    assert sha(a.checkpoint)==checkpoint_sha256, 'Inference modified its checkpoint'
    summary=dict(status='inference_complete',architecture=architecture,epoch=checkpoint_epoch,parent_epoch=parent_epoch,
        tasks=len(records),counts=counts,weights_unchanged=True,original_coordinate_export_verified=True,
        region_input=True,checkpoint_sha256=checkpoint_sha256,training_protocol_sha256=sha(a.training_protocol),
        manifest_sha256=sha(a.manifest),prediction_sha256=sha(a.output/'predictions.json'),
        sealed_prediction_sha256=sha(a.output/'sealed_predictions.json'),
        grounded_index_sha256=sha(a.grounded_root/'index.json'),
        elapsed_seconds=time.monotonic()-started,official_started=False,
        initial_yaw_histogram=dict(Counter(str(r['initial_yaw_degrees']) for r in records)),
        final_yaw_histogram=dict(Counter(str(r['final_yaw_degrees']) for r in records)),
        action_modes=dict(Counter(p['mode'] for r in records for p in r['path'])),
        changed_yaws=sum(abs(math.atan2(math.sin(r['yaw']-r['initial_yaw']),
                                        math.cos(r['yaw']-r['initial_yaw'])))>1e-6 for r in records),
        evaluation='fixed development validation150 with unchanged input identities; not untouched test',
        scope='two-task engineering smoke' if a.smoke else 'fixed_validation150')
    write(a.output/'inference_report.json',summary)
    write(a.output/'progress.json',summary)
    if a.smoke or a.inference_only:
        print(json.dumps(summary),flush=True)
        return
    (a.output/'tmp').mkdir();(a.output/'result').mkdir()
    cmd=read(B/'official_job.json')['command'].copy()
    tag=hashlib.sha256(str(a.output.resolve()).encode()).hexdigest()[:8]
    cmd[cmd.index('--name')+1]=f'refineplace-orient-pilot-{architecture.lower()}-{tag}-val150'
    cmd[cmd.index('--device')+1]=a.official_device
    for i in range(1,len(cmd)):
        if cmd[i-1]=='-v':
            src,dst,mode=cmd[i].rsplit(':',2)
            maps={'/tmp':a.output/'tmp','/app/mounted_scannet/scans':B/'scans',
                  '/app/val_input/predictions.json':a.output/'predictions.json','/app/val_output':a.output/'result'}
            if dst in maps:cmd[i]=str(maps[dst])+':'+dst+':'+mode
    job=subprocess.run(cmd,capture_output=True,text=True,timeout=90);job.check_returncode()
    cid=job.stdout.strip();write(a.output/'official_job.json',dict(container=cid,command=cmd,started=time.time()))
    write(a.output/'progress.json',dict(status='official_evaluation_running',architecture=architecture,
          tasks=len(tasks),container=cid,official_started=True))
    wait=subprocess.run(cmd[:3]+['wait',cid],capture_output=True,text=True,timeout=14400)
    wait.check_returncode();assert wait.stdout.strip()=='0'
    inspected=subprocess.run(cmd[:3]+['inspect',cid],capture_output=True,text=True,timeout=60)
    inspected.check_returncode();inspection=json.loads(inspected.stdout)[0]
    write(a.output/'official_inspect.json',inspection)
    logs=subprocess.run(cmd[:3]+['logs',cid],capture_output=True,text=True,timeout=60)
    logs.check_returncode();log_text=logs.stdout+'\n'+logs.stderr
    (a.output/'official.log').write_text(log_text,encoding='utf-8')
    state=inspection['State']
    assert not state.get('OOMKilled',False) and state.get('ExitCode')==0, 'Official evaluator failed or OOM-killed'
    assert 'Error evaluating constraint' not in log_text and 'Traceback (most recent call last)' not in log_text
    result=a.output/'result/val_results.json';assert result.exists()
    official=read(result)
    assert sum(int(v['total_placements']) for v in official['metrics_by_constraint_count'].values())==len(tasks)
    summary=dict(status='official_evaluation_complete',version=architecture,epoch=checkpoint_epoch,parent_epoch=parent_epoch,tasks=len(tasks),counts=counts,
                 checkpoint_sha256=sha(a.checkpoint),manifest_sha256=sha(a.manifest),official_result_sha256=sha(result),
                 grounded_index_sha256=sha(a.grounded_root/'index.json'),official=official,
                 official_started=True,official_exit_code=state['ExitCode'],official_oom_killed=state.get('OOMKilled',False),
                 evaluation='fixed development validation150, unmodified official evaluator and assets; not untouched test')
    write(a.output/'report.json',summary);write(a.output/'progress.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('checkpoint','training-protocol','manifest','output','grounded-root'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--epoch',type=int,default=2);p.add_argument('--smoke',action='store_true')
    mode=p.add_mutually_exclusive_group()
    mode.add_argument('--inference-only',dest='inference_only',action='store_true',help='Generate sealed validation predictions only (default)')
    mode.add_argument('--official',dest='inference_only',action='store_false',help='Explicitly start the unchanged official evaluator after inference')
    p.set_defaults(inference_only=True)
    p.add_argument('--official-device',default='nvidia.com/gpu=1')
    main(p.parse_args())
