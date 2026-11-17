"""Frozen new official563 joint checkpoint -> fixed native val50, real/masked pair.
No placement/relation labels read. Genuine image caches reused, not re-detected.
"""
import sys,os,json,time,hashlib,argparse,collections
from pathlib import Path
import numpy as np,torch
from torch import nn
M=Path('/path/to/research300/refineplace_v01_official563_modules_20261102_v1');sys.path.insert(0,str(M))
import train_official563_joint_v4_cpu12_fresh as d
sys.path.insert(0,'/path/to/research100/native_val50_current_model_20261102_v1')
import run_native_val50_allmask_v1 as old
import train_joint_pose_v6 as v6
B=Path('/path/to/research100/native_val50_current_model_20261102_v1');D=B/'allmask_current';R=B/'real_image_observations/reference_predictions'
sha=d.sha;read=d.read;write=d.write;clean=d.clean

def main(a):
 assert os.environ['CUDA_VISIBLE_DEVICES']=='1';torch.set_num_threads(2);torch.manual_seed(419);np.random.seed(419);a.output.mkdir(exist_ok=False)
 assert sha(B/'locked_tasks50.json')=='9e9a0c4cb32e2d76f29cb2eb95da382352cb979ab21f179f7820b32438fbc103';tasks=read(B/'locked_tasks50.json')['tasks'];assert len(tasks)==50;assert a.limit in (4,50);tasks=tasks[:a.limit]
 seal=read(R/'sealed_reference_outputs.json');assert sha(R/'sealed_reference_outputs.json')=='4f6c4246dd73a3a3dc0ebc24d829efab94ca0199a393f4d33a15e41fc34df0a3';assert all(sha(R/n)==h for n,h in seal['files'].items());registry=read(R/'selected_reference_registry.json');assert sha(R/'selected_reference_registry.json')=='d849dcab87d86927d7ae083722447e81cdb0a7ce59e2fe835b85d1792420df9a'
 cp=torch.load(a.checkpoint,map_location='cpu',weights_only=False);norm=read(d.F/'normalization.json');assert cp['normalization_sha256']==sha(d.F/'normalization.json')=='0b14a1b534fe71403862e056fb539281282f40b58bee8a8f021f36a5cb8c840d';assert cp['official_split_lock_sha256']==sha(d.LOCK)
 trainproto=read(a.training_protocol);assert cp['protocol_sha256']==sha(a.training_protocol);assert trainproto['source_sha256']==sha(d.__file__);lock=read(d.LOCK);assert all(t['scene'] in lock['validation_scenes'] and t['scene'] not in lock['train_scenes'] for t in tasks)
 oldnorm=read('/path/to/research100/prospective_train450_feature_cache_20261101_v1/normalization.json');geom=read(D/'geometry_provenance.json');types=read('/path/to/research300/relation_grounding_train_holdout_20261101_v2/run/protocol.json')['relation_types'];mean,std=np.asarray(norm['mean']),np.asarray(norm['std'])
 protocol=dict(source_sha256=sha(__file__),checkpoint_sha256=sha(a.checkpoint),checkpoint_updates=cp['updates'],training_protocol_sha256=sha(a.training_protocol),normalization_sha256=sha(d.F/'normalization.json'),task_manifest_sha256=sha(B/'locked_tasks50.json'),tasks=len(tasks),selection='Same frozen nativeval first50 manifest; smoke first4 only, no GT selection',two_steps=True,deterministic=True,epsilon=0.,new_model_strict_loaded=True,old450_model_or_c82_loaded=False,old450_normalization_used_only_for_raw_reconstruction_equality_audit=True,real_reference_cache_seal_sha256=sha(R/'sealed_reference_outputs.json'),image_detectors_rerun=False,projection_provisional=True,calibration_verified=False,placement_or_relation_labels_read=False,test_opened=False,geometry='CPU val adapter only at visited pose; no hard filtering/search/snap; train-only pool guard not bypassed')
 write(a.output/'protocol_before_forward.json',protocol)
 prior=nn.Sequential(nn.Linear(141,192),nn.SiLU(),nn.Linear(192,128),nn.SiLU(),nn.Linear(128,1));base=d.paired.BatchedJointModel(47,prior.state_dict(),768,.25);inner=d.paired.ReferenceJointPolicy(base,d.paired.ReferenceConditionedPositionHead(len(types)));model=d.joint.ReferencePosePolicy(inner,len(types),base.actor_body[0].in_features,preserve_candidate_logmass=False).cuda();model.load_state_dict(cp['joint_policy'],strict=True);model.eval().requires_grad_(False);before=d.tiny.digest(model.state_dict());del cp
 library=d.paired.load_mesh_manifest('/path/to/research300/placeit_mesh_v6/mesh_manifest_v6.json');scenes={};assets={};refs={};results={'real':[],'allmask':[]};calls=collections.Counter();hooks=collections.Counter();arm=['real'];hook=model.base.actor_body.register_forward_hook(lambda *_:hooks.update([arm[0]]));started=time.monotonic();reconstruction=[]
 for task in tasks:
  i=task['original_entry_id'];scene=task['scene'];assert i==len(results['real']);uid=task['asset_uuid'];dims=np.asarray(task['asset_size']);assert registry[i]['original_entry_id']==i
  with np.load(D/'observations'/f'{i:03d}.npz',allow_pickle=False) as z:cached={k:z[k].copy() for k in z.files}
  if scene not in scenes:
   payload=torch.load(d.b.P/'scans_full_refer_v1/val'/(scene+'_refer.pth'),map_location='cpu',weights_only=False);xyz=np.asarray(payload[0]).copy();rgb=np.asarray(payload[1]).copy();del payload
   g=geom[scene];A=np.asarray(g['A']);mp,me=library._file('scenes',scene);assert me['sha256']==g['mesh_sha256']
   with np.load(mp,allow_pickle=False) as z:world=z['vertices'].copy()
   assert np.max(np.linalg.norm(world-xyz-np.asarray(g['translation']),axis=1))<1e-5
   sx,sf=d.paired.load_scene_mesh(d.b.P,scene,'val',library,reference_xyz=xyz.astype(np.float32));scenes[scene]=dict(xyz=xyz,rgb=rgb,A=A,cache={},adapter=d.paired.MeshSceneAdapterV6(sx,sf,library))
  sd=scenes[scene];ids=cached['candidate_ids'];raw=old.raw_features(sd['rgb'],ids,dims,task['prompt']);raw,centers=v6.aligned_task(raw,sd['xyz'][ids],sd['xyz'],sd['A'],sd['cache']);assert np.array_equal(centers,cached['centers']);oldf=v6.make_features(raw,np.asarray(oldnorm['mean']),np.asarray(oldnorm['std']),'cpu').numpy();assert np.array_equal(oldf,cached['features']),'Raw47 reconstruction differs from sealed original observations'
  features=v6.make_features(raw,mean,std,'cpu').numpy();row=dict(scene=scene,split='val',group_id=i,features=features,centers=centers,candidate_ids=ids,text=cached['text'],dims=dims,asset_uuid=uid,asset_scene=task['asset_scene'],entry=task['entry'],prompt_sha256=task['prompt_sha256']);assert np.isfinite(features).all()
  rr=registry[i];identity={k:row[k] for k in ['scene','asset_scene','entry','group_id','prompt_sha256']};assert all(rr[k]==identity[k] for k in identity);assert d.candidate_context_fingerprint(identity,ids,centers)==rr['candidate_context_sha256'];path=Path(rr['selected_sidecar']);assert sha(path)==rr['selected_sidecar_sha256'];refs[scene,i]=[dict(frame=rr['selected_frame'],sidecar_path=str(path),sidecar_sha256=sha(path))]
  if uid not in assets:assets[uid]=torch.load(d.b.B/'placeit3d_reproduction_v1/placeit3d/data/pointbert_embeddings'/(uid+'.pt'),map_location='cpu',weights_only=True).reshape(-1).cuda()
  f,t,av,ctx,decisions=d.make_context([row],assets,refs,1,types);masked={k:v.clone() for k,v in ctx.items()};masked['reference_mask'].zero_();masked['relation_mask'].zero_();active=bool(ctx['reference_mask'].any() and ctx['relation_mask'].any());obs=[{k:row[k] for k in ['scene','split','asset_uuid','dims','centers']}]
  class Service:
   def query_many(self,qs):
    out=[]
    for q in qs:
     assert set(q)=={'scene','split','asset_uuid','dims','center','yaw'} and q['scene']==scene and q['split']=='val' and q['asset_uuid']==uid;calls[arm[0]]+=1;assert sum(calls.values())<=400;ad=sd['adapter'];ad.set_asset(uid,dims);out.append(ad.features(q['center'],q['dims'],q['yaw']))
    return out
  traces={}
  for name,context in [('real',ctx),('allmask',masked)]:
   arm[0]=name
   with torch.inference_mode(),model.reference_batch(**context):traces[name]=d.joint.rollout_batch(model,f,t,av,context['candidate_mask'],obs,Service(),steps=2,deterministic=True,apply_offsets=True,anti_cycle=True,mode_epsilon=0.)[0]
  fallback=not active
  if fallback:
   x,y=traces['real'],traces['allmask'];assert torch.equal(x['scores'],y['scores']) and np.array_equal(x['center'],y['center']) and x['yaw']==y['yaw']
  for name,tr in traces.items():
   record={k:task[k] for k in ['original_entry_id','scene','asset_scene','entry','asset_uuid','prompt_sha256']};record.update(dict(center=tr['center'],yaw=tr['yaw'],initial_center=tr['initial_center'],initial_yaw=tr['initial_yaw'],initial_candidate=tr['initial_candidate'],final_candidate=tr['final_candidate'],steps=tr['steps'],path=tr['path'],reference_active=active if name=='real' else False,fallback_exact=fallback,selected_frame=rr['selected_frame'],candidate_context_sha256=rr['candidate_context_sha256']));results[name].append(clean(record))
  reconstruction.append(dict(original_entry_id=i,raw47_reconstruction_matches_old_sealed_features_exact=True,centers_and_ordered_ids_unchanged=True,new563_features_sha256=hashlib.sha256(features.tobytes()).hexdigest()));write(a.output/'progress.json',dict(completed=i+1,total=len(tasks),seconds=time.monotonic()-started));print(json.dumps(dict(completed=i+1,total=len(tasks),reference_active=active)),flush=True)
 hook.remove();assert before==d.tiny.digest(model.state_dict());assert sha(a.checkpoint)==protocol['checkpoint_sha256']
 for name,records in results.items():write(a.output/(name+'_sealed_predictions.json'),records)
 write(a.output/'reconstruction_audit.json',reconstruction);write(a.output/'geometry_provenance.json',geom)
 report=dict(status='new_official563_nativeval50_frozen_checkpoint_paired_inference_complete',tasks=len(tasks),checkpoint_updates=protocol['checkpoint_updates'],checkpoint_sha256=protocol['checkpoint_sha256'],weights_unchanged=True,strict_state_dict_load=True,raw47_reconstruction_exact=len(reconstruction),reference_active=sum(x['reference_active'] for x in results['real']),missing_exact_fallback=sum(x['fallback_exact'] for x in results['real']),actor_hook_calls=dict(hooks),visited_geometry_queries=dict(calls),seconds=time.monotonic()-started,real_prediction_sha256=sha(a.output/'real_sealed_predictions.json'),allmask_prediction_sha256=sha(a.output/'allmask_sealed_predictions.json'),protocol_sha256=sha(a.output/'protocol_before_forward.json'),geometry_sha256=sha(a.output/'geometry_provenance.json'),placement_labels_read=False,test_opened=False,projection_provisional=True)
 write(a.output/'report.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--training-protocol',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--limit',type=int,choices=[4,50],required=True);main(p.parse_args())
