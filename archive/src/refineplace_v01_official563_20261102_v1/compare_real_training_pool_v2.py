"""Independent paused-checkpoint serial/8/12-process training-step comparison.
Never resumes or modifies the formal run. Plan uses observation order only;
training labels enter temporary losses only after each annotation-free rollout.
"""
import sys,json,hashlib,time,os,random,shutil,collections,resource,copy
from pathlib import Path
import numpy as np,torch
from torch import nn
sys.path[:0]=['/path/to/research300/refineplace_v01_official563_modules_20261102_v1','/path/to/research100/refineplace_persistent_grouped_pool_modules_v2']
import train_official563_joint_v2 as d
from persistent_grouped_geometry_v2 import PersistentGroupedGeometryService
import persistent_grouped_geometry_v2 as worker_module
from benchmark_private_geometry_v1 import compare as compare_geometry
O=Path('/path/to/research300/refineplace_v01_training_step_pool_comparison_v2_20261102_v1');LIVE=Path('/path/to/research300/refineplace_v01_official563_joint_full_20261102_v1');CP_SHA='be17c8c46ed37cabda5ece88a591a10af7576dc7aaae518d910cacae46f8cf4c';WRAP_SHA='d10475ec23c497336e52cdfbb6bc4936789a80478821ea82f8d3a97de6417bb1'
read=d.read;sha=d.sha;write=d.write
class ObservationScene(d.Scene):
 def __init__(self,s,lock,library):
  # Same declared observation reader, without an unused main-process BVH.
  d.guard(lock,s);self.s=s;self.lock=lock;self.path=d.F/s;self.m=read(self.path/'manifest.json');assert self.m['partition']=='official_train';self.a={k:d.arr(self.path,self.m,k) for k in ['features','centers','candidate_ids','candidate_offsets','group_ids','text','dims','asset_uuid']};self.registry=read(self.path/'task_registry.json');assert sha(self.path/'task_registry.json')==self.m['task_registry_sha256'];self.bygi={int(x):i for i,x in enumerate(self.a['group_ids'])};self.tm=read(d.TARGET/s/'manifest.json');assert sha(d.TARGET/s/'manifest.json')==self.m['source_identity']['observation_manifest'];self.labels_arrays=None;self.label_cache={};self.trees={};self.adapter=None

def restore_rng(r):
 torch.set_rng_state(r['torch_cpu']);torch.cuda.set_rng_state_all(r['torch_cuda']);np.random.set_state(r['numpy']);random.setstate(r['python'])
def rng_fingerprint():
 n=np.random.get_state();return dict(torch_cpu=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),torch_cuda=[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in torch.cuda.get_rng_state_all()],numpy_array=hashlib.sha256(n[1].tobytes()).hexdigest(),numpy_other=repr((n[0],*n[2:])),python=hashlib.sha256(repr(random.getstate()).encode()).hexdigest())
def optimizer_fingerprint(opt):
 s=opt.state_dict();h=hashlib.sha256(json.dumps(s['param_groups'],sort_keys=True).encode())
 for i,v in sorted(s['state'].items()):
  for k,x in sorted(v.items()):h.update(str((i,k)).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes() if torch.is_tensor(x) else repr(x).encode())
 return h.hexdigest()
def trace_fingerprint(tr):
 z=[dict(initial_candidate=r['initial_candidate'],final_candidate=r['final_candidate'],initial_center=r['initial_center'],center=r['center'],initial_yaw=r['initial_yaw'],yaw=r['yaw'],path=[dict(mode=q['mode'],center=q['center'],yaw=q['yaw']) for q in r['path']]) for r in tr];return hashlib.sha256(json.dumps(d.clean(z),sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='1';torch.set_num_threads(2);O.mkdir(exist_ok=False);assert read(LIVE/'summary.json')['status']=='stopped_checkpoint_saved' and (LIVE/'stop_after_current_batch').exists();assert sha(LIVE/'last_step.pt')==CP_SHA;shutil.copyfile(LIVE/'last_step.pt',O/'immutable_checkpoint572.pt');assert sha(O/'immutable_checkpoint572.pt')==CP_SHA;cp=torch.load(O/'immutable_checkpoint572.pt',map_location='cpu',weights_only=False);assert sha(d.F/'normalization.json')==cp['normalization_sha256'];cache_index=read(d.F/'index.json');assert cache_index['complete'] and cache_index['full_official_train563'];lock=read(d.LOCK);epoch,si,off=map(int,cp['next_cursor']);order=sorted(lock['train_scenes'],key=lambda s:hashlib.sha256(f'RefinePlace-v01:{epoch}:{s}'.encode()).hexdigest());plan=[];scenes=[]
 for j in range(si,len(order)):
  s=order[j];d.guard(lock,s);assert sha(d.F/s/'manifest.json')==cache_index['scene_manifests'][s];gids=np.load(d.F/s/'group_ids.npy',allow_pickle=False);pos=sorted(range(len(gids)),key=lambda i:hashlib.sha256(f'RefinePlace-v01:{epoch}:{s}:{int(gids[i])}'.encode()).hexdigest());scenes.append(s);scene_batches=0
  for offset in range(off if j==si else 0,len(pos),32):
   pp=pos[offset:offset+32];plan.append(dict(scene=s,positions=pp,task_ids=[[s,int(gids[i])] for i in pp]));scene_batches+=1
   if len(scenes)>=3 and (scene_batches>=2 or offset+32>=len(pos)):break
  if len(scenes)>=3:break
 assert len(scenes)>=3 and len(plan)<=32;assert sha(worker_module.__file__)==WRAP_SHA;proto=read(LIVE/'protocol_before_training.json');assert cp['protocol_sha256']==sha(LIVE/'protocol_before_training.json') and proto['source_sha256']==sha(d.__file__);catp=Path('/path/to/research100/official_train563_real_reference_catalog_20261102_v1/catalog.json');assert sha(catp)==proto['reference_catalog_sha256'];cat=read(catp);refs={(e['scene'],e['group_id']):e['frames'] for e in cat['entries']};types=read('/path/to/research300/relation_grounding_train_holdout_20261101_v2/run/protocol.json')['relation_types'];write(O/'plan_before_labels.json',dict(checkpoint_sha256=CP_SHA,checkpoint_updates=cp['updates'],next_cursor=cp['next_cursor'],source_sha256=sha(__file__),driver_source_sha256=sha(d.__file__),wrapper_sha256=WRAP_SHA,catalog_sha256=sha(catp),plan=plan,scope='Contiguous next paused-training batches through3scenes, input-only selection; temporary updates only',GPU_physical=1,formal_run_remains_stopped=True))
 # Explicitly reproduce the first harness failure's source without task labels:
 # CPU Adam step tensors can remain aliased after load into a CUDA optimizer.
 q=nn.Parameter(torch.ones(1));qo=torch.optim.AdamW([q],lr=.0001);q.grad=torch.ones_like(q);qo.step();state=copy.deepcopy(qo.state_dict());u=nn.Parameter(torch.ones(1,device='cuda:0'));uo=torch.optim.AdamW([u],lr=.0001);uo.load_state_dict(state);pid=next(iter(state['state']));before_step=float(state['state'][pid]['step']);alias=uo.state[u]['step'].data_ptr()==state['state'][pid]['step'].data_ptr();u.grad=torch.ones_like(u);uo.step();after_step=float(state['state'][pid]['step']);assert alias and after_step==before_step+1;write(O/'optimizer_alias_reproduction.json',dict(CPU_step_tensor_alias_after_CUDA_optimizer_load=alias,source_CPU_step_before=before_step,source_CPU_step_after=after_step,fix='Every arm freshly reloads the immutable on-disk checkpoint; no shared state dict across arms',first_harness_failure_preserved=True));del q,qo,u,uo,state
 reference_returns=[];arm_reports={};step_reference=[]
 for mode in ['serial','process8','process12']:
  arm_cp=torch.load(O/'immutable_checkpoint572.pt',map_location='cpu',weights_only=False) # fresh independent optimizer CPU step tensors per arm
  torch.manual_seed(419);np.random.seed(419);random.seed(419);library=d.paired.load_mesh_manifest('/path/to/research300/placeit_mesh_v6/mesh_manifest_v6.json');prior=nn.Sequential(nn.Linear(141,192),nn.SiLU(),nn.Linear(192,128),nn.SiLU(),nn.Linear(128,1));base=d.paired.BatchedJointModel(47,prior.state_dict(),768,.25);inner=d.paired.ReferenceJointPolicy(base,d.paired.ReferenceConditionedPositionHead(len(types)));model=d.joint.ReferencePosePolicy(inner,len(types),base.actor_body[0].in_features,preserve_candidate_logmass=False).cuda();model.load_state_dict(arm_cp['joint_policy'],strict=True);model.requires_grad_(True)
  for n,p in model.named_parameters():
   if n.startswith('yaw_reference.') or '.base.values.1.' in n or '.base.values.2.' in n:p.requires_grad_(False)
  model.train();opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=0.);opt.load_state_dict(arm_cp['optimizer']);restore_rng(arm_cp['rng']);start_model_sha=d.tiny.digest(model.state_dict());start_optimizer_sha=optimizer_fingerprint(opt);start_rng=rng_fingerprint();assets={};service=None if mode=='serial' else PersistentGroupedGeometryService(workers=int(mode.removeprefix('process')));source=None;current=None;query_index=0;steps=[];setup_total=0.;geometry_total=0.;training_total=0.;max_rss=0.;child_CPU=0.;arm_start=time.perf_counter();parent_cpu0=time.process_time();checks=[]
  try:
   for bi,desc in enumerate(plan):
    if desc['scene']!=current:
     if source is not None:source.close()
     t0=time.perf_counter();before_rng=rng_fingerprint();source=(d.Scene if mode=='serial' else ObservationScene)(desc['scene'],lock,library);assert rng_fingerprint()==before_rng;setup_total+=time.perf_counter()-t0;current=desc['scene']
    data_start=time.perf_counter();rows=source.inputs(desc['positions']);assert [[r['scene'],r['group_id']] for r in rows]==desc['task_ids'];keys=[(r['scene'],r['group_id']) for r in rows]
    for r in rows:
     uid=r['asset_uuid']
     if uid not in assets:assets[uid]=torch.load(d.b.B/'placeit3d_reproduction_v1/placeit3d/data/pointbert_embeddings'/(uid+'.pt'),map_location='cpu',weights_only=True).reshape(-1).cuda();library.asset(uid,r['dims'])
    f,t,av,ctx,_=d.make_context(rows,assets,refs,epoch,types);data_wall=time.perf_counter()-data_start;active=int((ctx['reference_mask'].any(-1)&ctx['relation_mask'].any(-1)).sum());geom=0.;call_checks=[];query_count=0
    class Service:
     def query_many(self,tasks):
      nonlocal geom,query_index,query_count,child_CPU,max_rss
      assert all(set(q)=={'scene','split','asset_uuid','dims','center','yaw'} and q['scene']==current and q['split']=='train' for q in tasks);start=time.perf_counter()
      if mode=='serial':
       out=[]
       for q in tasks:source.adapter.set_asset(q['asset_uuid'],q['dims']);out.append(source.adapter.features(q['center'],q['dims'],q['yaw']))
      else:
       before=service.process_snapshot();out=service.query_many(tasks);after=service.process_snapshot();child_CPU+=sum(v['cpu_seconds']-before.get(k,{}).get('cpu_seconds',0.) for k,v in after.items());max_rss=max(max_rss,sum(v.get('VmRSS',0) for v in after.values()))
      geom+=time.perf_counter()-start;query_count+=len(tasks)
      if mode=='serial':reference_returns.append((d.clean(tasks),out));check=dict(exactly_equal=True)
      else:
       expected_tasks,expected=reference_returns[query_index];assert d.clean(tasks)==expected_tasks,'Neural query trajectory diverged';check=compare_geometry(expected,out)
      query_index+=1;call_checks.append(check);assert check['exactly_equal'];return out
    torch.cuda.synchronize();start=time.perf_counter();opt.zero_grad(set_to_none=True)
    with model.reference_batch(**ctx):tr=d.joint.rollout_batch(model,f,t,av,ctx['candidate_mask'],[{k:r[k] for k in ['scene','split','asset_uuid','dims','centers']} for r in rows],Service(),steps=2,deterministic=False,apply_offsets=True,anti_cycle=True,mode_epsilon=.2)
    torch.cuda.synchronize();rollout_wall=time.perf_counter()-start;start=time.perf_counter();labels=source.labels(rows);quality=[source.quality(k,r['center']) for k,r in zip(keys,tr)];label_wall=time.perf_counter()-start;start=time.perf_counter();scores=torch.stack([r['scores'] for r in tr]);offsets=torch.stack([r['supervised_offsets'] for r in tr]);sup,known=d.supervised_safe(scores,offsets,labels,keys);sg=torch.autograd.grad(sup.mean(),scores,retain_graph=True)[0];assert bool((sg[(~known[:,None,:]).expand_as(sg)]==0).all());rl=d.joint.batched_rl_losses(tr,quality,geometry_weight=0.,rl_weight=.05);loss=(sup+rl['rl_loss']).mean();assert torch.isfinite(loss);loss.backward();torch.cuda.synchronize();backward_wall=time.perf_counter()-start
    if not active:
     for n,p in model.named_parameters():
      if n.startswith(('local_reference.','yaw_reference.')) or 'reference_head.' in n:p.grad=None
    grad_sha=d.tiny.digest({n:p.grad for n,p in model.named_parameters() if p.grad is not None});start=time.perf_counter();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();optimizer_wall=time.perf_counter()-start;train_wall=data_wall+rollout_wall+label_wall+backward_wall+optimizer_wall;training_total+=train_wall;geometry_total+=geom
    rec=dict(batch=bi,scene=current,task_ids=desc['task_ids'],tasks=len(rows),unique_assets=len({r['asset_uuid'] for r in rows}),active_reference_tasks=active,pose_action_trace_sha256=trace_fingerprint(tr),scores_sha256=d.tiny.digest({'scores':scores}),supervised_loss=float(sup.mean().detach()),RL_loss=float(rl['rl_loss'].mean().detach()),total_loss=float(loss.detach()),gradient_sha256=grad_sha,model_sha256=d.tiny.digest(model.state_dict()),optimizer_sha256=optimizer_fingerprint(opt),rng=rng_fingerprint(),query_count=query_count,geometry_exact=all(x['exactly_equal'] for x in call_checks),data_wall=data_wall,rollout_wall=rollout_wall,geometry_service_wall=geom,label_reward_wall=label_wall,backward_wall=backward_wall,optimizer_wall=optimizer_wall,training_step_wall_excluding_verification=train_wall)
    if mode=='serial':step_reference.append(rec)
    else:
     ref=step_reference[bi];fields=['task_ids','pose_action_trace_sha256','scores_sha256','supervised_loss','RL_loss','total_loss','gradient_sha256','model_sha256','optimizer_sha256','rng','query_count'];differences=[k for k in fields if rec[k]!=ref[k]];rec['differences_from_serial']=differences;assert not differences,repr(differences)
    steps.append(rec);write(O/(mode+'_progress.json'),dict(completed_batches=len(steps),total_batches=len(plan),last=rec));print(json.dumps(dict(mode=mode,batch=bi,scene=current,training_step_seconds=train_wall,geometry_seconds=geom,all_equal_to_serial=mode=='serial' or not rec['differences_from_serial'])),flush=True)
   assert query_index==len(reference_returns)
  finally:
   if source is not None:source.close()
   shutdown=time.perf_counter()
   if service is not None:service.close()
   shutdown=time.perf_counter()-shutdown
  torch.save(dict(joint_policy=model.state_dict(),optimizer=opt.state_dict()),O/(mode+'_temporary_final.pt'));report=dict(mode=mode,start_model_sha256=start_model_sha,start_optimizer_sha256=start_optimizer_sha,start_rng=start_rng,batches=len(steps),tasks=sum(x['tasks'] for x in steps),scenes=len(set(x['scene'] for x in steps)),all_geometry_forward_losses_gradients_parameters_optimizer_RNG_exact=mode=='serial' or all(not r['differences_from_serial'] for r in steps),training_step_wall_sum=training_total,scene_setup_wall=setup_total,production_like_wall_excluding_verification=training_total+setup_total+shutdown,geometry_service_wall_sum=geometry_total,total_research_wall=time.perf_counter()-arm_start,parent_CPU_seconds=time.process_time()-parent_cpu0,worker_CPU_seconds=child_CPU,worker_peak_RSS_sum_kib=max_rss,shutdown_wall=shutdown,steps=steps);write(O/(mode+'_report.json'),report);arm_reports[mode]=report;del model,opt;torch.cuda.empty_cache()
 assert len({r['start_model_sha256'] for r in arm_reports.values()})==1 and len({r['start_optimizer_sha256'] for r in arm_reports.values()})==1 and len({json.dumps(r['start_rng'],sort_keys=True) for r in arm_reports.values()})==1;assert sha(LIVE/'last_step.pt')==CP_SHA and read(LIVE/'summary.json')['status']=='stopped_checkpoint_saved';summary=dict(status='actual_multiscene_training_steps_parallel_exact',checkpoint_sha256=CP_SHA,plan_sha256=sha(O/'plan_before_labels.json'),batches_per_arm=len(plan),tasks_per_arm=sum(len(x['positions']) for x in plan),scene_ids=scenes,arms={k:{q:v for q,v in r.items() if q!='steps'} for k,r in arm_reports.items()},speedups={k:dict(training_step=arm_reports['serial']['training_step_wall_sum']/r['training_step_wall_sum'],including_scene_setup=arm_reports['serial']['production_like_wall_excluding_verification']/r['production_like_wall_excluding_verification']) for k,r in arm_reports.items() if k!='serial'},formal_run_unchanged_and_still_stopped=True,official_validation_or_test_opened=False,temporary_training_labels_only_after_rollout=True,limitations='Fixed paused-checkpoint next contiguous window through3officialTRAIN scenes, not whole-epoch speed or automatic backend switch; research checks/digests excluded from production-like timing');write(O/'summary.json',summary);print(json.dumps(summary),flush=True)
if __name__=='__main__':
 try:main()
 except Exception as error:
  O.mkdir(exist_ok=True);write(O/'failure.json',dict(type=type(error).__name__,message=str(error),formal_training_not_modified=True));raise
