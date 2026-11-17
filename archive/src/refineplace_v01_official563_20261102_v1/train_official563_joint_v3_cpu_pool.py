"""RefinePlace v0.1: official563 fresh random joint position/action training.
Only train labels enter loss/reward AFTER annotation-free rollout. Geometry is
weak observed evidence, never a hard filter or reward truth. Generic pretrained
RoBERTa/PointBERT vectors and externally predicted references remain frozen.
"""
import argparse,json,time,hashlib,sys,collections,os,resource,contextlib,math,random
from pathlib import Path
import numpy as np,torch,h5py
from torch import nn
from scipy.spatial import cKDTree
sys.path[:0]=['/path/to/research100/train8_actor_joint_modules_20261102_v1','/path/to/research100/train8_tiny_modules_20261102_v1','/path/to/research100/cross_category_real32_bound_modules_20261102_v1']
import train8_real_image_actor_joint_smoke_v1 as joint
from predicted_reference_sidecar_v1 import validate_predicted_reference,candidate_context_fingerprint
b,tiny,paired=joint.b,joint.tiny,joint.paired
F=Path('/path/to/research300/refineplace_v01_official563_feature_cache_20261102_v1');TARGET=Path('/path/to/research300/placeit_training_cache_v7');LOCK=Path('/path/to/research100/official_split_training_audit_20261102_v1/official563_training_split_lock.json');ADMIT=Path('/path/to/research100/official_split_training_audit_20261102_v1/official563_cache_training_admission.json')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_text())
def clean(x):
 if torch.is_tensor(x):return clean(x.detach().cpu().numpy())
 if isinstance(x,np.ndarray):return x.tolist()
 if isinstance(x,np.generic):return x.item()
 if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
 if isinstance(x,(list,tuple)):return [clean(v) for v in x]
 return x
def write(path,x):
 tmp=path.with_name(path.name+'.pending');tmp.write_text(json.dumps(clean(x),indent=2,allow_nan=False));os.replace(tmp,path)
def guard(lock,s):
 assert s in lock['train_scenes'] and s[:9] not in {x[:9] for x in lock['validation_scenes']+lock['test_scenes']}
def arr(path,m,k):
 p=path/(k+'.npy');assert sha(p)==m['arrays'][k]['sha256'];z=np.load(p,mmap_mode='r',allow_pickle=False);assert list(z.shape)==m['arrays'][k]['shape'] and str(z.dtype)==m['arrays'][k]['dtype'];return z
class Scene:
 def __init__(self,s,lock,library):
  guard(lock,s);self.s=s;self.lock=lock;self.path=F/s;self.m=read(self.path/'manifest.json');assert self.m['partition']=='official_train';self.a={k:arr(self.path,self.m,k) for k in ['features','centers','candidate_ids','candidate_offsets','group_ids','text','dims','asset_uuid']};self.registry=read(self.path/'task_registry.json');assert sha(self.path/'task_registry.json')==self.m['task_registry_sha256'];self.bygi={int(x):i for i,x in enumerate(self.a['group_ids'])};self.tm=read(TARGET/s/'manifest.json');assert sha(TARGET/s/'manifest.json')==self.m['source_identity']['observation_manifest'];self.labels_arrays=None;self.label_cache={};self.trees={}
  # Full scene observation only; not instance/placement arrays.
  obs=arr(TARGET/s,self.tm,'scene_xyz');mp,me=library._file('scenes',s)
  with np.load(mp,allow_pickle=False) as z:world=z['vertices'].copy()
  A=np.asarray(library.manifest['alignments'][s]);raw=np.asarray(obs,dtype=np.float64)@A[:3,:3];assert world.shape==raw.shape;translation=np.median(world-raw,axis=0);assert np.max(np.linalg.norm(world-raw-translation,axis=1))<1e-5;sx,sf=paired.load_scene_mesh(b.P,s,'train',library,reference_xyz=np.asarray(world-translation,dtype=np.float32));self.adapter=paired.MeshSceneAdapterV6(sx,sf,library)
 def inputs(self,positions):
  out=[]
  for pos in positions:
   q=self.registry[pos];assert q['group_id']==int(self.a['group_ids'][pos]);lo,hi=map(int,self.a['candidate_offsets'][pos:pos+2]);assert hi-lo==256
   out.append(dict(scene=self.s,split='train',group_id=q['group_id'],asset_scene=q['asset_scene'],entry=q['entry'],prompt_sha256=q['prompt_sha256'],candidate_ids=np.asarray(self.a['candidate_ids'][lo:hi]),features=np.asarray(self.a['features'][lo:hi]),centers=np.asarray(self.a['centers'][lo:hi]),text=np.asarray(self.a['text'][pos]),dims=np.asarray(self.a['dims'][pos]),asset_uuid=str(self.a['asset_uuid'][pos])))
  return out
 def labels(self,rows):
  if self.labels_arrays is None:
   self.labels_arrays={k:arr(TARGET/self.s,self.tm,k) for k in ['positive','rank_prob','offset_xy','offset_mask','group_ids','candidate_offsets','centers','annotation_rows','annotation_offsets','valid_ids','valid_mask8','scene_xyz']};assert np.array_equal(self.labels_arrays['group_ids'],self.a['group_ids']) and np.array_equal(self.labels_arrays['centers'],self.a['centers'])
  z=self.labels_arrays;out={};hpath=b.B/'placeit3d_reproduction_v1/downloads/data/placeit3d_train.h5'
  with h5py.File(hpath,'r') as h:
   for row in rows:
    gi=row['group_id'];key=self.s,gi
    if key not in self.label_cache:
     guard(self.lock,self.s);p=self.bygi[gi];lo,hi=map(int,z['candidate_offsets'][p:p+2]);e=h[row['asset_scene']+'/'+row['entry']];prompt=e['prompt'][()];prompt=prompt.decode() if isinstance(prompt,bytes) else str(prompt);assert hashlib.sha256(prompt.encode()).hexdigest()==row['prompt_sha256'];rot=np.asarray(e['rotations'][()],float);assert np.allclose(rot/45,np.rint(rot/45));known=np.zeros(8,bool);known[np.rint(rot/45).astype(int)%8]=True;an=int(z['annotation_rows'][p]);v,w=map(int,z['annotation_offsets'][an:an+2]);ids=np.asarray(z['valid_ids'][v:w]);masks=np.asarray(z['valid_mask8'][v:w]);positive_rows=masks.any(1)
     if an not in self.trees:self.trees[an]=cKDTree(np.asarray(z['scene_xyz'][ids[positive_rows]],float)) if positive_rows.any() else None
     self.label_cache[key]=dict(known=known,positive=np.asarray(z['positive'][lo:hi]),rank=np.asarray(z['rank_prob'][lo:hi]),offset_xy=np.asarray(z['offset_xy'][lo:hi]),offset_mask=np.asarray(z['offset_mask'][lo:hi]),annotation=an)
     assert not np.any(self.label_cache[key]['positive']&~known[None])
    out[key]=self.label_cache[key]
  return out
 def quality(self,key,center):
  # Positive-set proximity is a training proxy, not physical success or negative
  # evidence about unlisted placements. The full confirmed positive set is used.
  an=self.label_cache[key]['annotation']
  if self.trees[an] is None:return 0. # no confirmed positive reward evidence, not a negative label
  d=float(self.trees[an].query(center)[0]);return float(np.exp(-(d/.1)**2))
 def close(self):self.a.clear();self.label_cache.clear();self.trees.clear();self.labels_arrays=None;self.adapter=None

def make_context(rows,assets,refs,epoch,types):
 selected=[];decisions=[]
 for r in rows:
  key=(r['scene'],r['group_id']);frames=refs.get(key,[])
  if frames:
   z=frames[(epoch-1)%len(frames)];assert sha(z['sidecar_path'])==z['sidecar_sha256'];identity={k:r[k] for k in ['scene','asset_scene','entry','group_id','prompt_sha256']};identity['candidate_context_sha256']=candidate_context_fingerprint(identity,r['candidate_ids'],r['centers']);raw=read(z['sidecar_path']);checked=validate_predicted_reference(raw,identity,types,allow_provisional=True);checked={k:(v.copy() if isinstance(v,np.ndarray) else v) for k,v in checked.items()};null_rows=[];prob=None
   if checked['prediction_available']:
    mask=np.asarray(checked['reference_mask'],bool);logits=np.asarray(checked['reference_logits'],float);valid=np.r_[mask,True];v=np.where(valid[None],logits,-np.inf);prob=np.exp(v-v.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True);null_rows=(v.argmax(1)==len(mask))&checked['relation_mask'];checked['relation_mask'][null_rows]=False
    if not checked['relation_mask'].any():checked['reference_mask'][:]=False
   decisions.append(dict(scene=r['scene'],group_id=r['group_id'],frame=z['frame'],sidecar_path=z['sidecar_path'],sidecar_sha256=z['sidecar_sha256'],source_status=raw['status'],raw_reference_logits=raw['reference_logits'],raw_softmax_probabilities=None if prob is None else prob.tolist(),NULL_query_mask=np.asarray(null_rows).tolist(),policy_reference_active=bool(checked['reference_mask'].any() and checked['relation_mask'].any()),decision='NULL_top_policy_mask' if np.asarray(null_rows).any() and not checked['relation_mask'].any() else 'missing_or_unknown' if not checked['prediction_available'] else 'observed_reference'));selected.append(checked)
  else:selected.append(dict(reference_xyz=np.zeros((0,3)),reference_extents=np.zeros((0,3)),reference_logits=np.zeros((0,1)),reference_mask=np.zeros(0,bool),relation_types=np.zeros(0,np.int64),relation_mask=np.zeros(0,bool)))
 R=max(len(x['reference_mask']) for x in selected);Q=max(len(x['relation_mask']) for x in selected);B=len(rows);zeros=lambda *s,dtype=torch.float32:torch.zeros(*s,device='cuda:0',dtype=dtype);ctx=dict(candidate_xyz=torch.tensor(np.stack([r['centers'] for r in rows]),device='cuda:0'),candidate_mask=torch.ones((B,256),device='cuda:0',dtype=torch.bool),reference_xyz=zeros(B,R,3),reference_extents=zeros(B,R,3),reference_logits=zeros(B,Q,R+1),reference_mask=zeros(B,R,dtype=torch.bool),relation_types=zeros(B,Q,dtype=torch.long),relation_mask=zeros(B,Q,dtype=torch.bool))
 for i,z in enumerate(selected):
  n,q=len(z['reference_mask']),len(z['relation_mask'])
  if n:ctx['reference_xyz'][i,:n]=torch.as_tensor(z['reference_xyz'],device='cuda:0');ctx['reference_extents'][i,:n]=torch.as_tensor(z['reference_extents'],device='cuda:0');ctx['reference_mask'][i,:n]=torch.as_tensor(z['reference_mask'],device='cuda:0')
  if q:ctx['reference_logits'][i,:q,:n]=torch.as_tensor(z['reference_logits'][...,:n],device='cuda:0');ctx['reference_logits'][i,:q,-1]=torch.as_tensor(z['reference_logits'][...,-1],device='cuda:0');ctx['relation_types'][i,:q]=torch.as_tensor(z['relation_types'],device='cuda:0');ctx['relation_mask'][i,:q]=torch.as_tensor(z['relation_mask'],device='cuda:0')
 return torch.tensor(np.stack([r['features'] for r in rows]),device='cuda:0'),torch.tensor(np.stack([r['text'] for r in rows]),device='cuda:0'),torch.stack([assets[r['asset_uuid']] for r in rows]),ctx,decisions

def supervised_safe(scores,offsets,labels,keys):
 losses=[];knowns=[]
 for i,k in enumerate(keys):
  z=labels[k];known=np.asarray(z['known'],bool);knowns.append(known);valid=np.broadcast_to(known,scores.shape[1:]);zero=scores[i].sum()*0.+offsets[i].sum()*0.;loss=zero
  if valid.any():
   m=torch.tensor(valid.copy(),device=scores.device);v=scores[i][m];positive=np.asarray(z['positive'],bool)&valid
   if positive.any():loss=loss+v.logsumexp(0)-scores[i][torch.tensor(positive,device=scores.device)].logsumexp(0)
   rank=np.asarray(z['rank'],float)[valid];mass=rank.sum()
   if mass>0:loss=loss-.5*(torch.tensor(rank/mass,device=scores.device,dtype=scores.dtype)*v.log_softmax(0)).sum()
   om=np.asarray(z['offset_mask'],bool)&valid
   if om.any():
    mask=torch.tensor(om,device=scores.device);target=torch.tensor(np.asarray(z['offset_xy'])[om],device=scores.device,dtype=offsets.dtype);loss=loss+torch.nn.functional.smooth_l1_loss(offsets[i][mask]/.15,target/.15)
  losses.append(loss)
 return torch.stack(losses),torch.tensor(np.stack(knowns),device=scores.device)
def loss_boundary_tests():
 scores=torch.randn(2,3,8,device='cuda:0',requires_grad=True);offsets=torch.zeros(2,3,8,2,device='cuda:0',requires_grad=True);labels={};keys=[('synthetic',0),('synthetic',1)]
 for i,k in enumerate(keys):
  known=np.zeros(8,bool);known[0]=i==0;labels[k]=dict(known=known,positive=np.zeros((3,8),bool),rank=np.zeros((3,8)),offset_mask=np.zeros((3,8),bool),offset_xy=np.zeros((3,8,2)))
 loss,known=supervised_safe(scores,offsets,labels,keys);g=torch.autograd.grad(loss.sum(),scores)[0];assert torch.isfinite(loss).all() and torch.equal(loss,torch.zeros_like(loss)) and torch.equal(g,torch.zeros_like(g));labels[keys[0]]['positive'][0,0]=True;labels[keys[0]]['rank'][0,0]=1.;loss,known=supervised_safe(scores,offsets,labels,keys);g=torch.autograd.grad(loss.sum(),scores)[0];assert torch.isfinite(loss).all() and torch.isfinite(g).all() and bool((g[(~known[:,None,:]).expand_as(g)]==0).all());return dict(allunknown_and_empty_positive_finite_zero_loss=True,unknown_direct_logit_gradient_exact_zero=True,confirmed_positive_backward_finite=True)

def _run(a,geometry_backend):
 torch.set_num_threads(2);torch.manual_seed(419);np.random.seed(419);random.seed(419);assert sha(ADMIT)=='03064e6d3e27859dd57fc17ce7137823e7d89608d90df37adf705ea070d732c3';boundary_tests=loss_boundary_tests();admit=read(ADMIT);assert admit['training_admission_passed'] and admit['scenes']==563 and admit['tasks']==100505 and admit['feature_dim']==141 and admit['candidates_per_task']==256;assert sha(F/'normalization.json')==admit['normalization_sha256'] and sha(F/'index.json')==admit['cache_index_sha256'];lock=read(LOCK);assert sha(LOCK)==admit['official_split_lock_sha256'];index=read(F/'index.json');assert index['complete'] and index['full_official_train563'];types=read('/path/to/research300/relation_grounding_train_holdout_20261101_v2/run/protocol.json')['relation_types'];a.output.mkdir(parents=True,exist_ok=bool(a.resume))
 refs={};catalog_sha=None
 if a.reference_catalog:
  assert a.reference_catalog_sha256 and sha(a.reference_catalog)==a.reference_catalog_sha256;cat=read(a.reference_catalog);assert cat['normalization_sha256']==admit['normalization_sha256'] and cat['official_split_lock_sha256']==sha(LOCK) and cat['cache_index_sha256']==sha(F/'index.json');catalog_sha=sha(a.reference_catalog)
  for e in cat['entries']:
   guard(lock,e['scene']);key=(e['scene'],e['group_id']);assert key not in refs;refs[key]=e['frames']
 else:assert a.pilot_allmask and a.pilot_batches>0,'Formal training requires the locked real-image catalog, even if sparse'
 prior=nn.Sequential(nn.Linear(141,192),nn.SiLU(),nn.Linear(192,128),nn.SiLU(),nn.Linear(128,1));base=paired.BatchedJointModel(47,prior.state_dict(),768,.25);inner=paired.ReferenceJointPolicy(base,paired.ReferenceConditionedPositionHead(len(types)));model=joint.ReferencePosePolicy(inner,len(types),base.actor_body[0].in_features,preserve_candidate_logmass=False).cuda();model.requires_grad_(True)
 for n,p in model.named_parameters():
  if n.startswith('yaw_reference.') or '.base.values.1.' in n or '.base.values.2.' in n:p.requires_grad_(False)
 model.train();initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()};opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0001,weight_decay=0.);library=paired.load_mesh_manifest('/path/to/research300/placeit_mesh_v6/mesh_manifest_v6.json');assets={};asset_sha={};frozen=[n for n,p in model.named_parameters() if not p.requires_grad];trainable=[n for n,p in model.named_parameters() if p.requires_grad];scenes=sorted(lock['train_scenes']);counts=read(F/'normalization.json')['scene_counts'];updates_per_epoch=sum(math.ceil(counts[s]['tasks']/a.batch_size) for s in scenes)
 if a.pilot_batches:scenes=['scene0000_00','scene0001_00','scene0007_00','scene0008_00'];assert all(counts[s]['tasks']>=64 for s in scenes)
 protocol=dict(version='RefinePlace_v0.1_official563_joint_v3_CPU_pool_explicit_migration',source_sha256=sha(__file__),admission_sha256=sha(ADMIT),normalization_sha256=admit['normalization_sha256'],official_split_lock_sha256=sha(LOCK),cache_index_sha256=sha(F/'index.json'),reference_catalog_sha256=catalog_sha,initialization='All position/ranker/reference adapters/actor/critic parameters randomly seeded419; NO450/c82 task checkpoint loaded',initial_parameter_sha256=tiny.digest(initial),trainable_parameters=trainable,frozen_parameters=frozen,generic_frozen='Stored RoBERTa768/PointBERT768 vectors; externally predicted DINO/SAM/OOF contexts, no GT refs',epochs=1 if a.pilot_batches else a.epochs,batch_size=a.batch_size,steps=2,mode_epsilon_training=.2,mode_epsilon_inference=0.,lr=.0001,weight_decay=0.,checkpoint_selection='fixed final epoch, no val/test selection',loss='Known-yaw exact-positive-set CE +0.5 soft position-rank CE + offset SmoothL1 + existing actor-critic',RL_reward='Full annotated positive-position-set exp(-(nearest_distance/.1m)^2) minus existing move.002/switch.01 costs; incomplete weak proxy, unlisted placements are not negative labels',geometry_reward_weight=0.,geometry='weak observed mesh features at visited poses, no hard filters/search/project',pilot_batches=a.pilot_batches,pilot_scenes=scenes if a.pilot_batches else [],official_train_scenes=563,official_train_tasks=100505,updates_per_complete_epoch=updates_per_epoch,validation_or_test_opened=False,projection_provisional=True,checkpoint_every_updates=a.checkpoint_every,NULL_policy='Mask NULL-top relation queries; mask whole reference branch when all active queries NULL; original source logits and probabilities logged unchanged',resume_contract='Atomic model+optimizer+CPU/CUDA/NumPy/Python RNG+nextcursor+committed log-prefix hashes; restart data order unchanged')
 parent=Path(a.parent_checkpoint);assert parent.resolve()==Path('/path/to/research300/refineplace_v01_official563_joint_full_20261102_v1/last_step.pt').resolve()
 assert parent.resolve().parent!=a.output.resolve();assert sha(parent)=='be17c8c46ed37cabda5ece88a591a10af7576dc7aaae518d910cacae46f8cf4c'
 parent_protocol=parent.parent/'protocol_before_training.json';assert sha(parent_protocol)=='3785c7b0f48c12ab0d053d24ec628298f8a284cc96a126f1ddf0d47557631d4f'
 assert read(parent.parent/'summary.json')['status']=='stopped_checkpoint_saved' and (parent.parent/'stop_after_current_batch').exists()
 old=read(parent_protocol);allowed_changes={'version','source_sha256'}
 assert set(protocol)==set(old) and all(protocol[k]==old[k] for k in protocol if k not in allowed_changes),'Learning contract changed during CPU8 migration'
 from cpu_pool_backend_v1 import verify_sources
 proof=Path('/path/to/research300/refineplace_v01_cpu8_cold_resume_validation_20261102_v1/prepared_CPU8_continuation_NOT_launched.json');assert sha(proof)=='b061297e6e95ebb3988313ef9e3114addd3789a8e0a06672b361d0b8f8b13fe3'
 protocol['explicit_backend_migration']=dict(parent_checkpoint=str(parent),parent_checkpoint_sha256=sha(parent),parent_protocol_sha256=sha(parent_protocol),backend='CPU_persistent_grouped_v2',workers=a.geometry_workers,worker_sources=verify_sources(),backend_adapter_sha256=sha(Path(__file__).with_name('cpu_pool_backend_v1.py')),admission_plan_sha256=sha(proof),learning_fields_exactly_preserved=True)
 protocol_path=a.output/'protocol_before_training.json';history=[];log_chain=[];resume_state=None;start_epoch=1;start_scene=0;start_offset=0;restored_epoch_seen=set()
 if a.resume or a.parent_checkpoint:
  if a.resume:
   assert a.resume.resolve().parent==a.output.resolve();assert read(protocol_path)==protocol;input_checkpoint=a.resume
  else:
   assert not protocol_path.exists();write(protocol_path,protocol);write(a.output/'loss_boundary_tests.json',boundary_tests);input_checkpoint=parent
  resume_state=torch.load(input_checkpoint,map_location='cpu',weights_only=False)
  assert resume_state['protocol_sha256']==sha(protocol_path if a.resume else parent_protocol)
  assert resume_state['normalization_sha256']==admit['normalization_sha256'] and resume_state['official_split_lock_sha256']==sha(LOCK);model.load_state_dict(resume_state['joint_policy'],strict=True);opt.load_state_dict(resume_state['optimizer']);assert resume_state['initial_parameter_sha256']==tiny.digest(initial)
  log_chain=resume_state['log_chain'];assert len(log_chain)>0
  for item in log_chain:
   path=Path(item['path']);assert path.resolve().parent in {a.output.resolve(),parent.resolve().parent}
   with path.open('rb') as log:data=log.read(item['committed_bytes'])
   assert len(data)==item['committed_bytes'] and hashlib.sha256(data).hexdigest()==item['committed_sha256'];history.extend(json.loads(x) for x in data.splitlines() if x)
  assert len(history)==resume_state['updates'];start_epoch,start_scene,start_offset=map(int,resume_state['next_cursor']);restored_epoch_seen={tuple(k) for k in resume_state['next_epoch_seen']};torch.set_rng_state(resume_state['rng']['torch_cpu']);torch.cuda.set_rng_state_all(resume_state['rng']['torch_cuda']);np.random.set_state(resume_state['rng']['numpy']);random.setstate(resume_state['rng']['python']);write(a.output/f'resume_receipt_{time.time_ns()}.json',dict(checkpoint=str(input_checkpoint),checkpoint_sha256=sha(input_checkpoint),explicit_migration=not bool(a.resume),restored_updates=len(history),next_cursor=resume_state['next_cursor'],all_rng_restored=True))
 else:write(a.output/'loss_boundary_tests.json',boundary_tests);write(protocol_path,protocol)
 stopflag=a.output/'stop_after_current_batch'
 if a.resume and stopflag.exists():stopflag.rename(a.output/f'stop_request_consumed_by_resume_{time.time_ns()}')
 log_path=a.output/('steps.jsonl' if not a.resume else f'steps_resume_{time.time_ns()}.jsonl');start=time.perf_counter();cpu_start=time.process_time();total_queries=sum(h['geometry_queries'] for h in history);total_geom=sum(h['cpu_geometry_seconds'] for h in history);total_scene_setup=0.;seen={}
 for h in history:seen.setdefault(str(h['epoch']),[]).extend(h['task_ids'])
 prior_elapsed=resume_state['elapsed_totals'] if resume_state else dict(wall=0.,cpu=0.,scene_setup=0.);total_scene_setup=prior_elapsed['scene_setup'];stop_requested=False
 def checkpoint(next_cursor,next_seen,epoch,extra_name=None):
  torch.cuda.synchronize();chain=list(log_chain)
  if log_path.exists():
   data=log_path.read_bytes();chain.append(dict(path=str(log_path),committed_bytes=len(data),committed_sha256=hashlib.sha256(data).hexdigest()))
  state=dict(joint_policy=model.state_dict(),optimizer=opt.state_dict(),epoch=epoch,updates=len(history),normalization_sha256=admit['normalization_sha256'],official_split_lock_sha256=sha(LOCK),protocol_sha256=sha(protocol_path),initial_parameter_sha256=tiny.digest(initial),next_cursor=list(next_cursor),next_epoch_seen=[list(k) for k in sorted(next_seen)],log_chain=chain,elapsed_totals=dict(wall=prior_elapsed['wall']+time.perf_counter()-start,cpu=prior_elapsed['cpu']+time.process_time()-cpu_start,scene_setup=total_scene_setup),rng=dict(torch_cpu=torch.get_rng_state(),torch_cuda=torch.cuda.get_rng_state_all(),numpy=np.random.get_state(),python=random.getstate()))
  target=a.output/(extra_name or 'last_step.pt');tmp=target.with_name(target.name+'.pending');torch.save(state,tmp)
  if target.exists() and not extra_name:os.replace(target,a.output/'previous_step.pt')
  os.replace(tmp,target);write(a.output/'last_checkpoint.json',dict(path=str(target),sha256=sha(target),epoch=epoch,updates=len(history),next_cursor=list(next_cursor),committed_log_events=len(history)))
 profile=torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) if a.pilot_batches else contextlib.nullcontext()
 with profile as prof:
  for epoch in range(start_epoch,(1 if a.pilot_batches else a.epochs)+1):
   epochseen=set(restored_epoch_seen) if epoch==start_epoch else set();epochstart=time.perf_counter();order=scenes if a.pilot_batches else sorted(scenes,key=lambda s:hashlib.sha256(f'RefinePlace-v01:{epoch}:{s}'.encode()).hexdigest())
   for scene_index,scene in enumerate(order):
    if epoch==start_epoch and scene_index<start_scene:continue
    guard(lock,scene);ss=time.perf_counter();source=observation_scene_class(sys.modules[__name__])(scene,lock,library);setup=time.perf_counter()-ss;total_scene_setup+=setup;positions=list(range(len(source.registry)));positions=positions[:64] if a.pilot_batches else sorted(positions,key=lambda i:hashlib.sha256(f'RefinePlace-v01:{epoch}:{scene}:{int(source.a["group_ids"][i])}'.encode()).hexdigest())
    for offset in range(0,len(positions),a.batch_size):
     if epoch==start_epoch and scene_index==start_scene and offset<start_offset:continue
     bs=time.perf_counter();data_st=bs;rows=source.inputs(positions[offset:offset+a.batch_size]);keys=[(r['scene'],r['group_id']) for r in rows];assert not set(keys)&epochseen;epochseen.update(keys)
     for row in rows:
      uid=row['asset_uuid']
      if uid not in assets:
       ap=b.B/'placeit3d_reproduction_v1/placeit3d/data/pointbert_embeddings'/(uid+'.pt');asset_sha[uid]=sha(ap);assets[uid]=torch.load(ap,map_location='cpu',weights_only=True).reshape(-1).cuda();library.asset(uid,row['dims'])
     f,t,av,ctx,ref_decisions=make_context(rows,assets,refs,epoch,types);data_seconds=time.perf_counter()-data_st;active=int((ctx['reference_mask'].any(-1)&ctx['relation_mask'].any(-1)).sum());geom_seconds=0.;query_count=0
     class Service:
      def query_many(self,tasks):
       nonlocal geom_seconds,query_count
       for q in tasks:
        assert set(q)=={'scene','split','asset_uuid','dims','center','yaw'} and q['scene']==scene and q['split']=='train';guard(lock,q['scene'])
       ts=time.perf_counter();out=geometry_backend.query_many(tasks);geom_seconds+=time.perf_counter()-ts;query_count+=len(tasks)
       return out
     opt.zero_grad(set_to_none=True);torch.cuda.synchronize();r0=time.perf_counter()
     with model.reference_batch(**ctx):tr=joint.rollout_batch(model,f,t,av,ctx['candidate_mask'],[{k:r[k] for k in ['scene','split','asset_uuid','dims','centers']} for r in rows],Service(),steps=2,deterministic=False,apply_offsets=True,anti_cycle=True,mode_epsilon=.2)
     torch.cuda.synchronize();rollout_wall=time.perf_counter()-r0;l0=time.perf_counter();labels=source.labels(rows);quality=[source.quality(k,r['center']) for k,r in zip(keys,tr)];label_seconds=time.perf_counter()-l0;loss_st=time.perf_counter();scores=torch.stack([r['scores'] for r in tr]);offsets=torch.stack([r['supervised_offsets'] for r in tr]);sup,known=supervised_safe(scores,offsets,labels,keys);unknown_checked=bool(a.pilot_batches or len(history)%100==0)
     if unknown_checked:
      sg=torch.autograd.grad(sup.mean(),scores,retain_graph=True)[0];assert bool((sg[(~known[:,None,:]).expand_as(sg)]==0).all())
     rl=joint.batched_rl_losses(tr,quality,geometry_weight=0.,rl_weight=.05);loss=(sup+rl['rl_loss']).mean();assert torch.isfinite(loss);loss.backward()
     if not active:
      for n,p in model.named_parameters():
       if n.startswith(('local_reference.','yaw_reference.')) or 'reference_head.' in n:p.grad=None
     grad_tensors=collections.defaultdict(list)
     for n,p in model.named_parameters():
      if p.grad is not None:grad_tensors['prior' if '.base.prior.' in n else 'mean' if '.base.mean.' in n else 'actor' if any(x in n for x in ['actor_body.','.mode.','log_std']) else 'reference' if 'reference' in n else 'other'].append(p.grad.abs().sum())
     grads={k:float(torch.stack(v).sum()) for k,v in grad_tensors.items()}
     torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.,error_if_nonfinite=True);opt.step();torch.cuda.synchronize();update_seconds=time.perf_counter()-loss_st;total_queries+=query_count;total_geom+=geom_seconds;trace=[dict(initial_candidate=r['initial_candidate'],final_candidate=r['final_candidate'],initial_center=r['initial_center'],center=r['center'],initial_yaw=r['initial_yaw'],yaw=r['yaw'],path=[dict(mode=q['mode'],center=q['center'],yaw=q['yaw']) for q in r['path']]) for r in tr];trace_sha=hashlib.sha256(json.dumps(clean(trace),sort_keys=True,separators=(',',':')).encode()).hexdigest();h=dict(epoch=epoch,update=len(history)+1,pose_action_trace_sha256=trace_sha,scene=scene,task_ids=[list(k) for k in keys],tasks=len(rows),active_reference_tasks=active,selected_NULL_top_tasks=sum(d['decision']=='NULL_top_policy_mask' for d in ref_decisions),source_missing_or_unknown_tasks=sum(d['decision']=='missing_or_unknown' for d in ref_decisions),no_registered_reference_tasks=len(rows)-len(ref_decisions),data_seconds=data_seconds,rollout_wall_seconds=rollout_wall,cpu_geometry_seconds=geom_seconds,nongeometry_rollout_wall_seconds=max(0.,rollout_wall-geom_seconds),label_reward_seconds=label_seconds,supervised_backward_optimizer_wall_seconds=update_seconds,batch_seconds=time.perf_counter()-bs,geometry_queries=query_count,actions=dict(collections.Counter(s['mode'] for r in tr for s in r['path'])),supervised_loss=float(sup.mean().detach()),rl_loss=float(rl['rl_loss'].mean().detach()),position_quality_mean=float(np.mean(quality)),gradient_l1_by_group=dict(grads),known_yaw_supervised_unknown_gradient_exact_zero=True if unknown_checked else None);history.append(h);seen.setdefault(str(epoch),[]).extend(keys)
     with log_path.open('a') as log:log.write(json.dumps(h)+'\n')
     if ref_decisions:
      with (a.output/'reference_decisions.jsonl').open('a') as log:log.write(json.dumps(dict(update=len(history),decisions=ref_decisions))+'\n')
     write(a.output/'progress.json',dict(epoch=epoch,updates=len(history),tasks_this_epoch=len(epochseen),tasks_per_complete_epoch=100505,last=h,wall_seconds=time.perf_counter()-start));print(json.dumps(h),flush=True)
     next_epoch,next_scene,next_offset=epoch,scene_index,offset+a.batch_size
     if next_offset>=len(positions):next_scene+=1;next_offset=0
     if next_scene>=len(order):next_epoch+=1;next_scene=0
     next_seen=epochseen if next_epoch==epoch else set();stopping=(a.stop_after_updates>0 and len(history)>=a.stop_after_updates) or (a.output/'stop_after_current_batch').exists()
     if len(history)%a.checkpoint_every==0 or stopping:checkpoint((next_epoch,next_scene,next_offset),next_seen,epoch)
     if stopping:stop_requested=True;break
     if a.pilot_batches and len(history)>=a.pilot_batches:break
    source.close()
    if stop_requested or (a.pilot_batches and len(history)>=a.pilot_batches):break
   if stop_requested:break
   if not a.pilot_batches:assert len(epochseen)==100505
   checkpoint((epoch+1,0,0),set(),epoch,extra_name=(f'epoch{epoch}.pt' if not a.pilot_batches else 'pilot_last.pt'))
   if a.pilot_batches:break
 changes={n:float((v.detach().cpu()-initial[n]).abs().sum()) for n,v in model.state_dict().items()};assert all(changes[n]==0 for n in frozen);wall=prior_elapsed['wall']+time.perf_counter()-start;cpu=prior_elapsed['cpu']+time.process_time()-cpu_start;gpu_ms=None
 if a.pilot_batches:
  events=prof.key_averages();gpu_ms=sum(float(getattr(e,'self_device_time_total',getattr(e,'self_cuda_time_total',0.))) for e in events)/1000.;write(a.output/'gpu_kernel_profile.json',dict(summed_profile_GPU_kernel_milliseconds=gpu_ms,excludes_CPU_geometry_wait=True,note='Summed operator-attributed device time, not GPU utilization or wall-time speedup'))
 steady=history[2:] if len(history)>2 else history;batch_mean=float(np.mean([h['batch_seconds'] for h in steady]));setup_mean=total_scene_setup/len(set(h['scene'] for h in history));estimate=updates_per_epoch*batch_mean+563*setup_mean;summary=dict(status='stopped_checkpoint_saved' if stop_requested else 'pilot_complete' if a.pilot_batches else 'training_complete',updates=len(history),epochs=1 if a.pilot_batches else a.epochs,actual_train_task_exposures=sum(h['tasks'] for h in history),actual_unique_train_tasks=len(set(k for vv in seen.values() for k in map(tuple,vv))),active_reference_exposures=sum(h['active_reference_tasks'] for h in history),geometry_queries=total_queries,cpu_geometry_seconds=total_geom,scene_setup_seconds=total_scene_setup,wall_seconds=wall,process_CPU_seconds=cpu,average_CPU_cores=cpu/wall,host_logical_CPUs=os.cpu_count(),summed_GPU_kernel_milliseconds=gpu_ms,peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_host_RSS_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,steady_batch_mean_seconds=batch_mean,full_epoch_updates=updates_per_epoch,estimated_one_epoch_seconds=estimate,estimated_50_epochs_seconds=50*estimate,estimate_caveat='Extrapolation from prelocked small TRAIN batch/scene sample; scene complexity, cache reuse, reference availability, small final batches and profiling overhead vary. Not a completed epoch.',actions=dict(sum((collections.Counter(h['actions']) for h in history),collections.Counter())),parameter_change_l1=changes,frozen_parameters_unchanged=True,old_position_or_actor_weights_loaded=False,legacy_normalization_loaded=False,val_test_opened=False,protocol_sha256=sha(a.output/'protocol_before_training.json'),asset_embedding_sha256=asset_sha);write(a.output/'summary.json',summary);print(json.dumps({k:v for k,v in summary.items() if k!='parameter_change_l1'}),flush=True)
from cpu_pool_backend_v1 import CPUProcessGeometryBackend,observation_scene_class

def run(a):
 # Pool lifecycle includes failure paths. No parent BVH is constructed.
 with CPUProcessGeometryBackend(workers=a.geometry_workers) as backend:_run(a,backend)

if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--epochs',type=int,default=50);p.add_argument('--batch-size',type=int,default=32);p.add_argument('--pilot-batches',type=int,default=0);p.add_argument('--pilot-allmask',action='store_true');p.add_argument('--reference-catalog',type=Path);p.add_argument('--reference-catalog-sha256');p.add_argument('--resume',type=Path);p.add_argument('--parent-checkpoint',type=Path,required=True);p.add_argument('--geometry-workers',type=int,choices=[8,12],default=12);p.add_argument('--checkpoint-every',type=int,default=100);p.add_argument('--stop-after-updates',type=int,default=0);run(p.parse_args())
