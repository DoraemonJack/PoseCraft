"""Isolated GPU1 replay: immutable training checkpoint -> next TRAIN batch queries.
No labels, no optimizer update, no mutation or pause of the live GPU0 run.
"""
import sys,json,time,hashlib,shutil,os,random
from pathlib import Path
import numpy as np,torch
from torch import nn
sys.path.insert(0,'/path/to/research300/refineplace_v01_official563_modules_20261102_v1')
import train_official563_joint_v2 as d
O=Path('/path/to/research300/refineplace_v01_parallel_transition_audit_20261102_v1');LIVE=Path('/path/to/research300/refineplace_v01_official563_joint_full_20261102_v1');read=d.read;sha=d.sha

def main():
 torch.set_num_threads(2);O.mkdir(exist_ok=True);dest=O/'immutable_live_checkpoint.pt';assert not dest.exists();src=LIVE/'last_step.pt'
 # The live writer replaces checkpoint inodes atomically. Read one opened inode
 # to completion; subsequent last_step replacements cannot change this copy.
 with src.open('rb') as f,dest.open('xb') as g:
  st=os.fstat(f.fileno());shutil.copyfileobj(f,g,8*1024*1024)
 cp=torch.load(dest,map_location='cpu',weights_only=False);protocol=read(LIVE/'protocol_before_training.json');assert cp['protocol_sha256']==sha(LIVE/'protocol_before_training.json') and protocol['source_sha256']==sha(d.__file__);assert cp['normalization_sha256']=='0b14a1b534fe71403862e056fb539281282f40b58bee8a8f021f36a5cb8c840d';epoch,si,off=map(int,cp['next_cursor']);lock=read(d.LOCK);order=sorted(lock['train_scenes'],key=lambda s:hashlib.sha256(f'RefinePlace-v01:{epoch}:{s}'.encode()).hexdigest());scene=order[si];d.guard(lock,scene);library=d.paired.load_mesh_manifest('/path/to/research300/placeit_mesh_v6/mesh_manifest_v6.json');source=d.Scene(scene,lock,library);positions=sorted(range(len(source.registry)),key=lambda i:hashlib.sha256(f'RefinePlace-v01:{epoch}:{scene}:{int(source.a["group_ids"][i])}'.encode()).hexdigest());positions=positions[off:off+32];assert positions;rows=source.inputs(positions);assert source.labels_arrays is None
 types=read('/path/to/research300/relation_grounding_train_holdout_20261101_v2/run/protocol.json')['relation_types'];cat_path=Path('/path/to/research100/official_train563_real_reference_catalog_20261102_v1/catalog.json');assert sha(cat_path)==protocol['reference_catalog_sha256'];cat=read(cat_path);refs={(e['scene'],e['group_id']):e['frames'] for e in cat['entries']};assets={}
 for r in rows:
  uid=r['asset_uuid']
  if uid not in assets:assets[uid]=torch.load(d.b.B/'placeit3d_reproduction_v1/placeit3d/data/pointbert_embeddings'/(uid+'.pt'),map_location='cpu',weights_only=True).reshape(-1).cuda();library.asset(uid,r['dims'])
 torch.manual_seed(419);np.random.seed(419);random.seed(419);prior=nn.Sequential(nn.Linear(141,192),nn.SiLU(),nn.Linear(192,128),nn.SiLU(),nn.Linear(128,1));base=d.paired.BatchedJointModel(47,prior.state_dict(),768,.25);inner=d.paired.ReferenceJointPolicy(base,d.paired.ReferenceConditionedPositionHead(len(types)));model=d.joint.ReferencePosePolicy(inner,len(types),base.actor_body[0].in_features,preserve_candidate_logmass=False).cuda();model.load_state_dict(cp['joint_policy'],strict=True);model.requires_grad_(False);model.train();before=d.tiny.digest(model.state_dict());torch.set_rng_state(cp['rng']['torch_cpu']);torch.cuda.set_rng_state_all(cp['rng']['torch_cuda']);np.random.set_state(cp['rng']['numpy']);random.setstate(cp['rng']['python']);f,t,a,ctx,decisions=d.make_context(rows,assets,refs,epoch,types);calls=[];seconds=[];vectors=[]
 class Service:
  def query_many(self,tasks):
   assert all(set(q)=={'scene','split','asset_uuid','dims','center','yaw'} for q in tasks);calls.append(d.clean(tasks));start=time.perf_counter();out=[]
   for q in tasks:
    d.guard(lock,q['scene']);assert q['scene']==scene and q['split']=='train';source.adapter.set_asset(q['asset_uuid'],q['dims']);out.append(source.adapter.features(q['center'],q['dims'],q['yaw']))
   seconds.append(time.perf_counter()-start);vectors.append(np.stack([x[0] for x in out]));return out
 start=time.perf_counter()
 with torch.no_grad(),model.reference_batch(**ctx):tr=d.joint.rollout_batch(model,f,t,a,ctx['candidate_mask'],[{k:r[k] for k in ['scene','split','asset_uuid','dims','centers']} for r in rows],Service(),steps=2,deterministic=False,apply_offsets=True,anti_cycle=True,mode_epsilon=.2)
 assert d.tiny.digest(model.state_dict())==before and source.labels_arrays is None;asset_counts={uid:sum(r['asset_uuid']==uid for r in rows) for uid in set(r['asset_uuid'] for r in rows)};trace=dict(version='actual_next_training_batch_observation_query_trace_v1',scope='Immutable live checkpoint next TRAIN batch replay; not original GPU0 trace, no GT labels or optimizer update',checkpoint_sha256=sha(dest),checkpoint_updates=cp['updates'],next_cursor=cp['next_cursor'],scene=scene,task_ids=[[r['scene'],r['group_id']] for r in rows],asset_distribution=asset_counts,query_many_calls=calls,call_seconds=seconds,training_RNG_restored_in_isolated_process=True,geometry_backend='unchanged serial CPU V6',GPU_physical=1,live_GPU0_untouched=True,labels_opened=False,model_unchanged=True,rollout_seconds=time.perf_counter()-start);d.write(O/'actual_query_many_trace.json',trace);np.savez_compressed(O/'serial_observed_vectors.npz',**{f'call{i}':v for i,v in enumerate(vectors)});d.write(O/'trace_seal.json',dict(trace_sha256=sha(O/'actual_query_many_trace.json'),vectors_sha256=sha(O/'serial_observed_vectors.npz'),checkpoint_sha256=sha(dest),source_sha256=sha(__file__),live_checkpoint_inode_at_open=st.st_ino,live_checkpoint_size_at_open=st.st_size));summary={k:trace[k] for k in ['checkpoint_updates','next_cursor','scene','asset_distribution','call_seconds','rollout_seconds','labels_opened','model_unchanged']};summary.update(tasks=len(rows),query_many_call_sizes=[len(x) for x in calls],total_queries=sum(map(len,calls)),trace_path=str(O/'actual_query_many_trace.json'),trace_sha256=sha(O/'actual_query_many_trace.json'));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
