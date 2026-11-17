"""Official563 TRAIN-only raw47 statistics and label-free141 observation cache.
No validation/test arrays or placement labels are selected. Legacy group JSON
contains annotation-count metadata, which is parsed but not consumed. Existing raw refer
packages are deserialized but only XYZ is selected; instance fields discarded.
"""
import argparse,json,hashlib,os,tempfile,time,sys,shutil
from pathlib import Path
import numpy as np,torch
LOCK_SHA='ab291e92388ee563a015f8b41d8bffb93caa16fd6aa4f23ab49c3fd53d8f9aec'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def fp(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def atomic(p,x):
 t=p.with_name(p.name+'.pending');t.write_text(json.dumps(x,indent=2));os.replace(t,p)
def guard(lock,scene):
 assert scene in lock['train_scenes'] and scene not in lock['validation_scenes'] and scene not in lock['test_scenes']
 assert scene[:9] not in {s[:9] for s in lock['validation_scenes']+lock['test_scenes']}
def load_array(root,meta,name,verify=True):
 p=root/(name+'.npy');spec=meta['arrays'][name]
 if verify:assert sha(p)==spec['sha256'],str(p)
 z=np.load(p,mmap_mode='r',allow_pickle=False);assert list(z.shape)==spec['shape'] and str(z.dtype)==spec['dtype'];return z
class Source:
 def __init__(self,a,lock,scene,v6):
  guard(lock,scene);self.scene=scene;self.base=a.root/'rl_position_adjustment_v1';self.shard=self.base/'placeit_full_train_shards_v1'/scene;self.target=a.observation_source/scene;self.groups=read(self.shard/'groups.json');self.tm=read(self.target/'manifest.json');assert self.tm['scene']==scene and self.tm['split']=='train' and self.tm['test_reads']==0 and self.tm['complete'];self.geometry=self.base/'scans_full_refer_v1/train'/(scene+'_refer.pth');self.A=v6.validate_pure_yaw_alignment(read(a.alignment)['alignments'][scene]);self.cache={}
  identities={Path(x['path']).name:x['sha256'] for x in self.tm['dependencies']['sources']};self.identity={k:sha(p) for k,p in [('raw_npz',self.shard/'data.npz'),('groups',self.shard/'groups.json'),('geometry_package',self.geometry),('observation_manifest',self.target/'manifest.json')]}
  assert identities.get('data.npz')==self.identity['raw_npz'] and identities.get('groups.json')==self.identity['groups'] and identities.get(scene+'_refer.pth')==self.identity['geometry_package']
  self.observed={k:load_array(self.target,self.tm,k) for k in ['group_ids','candidate_offsets','centers','text','dims','asset_uuid']};self.identity['observation_arrays']={k:self.tm['arrays'][k]['sha256'] for k in self.observed};assert self.observed['text'].shape==(len(self.observed['group_ids']),768) and np.isfinite(self.observed['text']).all();assert self.observed['dims'].shape==(len(self.observed['group_ids']),3) and np.isfinite(self.observed['dims']).all() and (self.observed['dims']>0).all();assert np.isfinite(self.observed['centers']).all()
  with np.load(self.shard/'data.npz',allow_pickle=False) as z:
   self.raw=np.asarray(z['features']);self.raw_centers=np.asarray(z['centers']);self.rowids=np.asarray(z['group_ids']);assert set(z['split'])=={'train'}
  assert self.raw.ndim==2 and self.raw.shape[1]>=47 and len(self.raw)==len(self.raw_centers)==len(self.rowids);self.gids=np.unique(self.rowids);assert np.array_equal(self.gids,self.observed['group_ids']);self.registry=[]
  for gi in self.gids:
   g=self.groups[int(gi)];guard(lock,g['scene']);assert g['scene']==scene and g['asset_scene'].startswith(scene+'_') and g['entry'].startswith('entry_');self.registry.append(dict(group_id=int(gi),scene=scene,asset_scene=g['asset_scene'],entry=g['entry'],prompt=g['prompt'],prompt_sha256=hashlib.sha256(g['prompt'].encode()).hexdigest()))
  assert len({(x['asset_scene'],x['entry']) for x in self.registry})==len(self.gids)
  payload=torch.load(self.geometry,map_location='cpu',weights_only=False);self.xyz=np.asarray(payload[0],dtype=np.float32).copy();del payload;assert self.xyz.ndim==2 and self.xyz.shape[1]==3 and np.isfinite(self.xyz).all();self.v6=v6
 def blocks(self):
  for pos,gi in enumerate(self.gids):
   ix=np.flatnonzero(self.rowids==gi);assert len(ix)==256;g=self.groups[int(gi)];ids=np.asarray(g['point_indices'],dtype=np.int64);assert ids.shape==(256,) and len(np.unique(ids))==256 and ids.min()>=0 and ids.max()<len(self.xyz);assert np.max(np.linalg.norm(self.xyz[ids]-self.raw_centers[ix],axis=1))<1e-5
   x,c=self.v6.aligned_task(self.raw[ix],self.raw_centers[ix],self.xyz,self.A,self.cache);x=np.asarray(x,dtype=np.float64);lo,hi=map(int,self.observed['candidate_offsets'][pos:pos+2]);assert hi-lo==256 and np.array_equal(c,self.observed['centers'][lo:hi]);assert x.shape==(256,47) and np.isfinite(x).all()
   assert np.allclose(x[:,12:15],self.observed['dims'][pos],atol=1e-6);assert str(self.observed['asset_uuid'][pos])==g['asset_scene'].split('_')[-1];yield pos,int(gi),x,c,ids
 def close(self):self.observed.clear();self.cache.clear();self.raw=None;self.xyz=None

def run(a):
 start=time.monotonic();assert sha(a.split_lock)==LOCK_SHA;lock=read(a.split_lock);train=lock['train_scenes'];assert len(train)==563 and set(train)==set(lock['normalization_fit_scenes']);assert len(set(train))==563;assert lock['official_h5']['train']['tasks']==100505
 if a.smoke:
  assert a.scene and len(set(a.scene))==len(a.scene)<=2;scenes=a.scene
 else:assert not a.scene;scenes=sorted(train)
 for s in scenes:guard(lock,s)
 assert not a.output.exists() or a.resume;a.output.mkdir(exist_ok=True,parents=True);sys.path.insert(0,str(a.root/'rl_position_adjustment_v1'));import train_joint_pose_v6 as v6;torch.set_num_threads(2)
 meta_path=a.split_lock.parent/'official_split_metadata_audit.json';assert sha(meta_path)==lock['source_metadata_audit_sha256'];official_meta=read(meta_path);assert official_meta['official']['train']['sha256']==lock['official_h5']['train']['sha256'];counts=official_meta['official']['train']['tasks_per_scene'];seen=set();inventory={}
 for scene in scenes:
  guard(lock,scene);gp=a.root/'rl_position_adjustment_v1/placeit_full_train_shards_v1'/scene/'groups.json';groups=read(gp);assert len(groups)==counts[scene],(scene,len(groups),counts[scene])
  for g in groups:
   assert g['scene']==scene and g['asset_scene'].startswith(scene+'_') and g['entry'].startswith('entry_');key=(scene,g['asset_scene'],g['entry'],hashlib.sha256(g['prompt'].encode()).hexdigest());assert key not in seen;seen.add(key)
  inventory[scene]=dict(tasks=len(groups),groups_sha256=sha(gp),task_identity_fingerprint=fp(sorted((g['scene'],g['asset_scene'],g['entry'],hashlib.sha256(g['prompt'].encode()).hexdigest()) for g in groups)))
 assert len(seen)==sum(counts[s] for s in scenes)
 if not a.smoke:assert len(seen)==100505
 if not a.smoke:
  assert a.identity_audit is not None and a.identity_audit_sha256 and sha(a.identity_audit)==a.identity_audit_sha256
  ia=read(a.identity_audit);assert ia['complete_verified'] and ia['official_split_lock_sha256']==LOCK_SHA and ia['official_train_h5_sha256']==lock['official_h5']['train']['sha256'] and ia['source_h5_input_identity_only'] and ia['tasks']==100505 and ia['full_train_scenes']==563
  assert sha(lock['official_h5']['train']['source'])==lock['official_h5']['train']['sha256']
  for scene in scenes:
   q=ia['per_scene'][scene];assert q['groups_sha256']==inventory[scene]['groups_sha256'] and q['matched_tasks']==counts[scene] and q['observation_dims_asset_identity_verified'];assert q['native_task_identity_fingerprint']==q['raw_group_task_identity_fingerprint']==inventory[scene]['task_identity_fingerprint']
 inv=dict(full_official_train563=not a.smoke,scenes=len(scenes),tasks=len(seen),unique_task_identity_count=len(seen),per_scene=inventory,placement_labels_selected=False);atomic(a.output/'source_task_inventory_before_statistics.json',inv)
 header=dict(version='RefinePlace_v0.1_official_train563_observation_cache',official_split_lock_sha256=LOCK_SHA,official_train_h5_sha256=lock['official_h5']['train']['sha256'],fit_scenes=scenes,full_official_train563=not a.smoke,fit_task_count_expected=sum(counts[s] for s in scenes),source_task_inventory_sha256=sha(a.output/'source_task_inventory_before_statistics.json'),official_input_identity_audit_sha256=None if a.smoke else a.identity_audit_sha256,legacy_annotation_count_metadata_parsed_but_not_consumed=True,source_sha256=sha(__file__),aligned_module_sha256=sha(v6.__file__),alignment_sha256=sha(a.alignment),observation_fields_only=True,labels_consumed=False,validation_or_test_arrays_opened=False,legacy_normalization_or_weights_loaded=False,geometry_package_deserialized_but_instance_fields_not_selected=True)
 hp=a.output/'locked_input_protocol.json'
 if hp.exists():assert read(hp)==header
 else:atomic(hp,header)
 npth=a.output/'normalization.json';identities={};n=0;mean=np.zeros(47);m2=np.zeros(47);scene_counts={}
 if npth.exists():
  norm=read(npth);assert norm['header']==header;assert norm['fingerprint']==fp({k:v for k,v in norm.items() if k!='fingerprint'});identities=norm['sources'];scene_counts=norm['scene_counts']
 else:
  for index,s in enumerate(scenes,1):
   src=Source(a,lock,s,v6);assert src.identity['groups']==inventory[s]['groups_sha256'];assert len(src.gids)==counts[s],(s,len(src.gids),counts[s]);identities[s]=src.identity;rows=tasks=0
   for _,_,x,_,_ in src.blocks():
    k=len(x);xm=x.mean(0);delta=xm-mean;new=n+k;m2+=((x-xm)**2).sum(0)+delta*delta*n*k/new;mean+=delta*k/new;n=new;rows+=k;tasks+=1
   scene_counts[s]=dict(tasks=tasks,candidate_rows=rows);src.close();progress=dict(stage='fit_train_only_statistics',scenes_done=index,scenes_total=len(scenes),candidate_rows=n,tasks=sum(x['tasks'] for x in scene_counts.values()),seconds=time.monotonic()-start);atomic(a.output/'progress.json',progress);print(json.dumps(progress),flush=True)
  assert sum(x['tasks'] for x in scene_counts.values())==header['fit_task_count_expected'];std=np.maximum(np.sqrt(np.maximum(m2/n,0)),.01);norm=dict(header=header,sources=identities,scene_counts=scene_counts,fit_candidate_rows=n,mean=mean.tolist(),std=std.tolist(),variance='Population Welford merged across complete declared TRAIN tasks; std floor0.01');norm['fingerprint']=fp(norm);atomic(npth,norm)
 mean=np.asarray(norm['mean']);std=np.asarray(norm['std']);summary={}
 for index,s in enumerate(scenes,1):
  guard(lock,s);dst=a.output/s;src=Source(a,lock,s,v6);assert src.identity==identities[s] and src.identity['groups']==inventory[s]['groups_sha256']
  if dst.exists():
   assert a.resume;m=read(dst/'manifest.json');assert m['fingerprint']==fp({k:v for k,v in m.items() if k!='fingerprint'});assert sha(dst/'task_registry.json')==m['task_registry_sha256'] and read(dst/'task_registry.json')==src.registry;assert m['normalization_sha256']==sha(npth) and m['source_identity']==identities[s]
   for k,z in m['arrays'].items():assert sha(dst/(k+'.npy'))==z['sha256']
   summary[s]=m;src.close();continue
  nt=len(src.gids);nr=nt*256;tmp=Path(tempfile.mkdtemp(prefix=s+'.pending.',dir=a.output));spec=[('features',(nr,141),np.float32),('centers',(nr,3),np.float32),('candidate_ids',(nr,),np.int64),('candidate_offsets',(nt+1,),np.int64),('group_ids',(nt,),np.int64),('text',(nt,768),np.float32),('dims',(nt,3),np.float32),('asset_uuid',(nt,),'<U32')];arrays={k:np.lib.format.open_memmap(tmp/(k+'.npy'),mode='w+',shape=shape,dtype=dtype) for k,shape,dtype in spec};error=0.;cursor=0;arrays['candidate_offsets'][0]=0
  for pos,gi,x,c,ids in src.blocks():
   end=cursor+256;f=v6.make_features(x,mean,std,torch.device('cpu')).numpy();z=np.clip((x-mean)/std,-10,10).astype(np.float32);expected=np.concatenate((z,np.broadcast_to(z.mean(0),z.shape),np.broadcast_to(z.max(0),z.shape)),1);error=max(error,float(np.abs(expected-f).max()));assert f.shape==(256,141);arrays['features'][cursor:end]=f;arrays['centers'][cursor:end]=c;arrays['candidate_ids'][cursor:end]=ids;arrays['candidate_offsets'][pos+1]=end;arrays['group_ids'][pos]=gi
   for k in ['text','dims','asset_uuid']:arrays[k][pos]=src.observed[k][pos]
   cursor=end
  assert cursor==nr and error==0.;schema={}
  for k,v in arrays.items():v.flush();schema[k]=dict(shape=list(v.shape),dtype=str(v.dtype),sha256=sha(tmp/(k+'.npy')))
  del v;arrays.clear();atomic(tmp/'task_registry.json',src.registry);m=dict(scene=s,partition='official_train',tasks=nt,candidate_rows=nr,normalization_sha256=sha(npth),normalization_fingerprint=norm['fingerprint'],source_identity=identities[s],arrays=schema,task_registry_sha256=sha(tmp/'task_registry.json'),feature_and_pool_max_error=error,label_arrays_present=False);m['fingerprint']=fp(m);atomic(tmp/'manifest.json',m);os.replace(tmp,dst);summary[s]=m;src.close();progress=dict(stage='write_observation_cache',scenes_done=index,scenes_total=len(scenes),tasks=sum(x['tasks'] for x in summary.values()),seconds=time.monotonic()-start);atomic(a.output/'progress.json',progress);print(json.dumps(progress),flush=True)
 assert sum(x['tasks'] for x in summary.values())==header['fit_task_count_expected'];report=dict(complete=True,full_official_train563=not a.smoke,scenes=len(scenes),tasks=sum(x['tasks'] for x in summary.values()),candidate_rows=sum(x['candidate_rows'] for x in summary.values()),normalization_sha256=sha(npth),official_split_lock_sha256=LOCK_SHA,legacy_normalization_loaded=False,weights_loaded=False,labels_consumed=False,validation_or_test_arrays_opened=False,geometry_package_deserialization_caveat=True,feature_and_pool_error=0.,seconds=time.monotonic()-start,scene_manifests={s:sha(a.output/s/'manifest.json') for s in scenes});atomic(a.output/'index.json',report);print(json.dumps({k:v for k,v in report.items() if k!='scene_manifests'}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('/path/to/home/codex_3d_research_20261028'));p.add_argument('--split-lock',type=Path,default=Path('/path/to/research100/official_split_training_audit_20261102_v1/official563_training_split_lock.json'));p.add_argument('--alignment',type=Path,default=Path('/path/to/home/codex_3d_research_20261028/rl_position_adjustment_v1/pose_alignment_manifest_v4.json'));p.add_argument('--observation-source',type=Path,default=Path('/path/to/research300/placeit_training_cache_v7'));p.add_argument('--output',type=Path,required=True);p.add_argument('--smoke',action='store_true');p.add_argument('--scene',action='append');p.add_argument('--resume',action='store_true');p.add_argument('--identity-audit',type=Path);p.add_argument('--identity-audit-sha256');run(p.parse_args())
