"""Actual v3 CPU12 migration + same-directory resume, three temporary updates only."""
import os,sys,subprocess,json,hashlib,time
from pathlib import Path
M=Path('/path/to/research300/refineplace_v01_official563_modules_20261102_v1')
O=Path('/path/to/research300/refineplace_v01_v3_CPU12_migration_smoke_20261102_v1')
P=Path('/path/to/research300/refineplace_v01_official563_joint_full_20261102_v1')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert os.environ['CUDA_VISIBLE_DEVICES']=='1';O.mkdir(exist_ok=False)
 expected='be17c8c46ed37cabda5ece88a591a10af7576dc7aaae518d910cacae46f8cf4c';assert sha(P/'last_step.pt')==expected
 base=[sys.executable,'-u',str(M/'train_official563_joint_v3_cpu_pool.py'),'--output',str(O/'temporary_training'),'--epochs','50','--batch-size','32','--geometry-workers','12','--parent-checkpoint',str(P/'last_step.pt'),'--reference-catalog','/path/to/research100/official_train563_real_reference_catalog_20261102_v1/catalog.json','--reference-catalog-sha256','7e766ebe199a4dde76f2ae90669fa35107f522171f3107acd93dfae7d0a0e8f6']
 stages=[]
 for name,args in [('migrate_two_updates',['--stop-after-updates','574']),('resume_one_update',['--resume',str(O/'temporary_training/last_step.pt'),'--stop-after-updates','575'])]:
  with (O/(name+'.log')).open('wb') as f:r=subprocess.run(base+args,stdout=f,stderr=subprocess.STDOUT)
  stages.append(dict(stage=name,exit_code=r.returncode));(O/'stages.json').write_text(json.dumps(stages))
  assert r.returncode==0,name
 import torch,numpy as np,random
 sys.path.insert(0,str(M));import compare_real_training_pool_v2 as c
 cp=torch.load(O/'temporary_training/last_step.pt',map_location='cpu',weights_only=False)
 reference=json.loads(Path('/path/to/research300/refineplace_v01_training_step_pool_comparison_v2_20261102_v1/process12_report.json').read_text())['steps'][:3]
 def optsha(s):
  h=hashlib.sha256(json.dumps(s['param_groups'],sort_keys=True).encode())
  for i,v in sorted(s['state'].items()):
   for k,x in sorted(v.items()):h.update(str((i,k)).encode());h.update(x.detach().cpu().contiguous().numpy().tobytes() if torch.is_tensor(x) else repr(x).encode())
  return h.hexdigest()
 events=[]
 for item in cp['log_chain']:
  if Path(item['path']).parent==O/'temporary_training':events.extend(json.loads(x) for x in Path(item['path']).read_bytes()[:item['committed_bytes']].splitlines() if x)
 assert len(events)==3 and cp['updates']==575
 checks=[]
 for actual,ref in zip(events,reference):
  fields={'task_ids':actual['task_ids']==ref['task_ids'],'trace':actual['pose_action_trace_sha256']==ref['pose_action_trace_sha256'],'supervised_loss':actual['supervised_loss']==ref['supervised_loss'],'rl_loss':actual['rl_loss']==ref['RL_loss']}
  assert all(fields.values()),fields;checks.append(fields)
 c.restore_rng(cp['rng']);assert c.rng_fingerprint()==reference[-1]['rng']
 assert c.d.tiny.digest(cp['joint_policy'])==reference[-1]['model_sha256'] and optsha(cp['optimizer'])==reference[-1]['optimizer_sha256']
 assert sha(P/'last_step.pt')==expected and json.loads((P/'summary.json').read_text())['status']=='stopped_checkpoint_saved'
 report=dict(status='v3_CPU12_actual_migration_and_resume_passed',workers=12,temporary_updates=3,stage_exit_codes=stages,actual_updates=cp['updates'],next_cursor=cp['next_cursor'],all_task_order_trace_losses_exact=checks,final_model_optimizer_all_RNG_exact_to_frozen_serial_reference=True,parent_checkpoint_unchanged_and_formal_training_stopped=True,validation_test_opened=False,source_sha256=sha(M/'train_official563_joint_v3_cpu_pool.py'),new_protocol_sha256=sha(O/'temporary_training/protocol_before_training.json'),new_checkpoint_sha256=sha(O/'temporary_training/last_step.pt'))
 (O/'summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
if __name__=='__main__':main()
