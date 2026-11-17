from pathlib import Path
import json,subprocess,os,time,hashlib
import torch,numpy as np
M=Path('/path/to/research300/refineplace_v01_official563_modules_20261102_v1');O=Path('/path/to/research300/refineplace_v01_official563_resume_validation_20261102_v1');O.mkdir(exist_ok=False);A=O/'continuous';B=O/'interrupted';cat='/path/to/research100/official_train563_real_reference_catalog_20261102_v1/catalog.json';csha='7e766ebe199a4dde76f2ae90669fa35107f522171f3107acd93dfae7d0a0e8f6';assert hashlib.sha256(Path(cat).read_bytes()).hexdigest()==csha
base=['/path/to/home/codex_3d_research_20261028/.venv/bin/python','-u',str(M/'train_official563_joint_v2.py'),'--pilot-batches','4','--checkpoint-every','1','--reference-catalog',cat,'--reference-catalog-sha256',csha];env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');history=[]
for name,extra in [('continuous',['--output',str(A)]),('stop2',['--output',str(B),'--stop-after-updates','2']),('resume',['--output',str(B),'--resume',str(B/'last_step.pt')])]:
 start=time.time();r=subprocess.run(base+extra,env=env,cwd='/path/to/home/codex_3d_research_20261028/rl_position_adjustment_v1',stdout=open(O/(name+'.log'),'wb'),stderr=subprocess.STDOUT,timeout=1200);history.append(dict(stage=name,exit_code=r.returncode,seconds=time.time()-start));(O/'execution_status.json').write_text(json.dumps(history));print(json.dumps(history[-1]),flush=True)
 if r.returncode:raise RuntimeError('Actual resume validation child failed: '+name)
x=torch.load(A/'pilot_last.pt',map_location='cpu',weights_only=False);y=torch.load(B/'pilot_last.pt',map_location='cpu',weights_only=False);assert x['updates']==y['updates']==4
model_exact=all(torch.equal(x['joint_policy'][k],y['joint_policy'][k]) for k in x['joint_policy']);maxerr=max(float((x['joint_policy'][k]-y['joint_policy'][k]).abs().max()) for k in x['joint_policy']);assert maxerr<=1e-6
opt_exact=x['optimizer']['param_groups']==y['optimizer']['param_groups']
for k,v in x['optimizer']['state'].items():
 for n,t in v.items():opt_exact=opt_exact and (torch.equal(t,y['optimizer']['state'][k][n]) if torch.is_tensor(t) else t==y['optimizer']['state'][k][n])
rng_exact=torch.equal(x['rng']['torch_cpu'],y['rng']['torch_cpu']) and all(torch.equal(a,b) for a,b in zip(x['rng']['torch_cuda'],y['rng']['torch_cuda'])) and x['rng']['python']==y['rng']['python'] and all(np.array_equal(a,b) for a,b in zip(x['rng']['numpy'],y['rng']['numpy']));assert rng_exact
read=lambda p:json.loads(Path(p).read_text())
def rows(cp):
 result=[]
 for item in cp['log_chain']:
  with open(item['path'],'rb') as f:data=f.read(item['committed_bytes'])
  assert hashlib.sha256(data).hexdigest()==item['committed_sha256'];result.extend(json.loads(line) for line in data.splitlines())
 return result
ar,br=rows(x),rows(y);assert len(ar)==len(br)==4;trace_exact=[a['pose_action_trace_sha256'] for a in ar]==[b['pose_action_trace_sha256'] for b in br];assert trace_exact and [a['task_ids'] for a in ar]==[b['task_ids'] for b in br]
refgrad=sum(r['gradient_l1_by_group'].get('reference',0.) for r in ar);active=sum(r['active_reference_tasks'] for r in ar);summary=dict(status='actual_catalog_resume_validation_passed',updates_each=4,actual_task_exposures_each=sum(r['tasks'] for r in ar),model_parameters_bitwise_equal=model_exact,model_parameter_max_abs_difference=maxerr,optimizer_state_bitwise_equal=opt_exact,all_CPU_CUDA_NumPy_Python_RNG_equal=rng_exact,pose_action_traces_exact=trace_exact,task_order_exact=True,actual_reference_active_exposures=active,reference_gradient_l1_sum=refgrad,training_split='official563',val_test_opened=False,execution=history,uninterrupted_checkpoint_sha256=hashlib.sha256((A/'pilot_last.pt').read_bytes()).hexdigest(),resumed_checkpoint_sha256=hashlib.sha256((B/'pilot_last.pt').read_bytes()).hexdigest());assert model_exact and opt_exact and active>0 and refgrad>0;(O/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
