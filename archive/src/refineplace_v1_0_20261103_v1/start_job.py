"""Launch only explicitly selected v1 smoke or full eight-epoch pipeline."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
PY='/path/to/home/codex_3d_research_20261028/.venv/bin/python'

def launch(cmd,name):
    log=ROOT/(name+'.log');receipt=ROOT/(name+'_start.json')
    if receipt.exists():raise RuntimeError('Refusing duplicate launch: '+name)
    env=os.environ.copy();env.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
    with log.open('w') as fp:p=subprocess.Popen(cmd,cwd=ROOT,env=env,stdout=fp,stderr=subprocess.STDOUT,start_new_session=True)
    record=dict(pid=p.pid,command=cmd,log=str(log),started=time.time());receipt.write_text(json.dumps(record,indent=2));print(json.dumps(record))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['smoke','evaluation_smoke','full']);p.add_argument('--name');p.add_argument('--batch-size',type=int,default=64);a=p.parse_args()
    if a.mode in ('smoke','full'):
        name=a.name or ('smoke_train' if a.mode=='smoke' else 'train8')
        cmd=[PY,str(ROOT/'launch_v10.py'),'--output',str(ROOT/name),'--grounded-root',str(ROOT/'grounded_support/grounded_surfaces'),
             '--epochs','8','--batch-size',str(a.batch_size),'--workers','12']
        if a.mode=='smoke':cmd+=['--smoke-batches','3','--smoke-scene','scene0190_00']
        launch(cmd,name)
        if a.mode=='full':
            cmd=[PY,str(ROOT/'watch_epoch_evaluation.py'),'--training',str(ROOT/name),'--output',str(ROOT/(name+'_official150')),
                 '--manifest','/path/to/research300/refineplace_sync20_inferred_official150_20261102_v1/fixed_validation150.json',
                 '--grounded-root',str(ROOT/'validation_grounded_support/grounded_surfaces'),'--epochs','8']
            launch(cmd,name+'_evaluation_queue')
    else:
        name=a.name or 'smoke_eval'
        cmd=[PY,str(ROOT/'evaluate_v10.py'),'--checkpoint',str(ROOT/'smoke_gloo/rank0/smoke.pt'),
             '--training-protocol',str(ROOT/'smoke_gloo/protocol_before_training.json'),
             '--manifest','/path/to/research300/refineplace_sync20_inferred_official150_20261102_v1/fixed_validation150.json',
             '--output',str(ROOT/name),'--grounded-root',str(ROOT/'validation_grounded_support/grounded_surfaces'),'--epoch','0','--smoke']
        launch(cmd,name)
