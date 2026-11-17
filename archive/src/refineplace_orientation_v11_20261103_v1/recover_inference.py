import json, os, subprocess, sys, time
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE/'paired10k'
def status(stage, **kw):
    p=ROOT/'status.json';tmp=p.with_suffix('.pending')
    tmp.write_text(json.dumps(dict(stage=stage,time=time.time(),**kw)))
    tmp.replace(p)
jobs=[]
try:
    for gpu,arm in enumerate(('baseline','upgraded')):
        summary=json.loads((ROOT/arm/'summary.json').read_text())
        assert summary['status']=='training_complete' and summary['tasks']==10000
        output=ROOT/(arm+'_test300')
        if output.exists(): output.rename(ROOT/(arm+'_test300_failed_logging_'+str(int(time.time()))))
        cmd=[sys.executable,'-u',str(HERE/'infer_test300.py'),'--checkpoint',str(ROOT/arm/'final.pt'),
             '--training-protocol',str(ROOT/arm/'protocol_before_training.json'),'--output',str(output)]
        log=(ROOT/(arm+'_test300_recovery.log')).open('ab',buffering=0)
        jobs.append(subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu)),stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL))
    status('test300_inference',recovery='strict pose checks; nonfinite diagnostic fields serialized as null',pids=[p.pid for p in jobs])
    while any(p.poll() is None for p in jobs):
        if any(p.poll() not in (None,0) for p in jobs): raise RuntimeError('Inference child failed')
        time.sleep(5)
    assert all(p.returncode==0 for p in jobs)
    assert all(json.loads((ROOT/(a+'_test300')/'summary.json').read_text())['status']=='complete' for a in ('baseline','upgraded'))
    status('test300_predictions_ready')
except BaseException as e:
    for p in jobs:
        if p.poll() is None:p.terminate()
    status('failed',error=repr(e));raise
