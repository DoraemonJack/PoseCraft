"""One arm per GPU; train10k then fixed test300 and validation150 inference."""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
V10 = Path('/path/to/research300/refineplace_v1_0_20261103_v1')


def write(path, data):
    tmp = path.with_name(path.name + '.pending')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(tmp, path)


def run(a):
    a.output.mkdir(parents=True, exist_ok=False)
    plans = []
    processes = []
    logs = []
    start = time.time()
    try:
        for gpu, arm in enumerate(('baseline', 'upgraded')):
            out = a.output / arm
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            command = [sys.executable, '-u', str(HERE/'train_pilot.py'), '--arm', arm,
                '--output', str(out), '--manifest', str(a.manifest), '--parent', str(a.parent),
                '--grounded-root', str(V10/'grounded_support/grounded_surfaces'),
                '--batch-size', str(a.batch_size), '--workers', str(a.workers)]
            log = (a.output / (arm+'.log')).open('ab', buffering=0)
            logs.append(log)
            process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            processes.append(process)
            plans.append(dict(arm=arm, gpu=gpu, pid=process.pid, command=command))
        write(a.output/'status.json', dict(stage='training', plans=plans, started=start))
        while any(p.poll() is None for p in processes):
            bad = [(plans[i]['arm'], p.returncode) for i,p in enumerate(processes) if p.poll() not in (None,0)]
            if bad:
                raise RuntimeError('Training arm failed: '+repr(bad))
            time.sleep(5)
        for arm in ('baseline','upgraded'):
            summary = json.loads((a.output/arm/'summary.json').read_text())
            if summary['status']!='training_complete' or summary['tasks']!=10000:
                raise RuntimeError('Incomplete training arm: '+arm)
        write(a.output/'status.json', dict(stage='test300_inference', plans=plans, started=start))
        processes = []
        for gpu, arm in enumerate(('baseline','upgraded')):
            out = a.output / arm
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            cmd = [sys.executable, '-u', str(HERE/'infer_test300.py'), '--checkpoint', str(out/'final.pt'),
                '--training-protocol', str(out/'protocol_before_training.json'),
                '--output', str(a.output/(arm+'_test300'))]
            log = (a.output/(arm+'_test300.log')).open('ab', buffering=0)
            logs.append(log)
            processes.append(subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        while any(p.poll() is None for p in processes):
            bad = [p.returncode for p in processes if p.poll() not in (None,0)]
            if bad:
                raise RuntimeError('Test300 inference failed: '+repr(bad))
            time.sleep(5)
        requests = []
        for arm in ('baseline','upgraded'):
            report = json.loads((a.output/(arm+'_test300')/'summary.json').read_text())
            if report['status']!='complete' or report['tasks']!=300:
                raise RuntimeError('Missing test300 completion')
            requests.append(dict(arm=arm, partition='test', tasks=300,
                predictions=str(a.output/(arm+'_test300')/'predictions.json'), prediction_sha256=report['prediction_sha256']))
        write(a.output/'evaluation_requests.json', dict(status='test300_predictions_ready', requests=requests))
        write(a.output/'status.json', dict(stage='validation_inference', plans=plans, started=start))
        processes = []
        for gpu, arm in enumerate(('baseline','upgraded')):
            out = a.output / arm
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
            cmd = [sys.executable, '-u', str(HERE/'evaluate_pilot.py'), '--checkpoint', str(out/'final.pt'),
                '--training-protocol', str(out/'protocol_before_training.json'),
                '--manifest', str(a.validation_manifest), '--output', str(a.output/(arm+'_val150')),
                '--grounded-root', str(V10/'validation_grounded_support/grounded_surfaces'), '--inference-only']
            log = (a.output/(arm+'_val150.log')).open('ab', buffering=0)
            logs.append(log)
            processes.append(subprocess.Popen(cmd, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        while any(p.poll() is None for p in processes):
            bad = [p.returncode for p in processes if p.poll() not in (None,0)]
            if bad:
                raise RuntimeError('Validation inference failed: '+repr(bad))
            time.sleep(5)
        for arm in ('baseline','upgraded'):
            report = json.loads((a.output/(arm+'_val150')/'inference_report.json').read_text())
            if report['status']!='inference_complete':
                raise RuntimeError('Missing validation completion')
            requests.append(dict(arm=arm, partition='val', tasks=150, predictions=str(a.output/(arm+'_val150')/'predictions.json')))
        write(a.output/'evaluation_requests.json', dict(status='predictions_ready', requests=requests,
            test300='original_entry_id 0..299; baseline arm is continued V1.0; preserved original epoch2 predictions are an additional fixed reference'))
        write(a.output/'status.json', dict(stage='predictions_ready', plans=plans, started=start, elapsed_seconds=time.time()-start,
            official_evaluation_started=False, test300_evaluation_started=False))
    except BaseException as exc:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        write(a.output/'status.json', dict(stage='failed', error=repr(exc), started=start, plans=plans))
        raise
    finally:
        for log in logs:
            log.close()


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--manifest',type=Path,default=HERE/'manifest.json')
    p.add_argument('--parent',type=Path,default=V10/'train8_clean/rank0/epoch2.pt')
    p.add_argument('--validation-manifest',type=Path,default=Path('/path/to/research300/refineplace_sync20_inferred_official150_20261102_v1/fixed_validation150.json'))
    p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--workers',type=int,default=4)
    run(p.parse_args())
