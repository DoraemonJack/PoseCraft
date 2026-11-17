"""Measure real train-step throughput; dual case is independent replicas.

Not synchronous distributed training: the dual result measures a throughput
ceiling and must not be described as one shared-model training speed.
"""
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

MODULE = Path('/path/to/research300/refineplace_v01_official563_modules_20261102_v1')
ROOT = Path('/path/to/research300/refineplace_speedtest_32cpu_2gpu_20261102_v1')
CAT = Path('/path/to/research300/refineplace_reference_multiframe_assoc_20261102_v1/reference_catalog_compact_v2_manifest.json')
PY = '/path/to/home/codex_3d_research_20261028/.venv/bin/python'
SHA = hashlib.sha256(CAT.read_bytes()).hexdigest()

def start(name, gpu, workers):
    output = ROOT / name
    log = (ROOT / (name + '.log')).open('w')
    args = [PY, '-u', str(MODULE / 'train_official563_speedtest_v1.py'),
            '--fresh-start', '--output', str(output), '--pilot-batches', '4',
            '--batch-size', '32', '--geometry-workers', str(workers),
            '--reference-catalog', str(CAT), '--reference-catalog-sha256', SHA]
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return subprocess.Popen(args, cwd=MODULE, env=env, stdout=log, stderr=subprocess.STDOUT), log

def stage(name, jobs):
    t = time.monotonic(); procs = [(n, *start(n, g, w)) for n, g, w in jobs]
    (ROOT / 'status.json').write_text(json.dumps({'stage': name, 'pids': [p.pid for _, p, _ in procs]}))
    results = []
    for n, p, log in procs:
        code = p.wait(); log.close()
        path = ROOT / n / 'summary.json'
        result = json.loads(path.read_text()) if code == 0 and path.exists() else None
        results.append({'name': n, 'exit_code': code, 'summary': result})
    report = {'stage': name, 'wall_seconds': time.monotonic() - t, 'runs': results,
              'note': 'two_gpu uses independent unsynchronized models' if len(jobs) == 2 else 'single model'}
    (ROOT / (name + '.json')).write_text(json.dumps(report, indent=2))
    if any(x['exit_code'] != 0 for x in results):
        raise SystemExit('failed stage: ' + name)

def main():
    ROOT.mkdir(parents=True, exist_ok=False)
    stage('single12', [('single12', 0, 12)])
    stage('single24', [('single24', 0, 24)])
    stage('dual12each', [('dual12_gpu0', 0, 12), ('dual12_gpu1', 1, 12)])
    (ROOT / 'status.json').write_text(json.dumps({'stage': 'complete'}))

if __name__ == '__main__':
    main()
