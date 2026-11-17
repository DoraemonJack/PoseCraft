"""Run frozen full TEST reference detection, then CPU association/grounder scoring."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'references'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(value):
    tmp = OUT / 'reference_pipeline_progress.partial'
    tmp.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    tmp.replace(OUT / 'reference_pipeline_progress.json')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--adopt-detection-pid', type=int)
    args = parser.parse_args()
    started = time.time()
    required = ['tasks_manifest.json', 'text_features.npy', 'reference_detector.py', 'build_test_reference_catalog.py', 'reference_model_smoke.py',
                'geometry/mesh_manifest_v6.json', 'receivers/test_grounded_frames/frame_source_identity_before_models.json',
                'references/mapped_test_tasks.jsonl', 'references/cleaned_detection_jobs_v2.jsonl',
                'references/split_between_candidates_v1.jsonl', 'references/deepseek_reference_results.jsonl']
    identities = {path: sha(ROOT / path) for path in required}
    pairs = set()
    for name in ['cleaned_detection_jobs_v2.jsonl', 'split_between_candidates_v1.jsonl']:
        for line in (OUT / name).open(encoding='utf-8'):
            job = json.loads(line)
            if job['cleanup_status'] == 'detector_candidate': pairs.add((job['scene'], job['detector_phrase']))
    protocol = dict(scope='official_test_observation_inference', tasks=3500, scene_phrase_pairs=len(pairs),
                    frame_detection_jobs=4 * len(pairs), source_sha256=identities, gpu=1,
                    placement_labels_read=False, ground_truth_relationships_read=False,
                    train_weights_frozen=True, started_unix=started, pid=os.getpid(), python_executable=sys.executable)
    (OUT / 'reference_pipeline_protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='1', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               MKL_NUM_THREADS='1', REFERENCE_MAX_JOBS='100000', TOKENIZERS_PARALLELISM='false')
    stages = [('detection', 'reference_detector.py', 'reference_detection.log', sys.executable),
              ('association_and_grounder', 'build_test_reference_catalog.py', 'reference_catalog_build.log', sys.executable),
              ('model_schema_smoke', 'reference_model_smoke.py', 'reference_model_smoke.log', '/path/to/home/codex_3d_research_20261028/.venv/bin/python')]
    for stage, script, logfile, python in stages:
        save(dict(status='running', stage=stage, started_unix=started, updated_unix=time.time(), pid=os.getpid()))
        for name, expected in identities.items():
            if sha(ROOT / name) != expected: raise RuntimeError('Sealed reference input changed: ' + name)
        if stage == 'detection' and args.adopt_detection_pid:
            child = Path('/proc') / str(args.adopt_detection_pid)
            while child.exists():
                try:
                    state = (child / 'stat').read_text().split(') ', 1)[1].split()[0]
                    if state == 'Z': break
                except FileNotFoundError:
                    break
                time.sleep(5)
            # The builder independently verifies every expected detector file.
            continue
        with (OUT / logfile).open('ab') as log:
            result = subprocess.run([python, '-u', str(ROOT / script)], cwd=str(ROOT), env=env,
                                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            save(dict(status='failed', stage=stage, exit_code=result.returncode, log=logfile,
                      started_unix=started, updated_unix=time.time()))
            raise SystemExit(result.returncode)
    manifest = json.loads((OUT / 'reference_catalog_compact_v4_test_pending.json').read_text(encoding='utf-8'))
    smoke = json.loads((OUT / 'reference_model_smoke.json').read_text(encoding='utf-8'))
    assert smoke['status'] == 'passed' and smoke['catalog_sha256'] == manifest['data_sha256']
    manifest.update(status='complete', model_schema_smoke_sha256=sha(OUT / 'reference_model_smoke.json'))
    target = OUT / 'reference_catalog_compact_v4_test_manifest.json'
    temp = target.with_suffix('.partial')
    temp.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    temp.replace(target)
    assert manifest['task_count'] == 3500
    save(dict(status='complete', stage='complete', manifest=str(OUT / 'reference_catalog_compact_v4_test_manifest.json'),
              data_path=manifest['data_path'], data_sha256=manifest['data_sha256'], counts=manifest['counts'],
              started_unix=started, updated_unix=time.time(), elapsed_seconds=time.time() - started))


if __name__ == '__main__':
    main()
