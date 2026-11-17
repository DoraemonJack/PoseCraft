"""Wait for the isolated V13 paired-value fit, then export fixed test IDs 0..299.

No training is launched/resumed here, and no official evaluator is launched.
The frozen V11 baseline retains its learned policy; this V13 run fits action
values from paired supervision. It is not a new reinforcement-learning run.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from evaluate_test300_official_v13 import validate_inference

HERE = Path(__file__).resolve().parent
PARENT_SHA256 = '480bb902bbbeb174a36da395fdf531c1845386cc66d653478c2584081e41c774'
TASKS = 1000
EPOCHS = 3
SOURCE_FILES = ('train_paired_v13.py', 'model_v13.py', 'paired_value_losses.py',
                'refiner_v13.py', 'infer_test300_v13.py')
FAILURE_STATES = {'failed', 'audit_failed', 'stopped', 'cancelled', 'interrupted'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'),
                      parse_constant=lambda x: (_ for _ in ()).throw(ValueError('Nonfinite JSON: '+x)))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+'.pending')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def option(command, name):
    matches = [i for i, value in enumerate(command) if value == name]
    if len(matches) != 1 or matches[0]+1 >= len(command):
        raise ValueError('Training receipt must contain exactly one '+name)
    return command[matches[0]+1]


def check_receipt(path, train_dir):
    receipt = read(path)
    pid, command = receipt.get('pid'), receipt.get('command')
    if type(pid) is not int or pid <= 1 or pid == os.getpid():
        raise ValueError('Invalid training process ID')
    if not isinstance(command, list) or any(not isinstance(x, str) for x in command):
        raise ValueError('Invalid training command receipt')
    if len(command) < 3 or Path(command[2]).resolve() != HERE/'train_paired_v13.py':
        raise ValueError('Training receipt is not the isolated V13 trainer')
    if Path(option(command, '--output')).resolve() != train_dir.resolve():
        raise ValueError('Training receipt output mismatch')
    if option(command, '--epochs') != str(EPOCHS) or '--max-tasks' in command:
        raise ValueError('Expected the authorized full 1000-task, three-epoch pilot')
    return receipt


def process_token(pid):
    """Linux start time disambiguates a reused PID; reject a zombie trainer."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return None
    except PermissionError:
        return 'alive_without_proc_access'
    proc = Path('/proc')/str(pid)/'stat'
    if proc.exists():
        fields = proc.read_text().rsplit(')', 1)[1].split()
        if fields[0] == 'Z':
            return None
        return fields[19]  # field 22, after PID and parenthesized command
    return 'alive'


def check_failure(record):
    if record and (record.get('status') in FAILURE_STATES or record.get('stage') in FAILURE_STATES):
        raise RuntimeError('Training stopped without an acceptable complete checkpoint: '+
                           str(record.get('status', record.get('stage'))))


def validate_training(train_dir, source_hashes):
    summary = read(train_dir/'summary.json')
    protocol_path = train_dir/'protocol_before_training.json'
    protocol = read(protocol_path)
    required_summary = dict(status='training_complete', architecture='RefinePlaceV13',
        tasks=TASKS, epochs=EPOCHS, baseline_weights_unchanged=True,
        parent_checkpoint_unchanged=True, independent_test_used=False)
    required_protocol = dict(architecture='RefinePlaceV13',
        phase='paired_heading_and_one_step_action_value', tasks=TASKS, epochs=EPOCHS,
        baseline_frozen=True, full_horizon_q_learning=False, test_labels_read=False,
        validation_labels_read=False, original_full_training_resumed=False)
    for record, required in ((summary, required_summary), (protocol, required_protocol)):
        for key, value in required.items():
            if record.get(key) != value or type(record.get(key)) is not type(value):
                raise ValueError('Training admission rejected: '+key)
    if protocol.get('parent_checkpoint_sha256') != PARENT_SHA256:
        raise ValueError('Parent is not the pinned upgraded V11 checkpoint')
    if sha(protocol['parent_checkpoint']) != PARENT_SHA256:
        raise ValueError('Parent checkpoint changed on disk')
    if sha(train_dir/'final.pt') != summary.get('final_checkpoint_sha256'):
        raise ValueError('Final V13 checkpoint hash mismatch')
    for name, digest in source_hashes.items():
        if sha(HERE/name) != digest or protocol.get('source_hashes', {}).get(name) != digest:
            raise ValueError('Training/inference source changed: '+name)
    return dict(summary=summary, protocol=protocol, protocol_sha256=sha(protocol_path),
                summary_sha256=sha(train_dir/'summary.json'))


def wait_training(train_dir, pid, source_hashes, *, timeout, poll_seconds, progress):
    started, token = time.monotonic(), process_token(pid)
    while True:
        if (train_dir/'summary.json').is_file():
            check_failure(read(train_dir/'summary.json'))
            return validate_training(train_dir, source_hashes)
        state_path = train_dir/'progress.json'
        state = read(state_path) if state_path.exists() else None
        check_failure(state)
        current = process_token(pid)
        if token is None or current is None or current != token:
            # A completion can arrive between the first summary check and PID check.
            if (train_dir/'summary.json').is_file():
                return validate_training(train_dir, source_hashes)
            raise RuntimeError('Training process exited or its PID changed without a completed summary')
        elapsed = time.monotonic()-started
        if elapsed >= timeout:
            raise TimeoutError('Training completion deadline exceeded; no inference was started')
        progress('waiting_for_paired_value_training', training_pid=pid,
                 elapsed_seconds=elapsed, training_status=(state or {}).get('status'))
        time.sleep(min(poll_seconds, timeout-elapsed))


def inference_command(train_dir):
    return [sys.executable, '-u', str(HERE/'infer_test300_v13.py'),
            '--checkpoint', str(train_dir/'final.pt'),
            '--training-protocol', str(train_dir/'protocol_before_training.json'),
            '--output', str(train_dir/'test300'), '--limit', '300']


def main(a):
    if min(a.training_timeout_seconds, a.inference_timeout_seconds, a.poll_seconds) <= 0:
        raise ValueError('Timeouts and polling interval must be positive')
    a.output.mkdir(parents=True, exist_ok=False)
    def progress(stage, **fields):
        write(a.output/'progress.json', dict(stage=stage, time=time.time(),
            architecture='RefinePlaceV13', training_kind='paired_action_value_supervision',
            new_reinforcement_learning=False, official_evaluation_started=False, **fields))
    child = None
    try:
        progress('checking_training_receipt')
        receipt = check_receipt(a.training_receipt, a.train_dir)
        hashes = {name: sha(HERE/name) for name in SOURCE_FILES}
        protocol = dict(version='orientation_v13_continuation_v1', training_tasks=TASKS,
            training_epochs=EPOCHS, inference_tasks=300, original_entry_ids=list(range(300)),
            baseline_frozen=True, new_reinforcement_learning=False,
            training_kind='paired_action_value_supervision', official_evaluation_started=False,
            parent_checkpoint_sha256=PARENT_SHA256, training_pid=receipt['pid'],
            training_receipt_sha256=sha(a.training_receipt), source_hashes=hashes,
            train_directory=str(a.train_dir), inference_directory=str(a.train_dir/'test300'))
        write(a.output/'protocol.json', protocol)
        accepted = wait_training(a.train_dir, receipt['pid'], hashes,
            timeout=a.training_timeout_seconds, poll_seconds=min(a.poll_seconds, 30), progress=progress)
        if Path(accepted['protocol']['parent_checkpoint']).resolve() != Path(option(receipt['command'], '--parent')).resolve():
            raise ValueError('Training parent differs from launch receipt')
        # Inference owns creation of test300. Existing output is never overwritten.
        if (a.train_dir/'test300').exists():
            raise FileExistsError('test300 already exists; refuse to overwrite or relabel it')
        command = inference_command(a.train_dir)
        progress('inference_test300', command=command,
                 final_checkpoint_sha256=accepted['summary']['final_checkpoint_sha256'])
        env = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        with (a.output/'inference_test300.log').open('xb') as log:
            child = subprocess.Popen(command, cwd=HERE, env=env, stdout=log, stderr=subprocess.STDOUT)
            started = time.monotonic()
            while child.poll() is None:
                elapsed = time.monotonic()-started
                progress('inference_test300', pid=child.pid, elapsed_seconds=elapsed, command=command)
                if elapsed >= a.inference_timeout_seconds:
                    raise TimeoutError('Test300 inference deadline exceeded')
                try:
                    child.wait(timeout=min(a.poll_seconds, 30, a.inference_timeout_seconds-elapsed))
                except subprocess.TimeoutExpired:
                    pass
            if child.returncode != 0:
                raise RuntimeError('Test300 inference failed with exit code '+str(child.returncode))
        validate_training(a.train_dir, hashes)
        if sha(a.train_dir/'summary.json') != accepted['summary_sha256']:
            raise ValueError('Training summary changed during inference')
        output = a.train_dir/'test300'
        result, _ = validate_inference(output/'summary.json', output/'predictions.json',
                                       output/'protocol_before_forward.json')
        if result['checkpoint_sha256'] != accepted['summary']['final_checkpoint_sha256']:
            raise ValueError('Inference used a different checkpoint')
        if result['training_protocol_sha256'] != accepted['protocol_sha256']:
            raise ValueError('Inference used a different training protocol')
        progress('predictions_ready', status='complete', tasks=300,
            official_test300_complete=True, predictions=str(output/'predictions.json'),
            prediction_sha256=result['prediction_sha256'], checkpoint_sha256=result['checkpoint_sha256'],
            baseline_weights_unchanged=True, parent_checkpoint_unchanged=True,
            center_unchanged=True, baseline_first_two_steps_unchanged=True,
            frozen_v11_prediction_identical=True)
    except BaseException as exc:
        if child is not None and child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=15)
        progress('failed', status='failed', error=type(exc).__name__+': '+str(exc),
                 official_test300_complete=False)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-dir', type=Path, default=HERE/'refine1000_v1')
    parser.add_argument('--training-receipt', type=Path, default=HERE/'refine1000_train_receipt.json')
    parser.add_argument('--output', type=Path, default=HERE/'pipeline_v1')
    parser.add_argument('--training-timeout-seconds', type=float, default=12*3600)
    parser.add_argument('--inference-timeout-seconds', type=float, default=12*3600)
    parser.add_argument('--poll-seconds', type=float, default=10)
    main(parser.parse_args())
