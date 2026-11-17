"""Run one isolated V1.3 official evaluation of complete original test IDs 0..299.

Uses the existing official image/native-command template without replacing
/app/data or modifying native scoring. These 300 TEST entries were previously
viewed: this is an exposed diagnostic comparison, not an untouched holdout.
Only a completely audited run is published as a final result. Failed/partial
native artifacts are retained solely for diagnosis.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import subprocess
import time
import traceback


ROOT = Path('/path/to/home/refineplace_official_epoch2_test')
TEMPLATE = ROOT / 'multiprocess_pilot_v1/workers_1/command.json'
SHIM = ROOT / 'multiprocess_pilot_v1/workers_3/cpu_cap.so'
IMAGE = 'sha256:a209a5e38d5d82cd2a17fcbd8a05711db5d2213760d41b071c4d0769372664c6'
MEMORY_BYTES = 45 * 1024**3
MEMORY_SWAP_BYTES = 51 * 1024**3


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def reject_constant(value):
    raise ValueError('Non-finite JSON constant: ' + value)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'), parse_constant=reject_constant)


def write(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + '.pending')
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def validate_predictions(path):
    rows = read(path)
    if not isinstance(rows, list) or len(rows) != 300:
        raise ValueError('Expected exactly 300 original test predictions')
    if any(type(r.get('entry_id')) is not int for r in rows) or [r['entry_id'] for r in rows] != list(range(300)):
        raise ValueError('Predictions must be ordered by original entry_id 0..299')
    for row in rows:
        for key in ('tx', 'ty', 'tz', 'rotation'):
            value = row.get(key)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError('Invalid pose value for entry %s: %s' % (row['entry_id'], key))
        if row.get('dataset_idx', -1) != -1:
            raise ValueError('Expected original entry_id lookup with dataset_idx=-1')
    return rows



def validate_inference(summary_path, prediction_path, protocol_path):
    """Fail closed on a smoke run, modified export, or mismatched source seal."""
    summary, protocol = read(summary_path), read(protocol_path)
    required = dict(status='complete', tasks=300, architecture='RefinePlaceV13',
                    official_test300_complete=True, inference_complete=True,
                    weights_unchanged=True, run_kind='complete_test300',
                    prediction_filename='predictions.json', center_unchanged=True,
                    baseline_first_two_steps_unchanged=True, frozen_v11_prediction_identical=True)
    for key, value in required.items():
        if summary.get(key) != value or type(summary.get(key)) is not type(value):
            raise ValueError('Inference summary rejected: '+key)
    expected = list(range(300))
    for label, record in [('summary', summary), ('protocol', protocol)]:
        ids = record.get('original_entry_ids')
        if not isinstance(ids, list) or any(type(i) is not int for i in ids) or ids != expected:
            raise ValueError('Wrong original entry IDs in '+label)
        if record.get('version') != 'orientation_v13_test300_frozen_forward':
            raise ValueError('Wrong V1.3 complete inference version in '+label)
        if record.get('split') != 'test' or record.get('tasks') != 300:
            raise ValueError('Wrong inference split/count in '+label)
        for key in ('test_labels_read', 'ground_truth_relationships_read', 'instance_annotation_inputs', 'official_evaluator_modified'):
            if record.get(key) is not False:
                raise ValueError('Inference provenance rejected: '+label+'.'+key)
    if summary.get('protocol_sha256') != sha(protocol_path):
        raise ValueError('Inference protocol hash mismatch')
    for key, value in protocol.items():
        if summary.get(key) != value:
            raise ValueError('Summary and sealed protocol differ: '+key)
    if summary.get('prediction_sha256') != sha(prediction_path):
        raise ValueError('Inference prediction hash mismatch')
    if (summary.get('frozen_v11_prediction_sha256') != 'c511211848fd0d89ca6185b4789603679c7659643600e8dba1c97c535aa73c74'
            or protocol.get('v11_prediction_sha256') != summary['frozen_v11_prediction_sha256']
            or protocol.get('v11_checkpoint_sha256') != '480bb902bbbeb174a36da395fdf531c1845386cc66d653478c2584081e41c774'):
        raise ValueError('Frozen V11 baseline identity is not the original upgraded run')
    for key in ('checkpoint_sha256', 'manifest_sha256', 'training_protocol_sha256'):
        if not isinstance(summary.get(key), str) or not re.fullmatch('[0-9a-f]{64}', summary[key]):
            raise ValueError('Missing inference identity: '+key)
    validate_predictions(prediction_path)
    return summary, protocol


def option_index(command, name):
    indices = [i for i, value in enumerate(command) if value == name]
    if len(indices) != 1 or indices[0] + 1 >= len(command):
        raise ValueError('Expected one command option: ' + name)
    return indices[0] + 1


def mount_fields(value):
    fields = {}
    for part in value.split(','):
        key, separator, text = part.partition('=')
        if key in fields:
            raise ValueError('Duplicate mount field: ' + key)
        fields[key] = text if separator else True
    return fields


def construct_command(template, prediction, output, name):
    if not isinstance(template, list) or any(not isinstance(x, str) for x in template):
        raise ValueError('Command template must be a JSON list of strings')
    command = list(template)
    if command[:3] != ['docker', 'run', '-d'] or '--rm' in command:
        raise ValueError('Expected detached retained Docker container template')
    if any(x in command for x in ('-v', '--volume', '--volumes-from')):
        raise ValueError('Only explicitly inspected --mount binds are supported')
    replacements = {'--name': name, '--memory': str(MEMORY_BYTES),
                    '--memory-swap': str(MEMORY_SWAP_BYTES), '--cpus': '8',
                    '--num_entries': '3500'}
    # Keep the original complete HDF5 lookup range. The input JSON selects the
    # 300 evaluated placements, as in the previous audited 300-entry batch.
    for key, value in replacements.items():
        command[option_index(command, key)] = value
    entrypoint = command[option_index(command, '--entrypoint')]
    image_index = option_index(command, '--entrypoint') + 1
    if command[image_index] != IMAGE or not entrypoint.endswith('/run_benchmark.bin'):
        raise ValueError('Unexpected official image or native entrypoint')
    targets = {'/pilot/predictions.json': (prediction, True), '/pilot/out': (output, False),
               '/opt/cpu_count_cap.so': (SHIM, True)}
    found = set()
    native = None
    immutable_files = {}
    for i, value in enumerate(command):
        if value != '--mount':
            continue
        if i + 1 >= len(command):
            raise ValueError('Missing --mount specification')
        fields = mount_fields(command[i + 1])
        target = fields.get('target', fields.get('dst', fields.get('destination')))
        source = fields.get('source', fields.get('src'))
        if fields.get('type') != 'bind' or not target or not source:
            raise ValueError('Unsupported mount: ' + command[i + 1])
        target_path = PurePosixPath(target)
        protected = PurePosixPath('/app/data')
        if target_path == protected or target_path in protected.parents or protected in target_path.parents:
            raise ValueError('Official /app/data must not be covered by a bind mount')
        if target in targets:
            if target in found:
                raise ValueError('Duplicate output/input/shim mount')
            found.add(target)
            source, readonly = targets[target]
            command[i + 1] = 'type=bind,source=' + str(source) + ',target=' + target + (',readonly' if readonly else '')
        else:
            if not Path(source).exists():
                raise FileNotFoundError(source)
            if not fields.get('readonly', False):
                raise ValueError('Existing template dependency must remain read-only: ' + target)
            if PurePosixPath(entrypoint).is_relative_to(target_path):
                relative = PurePosixPath(entrypoint).relative_to(target_path)
                native = Path(source).joinpath(*relative.parts)
            if Path(source).is_file():
                immutable_files[str(source)] = sha(source)
    if found != set(targets) or native is None or not native.is_file():
        raise ValueError('Missing required mount or native program')
    immutable_files[str(native)] = sha(native)
    immutable_files[str(SHIM)] = sha(SHIM)
    return command, native, immutable_files


def docker(*args, check=True):
    result = subprocess.run(['docker', *args], capture_output=True, text=True)
    if check and result.returncode:
        raise RuntimeError('docker %s failed: %s' % (args[0], result.stderr.strip()))
    return result


def inspect_container(cid):
    records = json.loads(docker('inspect', cid).stdout)
    if len(records) != 1 or records[0]['Id'] != cid:
        raise RuntimeError('Container identity changed')
    return records[0]


def stop_own_container(cid):
    """Never stop/remove any container other than the one started by this run."""
    if cid and inspect_container(cid)['State']['Running']:
        docker('stop', '--time', '5', cid)


def audit_result(folder, info, cid, elapsed, observations, immutable, prediction_sha):
    logs = subprocess.run(['docker', 'logs', cid], capture_output=True, check=True)
    raw = logs.stdout + b'\n' + logs.stderr
    (folder / 'container.log').write_bytes(raw)
    write(folder / 'inspect.json', info)
    text = raw.decode(errors='replace')
    progress = re.findall(r'Processing placements:\s*100%[^\r\n]*?\b(\d+)/(\d+)\b', text)
    workers = re.findall(r'Using (\d+) processes for parallel computation', text)
    errors = re.findall(r'Error evaluating constraint\s+([^:\s]+)', text, flags=re.IGNORECASE)
    result_path = folder / 'out/test_results.json'
    official, accounted, parse_error = None, None, None
    try:
        if result_path.is_file():
            official = read(result_path)
            groups = official['metrics_by_constraint_count']
            counts = [g['total_placements'] for g in groups.values()]
            if not counts or any(type(n) is not int or n < 0 for n in counts):
                raise ValueError('Invalid official placement denominators')
            accounted = sum(counts)
    except (ValueError, TypeError, KeyError) as exc:
        parse_error = type(exc).__name__ + ': ' + str(exc)
    checks = dict(exit_zero=info['State']['ExitCode'] == 0,
                  no_oom=not (info['State']['OOMKilled'] or observations['oom'] or observations['oom_kill']),
                  memory_monitor_observed=observations['samples'] > 0,
                  image_unchanged=info['Image'] == IMAGE,
                  resource_limits_correct=(info['HostConfig']['Memory'] == MEMORY_BYTES
                                           and info['HostConfig']['MemorySwap'] == MEMORY_SWAP_BYTES
                                           and info['HostConfig']['NanoCpus'] == 8_000_000_000),
                  accounted_300=accounted == 300, workers_3=bool(workers) and set(workers) == {'3'},
                  progress_300=bool(progress) and progress[-1] == ('300', '300'),
                  no_constraint_errors=not errors, result_parsed=official is not None and parse_error is None,
                  prediction_unchanged=sha(folder / 'predictions.json') == prediction_sha,
                  official_runtime_unchanged=all(Path(p).is_file() and sha(p) == value for p, value in immutable.items()))
    receipt = dict(status='passed' if all(checks.values()) else 'audit_failed', tasks=300, container=cid,
                   first_entry_id=0, last_entry_id=299, prediction_sha256=prediction_sha,
                   elapsed_seconds=elapsed, exit_code=info['State']['ExitCode'],
                   oom_killed=bool(info['State']['OOMKilled']), memory_observations=observations,
                   workers_observed=workers, final_progress=progress[-1] if progress else None,
                   constraint_errors=errors, accounted_placements=accounted, result_parse_error=parse_error,
                   result_sha256=sha(result_path) if result_path.is_file() else None, checks=checks,
                   partial_results_are_not_final=True)
    write(folder / 'receipt.json', receipt)
    return receipt, official


def main(args):
    prediction = args.predictions.resolve(strict=True)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError('Refusing to overwrite existing evaluation output: ' + str(output))
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]{0,100}', args.name):
        raise ValueError('Invalid unique Docker container name')
    if ',' in str(output):
        raise ValueError('Output path cannot contain a Docker mount separator')
    inference_summary, inference_protocol = validate_inference(args.inference_summary, prediction, args.inference_protocol)
    source_hash = sha(prediction)
    template = read(TEMPLATE)
    command, native, immutable = construct_command(template, output / 'predictions.json', output / 'out', args.name)
    image = json.loads(docker('image', 'inspect', IMAGE).stdout)
    if len(image) != 1 or image[0]['Id'] != IMAGE:
        raise RuntimeError('Original official image identity mismatch')
    if docker('inspect', args.name, check=False).returncode == 0:
        raise RuntimeError('Container name already exists; use a fresh name')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'out').mkdir()
    (output / 'predictions.json').write_bytes(prediction.read_bytes())
    if sha(output / 'predictions.json') != source_hash:
        raise RuntimeError('Input prediction bytes changed during preparation')
    write(output / 'inference_summary.json', inference_summary)
    write(output / 'inference_protocol.json', inference_protocol)
    protocol = dict(version='orientation_v13_official_test300_v1', tasks=300, original_entry_ids=list(range(300)),
                    split='test', test_exposure='Previously viewed first 300 test entries; diagnostic comparison, not untouched holdout',
                    predictions_source=str(prediction), prediction_sha256=source_hash,
                    model_architecture='RefinePlaceV13', checkpoint_sha256=inference_summary['checkpoint_sha256'],
                    inference_summary_sha256=sha(args.inference_summary), inference_protocol_sha256=sha(args.inference_protocol),
                    template=str(TEMPLATE), template_sha256=sha(TEMPLATE), source_sha256=sha(__file__),
                    image=IMAGE, native_program=str(native), native_program_sha256=sha(native),
                    cpu_shim=str(SHIM), cpu_shim_sha256=sha(SHIM), workers=3, cpus=8,
                    memory_bytes=MEMORY_BYTES, memory_plus_swap_bytes=MEMORY_SWAP_BYTES,
                    official_evaluator_modified=False, official_data_overridden=False,
                    native_num_entries=3500, native_num_entries_reason='Preserve full official HDF5 identity range; input JSON contains only selected 300 placements',
                    failure_policy='Stop only this container on OOM; preserve evidence and publish no final metric for partial/failed runs',
                    immutable_dependency_sha256=immutable)
    write(output / 'protocol.json', protocol)
    write(output / 'command.json', command)
    cid = None
    started = time.monotonic()
    observations = dict(samples=0, oom=0, oom_kill=0, peak_observed_bytes=0, memory_read_failures=0)

    def status(stage, **values):
        write(output / 'status.json', dict(stage=stage, tasks=300, time=time.time(),
                                           elapsed_seconds=time.monotonic() - started, container=cid, **values))

    try:
        status('starting')
        cid = subprocess.check_output(command, text=True).strip()
        if not re.fullmatch(r'[0-9a-f]{64}', cid):
            raise RuntimeError('Docker did not return an unambiguous container ID')
        write(output / 'launch_receipt.json', dict(container=cid, name=args.name, prediction_sha256=source_hash,
                                                   command_sha256=sha(output / 'command.json'), started_at=time.time()))
        while True:
            info = inspect_container(cid)
            if not info['State']['Running']:
                break
            events = docker('exec', cid, 'cat', '/sys/fs/cgroup/memory.events', check=False)
            if events.returncode == 0:
                counters = dict(line.split() for line in events.stdout.splitlines() if len(line.split()) == 2)
                observations['samples'] += 1
                observations['oom'] = max(observations['oom'], int(counters.get('oom', 0)))
                observations['oom_kill'] = max(observations['oom_kill'], int(counters.get('oom_kill', 0)))
            else:
                observations['memory_read_failures'] += 1
            memory = docker('exec', cid, 'cat', '/sys/fs/cgroup/memory.current', check=False)
            if memory.returncode == 0 and memory.stdout.strip().isdigit():
                observations['peak_observed_bytes'] = max(observations['peak_observed_bytes'], int(memory.stdout.strip()))
            status('running', memory_observations=observations)
            if observations['oom'] or observations['oom_kill']:
                status('stopping_own_container_after_oom', memory_observations=observations)
                stop_own_container(cid)
                break
            if observations['samples'] == 0 and observations['memory_read_failures'] >= 3:
                raise RuntimeError('Cannot verify cgroup OOM monitoring')
            time.sleep(5)
        info = inspect_container(cid)
        receipt, official = audit_result(output, info, cid, time.monotonic() - started,
                                         observations, immutable, source_hash)
        if receipt['status'] != 'passed':
            write(output / 'summary.json', dict(status='audit_failed', tasks=300, final_metrics_available=False,
                                                receipt_sha256=sha(output / 'receipt.json'),
                                                failed_checks=[key for key, value in receipt['checks'].items() if not value],
                                                test_exposure=protocol['test_exposure']))
            status('audit_failed', final_metrics_available=False)
            return 2
        summary = dict(status='complete', tasks=300, final_metrics_available=True, audited=True,
                       image=IMAGE, prediction_sha256=source_hash, protocol_sha256=sha(output / 'protocol.json'),
                       receipt_sha256=sha(output / 'receipt.json'), test_exposure=protocol['test_exposure'],
                       official_metrics=official, official_evaluator_modified=False, workers=3,
                       model_architecture='RefinePlaceV13', checkpoint_sha256=inference_summary['checkpoint_sha256'])
        write(output / 'summary.json', summary)
        status('complete', final_metrics_available=True, result=str(output / 'out/test_results.json'))
        print(json.dumps(dict(status='complete', tasks=300, summary=str(output / 'summary.json')), ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:
        cleanup_error = None
        if cid and re.fullmatch(r'[0-9a-f]{64}', cid):
            try:
                stop_own_container(cid)
                audit_result(output, inspect_container(cid), cid, time.monotonic() - started,
                             observations, immutable, source_hash)
            except Exception as cleanup:
                cleanup_error = type(cleanup).__name__ + ': ' + str(cleanup)
        (output / 'failure.log').write_text(traceback.format_exc(), encoding='utf-8')
        write(output / 'summary.json', dict(status='failed', tasks=300, final_metrics_available=False,
                                            error=type(exc).__name__ + ': ' + str(exc), cleanup_error=cleanup_error,
                                            test_exposure=protocol['test_exposure']))
        status('failed', final_metrics_available=False, error=type(exc).__name__ + ': ' + str(exc))
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--inference-summary', type=Path, required=True)
    parser.add_argument('--inference-protocol', type=Path, required=True)
    parser.add_argument('--name', required=True, help='Fresh unique Docker name; no existing container is stopped')
    raise SystemExit(main(parser.parse_args()))
