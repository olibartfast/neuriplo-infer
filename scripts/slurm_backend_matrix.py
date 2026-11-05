#!/usr/bin/env python3
"""Submit independent backend commands to Slurm and preserve per-case evidence.

Stage the manifest, executables and fixtures at identical absolute paths on each
selected worker first. Run submit on the controller using its matching client.
Commands are argument arrays, never shell fragments. See --help.
"""
import argparse
import datetime
import hashlib
import json
import os
import resource
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def execute(manifest, output, index):
    row = json.loads(manifest.read_text())[index]
    work = output / f'{os.environ.get("SLURM_JOB_ID", "local")}-{index}'
    work.mkdir(parents=True, exist_ok=False)
    record = dict(row, job_id=os.environ.get('SLURM_JOB_ID'),
                  node=os.uname().nodename, start=timestamp(), status='FAILED')
    started = time.monotonic()
    try:
        record['sha256'] = {}
        for filename in row['artifacts']:
            with open(filename, 'rb') as stream:
                record['sha256'][filename] = hashlib.file_digest(stream, 'sha256').hexdigest()
        environment = os.environ | row.get('env', {})
        memory_bytes = row.get('memory_mib', 8192) * 1024 * 1024
        if memory_bytes <= 0:
            raise ValueError('memory_mib must be positive')
        record['memory_mib'] = memory_bytes // (1024 * 1024)

        def limit_memory():
            resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))

        with (work / 'runtime.log').open('w') as log:
            process = subprocess.Popen(row['command'], cwd=work, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, preexec_fn=limit_memory)
            try:
                code = process.wait(timeout=row.get('timeout_seconds', 300))
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                code = 124
        record.update(exit_code=code, status='RUNTIME_PASS' if code == 0 else 'FAILED')
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        record.update(error=str(error), exit_code=125)
    finally:
        record.update(end=timestamp(), duration_seconds=time.monotonic() - started)
        (work / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    # Runtime success is not a numerical coverage PASS.
    return 0 if record['status'] == 'RUNTIME_PASS' else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True,
                        help='JSON list: name, node, command[], artifacts[], optional env{}, timeout_seconds')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partition', default='cpu')
    parser.add_argument('--cpus', type=int, default=2)
    parser.add_argument('--minutes', type=int, default=20)
    parser.add_argument('--case-index', type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    manifest = args.manifest.resolve()
    output = args.output.resolve()
    rows = json.loads(manifest.read_text())
    if not rows or args.cpus < 1 or args.minutes < 1:
        parser.error('nonempty manifest and positive resource limits required')
    for row in rows:
        if (not isinstance(row['command'], list) or not row['command']
                or not all(isinstance(argument, str) for argument in row['command'])):
            parser.error('each command must be a nonempty argument array')
        if not isinstance(row['artifacts'], list) or not row['artifacts']:
            parser.error('each case must declare executable/model artifacts for hashing')
    if args.case_index is not None:
        if not 0 <= args.case_index < len(rows):
            parser.error('case index outside manifest')
        return execute(manifest, output, args.case_index)
    output.mkdir(parents=True, exist_ok=True)
    submissions = []
    for index, row in enumerate(rows):
        command = [sys.executable, str(Path(__file__).resolve()), '--manifest', str(manifest),
                   '--output', str(output), '--case-index', str(index)]
        result = subprocess.run(['sbatch', '--parsable', '--partition', args.partition,
                                 '--nodelist', row['node'], '--nodes=1', '--ntasks=1',
                                 f'--cpus-per-task={args.cpus}', f'--time={args.minutes}',
                                 '--job-name', row['name'], '--output', str(output / '%x-%j.log'),
                                 '--wrap', shlex.join(command)],
                                check=True, capture_output=True, text=True)
        submissions.append(dict(index=index, name=row['name'], job_id=result.stdout.strip()))
        (output / 'submitted.json').write_text(json.dumps(submissions, indent=2) + '\n')
        print(result.stdout.strip(), row['name'], flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
