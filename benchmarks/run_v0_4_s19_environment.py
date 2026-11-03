#!/usr/bin/env python3
"""Isolated S-19 wheel/RECORD inspection and node preflight, CPU-rehearsable."""
from __future__ import annotations

import argparse
import base64
import csv
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import sysconfig

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks._artifact_io import atomic_write_json, file_sha256 as digest_file, runtime_environment
from benchmarks._s19_compare import HardQualificationError
from benchmarks.plan_v0_4_s19_h100 import build_plan
from benchmarks.run_v0_4_s19_self_compare import _installed_identity


def inspect_role(role, provider_wheel, consumer_wheel):
    identity = _installed_identity('baseline_a' if role == 'baseline' else 'candidate', provider_wheel)
    purelib = Path(sysconfig.get_path('purelib')).resolve()
    names = ['pssolver']
    if role == 'consumer':
        import pssolver_control
        if (not Path(pssolver_control.__file__).resolve().is_relative_to(purelib)
                or metadata.version('pssolver-control') != build_plan()['consumer']['version']
                or digest_file(consumer_wheel) != build_plan()['consumer']['wheel_sha256']):
            raise HardQualificationError('Consumer wheel/import/version')
        names.append('pssolver-control')
    records = {}
    for name in names:
        dist = metadata.distribution(name)
        record = Path(dist._path) / 'RECORD'
        if not record.resolve().is_relative_to(purelib):
            raise HardQualificationError('RECORD outside own venv')
        rows = list(csv.reader(record.open()))
        checked = 0
        for relative, digest, size in rows:
            if not digest:
                continue
            file = Path(dist.locate_file(relative))
            algorithm, value = digest.split('=', 1)
            if algorithm != 'sha256':
                raise HardQualificationError('unexpected RECORD digest algorithm')
            actual = base64.urlsafe_b64encode(bytes.fromhex(digest_file(file))).decode().rstrip('=')
            if actual != value or file.stat().st_size != int(size):
                raise HardQualificationError(f'RECORD mismatch: {name}/{relative}')
            checked += 1
        records[name] = dict(path=str(record), sha256=digest_file(record), entries=len(rows), checked=checked)
    return dict(role=role, identity=identity, records=records, environment=runtime_environment(),
                versions={d.metadata['Name']: d.version for d in metadata.distributions()},
                python=runtime_environment()['python_version'], prefix=sys.prefix, purelib=str(purelib))


def run(args):
    plan = build_plan()
    reports = {}
    for role in ('baseline', 'candidate', 'consumer'):
        python = getattr(args, role + '_python')
        wheel = args.baseline_wheel if role == 'baseline' else args.candidate_wheel
        command = [str(python), '-I', '-B', str(Path(__file__).resolve()), '--inspect-role', role,
                   '--provider-wheel', str(wheel), '--consumer-wheel', str(args.consumer_wheel)]
        env = dict(os.environ, PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_CACHE_DIR='1')
        env.pop('PYTHONPATH', None)
        env.pop('PYTHONHOME', None)
        proc = subprocess.run(command, capture_output=True, text=True, env=env)
        atomic_write_json(args.output.parent / f'{role}_inspection_process.json',
                          dict(command=command, returncode=proc.returncode, stdout=proc.stdout, stderr=proc.stderr), overwrite=False)
        if proc.returncode:
            raise HardQualificationError(f'{role} isolated inspection failed')
        report = json.loads(proc.stdout)
        # Exact C/D-0021 package versions; no +cpu substitution, no matplotlib import.
        expected = json.loads(args.mirror_versions.read_text())
        if 'versions' in expected:
            expected = expected['versions']
        normalized = {'-'.join(k.lower().split('_')): v for k, v in report['versions'].items()}
        for name, version in expected.items():
            if normalized.get('-'.join(name.lower().split('_'))) != version:
                raise HardQualificationError(f'C package version differs: {role}/{name}')
        if report['python'] != '3.10.18':
            raise HardQualificationError('C Python version differs')
        check = subprocess.run([str(python), '-I', '-B', '-m', 'pip', '--disable-pip-version-check',
                                '--no-cache-dir', 'check'], capture_output=True, text=True, env=env)
        report['pip_check'] = dict(returncode=check.returncode, stdout=check.stdout, stderr=check.stderr)
        atomic_write_json(args.output.parent / f'{role}_environment.json', report, overwrite=False)
        if check.returncode:
            raise HardQualificationError('pip check failed')
        reports[role] = report
    import torch
    from benchmarks.run_rc5_g1_numerical import _tf32_off
    _tf32_off()
    compiler, tmp = {}, {}
    if args.device == 'cuda':
        if (not torch.cuda.is_available() or torch.cuda.device_count() != 1
                or 'H100' not in torch.cuda.get_device_name(0)
                or str(torch.empty((), device='cuda').device) != 'cuda:0'):
            raise HardQualificationError('one allocated cuda:0 H100 required')
        from benchmarks.run_v0_3_s09_environment import _host_compiler_and_compile_smokes
        compiler = _host_compiler_and_compile_smokes(args.work / 'compiler')
        tmp = json.loads(args.tmp_preflight.read_text())
        if tmp.get('passed') is not True or tmp.get('tmpdir_matches') is not True:
            raise HardQualificationError('node-local tmp preflight failed')
    elif os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise HardQualificationError('CPU rehearsal must hide CUDA')
    return dict(name='node_local_tmp_environment_and_host_compiler', classification='PASS',
                provider_candidate_commit=plan['provider']['candidate']['commit'],
                candidate_wheel_sha256=plan['provider']['candidate']['wheel_sha256'],
                baseline_wheel_sha256=plan['provider']['baseline']['wheel_sha256'],
                consumer_wheel_sha256=plan['consumer']['wheel_sha256'], execution_device=args.device,
                finite=True, fallback_used=False, installed_wheels_only=True,
                source_shadow_import_false=True, pip_check=True, node_local_tmp=bool(tmp),
                host_compiler=bool(compiler), empty_compile_caches=bool(compiler),
                cuda_available=args.device == 'cuda', single_visible_h100=args.device == 'cuda',
                allocated_cuda_0=args.device == 'cuda', observations=reports, compiler=compiler,
                tmp_preflight=tmp, formal_h100_contract=args.device == 'cuda',
                cpu_rehearsal_exclusions=['CUDA/compiler/node-local preflight'] if args.device == 'cpu' else [])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspect-role', choices=('baseline', 'candidate', 'consumer'))
    parser.add_argument('--device', choices=('cpu', 'cuda'))
    for name in ('provider-wheel', 'baseline-wheel', 'candidate-wheel', 'consumer-wheel', 'baseline-python',
                 'candidate-python', 'consumer-python', 'output', 'work', 'mirror-versions', 'tmp-preflight'):
        parser.add_argument('--' + name, type=Path)
    args = parser.parse_args()
    if args.inspect_role:
        print(json.dumps(inspect_role(args.inspect_role, args.provider_wheel, args.consumer_wheel), allow_nan=False))
    else:
        atomic_write_json(args.output, run(args), overwrite=False)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
