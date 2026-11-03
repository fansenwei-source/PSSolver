#!/usr/bin/env python3
"""Read-only input validation; isolated venv inspection writes only its report.

The manifest hash is frozen by the preparation review, supplied explicitly by
the confirmed execution command, not trusted merely because a file exists.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks._artifact_io import atomic_write_json, file_sha256, git_identity
from benchmarks._s19_compare import HardQualificationError
from benchmarks.plan_v0_4_s19_h100 import build_plan
from benchmarks.run_v0_4_s19_environment import validate_mirror


def validate_tar_layout(path, prefix):
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            parts = Path(member.name).parts
            if (not parts or parts[0] != prefix or '..' in parts
                    or not (member.isfile() or member.isdir())):
                raise HardQualificationError(f'unsafe archive layout: {path.name}/{member.name}')


def validate_inputs(root, manifest_path, expected_sha256):
    if file_sha256(manifest_path) != expected_sha256:
        raise HardQualificationError('deployment manifest hash')
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest['inputs'].items():
        name = Path(relative)
        if name.is_absolute() or '..' in name.parts:
            raise HardQualificationError('input path escapes control directory')
        path = root / name
        if (path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve())
                or path.stat().st_size != expected['size'] or file_sha256(path) != expected['sha256']):
            raise HardQualificationError(f'uploaded input mismatch: {relative}')
        if expected.get('tar_prefix'):
            validate_tar_layout(path, expected['tar_prefix'])
    return manifest


def ready(root, manifest, output):
    plan = build_plan()
    if str(root) != plan['deployment']['control_root']:
        raise HardQualificationError('control directory differs from frozen plan')
    for key, expected in manifest['environment'].items():
        if os.environ.get(key) != expected:
            raise HardQualificationError(f'job environment differs: {key}')
    result_root = Path(os.environ['S19_OUTPUT_ROOT'])
    if result_root.exists() or result_root.is_symlink() or not (root / 'logs').is_dir():
        raise HardQualificationError('output must be absent and control logs present')
    support = Path(os.environ['S19_PROVIDER_SUPPORT'])
    if any(p.name in ('pssolver', 'pssolver_control') for p in support.rglob('*') if p.is_dir()):
        raise HardQualificationError('support includes runtime package directory')
    consumer = Path(os.environ['S19_CONSUMER_REFERENCE'])
    identity = git_identity(consumer)
    if identity['head'] != plan['consumer']['commit'] or not identity['clean']:
        raise HardQualificationError('Consumer provenance reference identity')
    source = Path(os.environ['S19_PROVIDER_REFERENCE'])
    for relative, expected in manifest['provider_source_files'].items():
        if file_sha256(source / relative) != expected:
            raise HardQualificationError(f'Provider source reference differs: {relative}')
    reports = {}
    for role in ('baseline', 'candidate', 'consumer'):
        python = os.environ[f'S19_{role.upper()}_PYTHON']
        provider_role = 'BASELINE' if role == 'baseline' else 'CANDIDATE'
        command = [python, '-I', '-B', str(support / 'benchmarks/run_v0_4_s19_environment.py'),
                   '--inspect-role', role, '--provider-wheel', os.environ[f'S19_{provider_role}_WHEEL'],
                   '--consumer-wheel', os.environ['S19_CONSUMER_WHEEL']]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES='', PIP_DISABLE_PIP_VERSION_CHECK='1', PIP_NO_CACHE_DIR='1')
        run = subprocess.run(command, env=env, capture_output=True, text=True)
        check = subprocess.run([python, '-I', '-B', '-m', 'pip', '--disable-pip-version-check', '--no-cache-dir', 'check'],
                               env=env, capture_output=True, text=True)
        reports[role] = dict(command=command, returncode=run.returncode, stdout=run.stdout,
                             stderr=run.stderr, pip_check=dict(returncode=check.returncode, stdout=check.stdout, stderr=check.stderr))
        if run.returncode or check.returncode:
            atomic_write_json(output, dict(passed=False, reports=reports), overwrite=False)
            raise HardQualificationError('submission preflight venv/RECORD/pip failed')
        try:
            validate_mirror(json.loads(run.stdout), json.loads(Path(os.environ['S19_MIRROR_VERSIONS']).read_text()))
        except Exception:
            atomic_write_json(output, dict(passed=False, reports=reports), overwrite=False)
            raise
    atomic_write_json(output, dict(passed=True, consumer_reference=identity, reports=reports), overwrite=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--ready-output', type=Path)
    args = parser.parse_args()
    manifest = validate_inputs(args.root, args.manifest, args.sha256)
    if args.ready_output:
        ready(args.root, manifest, args.ready_output)
    print('PASS_INPUTS' if not args.ready_output else 'PASS_SUBMISSION_PREFLIGHT')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
