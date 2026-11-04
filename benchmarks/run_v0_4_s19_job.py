#!/usr/bin/env python3
"""One sequential S-19 execution, also used end-to-end by CPU rehearsal.

This is not a scheduler submission command. Formal invocation requires the
separately confirmed job. Every child log is retained, H stops, N is latched.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import statistics
import traceback
import types

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks._artifact_io import atomic_write_json, atomic_write_text, file_sha256
from benchmarks._s19_compare import HardQualificationError, final_verdict, KNOWN_NONFINITE_PCG_RECORDS
from benchmarks.plan_v0_4_s19_h100 import build_plan


def launch(command, output, label, *, env=None):
    result = subprocess.run([str(x) for x in command], capture_output=True, text=True, env=env)
    atomic_write_json(output / 'launches' / f'{label}.json',
                      dict(command=[str(x) for x in command], returncode=result.returncode), overwrite=False)
    atomic_write_text(output / 'logs' / f'{label}.stdout', result.stdout, overwrite=False)
    atomic_write_text(output / 'logs' / f'{label}.stderr', result.stderr, overwrite=False)
    if result.returncode:
        raise HardQualificationError(f'child {label} failed ({result.returncode}); see retained log')


def validate_profile_h(profile, entry, plan):
    # These are the frozen analyzer's per-profile validity checks, performed
    # immediately after each child. Ratio gates remain in the final analyzer.
    from benchmarks import analyze_v0_3_s09_h100 as frozen

    def require(condition, message):
        if not condition:
            raise HardQualificationError('profile ' + message)

    require(profile.get('schema') == frozen.PROFILE_SCHEMA, 'schema')
    for name in ('config', 'provider_identity', 'runtime_identity', 'environment',
                 'timing', 'memory', 'transform_calls', 'compiler'):
        require(isinstance(profile.get(name), dict), name + ' missing')
    expected = dict(family=entry['family'], trial=entry['trial'], device='cuda', dtype=entry['dtype'],
                    **plan['grid_contracts'][entry['grid']])
    for key in ('family', 'trial', 'device', 'dtype', 'shape', 'lengths'):
        if profile.get('config', {}).get(key) != expected[key]:
            raise HardQualificationError(f'profile config {key}')
    for key, value in (('role', entry['provider_role']), ('commit', entry['provider_commit']),
                       ('wheel_sha256', entry['provider_wheel_sha256'])):
        if profile.get('provider_identity', {}).get(key) != value:
            raise HardQualificationError(f'profile provider {key}')
    runtime, environment = profile.get('runtime_identity', {}), profile.get('environment', {})
    if (profile.get('finite') is not True or runtime.get('fallback_used') is not False
            or runtime.get('requested') != runtime.get('effective')
            or environment.get('allocated_device') != 'cuda:0' or 'H100' not in environment.get('device_name', '')
            or profile.get('compiler', {}).get('compile_fallback') is not False):
        raise HardQualificationError('profile device/finite/fallback')
    off = dict(float32_matmul_precision='highest', cuda_matmul_allow_tf32=False,
               cudnn_allow_tf32=False)
    require(environment.get('tf32_caller_before') == off, 'TF32 before differs')
    require(environment.get('tf32_caller_after') == off, 'TF32 after differs')
    require(profile['compiler'].get('graph_breaks') == 0, 'graph break')
    try:
        steps = frozen._integer(profile.get('completed_profile_steps'), 'steps')
        require(steps == entry['timed_steps'], 'step count')
        require(profile.get('runtime_completed_steps') == entry['warmup_steps'] + steps,
                'runtime count')
        transforms = profile['transform_calls']
        forward = frozen._integer(transforms.get('forward'), 'forward')
        inverse = frozen._integer(transforms.get('inverse'), 'inverse')
        require(forward == entry['expected_forward_raw_calls'], 'raw forward')
        require(inverse == entry['expected_inverse_raw_calls'], 'raw inverse')
        require(transforms.get('forward_per_step') == forward / steps, 'forward rate')
        require(transforms.get('inverse_per_step') == inverse / steps, 'inverse rate')
        require(forward - entry['scheduled_refreshes_in_timed_window'] ==
                entry['steady_forward_transforms_per_step'] * steps, 'steady forward')
        require(inverse == entry['steady_inverse_transforms_per_step'] * steps, 'steady inverse')
        timing = profile['timing']
        samples = timing.get('samples_seconds')
        require(isinstance(samples, list) and len(samples) == 60, 'timing samples')
        clean = [frozen._positive(value, f'sample {i}') for i, value in enumerate(samples)]
        require(frozen._positive(timing.get('mean_timestep_seconds'), 'mean') ==
                statistics.fmean(clean), 'mean is not raw-sample mean')
        frozen._positive(timing.get('median_timestep_seconds'), 'median')
        for key in ('peak_allocated_bytes', 'peak_reserved_bytes'):
            require(frozen._integer(profile['memory'].get(key), key) > 0, key)
    except (ValueError, TypeError, ZeroDivisionError) as error:
        raise HardQualificationError(f'profile validity: {error}') from error
    for key in ('initial_state_sha256', 'final_state_sha256'):
        require(isinstance(profile.get(key), str) and len(profile[key]) == 64, key)


def profiles(args, output):
    from benchmarks import run_v0_3_s09_profile_matrix as frozen
    namespace = dict(frozen.build_command_plan.__globals__, build_plan=build_plan)
    make = types.FunctionType(frozen.build_command_plan.__code__, namespace)
    commands = make(baseline_python=args.baseline_python, candidate_python=args.candidate_python,
                    support_root=args.support, output_root=output / 'profile_matrix')
    frozen._preflight(commands)
    atomic_write_json(output / 'profile_command_plan.json', commands, overwrite=False)
    if args.device == 'cpu':
        return dict(classification='PASS', formal_profiles_excluded=True, planned_profiles=len(commands['rows']))
    plan = build_plan()
    for row, entry in zip(commands['rows'], plan['profile_matrix']):
        cache = Path(row['compiler_cache'])
        for name in ('triton', 'torchinductor'):
            (cache / name).mkdir(parents=True, exist_ok=False)
        env = dict(os.environ, PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1', NVIDIA_TF32_OVERRIDE='0',
                   PYTHONPATH=str(args.support), TRITON_CACHE_DIR=str(cache / 'triton'),
                   TORCHINDUCTOR_CACHE_DIR=str(cache / 'torchinductor'))
        launch(row['command'], output, 'profile_' + row['key'], env=env)
        frozen._validate_tf32_bootstrap(Path(row['tf32_bootstrap_output']), Path(row['profile_script']))
        validate_profile_h(json.loads(Path(row['output']).read_text()), entry, plan)
    return dict(classification='PASS', planned_profiles=192, executed_profiles=192)


def run(args):
    plan = build_plan()
    output = args.output.absolute()
    output.mkdir(parents=True, exist_ok=False)
    support = args.support.resolve()
    if (support / 'pssolver').exists() or (support / 'pssolver_control').exists():
        raise HardQualificationError('support contains an importable runtime package')
    # Consumer main needs reachable historical objects and src files for its
    # immutable provenance tests. The unchanged Consumer conftest prepends the
    # Provider reference, but packages are preloaded from the installed wheels
    # and every loaded package module is checked against its own venv afterward.
    if args.device == 'cpu' and os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        raise HardQualificationError('CPU rehearsal must hide CUDA')
    stages = []
    try:
        env_output = output / 'environment/stage.json'
        command = [args.candidate_python, '-I', '-B', support / 'benchmarks/run_v0_4_s19_environment.py',
                   '--device', args.device, '--output', env_output, '--work', output / 'environment',
                   '--mirror-versions', args.mirror_versions]
        for key in ('baseline_python', 'candidate_python', 'consumer_python', 'baseline_wheel', 'candidate_wheel', 'consumer_wheel'):
            command.extend(['--' + '-'.join(key.split('_')), getattr(args, key)])
        if args.tmp_preflight:
            command.extend(['--tmp-preflight', args.tmp_preflight])
        launch(command, output, 'environment')
        stages.append(json.loads(env_output.read_text()))
        for stage in ('cuda_tests', 'long_horizon', 'tf32', 'checkpoints'):
            path = output / 'stages' / f'{stage}.json'
            launch([args.candidate_python, '-I', '-B', support / 'benchmarks/run_v0_4_s19_stage.py', stage,
                    '--device', args.device, '--support', support, '--provider-wheel', args.candidate_wheel,
                    '--output', path, '--work', output / 'work' / stage], output, stage)
            stages.append(json.loads(path.read_text()))
        numerical_root = output / 'numerical'
        launch([args.candidate_python, '-I', '-B', support / 'benchmarks/run_v0_4_s19_self_compare.py',
                '--device', args.device, '--support-root', support, '--output-root', numerical_root,
                '--work', output / 'calculation/s19_shared_calculation',
                '--baseline-python', args.baseline_python, '--candidate-python', args.candidate_python,
                '--baseline-wheel', args.baseline_wheel, '--candidate-wheel', args.candidate_wheel], output, 'numerical')
        summary = json.loads((numerical_root / 'summary.json').read_text())
        cases = summary['cases']
        if (len(cases) != 47 or sum(x['kind'] == 'trajectory' for x in cases) != 40
                or sum(x['kind'] == 'pcg' for x in cases) != 1
                or sum(x['kind'] == 'public_dispatch' for x in cases) != 6 or summary['launch_count'] != 141):
            raise HardQualificationError('numerical execution coverage')
        nonfinite = sorted({tuple(field['pcg_record']) for case in cases
                            for comparison in case['comparisons'].values()
                            for field in comparison['fields'] if field.get('decision') == 'D-0022'})
        numerical = dict(name='numerical_gradient_taylor_and_replay', classification=summary['classification'],
                         provider_candidate_commit=plan['provider']['candidate']['commit'],
                         candidate_wheel_sha256=plan['provider']['candidate']['wheel_sha256'],
                         execution_device=args.device, finite=not nonfinite,
                         nonfinite_contract='D-0022 exact two oracle gradient records; all other fields finite',
                         allowed_nonfinite_records=[list(x) for x in sorted(KNOWN_NONFINITE_PCG_RECORDS)],
                         nonfinite_records=[list(x) for x in nonfinite],
                         fallback_used=False, same_absolute_work_path=True, independent_empty_caches=True,
                         baseline_self_comparison=True, public_dispatch_completed=True, four_pcg_paths=True,
                         all_fields_and_steps_recorded=True, trajectory_cases=40, trajectory_steps=20,
                         report=str(numerical_root / 'summary.json'), report_sha256=file_sha256(numerical_root / 'summary.json'))
        stages.append(numerical)
        consumer_output = output / 'stages/consumer.json'
        launch([args.consumer_python, '-I', '-B', support / 'benchmarks/run_v0_4_s19_stage.py', 'consumer',
                '--device', args.device, '--support', support, '--consumer-support', args.consumer_reference,
                '--provider-reference', args.provider_reference, '--consumer-wheel', args.consumer_wheel,
                '--provider-wheel', args.candidate_wheel, '--output', consumer_output,
                '--work', output / 'work/consumer', '--login-preflight', args.login_preflight,
                '--login-preflight-sha256', args.login_preflight_sha256], output, 'consumer')
        stages.append(json.loads(consumer_output.read_text()))
        auxiliary = dict(stages=stages, formal_h100_submission_count=1 if args.device == 'cuda' else 0,
                         automatic_retry=False, second_job_submitted=False)
        atomic_write_json(output / 'auxiliary.json', auxiliary, overwrite=False)
        matrix = profiles(args, output)
        if args.device == 'cuda':
            from benchmarks.analyze_v0_4_s19_h100 import analyze
            result = analyze(output / 'profile_matrix/profiles', auxiliary)
        else:
            result = final_verdict(stages + [matrix], formal_h100=False)
        atomic_write_json(output / 'analysis.json', result, overwrite=False)
        success = result.get('qualification_complete') is True or result.get('cpu_rehearsal_passed') is True
        atomic_write_text(output / ('COMPLETE' if success else 'FAILED'), result['classification'] + '\n', overwrite=False)
        return 0 if success else 1
    except Exception:
        atomic_write_text(output / 'FAILED', traceback.format_exc(), overwrite=False)
        raise
    finally:
        # Caches remain diagnostic artifacts, not qualification outputs.
        excluded = [str(p.relative_to(output)) for p in output.rglob('*') if p.is_file() and
                    ('compiler_cache' in p.parts or '__pycache__' in p.parts)]
        atomic_write_json(output / 'cache_inventory.json', dict(excluded_from_manifest=excluded,
                          reason='compiler caches; all commands, raw profiles and results retained'), overwrite=False)
        files = sorted(p for p in output.rglob('*') if p.is_file() and p.name != 'checksums.sha256'
                       and str(p.relative_to(output)) not in excluded)
        atomic_write_text(output / 'checksums.sha256', ''.join(f'{file_sha256(p)}  {p.relative_to(output)}\n' for p in files), overwrite=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--login-preflight', type=Path, required=True)
    parser.add_argument('--login-preflight-sha256', required=True)
    parser.add_argument('--device', required=True, choices=('cpu', 'cuda'))
    for name in ('baseline-python', 'candidate-python', 'consumer-python', 'baseline-wheel', 'candidate-wheel',
                 'consumer-wheel', 'support', 'provider-reference', 'consumer-reference', 'mirror-versions', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--tmp-preflight', type=Path)
    return run(parser.parse_args())


if __name__ == '__main__':
    raise SystemExit(main())
