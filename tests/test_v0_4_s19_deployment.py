"""Preparation-only deployment gates; no SSH, scheduler or GPU invocation."""
import base64
import csv
import hashlib
import inspect
import io
import json
from pathlib import Path
import subprocess
import tarfile
from types import SimpleNamespace

import pytest

from benchmarks import run_v0_4_s19_environment as environment
from benchmarks import run_v0_4_s19_job as job
from benchmarks import validate_v0_4_s19_deployment as deployment
from benchmarks._s19_compare import HardQualificationError
from benchmarks.plan_v0_4_s19_h100 import build_plan


def test_manifest_rejects_tamper_escape_symlink_and_bad_tar(tmp_path):
    artifact = tmp_path / 'source.tar'
    with tarfile.open(artifact, 'w') as archive:
        info = tarfile.TarInfo('provider/a.py')
        info.size = 1
        archive.addfile(info, io.BytesIO(b'a'))
    binding = dict(size=artifact.stat().st_size, sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(), tar_prefix='provider')
    manifest = tmp_path / 'deployment.json'

    def validate(value):
        manifest.write_text(json.dumps(value))
        return deployment.validate_inputs(tmp_path, manifest, hashlib.sha256(manifest.read_bytes()).hexdigest())

    assert validate(dict(inputs={'source.tar': binding}))['inputs']['source.tar'] == binding
    for key, value in [('size', 1), ('sha256', 'bad'), ('tar_prefix', 'other')]:
        with pytest.raises(HardQualificationError):
            validate(dict(inputs={'source.tar': dict(binding, **{key: value})}))
    with pytest.raises(HardQualificationError):
        validate(dict(inputs={'../source.tar': binding}))
    (tmp_path / 'link.tar').symlink_to(artifact)
    with pytest.raises(HardQualificationError):
        validate(dict(inputs={'link.tar': binding}))
    with pytest.raises(HardQualificationError, match='manifest hash'):
        deployment.validate_inputs(tmp_path, manifest, 'bad')


def test_d0021_uses_exact_two_versions_and_validates_actual_files(tmp_path, monkeypatch):
    dists = []
    for name, version in [('packaging', '26.3'), ('typing_extensions', '4.16.0')]:
        payload = tmp_path / f'{name}.py'
        payload.write_bytes(b'actual module')
        directory = tmp_path / f'{name}-{version}.dist-info'
        directory.mkdir()
        digest = base64.urlsafe_b64encode(hashlib.sha256(payload.read_bytes()).digest()).decode().rstrip('=')
        with (directory / 'RECORD').open('w') as stream:
            csv.writer(stream).writerow([payload.name, 'sha256=' + digest, str(payload.stat().st_size)])
        dists.append(SimpleNamespace(metadata={'Name': name}, version=version, _path=directory,
                                     locate_file=lambda relative: tmp_path / relative))
    monkeypatch.setattr(environment.metadata, 'distributions', lambda: iter(dists))
    monkeypatch.setattr(environment.metadata, 'version', lambda name: {'packaging': '26.2', 'typing_extensions': '4.12.2'}[name])
    versions, proof = environment.effective_versions()
    assert versions == {'packaging': '26.3', 'typing_extensions': '4.16.0'}
    assert proof['reported_versions']['packaging'] == '26.2'
    assert len(proof['d0021_file_proofs']) == 2
    (tmp_path / 'packaging.py').write_bytes(b'tampered')
    with pytest.raises(HardQualificationError, match='actual file mismatch'):
        environment.effective_versions()


def test_mirror_versions_reject_cpu_wheel_and_python_drift():
    expected = {'torch': '2.5.1+cu121', 'typing_extensions': '4.16.0'}
    value = dict(python='3.10.18', versions={'torch': '2.5.1+cu121', 'typing-extensions': '4.16.0'})
    environment.validate_mirror(value, expected)
    with pytest.raises(HardQualificationError):
        environment.validate_mirror(dict(value, python='3.10.22'), expected)
    value['versions']['torch'] = '2.5.1+cpu'
    with pytest.raises(HardQualificationError):
        environment.validate_mirror(value, expected)


def test_job_is_parseable_and_contains_no_scheduler_submission():
    path = Path(__file__).parents[1] / 'jobs/v0_4_s19_single_h100.sbatch'
    subprocess.run(['bash', '-n', str(path)], check=True)
    source = path.read_text()
    assert build_plan()['deployment']['control_root'] in source
    assert 'sbatch ' not in source and 'srun ' not in source
    assert '--no-requeue' in source and '--preflight-only' in source
    assert 'validate_v0_4_s19_deployment.py' in source
    assert 'module load cuda/12.1' in source and 'unset LD_LIBRARY_PATH' in source
    assert source.index('--ready-output') < source.index('run_v0_4_s19_job.py')
    assert 'case "$SLURM_JOB_ID"' in source


def test_driver_child_crash_is_h_and_preserves_logs(tmp_path):
    with pytest.raises(HardQualificationError, match='child injected failed'):
        job.launch(['bash', '-c', 'printf diagnostic; exit 9'], tmp_path, 'injected')
    assert (tmp_path / 'logs/injected.stdout').read_text() == 'diagnostic'
    assert json.loads((tmp_path / 'launches/injected.json').read_text())['returncode'] == 9


def test_driver_orders_control_before_profiles_and_has_one_numerical_run():
    source = inspect.getsource(job.run)
    assert source.index("'numerical')") < source.index("'consumer')") < source.index('matrix = profiles(')
    assert source.count('run_v0_4_s19_self_compare.py') == 1
    assert 'summary[\'launch_count\'] != 141' in source
    assert 'second_job_submitted=False' in source and 'automatic_retry=False' in source
