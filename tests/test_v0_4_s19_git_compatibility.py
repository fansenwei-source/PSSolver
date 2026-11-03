"""Git 1.8-compatible provenance remains fail-closed (local CPU tests only)."""
import os
from pathlib import Path
import subprocess

import pytest

from benchmarks import _artifact_io as artifacts
from benchmarks import validate_v0_4_s19_deployment as deployment


def _git(path, *args):
    return subprocess.run(['git', *args], cwd=path, check=True,
                          capture_output=True, text=True).stdout.strip()


@pytest.fixture
def repository(tmp_path):
    path = tmp_path / 'repository with spaces'
    path.mkdir()
    _git(path, 'init')
    _git(path, 'config', 'user.name', 'Local compatibility test')
    _git(path, 'config', 'user.email', 'local-test@example.invalid')
    (path / 'tracked.txt').write_text('original\n')
    _git(path, 'add', 'tracked.txt')
    _git(path, 'commit', '-m', 'local fixture')
    return path


@pytest.mark.parametrize('state', ['clean', 'dirty', 'detached'])
def test_git_identity_keeps_fields_and_process_cwd(repository, monkeypatch, state):
    if state == 'dirty':
        (repository / 'tracked.txt').write_text('changed\n')
    elif state == 'detached':
        _git(repository, 'checkout', '--detach', 'HEAD')
    expected = dict(root=str(repository.resolve()), head=_git(repository, 'rev-parse', 'HEAD'),
                    branch=_git(repository, 'rev-parse', '--abbrev-ref', 'HEAD'),
                    clean=state != 'dirty', status=_git(repository, 'status', '--short'))
    original = subprocess.run
    calls = []

    def rejecting_old_git(argv, **kwargs):
        assert argv[0] == 'git'
        assert '-C' not in argv, 'old Git rejects global -C'
        assert Path(kwargs['cwd']) == repository
        calls.append(list(argv))
        return original(argv, **kwargs)

    monkeypatch.setattr(artifacts.subprocess, 'run', rejecting_old_git)
    before = Path.cwd()
    assert artifacts.git_identity(repository) == expected
    assert Path.cwd() == before
    assert len(calls) == 4


@pytest.mark.parametrize('kind,error', [
    ('missing', FileNotFoundError), ('file', NotADirectoryError),
    ('non_repository', subprocess.CalledProcessError),
])
def test_invalid_git_directory_fails_before_venv_or_success_report(tmp_path, monkeypatch, kind, error):
    root = tmp_path / 'control'
    (root / 'logs').mkdir(parents=True)
    support = root / 'support'
    support.mkdir()
    consumer = root / 'invalid'
    if kind == 'file':
        consumer.write_text('not a directory')
    elif kind == 'non_repository':
        consumer.mkdir()
        # Disable discovery outside the fixture, including a containing checkout.
        monkeypatch.setenv('GIT_CEILING_DIRECTORIES', str(consumer.parent))
    output = root / 'logs/preflight_login.json'
    monkeypatch.setenv('S19_OUTPUT_ROOT', str(root / 'absent_output'))
    monkeypatch.setenv('S19_PROVIDER_SUPPORT', str(support))
    monkeypatch.setenv('S19_CONSUMER_REFERENCE', str(consumer))
    monkeypatch.setattr(deployment, 'build_plan', lambda: {
        'deployment': {'control_root': str(root)}, 'consumer': {'commit': '0' * 40}})
    # No Python/venv/source settings: proceeding past failed identity would fail
    # this assertion by raising a different exception or publishing a report.
    before = Path.cwd()
    with pytest.raises(error):
        deployment.ready(root, {'environment': {}}, output)
    assert not output.exists()
    assert Path.cwd() == before


def test_remote_remove_in_subshell_preserves_parent_directory(repository):
    _git(repository, 'remote', 'add', 'origin', '/not/contacted/local.bundle')
    command = '(cd "$1" && git remote remove origin); test "$PWD" = "$2"'
    subprocess.run(['bash', '-e', '-c', command, 'test', str(repository), os.getcwd()], check=True)
    assert _git(repository, 'remote') == ''
