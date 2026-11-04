"""D-0027 immutable full-tree attestation and lossless test partition."""
import copy
import json
from pathlib import Path
import subprocess

import pytest

from benchmarks._artifact_io import file_sha256
from benchmarks._s19_compare import HardQualificationError
from benchmarks._s19_consumer_attestation import (
    LOGIN_TESTS, ConsumerPartition, check_login_report, coverage_union, tree_manifest,
)
from benchmarks.plan_v0_4_s19_h100 import CONSUMER_RUNTIME_COMMIT


def records():
    node = [f'tests/test_other.py::test_{i:03d}' for i in range(302)]
    full = sorted([*LOGIN_TESTS, *node])
    login = dict(collected=full, selected=list(LOGIN_TESTS), deselected=node,
                 passed=list(LOGIN_TESTS), failed=[], skipped=[], xfailed=[], xpassed=[],
                 collection_errors=[], returncode=0)
    execution = dict(collected=full, selected=node, deselected=list(LOGIN_TESTS),
                     passed=node, skipped=[], failed=[], xfailed=[], xpassed=[],
                     collection_errors=[], returncode=0)
    return login, execution


@pytest.fixture
def attestation(tmp_path):
    root = tmp_path / 'consumer'
    (root / '.git/objects').mkdir(parents=True)
    (root / '.git/config').write_text('[core]\n')
    (root / 'test.py').write_text('pass\n')
    report = dict(passed=True, mode='login', consumer_reference={
        'head': CONSUMER_RUNTIME_COMMIT, 'clean': True}, consumer_tree=tree_manifest(root),
        consumer_login_tests=records()[0])
    path = tmp_path / 'login.json'
    path.write_text(json.dumps(report))
    return root, path, report


def test_node_attestation_reads_no_git(attestation, monkeypatch):
    root, path, report = attestation
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: pytest.fail('node invoked subprocess/git'))
    assert check_login_report(path, file_sha256(path), root) == report
    assert '.git/config' in report['consumer_tree'] and '.git/objects' in report['consumer_tree']


@pytest.mark.parametrize('mutation', ['content', 'config', 'added', 'removed', 'symlink', 'mode', 'empty_directory'])
def test_reference_changes_fail_closed(attestation, mutation):
    root, path, _ = attestation
    if mutation == 'content':
        (root / 'test.py').write_text('fail\n')
    elif mutation == 'config':
        (root / '.git/config').write_text('different')
    elif mutation == 'added':
        (root / '.git/unexpected').write_bytes(b'')
    elif mutation == 'removed':
        (root / 'test.py').unlink()
    elif mutation == 'symlink':
        (root / 'link').symlink_to('test.py')
    elif mutation == 'mode':
        (root / 'test.py').chmod(0o700)
    else:
        (root / 'empty').mkdir()
    with pytest.raises(HardQualificationError):
        check_login_report(path, file_sha256(path), root)


@pytest.mark.parametrize('mutation', ['hash', 'passed', 'mode', 'head', 'clean', 'failed_test', 'missing_test', 'extra_test'])
def test_login_evidence_cannot_be_forged_or_partial(attestation, mutation):
    root, path, report = attestation
    old_hash = file_sha256(path)
    if mutation in ('hash', 'passed'):
        report['passed'] = False
    elif mutation == 'mode':
        report['mode'] = 'job'
    elif mutation in ('head', 'clean'):
        report['consumer_reference'][mutation] = 'bad' if mutation == 'head' else False
    elif mutation == 'failed_test':
        report['consumer_login_tests']['failed'] = ['error']
    elif mutation == 'missing_test':
        report['consumer_login_tests']['passed'].pop()
    else:
        report['consumer_login_tests']['collected'].append('unexpected')
    path.write_text(json.dumps(report))
    with pytest.raises(HardQualificationError):
        check_login_report(path, old_hash if mutation == 'hash' else file_sha256(path), root)


def test_exact_disjoint_union_keeps_all_tests_not_skip_or_xfail():
    login, node = records()
    result = coverage_union(login, node)
    assert len(result['collected']) == 304
    assert result['exact_disjoint_union'] is True
    assert len(result['login_passed']) == 2 and len(result['node_selected']) == 302


@pytest.mark.parametrize('mutation', ['missing', 'duplicate', 'wrong_partition', 'unreported', 'failed', 'collection', 'extra'])
def test_coverage_cannot_pass_by_count_alone(mutation):
    login, node = records()
    if mutation == 'missing':
        node['selected'].pop()
    elif mutation == 'duplicate':
        node['selected'][0] = node['selected'][1]
    elif mutation == 'wrong_partition':
        node['deselected'] = []
    elif mutation == 'unreported':
        node['passed'].pop()
    elif mutation == 'failed':
        node['failed'] = ['failure']
    elif mutation == 'collection':
        node['collected'] = node['collected'][1:]
    else:
        node['passed'].append('unexpected')
    with pytest.raises(HardQualificationError):
        coverage_union(login, node)


def test_partition_selects_the_exact_two_tests_in_both_phases():
    from types import SimpleNamespace
    login, node = records()
    for phase, expected in [('login', login), ('node', node)]:
        items = [SimpleNamespace(nodeid=n) for n in login['collected']]
        deselected = []
        config = SimpleNamespace(hook=SimpleNamespace(
            pytest_deselected=lambda items: deselected.extend(i.nodeid for i in items)))
        partition = ConsumerPartition(phase)
        partition.pytest_collection_modifyitems(None, config, items)
        assert sorted(i.nodeid for i in items) == expected['selected']
        assert sorted(deselected) == expected['deselected']


def test_synthetic_analyzer_rejects_changed_consumer_partition():
    from tests.test_analyze_v0_4_s19_h100 import _auxiliary
    from benchmarks.analyze_v0_4_s19_h100 import validate_auxiliary
    from benchmarks.plan_v0_4_s19_h100 import build_plan
    value = _auxiliary()
    validate_auxiliary(value, build_plan())
    tampered = copy.deepcopy(value)
    tampered['stages'][6]['coverage']['login_passed'] = []
    with pytest.raises(HardQualificationError):
        validate_auxiliary(tampered, build_plan())
