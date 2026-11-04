"""D-0027 login/node Consumer evidence partition; standard library only.

The node consumes a hash-bound login report, never a Git working-copy query.
The manifest covers the complete reference, including .git and empty folders.
"""
from __future__ import annotations

import json
from pathlib import Path
import stat

from benchmarks._artifact_io import file_sha256
from benchmarks._s19_compare import HardQualificationError

LOGIN_TESTS = (
    "tests/test_p9_8_stable_functional_api_migration.py::test_p983_record_binds_sources_and_preserves_historical_evidence",
    "tests/test_pssolver_rc4_pressure_migration.py::test_rc446_record_binds_live_consumer_sources",
)


def tree_manifest(root):
    root = Path(root)
    if root.is_symlink() or not root.is_dir() or not (root / '.git').is_dir():
        raise HardQualificationError('Consumer reference must be a real Git directory')
    result = {}
    for path in sorted(root.rglob('*')):
        before = path.lstat()
        kind = stat.S_IFMT(before.st_mode)
        entry = {'mode': stat.S_IMODE(before.st_mode)}
        if kind == stat.S_IFDIR:
            entry['type'] = 'directory'
        elif kind == stat.S_IFREG:
            entry.update(type='file', size=before.st_size, sha256=file_sha256(path))
        else:
            raise HardQualificationError(f'nonregular Consumer reference entry: {path}')
        after = path.lstat()
        if (before.st_ino, before.st_size, before.st_mtime_ns, before.st_mode) != (
                after.st_ino, after.st_size, after.st_mtime_ns, after.st_mode):
            raise HardQualificationError('Consumer reference changed during inventory')
        result[path.relative_to(root).as_posix()] = entry
    return result


def check_login_report(path, expected_sha256, consumer_root):
    if (not isinstance(expected_sha256, str) or len(expected_sha256) != 64
            or file_sha256(path) != expected_sha256):
        raise HardQualificationError('login preflight report hash')
    report = json.loads(Path(path).read_text())
    from benchmarks.plan_v0_4_s19_h100 import CONSUMER_RUNTIME_COMMIT
    if (report.get('passed') is not True or report.get('mode') != 'login'
            or report.get('consumer_reference', {}).get('head') != CONSUMER_RUNTIME_COMMIT
            or report['consumer_reference'].get('clean') is not True
            or report.get('consumer_tree') != tree_manifest(consumer_root)):
        raise HardQualificationError('login attestation / Consumer full tree mismatch')
    login = report.get('consumer_login_tests', {})
    if (login.get('returncode') != 0 or login.get('passed') != list(LOGIN_TESTS)
            or any(login.get(k) for k in ('failed', 'skipped', 'xfailed', 'xpassed', 'collection_errors'))
            or login.get('selected') != list(LOGIN_TESTS)):
        raise HardQualificationError('login Consumer test outcomes')
    collected = login.get('collected', [])
    if (len(collected) != 304 or len(set(collected)) != 304
            or not set(LOGIN_TESTS) <= set(collected)
            or login.get('deselected') != sorted(set(collected) - set(LOGIN_TESTS))):
        raise HardQualificationError('login Consumer complete collection')
    return report


def coverage_union(login, node):
    """Require exact node IDs and outcomes, not just a compensating count."""
    collected = login['collected']
    selected = node.get('selected', [])
    skipped = [row['nodeid'] for row in node.get('skipped', [])]
    if (node.get('collected') != collected
            or node.get('deselected') != list(LOGIN_TESTS)
            or sorted(selected) != sorted(set(collected) - set(LOGIN_TESTS))
            or len(selected) != len(set(selected))
            or sorted(node.get('passed', []) + skipped) != sorted(selected)
            or login.get('passed') != list(LOGIN_TESTS)
            or any(node.get(k) for k in ('failed', 'xfailed', 'xpassed', 'collection_errors'))
            or node.get('returncode') != 0):
        raise HardQualificationError('login + node Consumer coverage is not exactly the full suite')
    return {'collected': collected, 'login_passed': list(LOGIN_TESTS),
            'node_selected': sorted(selected), 'node_passed': node['passed'],
            'node_skipped': skipped, 'exact_disjoint_union': True}


class ConsumerPartition:
    def __init__(self, phase):
        if phase not in ('login', 'node'):
            raise ValueError(phase)
        self.phase = phase
        self.report = {}

    def pytest_collection_modifyitems(self, session, config, items):
        collected = sorted(item.nodeid for item in items)
        if len(collected) != 304 or len(set(collected)) != 304 or not set(LOGIN_TESTS) <= set(collected):
            raise HardQualificationError('Consumer full collection changed')
        selected, deselected = [], []
        for item in items:
            keep = (item.nodeid in LOGIN_TESTS) == (self.phase == 'login')
            (selected if keep else deselected).append(item)
        self.report = dict(collected=collected, selected=sorted(i.nodeid for i in selected),
                           deselected=sorted(i.nodeid for i in deselected))
        config.hook.pytest_deselected(items=deselected)
        items[:] = selected
