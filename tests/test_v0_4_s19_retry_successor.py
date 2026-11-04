"""Repository-only forward bindings and explicit D-0027 scope."""
import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor

ROOT = Path(__file__).resolve().parents[1]


def test_retry_successor_is_support_only_and_forward_bound():
    record = json.loads((ROOT / 'notes/PSSolver_v0_3_0_s03_successor_123_s19_retry.json').read_text())
    assert record['decision'] == 'D-0027'
    assert record['runtime_changed'] is False
    assert record['runtime_commit'] == 'c7c940e693719becccef80a918142a27aa050510'
    assert record['remote_execution_authorized_by_this_record'] is False
    for path, row in record['historical_successor_sources'].items():
        assert path.startswith(('benchmarks/', 'jobs/', 'tests/', 'notes/'))
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        assert actual == resolve_s03_successor(path, row['historical_sha256'])
    for path, expected in record['introduced_sources'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == resolve_s03_successor(path, expected)


def test_command_inventory_binds_node_evidence_without_reading_external_paths():
    catalog = json.loads((ROOT / 'notes/PSSolver_v0_4_s19_retry_node_commands.json').read_text())
    job = (ROOT / 'jobs/v0_4_s19_single_h100.sbatch').read_text()
    for row in catalog['required_commands']:
        assert row['historical_job'] in (10911538, 10912583)
        assert row['node'] == 'gpu-h100-4-0'
        assert row['evidence'] and row['interpretation'] and row['current_sites']
        for evidence in row['evidence']:
            assert len(evidence['sha256']) == 64
            assert evidence['lines'][0] >= 1
    assert 'MISSING_REQUIRED_COMMAND' in job
    assert 'command -v module' in job
    assert 'command -v "$assembler"' in job
    assert 'export LC_ALL=C' in job
    assert '--mode job' in job and '--mode login' in job
    assert 'S19_LOGIN_PREFLIGHT_SHA256' in job
    assert 'git ' not in job
