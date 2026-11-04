"""Read only the frozen import snapshot, allowing future registry progression."""
import hashlib
import json
from pathlib import Path
from tests.v0_3_s03_successor import resolve_s03_successor

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / 'notes/PSSolver_v0_3_0_s03_successor_128_r0064_import.json'


def test_r0064_is_interim_not_gpu_acceptance():
    record = json.loads(RECORD.read_text())
    assert record['review']['classification'] == 'INTERIM_PASS_S19_LOGIN_COMPUTE_COMMAND_BOUNDARY_PREPARATION'
    assert record['git_identity']['reviewed_support'] == 'ef71e5439f87796646bde3635ab44b7f2e28888f'
    assert not any(record['scope'].values())


def test_r0064_snapshot_verifies_v64_v65_only():
    record = json.loads(RECORD.read_text())
    rows = record['registry_snapshot']
    for key, fix in [('V-64', 'fed03fc969ef6926065d43b1d5c14872dca875f6'),
                     ('V-65', '7c6aca1ff11fa1a87cc4506ad9f05b91b01a0726')]:
        assert rows[key]['status'] == 'verified'
        assert rows[key]['verified_by'] == 'R-0064'
        assert rows[key]['fixed_in'] == fix
    assert rows['V-62']['status'] == 'verified' and rows['V-62']['verified_by'] == 'R-0060'
    assert rows['V-63']['status'] == 'open'


def test_successor_128_resolves_forward_without_external_reads():
    record = json.loads(RECORD.read_text())
    bindings = {p: row['current_sha256'] for p, row in record['historical_successor_sources'].items()}
    bindings.update(record['introduced_sources'])
    for path, digest in bindings.items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == resolve_s03_successor(path, digest)
