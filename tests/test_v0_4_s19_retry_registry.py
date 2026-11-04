"""Historical preparation snapshot; do not pin future registry states."""
import hashlib
import json
from pathlib import Path
from tests.v0_3_s03_successor import resolve_s03_successor

ROOT = Path(__file__).resolve().parents[1]


def test_retry_claim_is_only_v64_and_preserves_v62_v63():
    record = json.loads((ROOT / 'notes/PSSolver_v0_3_0_s03_successor_124_s19_retry_registry.json').read_text())
    old = json.loads((ROOT / 'notes/PSSolver_v0_3_0_s03_successor_122_r0062_import.json').read_text())
    assert record['registry_snapshot']['V-64']['status'] == 'claimed_fixed'
    assert record['registry_snapshot']['V-64']['fixed_in'] == record['git_identity']['implementation']
    for key in ('V-62', 'V-63'):
        assert record['registry_snapshot'][key] == old['registry_snapshot'][key]
    for path, row in record['historical_successor_sources'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == resolve_s03_successor(path, row['historical_sha256'])
    for path, expected in record['introduced_sources'].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == resolve_s03_successor(path, expected)
