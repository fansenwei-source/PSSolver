"""Forward-only provenance for the S-19 Git compatibility correction."""
import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor

ROOT = Path(__file__).resolve().parents[1]


def test_s19_git_compatibility_successor_preserves_old_records():
    record = json.loads((ROOT / 'notes/PSSolver_v0_3_0_s03_successor_121_s19_git_compatibility.json').read_text())
    assert record['successor'] == 121
    assert record['scope']['runtime_changed'] is False
    assert record['scope']['slurm_submitted'] is False
    assert set(record['historical_successor_sources']) == {'benchmarks/_artifact_io.py'}
    for relative, row in record['historical_successor_sources'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == resolve_s03_successor(relative, row['historical_sha256'])
    for relative, expected in record['introduced_sources'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == resolve_s03_successor(relative, expected)
    for relative, expected in record['unchanged_bindings'].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == resolve_s03_successor(relative, expected)
