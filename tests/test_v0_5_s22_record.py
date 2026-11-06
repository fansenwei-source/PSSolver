"""Forward bindings for the approved S-22 foundation; historical records stay immutable."""

import hashlib
import json
import subprocess
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_134_s22_foundation.json"


def test_s22_forward_bindings_cover_changed_sources():
    record = json.loads(RECORD.read_text())
    assert record["decisions"] == ["D-0042", "D-0043"]
    assert record["historical_records_rewritten"] is False
    for path, item in record["historical_successor_sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == resolve_s03_successor(
            path, item["historical_sha256"]
        )
    for path, digest in record["introduced_sources"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == resolve_s03_successor(
            path, digest
        )


def test_s22_preserves_the_frozen_numerical_source_tree():
    record = json.loads(RECORD.read_text())
    base = record["git_identity"]["implementation_parent"]
    for path, expected in record["unchanged_runtime_sha256"].items():
        historical = subprocess.check_output(["git", "show", f"{base}:{path}"], cwd=ROOT)
        assert hashlib.sha256(historical).hexdigest() == expected
    assert record["allowed_runtime_changes"] == ["pssolver/_version.py", "pssolver/visualizer.py"]
    assert record["registry_updated"] is False
