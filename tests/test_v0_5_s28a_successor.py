"""Forward-only source and protocol bindings for the Channel migration."""

import hashlib
import json
import subprocess
from pathlib import Path

from tests.v0_3_s03_successor import read_registered_source, resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_147_s28a_channel.json"


def test_s28a_records_every_retirement_and_changed_source():
    record = json.loads(RECORD.read_bytes())
    authority = json.loads((ROOT / "qualification/s28a_tables.json").read_bytes())
    assert record["historical_records_rewritten"] is False
    assert {f"{module.replace('.', '/')}.py" for module in authority["retired_modules"]} == (
        record["retired_sources"].keys()
    )
    for path, row in record["retired_sources"].items():
        assert hashlib.sha256(read_registered_source(path)).hexdigest() == row["sha256"]
        assert row["source_commit"] == record["git_identity"]["implementation_parent"]
    for path, row in record["historical_successor_sources"].items():
        expected = resolve_s03_successor(path, row["current_sha256"])
        assert hashlib.sha256(read_registered_source(path)).hexdigest() == expected
    for path, sha in record["introduced_sources"].items():
        expected = resolve_s03_successor(path, sha)
        assert hashlib.sha256(read_registered_source(path)).hexdigest() == expected
    assert "qualification/s28a_guard.py" in record["historical_successor_sources"]
    assert "qualification/s28a_tables.json" in record["historical_successor_sources"]


def test_s28a_preserves_accepted_frozen_sources():
    record = json.loads(RECORD.read_bytes())
    baseline = record["git_identity"]["implementation_parent"]
    table = json.loads((ROOT / "qualification/s28a_tables.json").read_bytes())
    assert hashlib.sha256((ROOT / "qualification/s28a_guard.py").read_bytes()).hexdigest() == (
        table["gate_sha256"]
    )
    for path, expected in table["frozen_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    manifest = json.loads((ROOT / "qualification/s28a_expected_unimportable.json").read_bytes())
    for row in manifest["modules"].values():
        path = row["path"]
        original = subprocess.check_output(["git", "show", f"{baseline}:{path}"], cwd=ROOT)
        assert (ROOT / path).read_bytes() == original
        assert hashlib.sha256(original).hexdigest() == row["sha256"]
