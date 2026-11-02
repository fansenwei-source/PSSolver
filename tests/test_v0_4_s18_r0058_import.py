"""Repository-only, forward-resolving contract for the R-0058 import."""

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor

ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_110_r0058_import.json"


def _read():
    return json.loads(RECORD.read_text())


def test_r0058_import_binds_independent_result_and_archive():
    record = _read()
    assert record["classification"] == "R0058_PASS_IMPORTED_S18_R5_COMPLETE_PENDING_PUSH"
    assert record["review"]["result_sha256"] == "2634066cf00a33485dde4c0193f201413cd5cd7663f6e954822e169a6194021c"
    assert record["review"]["archive"]["checksums_entries"] == 375
    assert record["review"]["archive"]["checksums_sha256"] == "cf3994d2246438dcae3708ec6c79c34b5d84e703b3fff1e7a1865ced6035a640"
    assert record["review"]["classification"] == "PASS"


def test_r0058_import_snapshot_preserves_fix_and_qualification_sources():
    record = _read()
    assert record["registry_snapshot"]["V-61"] == {
        "status": "verified", "verified_by": "R-0058",
        "fixed_in": "3c519ccb63d523549cce161b05b62803c727c978",
    }
    assert record["git_identity"]["reviewed_candidate"] == "77a6eb1d92b414579e3b67aaa565f2526bd7a3de"
    assert record["git_identity"]["record_commit_is_qualified_source"] is False
    assert record["new_findings"] == []


def test_successor_110_sources_resolve_forward_without_external_reads():
    record = _read()
    bindings = {p: v["current_sha256"] for p, v in record["historical_successor_sources"].items()}
    bindings.update(record["introduced_sources"])
    for relative, expected in bindings.items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == resolve_s03_successor(relative, expected)


def test_r0058_forward_erratum_records_ignored_child_bytecode():
    correction = _read()["pruning_erratum"]
    assert correction["original_archive_modified"] is False
    assert correction["ignored_pyc_per_clone"] == {"conda": 116, "a2": 116, "blocked": 116}
    assert correction["cause"] == "python -I ignores PYTHONDONTWRITEBYTECODE and PYTHONPYCACHEPREFIX"
    assert all(len(item["sha256"]) == 64 for item in correction["sources"])


def test_r0058_closes_r5_only_without_new_execution_authority():
    record = _read()
    assert record["remaining_batch"] == "R1 (shadow registry first)"
    assert record["aggregate_r1_r9_closed"] is False
    assert not any(record["scope"].values())
