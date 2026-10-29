"""Bindings for the passing R-0030 review and records-only V-42 fix."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_45_r0030_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
RESULT = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0030.result.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0030_result_and_independent_archive_are_bound() -> None:
    from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration

    record = _read(RECORD)
    assert_pytest_migration(
        "tests/test_v0_3_s09_r0030_import.py::"
        "test_r0030_result_and_independent_archive_are_bound"
    )
    assert record["classification"] == (
        "PASS_R0030_DIAGNOSTIC_ENVIRONMENT_REVIEW_IMPORTED"
    )
    assert len(record["sync_result"]["sha256"]) == 64
    assert record["sync_result"]["classification"] == "PASS"
    archive = record["independent_archive"]
    assert archive["complete_present"] is True
    assert archive["checksums_entries"] == 410
    assert len(archive["checksums_sha256"]) == 64


def test_r0030_record_preserves_its_historical_v41_v42_boundary() -> None:
    record = _read(RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert record["registry_import"] == {
        "registered": ["V-42"],
        "v41_status": "registered",
        "v42_status": "registered",
        "v42_review_target": "R-0031",
        "s09_complete": False,
    }
    assert issues["V-41"]["status"] in {"registered", "accepted"}
    assert issues["V-42"]["status"] in {"registered", "verified"}
    assert issues["V-42"]["source_report"] == (
        "PSSolver_sync/results/R-0030.result.json"
    )


def test_r0030_citable_identities_match_the_passing_result() -> None:
    from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration

    record = _read(RECORD)
    assert_pytest_migration(
        "tests/test_v0_3_s09_r0030_import.py::"
        "test_r0030_citable_identities_match_the_passing_result"
    )
    assert set(record["citable_identities"]) >= {"runtime", "support_commit"}
    assert record["citable_identities"]["runtime"] == (
        "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
    )


def test_r0030_import_successors_are_forward_only_and_complete() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        actual = _sha256(ROOT / relative)
        assert resolve_s03_successor(relative, successor["historical_sha256"]) == actual
        assert resolve_s03_successor(relative, successor["current_sha256"]) == actual


def test_r0030_import_changes_no_runtime_and_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    scope = record["record_scope"]
    assert scope["evidence_only"] is True
    assert all(value is False for key, value in scope.items() if key != "evidence_only")
    assert record["authorization"] == {
        "local_evidence_import": True,
        "push": False,
        "hpcc_write": False,
        "slurm_or_h100": False,
        "tag_or_release": False,
    }
