"""Bindings for the passing R-0035 S-09 closure review import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_52_r0035_import.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
RESULT = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0035.result.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0035_result_and_independent_archive_are_bound() -> None:
    from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration

    record = _read(RECORD)
    assert_pytest_migration(
        "tests/test_v0_3_s09_r0035_import.py::"
        "test_r0035_result_and_independent_archive_are_bound"
    )
    assert record["classification"] == "PASS_R0035_S09_CLOSURE_REVIEW_IMPORTED"
    assert len(record["sync_result"]["sha256"]) == 64
    assert record["sync_result"]["classification"] == "PASS"
    archive = record["independent_archive"]
    assert archive["complete_present"] is True
    assert archive["checksums_entries"] == 50
    for field in ("checksums_sha256", "final_report_sha256", "summary_sha256"):
        assert len(archive[field]) == 64


def test_r0035_verifies_v45_at_the_actual_fix_commit() -> None:
    record = _read(RECORD)
    issue = {item["id"]: item for item in _read(REGISTRY)["issues"]}["V-45"]
    expected = {
        "status": "verified",
        "fixed_in": "e052f5b87e1424fbd8d9bd097a893469f9c72658",
        "verified_by": "R-0035",
    }
    assert record["registry_import"]["V-45"] == expected
    assert {key: issue[key] for key in expected} == expected


def test_r0035_binds_the_narrow_d0013_closure_conclusion() -> None:
    conclusion = _read(RECORD)["reviewed_conclusion"]
    assert conclusion["s09_complete"] is True
    assert conclusion["noninterrupting_recompute_sha256"] == (
        "1757374241e033c77a5fbe0b31df29e6f33a0dcae3e777c4ae45754d5927df01"
    )
    assert conclusion["recorded_failures"] == [
        "channel_legacy/channel_R512x40x40/float32/peak_reserved",
        "channel_compiled_v2/channel_R512x40x40/float32/peak_reserved",
    ]
    assert conclusion["fail_fast_stopped_at_first_failure"] is True
    assert conclusion["five_negative_controls_failed_closed"] is True
    assert conclusion["bounded_exception_decision"] == "D-0013"


def test_r0035_import_successors_are_forward_only_and_complete() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        actual = _sha256(ROOT / relative)
        assert resolve_s03_successor(relative, successor["historical_sha256"]) == actual
        assert resolve_s03_successor(relative, successor["current_sha256"]) == actual


def test_r0035_import_is_records_only_and_authorizes_no_external_action() -> None:
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
