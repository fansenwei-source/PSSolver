"""Evidence-only bindings for the R-0037 S-10 final review import."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_60_r0037_import.json"
RESULT = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0037.result.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0037_import_binds_the_exact_independent_pass() -> None:
    from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration

    record = _read(RECORD)
    assert_pytest_migration(
        "tests/test_v0_3_s10_r0037_import.py::"
        "test_r0037_import_binds_the_exact_independent_pass"
    )
    assert record["plan_step"] == "S-10"
    assert record["sync_result"] == {
        "id": "R-0037",
        "path": "PSSolver_sync/results/R-0037.result.json",
        "sha256": "878feb76d315a9292ebbb1883a3c4699771ffa354d585a444c0ac2b7bfa57c95",
        "sync_commit": "dfe96f327ee7e8c944885f5b5da534a8a26a3fa6",
        "classification": "PASS",
    }
    assert len(record["sync_result"]["sha256"]) == 64
    archive = record["independent_archive"]
    assert archive["complete_present"] is True
    assert archive["checksums_entries"] == 51
    for field in ("checksums_sha256", "final_report_sha256", "summary_sha256"):
        assert len(archive[field]) == 64


def test_r0037_import_records_the_terminal_v46_update_without_a_live_pin() -> None:
    record = _read(RECORD)
    assert record["registry_import"] == {
        "V-46": {
            "status": "verified",
            "fixed_in": "5254ff32af5d68471f90f5fa8055d040598090f0",
            "verified_by": "R-0037",
            "fix_commit_removed": True,
        }
    }
    assert record["forward_durable_guards"]["prefix_only"] is True
    assert record["forward_durable_guards"][
        "live_registry_status_is_not_pinned_by_historical_tests"
    ] is True


def test_r0037_import_is_a_forward_only_complete_successor() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        live = _sha256(ROOT / relative)
        assert resolve_s03_successor(relative, successor["historical_sha256"]) == live
        assert resolve_s03_successor(relative, successor["current_sha256"]) == live


def test_r0037_import_preserves_records_only_scope() -> None:
    record = _read(RECORD)
    assert record["scope"]["records_and_tests_only"] is True
    assert all(
        value is False
        for key, value in record["scope"].items()
        if key != "records_and_tests_only"
    )
    assert record["authorization"]["local_import"] is True
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_import"
    )
    assert record["local_archive_refs"]["must_not_be_pushed"] is True
    assert record["next_plan_step"] == "S-11"
