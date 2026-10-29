"""Evidence-only bindings for the R-0036 S-10 integration rework."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_59_r0036_rework.json"
RESULT = Path("/home/fansenwei/Desktop/Develop/PSSolver_sync/results/R-0036.result.json")


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_r0036_rework_binds_the_exact_independent_review() -> None:
    from benchmarks.verify_v0_4_local_evidence import assert_pytest_migration

    record = _read(RECORD)
    assert_pytest_migration(
        "tests/test_v0_3_s10_r0036_rework.py::"
        "test_r0036_rework_binds_the_exact_independent_review"
    )
    assert record["plan_step"] == "S-10"
    assert record["sync_result"] == {
        "id": "R-0036",
        "path": "PSSolver_sync/results/R-0036.result.json",
        "sha256": "d7572772516ba1d6adbe0e34359cb1a8d9778425cdac47dcf75977964288580e",
        "sync_commit": "57c8dce915dc033480e49ceebed3ad09f19a9322",
        "classification": "REWORK_REQUIRED",
    }
    assert len(record["sync_result"]["sha256"]) == 64
    archive = record["independent_archive"]
    assert archive["complete_present"] is True
    assert archive["checksums_entries"] == 57
    for field in ("checksums_sha256", "final_report_sha256", "summary_sha256"):
        assert len(archive[field]) == 64


def test_r0036_registry_updates_and_new_finding_are_exact() -> None:
    record = _read(RECORD)
    assert record["registry_import"] == {
        "V-33": {
            "status": "verified",
            "verified_by": "R-0033",
            "reviewed_original": "61838af7910d955b27268abd4a79e4d32f235f58",
            "fixed_in": "a54074013e5f0fb9bfe278baf944f805bf86eef4",
        },
        "V-43": {
            "status": "verified",
            "verified_by": "R-0033",
            "reviewed_original": "61838af7910d955b27268abd4a79e4d32f235f58",
            "fixed_in": "a54074013e5f0fb9bfe278baf944f805bf86eef4",
        },
        "V-46": {
            "status": "claimed_fixed",
            "source_report": "PSSolver_sync/results/R-0036.result.json",
            "target_release": "0.3.0 (S-10 integration)",
        },
    }


def test_r0036_rework_is_a_forward_only_complete_successor() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        live = _sha256(ROOT / relative)
        assert resolve_s03_successor(relative, successor["historical_sha256"]) == live
        assert resolve_s03_successor(relative, successor["current_sha256"]) == live


def test_r0036_rework_preserves_local_only_scope() -> None:
    record = _read(RECORD)
    assert record["replay_identity_correction"]["expected_mapping_entries"] == 10
    assert record["replay_identity_correction"]["archive_ref_must_not_be_pushed"] is True
    assert record["replay_identity_correction"][
        "fresh_clone_gate_required_after_replay_rebase_or_amend"
    ] is True
    assert record["scope"]["records_and_tests_only"] is True
    assert all(
        value is False
        for key, value in record["scope"].items()
        if key != "records_and_tests_only"
    )
    assert record["authorization"]["local_rework"] is True
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_rework"
    )
    assert record["next_request"] == "R-0037"
