"""Final local bindings for the S-07 rework."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_21_s07_rework_close.json"
CLAIMS = {"A1-07", "S1-15/NEW-13", "S1-04", "OBS-4", "B30", "V-26", "V-30"}


def _read() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_s07_rework_close_binds_exact_sources_and_issue_state() -> None:
    record = _read()
    assert record["classification"] == "PASS_V0_3_S07_REWORK_LOCAL_QUALIFICATION"
    assert record["git_identity"]["runtime_source"] == (
        "8aceaab446e57a13866090ec06ecdf30e33ef490"
    )
    assert record["git_identity"]["qualification_support_source"] == (
        "b93037a966da65d1fba3ec6ab1f226c2d71a1ed8"
    )
    assert set(record["issue_state"]["claimed_fixed"]) == CLAIMS
    assert record["issue_state"]["verified"] == []


def test_s07_rework_close_records_all_local_gates() -> None:
    validation = _read()["validation"]
    assert validation["focused_source_pytest"] == {"passed": 32, "failed": 0, "skipped": 0}
    assert validation["installed_wheel_pytest"] == {"passed": 7, "failed": 0, "skipped": 0}
    assert validation["full_cpu_suite"]["passed"] == 4536
    assert validation["full_cpu_suite"]["failed"] == 0
    assert validation["full_cpu_suite"]["skipped"] == 1
    assert validation["full_cpu_suite"]["deselected"] == 7
    assert validation["full_cpu_suite"]["environment_is_hardlink_clone"] is True
    assert validation["qualification_complete"] is True
    assert validation["eligible_for_independent_review"] is True


def test_s07_rework_close_authorizes_no_external_action() -> None:
    record = _read()
    assert record["roadmap"]["independent_review_pending"] is True
    scope = record["scope"]
    assert all(value is False for value in scope.values())
