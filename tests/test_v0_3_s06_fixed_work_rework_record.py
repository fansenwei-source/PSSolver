"""Bindings for the S-06 option-B fixed-work pressure rework."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_3_0_s03_successor_14_s06_fixed_work_rework.json"
)
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
IMPLEMENTATION = "4c3a2d8bca191b6aa361c37920826e63582a4774"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s06_rework_binds_runtime_and_option_b() -> None:
    record = _read(RECORD)
    assert record["git_identity"]["implementation_commit"] == IMPLEMENTATION
    assert record["roadmap"]["plan_step"] == "S-06"
    assert set(record["roadmap"]["reworked_issue_ids"]) == {"S2-17", "V-28"}
    contract = record["contract"]
    assert contract["selected_option"] == "B"
    assert contract["latest_at_or_below_floor_state_preserved"] is True
    assert contract["represented_return_residual_recomputed"] is True
    assert contract["represented_return_residual_must_not_exceed_floor"] is True
    assert contract["option_a_stop_at_first_floor_crossing_used"] is False
    assert contract["unchanged_floor_real_dtype_epsilons"] == 8
    assert set(contract["paths"]) == {
        "production",
        "functional_transpose",
        "implicit_primal",
        "unrolled_oracle",
    }
    assert contract["regression_budgets"] == [200, 250, 300, 400]
    for relative, expected in record["runtime_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative, expected
        )


def test_s06_rework_items_are_verified_at_exact_fix_sources() -> None:
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    claimed = {
        issue_id
        for issue_id, item in issues.items()
        if item["status"] == "claimed_fixed"
    }
    assert claimed == {"V-39", "V-40"}
    for issue_id in ("A1-07", "S1-15/NEW-13", "S1-04", "OBS-4", "B30", "V-26", "V-30"):
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["verified_by"] == "R-0018"
    for issue_id in ("S2-17", "V-28", "V-29"):
        assert issues[issue_id]["fixed_in"] == (
            "f6da20f71ed2b7ae63ab07b19103eb6c29a95829"
        )
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["verified_by"] == "R-0016"
        assert "R-0016" in issues[issue_id]["notes"]
    for issue_id in ("C1-05", "C2-18", "V-27"):
        assert issues[issue_id]["fixed_in"] == (
            "bbb7077fa0f0cc7206a0939f6bff34ddc1ff823a"
        )
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["verified_by"] == "R-0016"


def test_s06_rework_successors_are_forward_only_and_complete() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_s06_rework_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["scope"]["runtime_changed"] is True
    assert record["scope"]["public_minimum_tolerance_changed"] is False
    assert record["scope"]["threshold_relaxed"] is False
    assert record["scope"]["skip_or_xfail_added"] is False
    assert record["authorization"]["local_implementation_authorized"] is True
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_implementation_authorized"
    )
    validation = record["validation"]
    assert validation["status"] == (
        "PASS_V0_3_S06_FIXED_WORK_OPTION_B_LOCAL_QUALIFICATION"
    )
    assert validation["archive_checksums_entries"] == 17
    assert validation["archive_checksums_sha256"] == (
        "5f2a2deba478bd0759fd587b3f393c8860112132c03b67195af3a832ac4019a0"
    )
    assert validation["archive_complete_present"] is True
    assert validation["environment_clone_method"] == "cp -al"
    assert validation["environment_clones_pruned"] is True
    assert validation["focused_pytest"]["passed"] == 101
    assert validation["full_cpu_suite"] == {
        "passed": 4489,
        "failed": 0,
        "skipped": 1,
        "deselected": 7,
        "subtests_passed": 8,
        "log_sha256": "1c16dd13828b15fbf6fa943786ed18a3b725b0bef802f1404b826fa9facbb4b7",
        "junit_sha256": "082524be2688e4bdb5c9652cf340a57aca664f65c2854084e93cbcc8a6138a76",
    }
    assert validation["installed_wheel"]["passed"] == 40
    assert validation["installed_wheel"]["pip_check"] == "PASS"
    assert validation["installed_wheel"]["source_shadow_import"] is False
    assert validation["stress_scan"][
        "silent_acceptable_represented_residual_above_1e_3"
    ] == 0
    assert validation["qualification_complete"] is True
    assert validation["eligible_for_independent_review"] is True
