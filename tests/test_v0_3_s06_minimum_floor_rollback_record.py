"""Bindings for the refined S-06 minimum-floor rollback candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_3_0_s03_successor_16_s06_minimum_floor_rollback.json"
)
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
IMPLEMENTATION = "f6da20f71ed2b7ae63ab07b19103eb6c29a95829"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s06_minimum_floor_contract_is_exact() -> None:
    record = _read(RECORD)
    assert record["git_identity"]["implementation_commit"] == IMPLEMENTATION
    assert record["roadmap"]["plan_step"] == "S-06"
    assert record["roadmap"]["review_source"] == "R-0015"
    contract = record["contract"]
    assert contract["rollback_candidate"] == (
        "minimum_recurrence_residual_at_or_below_floor"
    )
    assert contract[
        "actual_rollback_requires_strict_represented_residual_postcondition"
    ] is True
    assert contract["normal_completion_preserves_pre_s06_acceptance"] is True
    assert contract["normal_completion_maximum_represented_floor_multiple"] == 2.0
    assert contract["saved_strict_candidate_preferred_before_completion_margin"] is True
    assert contract["option_a_stop_at_first_floor_crossing_used"] is False


def test_s06_minimum_floor_runtime_and_registry_are_bound() -> None:
    record = _read(RECORD)
    for relative, expected in record["runtime_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(
            relative,
            expected,
        )
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    claimed = {
        issue_id
        for issue_id, item in issues.items()
        if item["status"] == "claimed_fixed"
    }
    assert claimed == set()
    assert set(record["roadmap"]["claimed_issue_ids"]) == {
        "C1-05", "C2-18", "S2-17", "V-27", "V-28", "V-29"
    }
    for issue_id in ("S2-17", "V-28", "V-29"):
        assert issues[issue_id]["fixed_in"] == IMPLEMENTATION
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["verified_by"] == "R-0016"


def test_s06_minimum_floor_scans_record_fail_closed_counts() -> None:
    validation = _read(RECORD)["validation"]
    pressure = validation["pressure_scan"]
    assert pressure["runs"] == 2160
    assert pressure["fail_closed"] == 0
    assert pressure["acceptable_represented_residual_above_floor"] == 0
    assert pressure["maximum_acceptable_represented_residual"] <= pressure[
        "precision_floor"
    ]
    crossing = validation["floor_crossing_probe"]
    assert crossing["runs"] == 360
    assert crossing["fail_closed"] == 0
    assert crossing["maximum_accepted_floor_multiple"] <= 2.0


def test_s06_minimum_floor_successors_are_forward_only_and_complete() -> None:
    for relative, successor in _read(RECORD)[
        "historical_successor_sources"
    ].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_s06_minimum_floor_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["scope"]["runtime_changed"] is True
    assert record["scope"]["public_minimum_tolerance_changed"] is False
    assert record["scope"]["skip_or_xfail_added"] is False
    assert record["authorization"]["local_implementation_authorized"] is True
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_implementation_authorized"
    )
    validation = record["validation"]
    assert validation["status"] == (
        "PASS_V0_3_S06_MINIMUM_FLOOR_ROLLBACK_LOCAL_QUALIFICATION"
    )
    assert validation["archive_checksums_entries"] == 28
    assert validation["archive_checksums_sha256"] == (
        "fbaf4db2d57c8269fb150501398f7e0bd26bb3d0434b72be8c3b3750ab11a573"
    )
    assert validation["archive_complete_present"] is True
    assert validation["environment_clone_method"] == "cp -al"
    assert validation["environment_clones_pruned"] is True
    assert validation["focused_pytest"]["passed"] == 154
    assert validation["full_cpu_suite"]["passed"] == 4502
    assert validation["installed_wheel"]["passed"] == 43
    assert validation["qualification_complete"] is True
    assert validation["eligible_for_independent_review"] is True
