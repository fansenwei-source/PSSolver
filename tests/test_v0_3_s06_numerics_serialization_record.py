"""Bindings for the S-06 numerics and serialization candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_3_0_s03_successor_12_"
    "s06_numerics_serialization.json"
)
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
IMPLEMENTATION = "bbb7077fa0f0cc7206a0939f6bff34ddc1ff823a"
ISSUES = {"C1-05", "C2-18", "S2-17", "V-27"}


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s06_record_binds_the_roadmap_and_runtime_sources() -> None:
    record = _read(RECORD)
    assert record["classification"] == (
        "V0_3_S06_NUMERICS_SERIALIZATION_QUALIFICATION_CANDIDATE"
    )
    assert record["git_identity"]["implementation_commit"] == IMPLEMENTATION
    assert record["roadmap"]["decision"] == "D-0010"
    assert record["roadmap"]["plan_step"] == "S-06"
    assert set(record["roadmap"]["issue_ids"]) == ISSUES
    for relative, expected in record["runtime_sources"].items():
        assert _sha256(ROOT / relative) == resolve_s03_successor(relative, expected)


def test_s06_registry_advances_all_review_items_through_r0016() -> None:
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    claimed = {
        issue_id
        for issue_id, item in issues.items()
        if item["status"] == "claimed_fixed"
    }
    assert claimed == {"A2-25", "N5"}
    for issue_id in ISSUES | {"V-28", "V-29"}:
        expected = (
            "f6da20f71ed2b7ae63ab07b19103eb6c29a95829"
            if issue_id in {"S2-17", "V-28", "V-29"}
            else IMPLEMENTATION
        )
        assert issues[issue_id]["fixed_in"] == expected
        assert issues[issue_id]["status"] == "verified"
        assert issues[issue_id]["verified_by"] == "R-0016"
        assert issues[issue_id]["target_release"] == "0.3"
        assert "R-0016" in issues[issue_id]["notes"]


def test_s06_contract_preserves_compatibility_boundaries() -> None:
    record = _read(RECORD)
    claims = record["claims"]
    assert claims["C1-05"]["public_floor_real_dtype_epsilons"] == 8
    assert claims["C1-05"][
        "represented_state_residual_verified_before_tolerance_acceptance"
    ] is True
    assert claims["S2-17"]["ordinary_fixed_work_unchanged"] is True
    assert claims["S2-17"]["breakdown_above_floor_remains_fail_closed"] is True
    assert claims["C2-18"]["exact_public_subclass_preserved"] is True
    assert claims["V-27"]["current_checkout_root_comparison_removed"] is True


def test_s06_successors_are_forward_only_and_complete() -> None:
    record = _read(RECORD)
    for relative, successor in record["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative,
            successor["historical_sha256"],
        ) == _sha256(ROOT / relative)


def test_s06_candidate_authorizes_no_external_action() -> None:
    record = _read(RECORD)
    assert record["scope"]["runtime_changed"] is True
    assert record["scope"]["public_minimum_tolerance_changed"] is False
    assert record["scope"]["ordinary_fixed_work_path_changed"] is False
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_implementation_authorized"
    )
    assert record["authorization"]["local_implementation_authorized"] is True
    assert record["roadmap"]["next_request"] == "R-0014"


def test_s06_local_qualification_is_bound_for_independent_review() -> None:
    validation = _read(RECORD)["validation"]
    assert validation["archive"] == (
        "/home/fansenwei/"
        "pssolver_v0_3_s06_pressure_serialization_bbb7077_20261022_v1"
    )
    assert validation["archive_complete_present"] is True
    assert validation["archive_checksums_entries"] == 23
    assert validation["archive_checksums_sha256"] == (
        "ef20d7d4be841d6821fb2932ae1355c7ebbfd397919640e8ad673df4e553f13b"
    )
    assert validation["environment_clone_pruned"] is True
    assert validation["focused_pytest"] == {
        "passed": 128,
        "failed": 0,
        "skipped": 0,
    }
    assert validation["full_cpu_suite"] == {
        "passed": 4448,
        "failed": 0,
        "skipped": 1,
        "deselected": 7,
        "subtests_passed": 8,
    }
    assert validation["installed_wheel"]["passed"] == 8
    assert validation["installed_wheel"]["pip_check"] == "PASS"
    assert validation["installed_wheel"]["source_shadow_import"] is False
    assert validation["qualification_complete"] is True
    assert validation["eligible_for_independent_review"] is True
