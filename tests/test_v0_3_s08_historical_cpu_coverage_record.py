"""Evidence bindings for the S-08 historical CPU coverage candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_23_s08_historical_cpu_coverage.json"
REWORK_RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_25_s08_rework_candidate.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
MANIFEST = ROOT / "tests/fixtures/historical_cpu_coverage_v1/manifest.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s08_record_binds_exact_release_sources_and_fixture_trees() -> None:
    record = _read(RECORD)
    rework = _read(REWORK_RECORD)
    manifest = _read(MANIFEST)
    assert record["classification"] == "PASS_V0_3_S08_LOCAL_CPU_COVERAGE_CANDIDATE"
    assert record["git_identity"]["implementation_commit"] == (
        "8bf898f5b10056c0b70e1d3acbeaad2c0d8c3851"
    )
    assert record["roadmap"] == {
        "decision": "D-0010",
        "plan_step": "S-08",
        "issues": ["A2-25", "N5"],
        "next_request": "R-0019",
        "independent_review_pending": True,
    }
    # Successor 23 remains an immutable record of the pre-review candidate.
    assert record["fixture_identity"]["generator_sha256"] == (
        "344ea581a5bfb3ddcecc5b9b4cb81c3eac4327d3466c4017023d5c14271c9f87"
    )
    assert rework["fixture_identity"]["generator_sha256"] == _sha256(
        ROOT / manifest["generator"]["path"]
    )
    assert rework["fixture_identity"]["released_sources"] == manifest["released_sources"]
    assert rework["fixture_identity"]["numerical_build_fingerprint"] == (
        manifest["generation_environment"]["numerical_build_fingerprint"]
    )
    assert rework["fixture_identity"]["tolerance_policy"] == (
        manifest["continuation_tolerance_policy"]
    )
    for name, evidence in rework["fixture_identity"]["cases"].items():
        for key, value in evidence.items():
            assert manifest["cases"][name][key] == value


def test_s08_record_states_the_rc1_boundary_without_reopening_lifting() -> None:
    record = _read(RECORD)
    rc1 = record["adjudication"]["A2-25"]["rc1"]
    rc2 = record["adjudication"]["A2-25"]["rc2"]
    assert rc1["migration_kind"] == "state_only_migration"
    assert rc1["trajectory_equivalent"] is False
    assert rc1["continuation_exact"] is False
    assert rc1["physical_max_abs_difference"] > 0.0
    assert rc1["spectral_max_abs_difference"] > 0.0
    assert rc2["migration_kind"] == "exact_legacy_upgrade"
    assert rc2["trajectory_equivalent"] is True
    assert rc2["continuation_exact"] is True
    lifted = record["adjudication"]["N5"]
    assert lifted["source_unchanged"] is True
    assert lifted["target_absent"] is True
    assert lifted["public_lifting_reopened"] is False
    assert record["scope"]["public_lifting_reopened"] is False


def test_s08_registry_claims_only_the_two_cpu_coverage_items() -> None:
    record = _read(RECORD)
    rework = _read(REWORK_RECORD)
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert record["issue_state"]["claimed_fixed"] == ["A2-25", "N5"]
    assert rework["issue_state"]["claimed_fixed"] == [
        "A2-25",
        "N5",
        "V-31",
        "V-32",
    ]
    for issue_id in rework["issue_state"]["claimed_fixed"]:
        assert issues[issue_id]["status"] == "claimed_fixed"
        assert issues[issue_id]["fixed_in"] == rework["issue_state"]["fixed_in"][issue_id]
        assert "verified_by" not in issues[issue_id]
    assert issues["A3-18"]["status"] == "coverage_gap"


def test_s08_successors_are_forward_only_and_complete() -> None:
    for source in (RECORD, REWORK_RECORD):
        for relative, successor in _read(source)["historical_successor_sources"].items():
            assert resolve_s03_successor(
                relative, successor["historical_sha256"]
            ) == _sha256(ROOT / relative)


def test_s08_record_authorizes_no_external_action() -> None:
    record = _read(REWORK_RECORD)
    assert record["authorization"]["local_s08_authorized"] is True
    assert all(
        value is False
        for key, value in record["authorization"].items()
        if key != "local_s08_authorized"
    )
    for key in (
        "push_executed",
        "HPCC_executed",
        "GPU_executed",
        "H100_executed",
        "skip_or_xfail_added",
        "threshold_changed",
        "production_default_changed",
    ):
        assert record["scope"][key] is False
