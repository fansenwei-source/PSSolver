"""Evidence bindings for the S-08 historical CPU coverage candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_3_0_s03_successor_23_s08_historical_cpu_coverage.json"
REGISTRY = ROOT / "notes/pssolver_v0_2_x_open_issues.json"
MANIFEST = ROOT / "tests/fixtures/historical_cpu_coverage_v1/manifest.json"


def _read(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_s08_record_binds_exact_release_sources_and_fixture_trees() -> None:
    record = _read(RECORD)
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
    assert record["fixture_identity"]["generator_sha256"] == _sha256(
        ROOT / manifest["generator"]["path"]
    )
    assert record["fixture_identity"]["released_sources"] == manifest["released_sources"]
    for name, evidence in record["fixture_identity"]["cases"].items():
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
    issues = {item["id"]: item for item in _read(REGISTRY)["issues"]}
    assert record["issue_state"]["claimed_fixed"] == ["A2-25", "N5"]
    for issue_id in record["issue_state"]["claimed_fixed"]:
        assert issues[issue_id]["status"] == "claimed_fixed"
        assert issues[issue_id]["fixed_in"] == record["issue_state"]["fixed_in"]
        assert "verified_by" not in issues[issue_id]
    assert issues["A3-18"]["status"] == "coverage_gap"


def test_s08_successors_are_forward_only_and_complete() -> None:
    for relative, successor in _read(RECORD)["historical_successor_sources"].items():
        assert resolve_s03_successor(
            relative, successor["historical_sha256"]
        ) == _sha256(ROOT / relative)


def test_s08_record_authorizes_no_external_action() -> None:
    record = _read(RECORD)
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
