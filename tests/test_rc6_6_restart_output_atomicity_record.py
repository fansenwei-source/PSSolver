"""Closure-record checks for RC6-6 restart output atomicity."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_6_restart_output_atomicity.json"
RC6_FREEZE_RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_source_freeze.json"
FINAL_RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_0.json"
V021_N1_RECORD = ROOT / (
    "notes/PSSolver_v0_2_1_n1_checkpoint_nfs_implementation.json"
)
V021_RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_1.json"
V022_V17_RECORD = ROOT / (
    "notes/PSSolver_v0_2_2_v17_checkpoint_symlink_parent.json"
)
V022_RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_2.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _freeze_successor(relative: str, historical: str) -> str:
    registration = json.loads(RC6_FREEZE_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
    registration = json.loads(FINAL_RELEASE_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
    registration = json.loads(V021_N1_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
    registration = json.loads(V021_RELEASE_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
    registration = json.loads(V022_V17_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
    registration = json.loads(V022_RELEASE_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
    return historical


def test_rc6_6_record_binds_implementation_and_regression_sources():
    record = _record()
    assert record["status"] == "implementation_complete_cpu_verified"
    assert record["classification"] == (
        "PASS_RC6_6_RESTART_OUTPUT_ATOMICITY_CPU_CLOSURE"
    )
    assert record["git_identity"]["implementation_commit"] == (
        "80c6e17db5838ef4f142add4955221637efcdf3e"
    )
    assert record["git_identity"]["record_commit_is_qualified_source"] is False
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(relative) == expected


def test_rc6_6_contract_covers_all_public_restart_workflows():
    contract = _record()["contract"]
    for key in (
        "restart_preflight_precedes_missing_output_materialization",
        "rejected_missing_restart_target_remains_absent",
        "periodic_restart_rejection_covered",
        "channel_complete_stress_restart_rejection_covered",
        "plane_restart_rejection_covered",
        "legacy_channel_restart_rejection_covered",
        "successful_restart_creates_output_before_publication",
        "fresh_run_output_behavior_preserved",
        "preexisting_nonempty_output_rejection_preserved",
    ):
        assert contract[key] is True
    for key in (
        "checkpoint_schema_changed",
        "functional_api_version_changed",
        "runtime_defaults_changed",
    ):
        assert contract[key] is False


def test_rc6_6_registers_exact_forward_only_successors():
    successors = _record()["historical_successor_sources"]
    assert len(successors) == 54
    assert set(_record()["implementation_files"]) < set(successors)
    assert set(_record()["regression_tests"]) < set(successors)
    assert "CHANGELOG.md" in successors
    assert "tests/test_rc6_5_checkpoint_control_flow_record.py" in successors
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert registration["introduced_by_phase"] == "RC6-6"
        assert registration["reason"]
        assert _sha256(relative) == _freeze_successor(
            relative, registration["current_sha256"]
        )


def test_rc6_6_scope_is_local_cpu_only_and_numerically_neutral():
    scope = _record()["scope"]
    assert scope["restart_output_materialization_order_changed"] is True
    for key in (
        "scientific_equations_changed",
        "integrator_changed",
        "transform_algorithm_changed",
        "gradient_algorithm_changed",
        "checkpoint_schema_changed",
        "runtime_default_changed",
        "production_default_changed",
        "PSSolver_Control_modified",
        "nematics3d_modified",
        "H100_required",
        "H100_executed",
        "push_executed",
    ):
        assert scope[key] is False


def test_rc6_6_authorizes_only_the_next_local_phase():
    assert _record()["authorization"] == {
        "rc6_6_complete": True,
        "eligible_for_rc6_7": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }
