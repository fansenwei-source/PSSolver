"""Closure-record checks for RC6-4 public execution failures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_4_channel_failure_contract.json"
RC6_R6_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_6_restart_output_atomicity.json"
)
RC6_R5_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_5_checkpoint_control_flow.json"
)
RC6_FREEZE_RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_source_freeze.json"
FINAL_RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_0.json"
V021_N1_RECORD = ROOT / (
    "notes/PSSolver_v0_2_1_n1_checkpoint_nfs_implementation.json"
)
V021_RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_1.json"
V022_V17_RECORD = ROOT / (
    "notes/PSSolver_v0_2_2_v17_checkpoint_symlink_parent.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _rc6_5_successor(relative: str, historical: str) -> str:
    registration = json.loads(RC6_R5_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        return registration["current_sha256"]
    return historical


def _rc6_6_successor(relative: str, historical: str) -> str:
    registration = json.loads(RC6_R6_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
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
    return historical


def test_rc6_4_record_binds_implementation_and_regression_sources():
    record = _record()
    assert record["status"] == "implementation_complete_cpu_verified"
    assert record["classification"] == (
        "PASS_RC6_4_CHANNEL_FAILURE_AND_METADATA_CPU_CLOSURE"
    )
    assert record["git_identity"]["implementation_commit"] == (
        "4660e1d4051102613ce8b1c3df707bac672f4827"
    )
    assert record["git_identity"]["record_commit_is_qualified_source"] is False
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            expected = _rc6_5_successor(relative, expected)
            assert _sha256(relative) == _rc6_6_successor(relative, expected)


def test_rc6_4_contract_preserves_types_and_finalizes_started_runs():
    contract = _record()["contract"]
    assert contract["pressure_error_type_preserved"] == (
        "ChannelPressureConvergenceError"
    )
    assert contract["pressure_error_public_base"] == "SimulationExecutionError"
    assert contract["started_plane_failure_status"] == "failed"
    assert contract["started_legacy_channel_failure_status"] == "failed"
    for key in (
        "pressure_details_mapping_compatibility_preserved",
        "functional_convergence_dual_hierarchy_preserved",
        "failure_metadata_is_structured",
        "workflow_completed_step_clock_persisted",
        "metadata_write_is_atomic",
    ):
        assert contract[key] is True
    assert contract["complete_marker_written_on_failure"] is False
    assert contract["nonfinite_arrays_published"] is False


def test_rc6_4_registers_exact_forward_only_successors():
    successors = _record()["historical_successor_sources"]
    assert set(successors) == {
        "CHANGELOG.md",
        "pssolver/errors.py",
        "pssolver/linear_solvers/stokes/channel_no_slip.py",
        "pssolver/workflows/channel_active_nematics.py",
        "pssolver/workflows/plane_beris_edwards.py",
        "tests/test_phase7_channel_local_closure.py",
        "tests/test_rc447_pressure_bounded_operator_recovery.py",
        "tests/test_rc447_python310_compatibility.py",
        "tests/test_rc455_public_robustness.py",
        "tests/test_rc456_tf32_execution_scope.py",
        "tests/test_rc457_installed_consumer_impact.py",
        "tests/test_rc457_installed_consumer_recovery.py",
        "tests/test_rc458_h100_closure_record.py",
        "tests/test_rc4_lifting_checkpoint_integration_record.py",
        "tests/test_rc4_plane_checkpoint_identity_record.py",
        "tests/test_rc4_post_release_findings.py",
        "tests/test_rc4_pressure_contract_plan.py",
        "tests/test_rc4_pressure_policy_implementation_record.py",
        "tests/test_rc4_pressure_public_contract.py",
        "tests/test_rc5_g1_checkpoint_glibc_compatibility_record.py",
        "tests/test_rc5_g1_historical_audit_portability.py",
        "tests/test_rc5_legacy_functional_checkpoint_record.py",
        "tests/test_rc5_r510_channel_checkpoint_observation_record.py",
        "tests/test_rc5_r511_metadata_strict_record.py",
        "tests/test_rc5_r512_compiled_simulation_integrity_record.py",
        "tests/test_rc5_r513_optional_hygiene_record.py",
        "tests/test_rc5_r52_finite_workflow_record.py",
        "tests/test_rc5_r53_channel_checkpoint_record.py",
        "tests/test_rc5_r55_tf32_state_restore.py",
        "tests/test_rc5_r56_public_error_boundary_record.py",
        "tests/test_rc5_r57_device_normalization_record.py",
        "tests/test_rc5_r58_checkpoint_integrity_record.py",
        "tests/test_rc5_r59_pressure_postcondition_record.py",
        "tests/test_rc5_w1_documentation.py",
        "tests/test_rc6_1_plane_diagnostic_error_record.py",
        "tests/test_rc6_2_pressure_floor_record.py",
        "tests/test_rc6_3_compile_preflight_record.py",
        "tests/test_release_v0_2.py",
    }
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert registration["introduced_by_phase"] == "RC6-4"
        assert registration["reason"]
        resolved = _rc6_5_successor(relative, registration["current_sha256"])
        assert _sha256(relative) == _rc6_6_successor(relative, resolved)


def test_rc6_4_scope_is_cpu_only_and_preserves_numerical_contracts():
    scope = _record()["scope"]
    assert scope["public_error_hierarchy_changed"] is True
    assert scope["failed_run_metadata_finalization_added"] is True
    for key in (
        "pressure_algorithm_changed",
        "scientific_equations_changed",
        "integrator_changed",
        "transform_algorithm_changed",
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


def test_rc6_4_authorizes_only_the_next_local_phase():
    assert _record()["authorization"] == {
        "rc6_4_complete": True,
        "eligible_for_rc6_5": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }
