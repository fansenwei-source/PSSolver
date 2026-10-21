"""Closure-record checks for the RC6-2 Channel pressure contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_2_pressure_floor_contract.json"
RC6_R3_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_3_compile_preflight_contract.json"
)
RC6_R4_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_4_channel_failure_contract.json"
)
RC6_R5_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_5_checkpoint_control_flow.json"
)
RC6_R6_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_6_restart_output_atomicity.json"
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
V022_RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_2.json"


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _rc6_3_successor(relative: str, historical: str) -> str:
    registration = json.loads(RC6_R3_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
    registration = json.loads(RC6_R4_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
    registration = json.loads(RC6_R5_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
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
    registration = json.loads(V022_RELEASE_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
    return historical


def test_rc6_2_record_binds_implementation_and_regressions():
    record = _record()
    assert record["status"] == "implementation_complete_cpu_verified"
    assert record["classification"] == (
        "PASS_RC6_2_CHANNEL_PRESSURE_FLOOR_CPU_CLOSURE"
    )
    assert record["git_identity"]["implementation_commit"] == (
        "a134658d04c9a6c0aa2cffd786982548e00f18f7"
    )
    assert record["git_identity"]["record_commit_is_qualified_source"] is False
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(relative) == _rc6_3_successor(relative, expected)


def test_rc6_2_contract_is_explicit_and_fail_closed():
    contract = _record()["contract"]
    assert contract["tolerance_mode_floor_definition"] == (
        "8*finfo(real_dtype).eps"
    )
    assert contract["normalized_residual_definition"] == (
        "active_rms(projected_pcg_recurrence_residual)/"
        "active_rms(projected_rhs)"
    )
    for key in (
        "relative_tolerance_met_is_recurrence_postcondition",
        "legacy_channel_public_compiler_rejects_sub_floor",
        "complete_stress_channel_public_compiler_rejects_sub_floor",
        "rejection_precedes_runtime_construction_and_output",
        "exact_floor_accepted",
        "fixed_work_sub_floor_diagnostic_tolerance_preserved",
    ):
        assert contract[key] is True
    assert contract["independently_recomputed_residual_claimed"] is False
    assert contract["second_pressure_operator_application_added_to_hot_path"] is False
    assert contract["public_rejection_code"] == "discretization_contract"


def test_rc6_2_registers_exact_forward_only_successors():
    successors = _record()["historical_successor_sources"]
    assert set(successors) == {
        "CHANGELOG.md",
        "pssolver/configuration/public_channel_beris_edwards_compiler.py",
        "pssolver/configuration/public_channel_simulation_compiler.py",
        "pssolver/configuration/public_option_validation.py",
        "pssolver/functional/pressure_metadata.py",
        "pssolver/linear_solvers/stokes/channel_no_slip.py",
        "tests/test_rc447_pressure_bounded_operator_recovery.py",
        "tests/test_rc447_python310_compatibility.py",
        "tests/test_rc455_public_robustness.py",
        "tests/test_rc456_tf32_execution_scope.py",
        "tests/test_rc457_installed_consumer_impact.py",
        "tests/test_rc457_installed_consumer_recovery.py",
        "tests/test_rc458_h100_closure_record.py",
        "tests/test_rc4_lifting_capability_restriction_record.py",
        "tests/test_rc4_post_release_findings.py",
        "tests/test_rc4_pressure_contract_plan.py",
        "tests/test_rc4_pressure_metadata_identity_connection_record.py",
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
        "tests/test_rc5_r59_pressure_postcondition_errors.py",
        "tests/test_rc5_r59_pressure_postcondition_record.py",
        "tests/test_rc5_w1_documentation.py",
        "tests/test_rc6_1_plane_diagnostic_error_record.py",
        "tests/test_release_v0_2.py",
    }
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert registration["introduced_by_phase"] == "RC6-2"
        assert registration["reason"]
        assert _sha256(relative) == _rc6_3_successor(
            relative, registration["current_sha256"]
        )


def test_rc6_2_scope_is_cpu_only_and_does_not_claim_release_authority():
    record = _record()
    scope = record["scope"]
    assert scope["pressure_tolerance_contract_changed"] is True
    assert scope["public_compilation_validation_changed"] is True
    for key in (
        "scientific_equations_changed",
        "integrator_changed",
        "transform_algorithm_changed",
        "pressure_algorithm_changed",
        "pressure_hot_path_changed",
        "checkpoint_schema_changed",
        "production_default_changed",
        "PSSolver_Control_modified",
        "nematics3d_modified",
        "H100_required",
        "H100_executed",
        "push_executed",
    ):
        assert scope[key] is False
    assert record["authorization"]["eligible_for_automatic_merge"] is False
    assert record["authorization"]["eligible_for_default_promotion"] is False
    assert record["authorization"]["rc6_2_complete"] is True
    assert record["authorization"]["eligible_for_rc6_3"] is True
