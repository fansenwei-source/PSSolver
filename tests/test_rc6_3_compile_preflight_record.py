"""Closure-record checks for RC6-3 complete-stress compile preflight."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_3_compile_preflight_contract.json"
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
V03_INTEGRATION_RECORD = ROOT / (
    "notes/PSSolver_v0_3_0_n1_v17_checkpoint_integration.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _rc6_4_successor(relative: str, historical: str) -> str:
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
    registration = json.loads(V03_INTEGRATION_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
    return historical


def test_rc6_3_record_binds_implementation_and_regression_sources():
    record = _record()
    assert record["status"] == "implementation_complete_cpu_verified"
    assert record["classification"] == (
        "PASS_RC6_3_COMPLETE_STRESS_COMPILE_PREFLIGHT_CPU_CLOSURE"
    )
    assert record["git_identity"]["implementation_commit"] == (
        "3070d852685326e7efa051bb76ae4ef6e3458962"
    )
    assert record["git_identity"]["record_commit_is_qualified_source"] is False
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(relative) == _rc6_4_successor(relative, expected)


def test_rc6_3_contract_is_structured_and_preserves_planning():
    contract = _record()["contract"]
    assert contract["missing_parameter_public_exception"] == (
        "PublicSimulationCompilationError"
    )
    assert contract["missing_parameter_public_code"] == "malformed_application_spec"
    assert contract["snapshot_shape_public_exception"] == (
        "PublicSimulationCompilationError"
    )
    assert contract["snapshot_shape_public_code"] == "initial_condition_contract"
    assert contract["accepted_existing_q_snapshot_layouts"] == [
        "(*grid_shape, 5)",
        "(5, *grid_shape)",
    ]
    for key in (
        "existing_snapshot_header_inspected_without_full_payload_allocation",
        "unavailable_snapshot_remains_valid_for_planning_only_compilation",
        "runtime_dtype_finiteness_and_shape_backstop_preserved",
        "rejection_precedes_runtime_construction_timestep_and_output",
    ):
        assert contract[key] is True


def test_rc6_3_registers_exact_forward_only_successors():
    successors = _record()["historical_successor_sources"]
    assert set(successors) == {
        "CHANGELOG.md",
        "pssolver/configuration/public_periodic_simulation_compiler.py",
        "pssolver/configuration/public_channel_beris_edwards_compiler.py",
        "tests/test_rc455_public_robustness.py",
        "tests/test_rc4_lifting_capability_restriction_record.py",
        "tests/test_rc4_pressure_contract_plan.py",
        "tests/test_rc4_pressure_metadata_identity_connection_record.py",
        "tests/test_rc4_pressure_public_contract.py",
        "tests/test_rc5_r56_public_error_boundary_record.py",
        "tests/test_rc5_w1_documentation.py",
        "tests/test_rc6_2_pressure_floor_record.py",
        "tests/test_rc456_tf32_execution_scope.py",
        "tests/test_rc457_installed_consumer_impact.py",
        "tests/test_rc457_installed_consumer_recovery.py",
        "tests/test_rc458_h100_closure_record.py",
        "tests/test_rc4_post_release_findings.py",
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
        "tests/test_rc5_r57_device_normalization_record.py",
        "tests/test_rc5_r58_checkpoint_integrity_record.py",
        "tests/test_rc5_r59_pressure_postcondition_record.py",
        "tests/test_rc6_1_plane_diagnostic_error_record.py",
        "tests/test_release_v0_2.py",
    }
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert registration["introduced_by_phase"] == "RC6-3"
        assert registration["reason"]
        assert _sha256(relative) == _rc6_4_successor(
            relative, registration["current_sha256"]
        )


def test_rc6_3_scope_is_cpu_only_and_does_not_claim_release_authority():
    scope = _record()["scope"]
    assert scope["public_compilation_validation_changed"] is True
    assert scope["existing_snapshot_header_inspection_added"] is True
    for key in (
        "full_snapshot_payload_allocation_added",
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


def test_rc6_3_authorizes_only_the_next_local_phase():
    authorization = _record()["authorization"]
    assert authorization == {
        "rc6_3_complete": True,
        "eligible_for_rc6_4": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }
