"""Closure record for the RC6-1 Plane compiled diagnostic error repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_1_plane_diagnostic_errors.json"
RC6_R2_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_2_pressure_floor_contract.json"
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _resolve_rc6_2(relative: str, predecessor: str) -> str:
    registration = json.loads(RC6_R2_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is None:
        return predecessor
    accepted = registration.get(
        "accepted_predecessor_sha256",
        [registration["historical_sha256"]],
    )
    if predecessor in accepted:
        return registration["current_sha256"]
    return predecessor


def test_rc6_1_record_binds_the_implementation_and_regression_test():
    record = _record()
    assert record["status"] == "complete"
    assert record["classification"] == (
        "PASS_RC6_1_PLANE_COMPILED_DIAGNOSTIC_PUBLIC_ERROR_CPU_CLOSURE"
    )
    assert record["git_identity"]["implementation_commit"] == (
        "17a026650bbbd7382b7dfea0b9ca195985e46f9e"
    )
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(relative) == expected


def test_rc6_1_contract_is_the_stable_public_state_error():
    contract = _record()["contract"]
    assert contract == {
        "application_family": "plane_complete_stress",
        "stage": "diagnostic",
        "public_error_base": "SimulationExecutionError",
        "nonfinite_error": "SimulationStateError",
        "machine_code": "simulation_state_error",
        "step_source": "compiled workflow completed_steps",
        "normal_force_nonfinite_field": "projected_normal_force",
        "pressure_nonfinite_fields_are_reported_individually": True,
        "disabled_diagnostic_error_unchanged": "RuntimeError",
        "unavailable_diagnostic_error_unchanged": "RuntimeError",
        "fail_closed": True,
    }


def test_rc6_1_registers_forward_only_historical_successors():
    successors = _record()["historical_successor_sources"]
    assert set(successors) == {
        "pssolver/workflows/plane_compiled_v2.py",
        "tests/test_rc455_public_robustness.py",
        "tests/test_rc456_tf32_execution_scope.py",
        "tests/test_rc457_installed_consumer_impact.py",
        "tests/test_rc457_installed_consumer_recovery.py",
        "tests/test_rc458_h100_closure_record.py",
        "tests/test_rc4_lifting_checkpoint_integration_record.py",
        "tests/test_rc4_plane_checkpoint_identity_record.py",
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
        "tests/test_rc5_r56_public_error_boundary_record.py",
        "tests/test_rc5_r57_device_normalization_record.py",
        "tests/test_rc5_r58_checkpoint_integrity_record.py",
        "tests/test_rc5_r59_pressure_postcondition_record.py",
        "tests/test_rc5_w1_documentation.py",
        "tests/test_release_v0_2.py",
    }
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert _sha256(relative) == _resolve_rc6_2(
            relative, registration["current_sha256"]
        )


def test_rc6_1_scope_authorizes_only_the_next_local_work_item():
    record = _record()
    scope = record["scope"]
    assert scope["public_exception_behavior_changed"] is True
    for key in (
        "scientific_equations_changed",
        "integrator_changed",
        "transform_algorithm_changed",
        "pressure_algorithm_changed",
        "checkpoint_schema_changed",
        "production_default_changed",
        "PSSolver_Control_modified",
        "nematics3d_modified",
        "H100_required",
        "H100_executed",
        "push_executed",
    ):
        assert scope[key] is False
    assert record["authorization"] == {
        "rc6_1_complete": True,
        "eligible_for_rc6_2": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }
