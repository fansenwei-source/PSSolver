"""R5-12 evidence record and fail-closed scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_12_compiled_simulation_integrity.json"
)
R5_13_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_13_optional_hygiene.json"
G1_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_g1_source_freeze.json"
G1_GLIBC_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_g1_checkpoint_glibc_compatibility.json"
)
RC6_R1_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_1_plane_diagnostic_errors.json"
)
RC6_R2_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc6_rc6_2_pressure_floor_contract.json"
)
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


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _r513(relative: str, historical: str) -> str:
    resolved = historical
    for path in (
        R5_13_RECORD,
        G1_RECORD,
        G1_GLIBC_RECORD,
        RC6_R1_RECORD,
        RC6_R2_RECORD,
        RC6_R3_RECORD,
        RC6_R4_RECORD,
        RC6_R5_RECORD,
        RC6_R6_RECORD,
        RC6_FREEZE_RECORD,
    ):
        registration = json.loads(path.read_text(encoding="utf-8"))[
            "historical_successor_sources"
        ].get(relative)
        accepted = (
            []
            if registration is None
            else registration.get(
                "accepted_predecessor_sha256",
                [registration["historical_sha256"]],
            )
        )
        if registration is not None and resolved in accepted:
            resolved = registration["current_sha256"]
    return resolved


def test_r512_record_binds_implementation_and_regression_sources():
    record = _record()
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(relative) == _r513(relative, expected), relative


def test_r512_compilation_identity_binds_every_executable_component():
    contract = _record()["contract"]
    assert contract["identity_schema_version"] == 1
    assert contract["bound_components"] == [
        "source_simulation_sha256",
        "application",
        "application_simulation_sha256",
        "lowering_plan_sha256",
        "construction_plan_sha256",
        "application_request_type",
        "application_request_sha256",
        "normalization",
    ]
    assert contract["validated_before_application_dispatch"] is True
    assert contract["validated_before_metadata_emission"] is True
    assert contract["recompiles_at_run_time"] is False
    assert contract["tampered_product_rejected"] is True


def test_r512_successor_registry_is_forward_only_and_exact():
    successors = _record()["historical_successor_sources"]
    assert set(successors) == {
        "pssolver/api/runner.py",
        "tests/test_phase8_p844_lifting_workflow_restart.py",
        "tests/test_rc4_lifting_checkpoint_integration_record.py",
        "tests/test_rc455_public_robustness.py",
        "tests/test_rc456_tf32_execution_scope.py",
        "tests/test_rc457_installed_consumer_impact.py",
        "tests/test_rc457_installed_consumer_recovery.py",
        "tests/test_rc458_h100_closure_record.py",
        "tests/test_rc4_post_release_findings.py",
        "tests/test_rc5_r510_channel_checkpoint_observation_record.py",
        "tests/test_rc5_r52_finite_workflow_record.py",
        "tests/test_rc5_r53_channel_checkpoint_record.py",
        "tests/test_rc5_r55_tf32_state_restore.py",
        "tests/test_rc5_r56_public_error_boundary_record.py",
        "tests/test_rc5_r57_device_normalization_record.py",
        "tests/test_rc5_r58_checkpoint_integrity_record.py",
        "tests/test_rc5_r59_pressure_postcondition_record.py",
        "tests/test_rc5_legacy_functional_checkpoint_record.py",
        "tests/test_rc5_w1_documentation.py",
        "tests/test_release_v0_2.py",
        "tests/test_rc5_r511_metadata_strict_record.py",
    }
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert _sha256(relative) == _r513(
            relative, registration["current_sha256"]
        )
        assert registration["introduced_by_phase"] == "R5-12"
        assert registration["reason"]


def test_r512_scope_and_completion_are_fail_closed():
    record = _record()
    scope = record["scope"]
    assert scope["public_compiled_product_contract_changed"] is True
    for key in (
        "scientific_equations_changed",
        "numerical_hot_path_changed",
        "checkpoint_schema_changed",
        "production_default_changed",
        "PSSolver_Control_modified",
        "nematics3d_modified",
        "H100_required",
        "H100_executed",
        "push_executed",
    ):
        assert scope[key] is False

    completed = record["status"] == "implementation_complete_cpu_verified"
    if completed:
        assert record["classification"] == (
            "PASS_R5_12_COMPILED_SIMULATION_INTEGRITY_CPU_CLOSURE"
        )
        assert record["installed_wheel_evidence"] is not None
        assert record["authorization"]["r5_12_cpu_closure_complete"] is True
        assert record["authorization"]["eligible_for_next_work_item"] is True
    else:
        assert record["status"] == "implementation_complete_validation_pending"
        assert record["classification"].startswith("PENDING_")
        assert record["installed_wheel_evidence"] is None
        assert record["authorization"]["r5_12_cpu_closure_complete"] is False
