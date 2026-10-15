"""R5-12 evidence record and fail-closed scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_12_compiled_simulation_integrity.json"
)


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_r512_record_binds_implementation_and_regression_sources():
    record = _record()
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(relative) == expected, relative


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
        "tests/test_rc5_r510_channel_checkpoint_observation_record.py",
        "tests/test_rc5_r52_finite_workflow_record.py",
        "tests/test_rc5_r53_channel_checkpoint_record.py",
        "tests/test_rc5_r55_tf32_state_restore.py",
        "tests/test_rc5_r56_public_error_boundary_record.py",
        "tests/test_rc5_r57_device_normalization_record.py",
        "tests/test_rc5_r58_checkpoint_integrity_record.py",
        "tests/test_rc5_r59_pressure_postcondition_record.py",
        "tests/test_rc5_r511_metadata_strict_record.py",
    }
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert _sha256(relative) == registration["current_sha256"]
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
