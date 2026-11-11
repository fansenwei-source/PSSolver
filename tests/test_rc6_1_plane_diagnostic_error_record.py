"""Closure record for the RC6-1 Plane compiled diagnostic error repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc6_rc6_1_plane_diagnostic_errors.json"
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


def _resolve_rc6_3(relative: str, predecessor: str) -> str:
    registration = json.loads(RC6_R3_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == predecessor:
        return registration["current_sha256"]
    return predecessor


def _resolve_rc6_6(relative: str, predecessor: str) -> str:
    registration = json.loads(RC6_R6_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == predecessor:
        return registration["current_sha256"]
    return predecessor


def _resolve_rc6_freeze(relative: str, predecessor: str) -> str:
    registration = json.loads(RC6_FREEZE_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == predecessor:
        predecessor = registration["current_sha256"]
    registration = json.loads(FINAL_RELEASE_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == predecessor:
        predecessor = registration["current_sha256"]
    registration = json.loads(V021_N1_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == predecessor:
        predecessor = registration["current_sha256"]
    registration = json.loads(V021_RELEASE_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == predecessor:
        predecessor = registration["current_sha256"]
    registration = json.loads(V022_V17_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == predecessor:
        predecessor = registration["current_sha256"]
    registration = json.loads(V022_RELEASE_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == predecessor:
        predecessor = registration["current_sha256"]
    return resolve_s03_successor(relative, predecessor)


def _resolve_rc6_4(relative: str, predecessor: str) -> str:
    registration = json.loads(RC6_R4_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == predecessor:
        return registration["current_sha256"]
    return predecessor


def _resolve_rc6_5(relative: str, predecessor: str) -> str:
    registration = json.loads(RC6_R5_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == predecessor:
        return registration["current_sha256"]
    return predecessor


def _resolve_from_rc6_1(relative: str, predecessor: str) -> str:
    resolved = _resolve_rc6_2(relative, predecessor)
    resolved = _resolve_rc6_3(relative, resolved)
    resolved = _resolve_rc6_4(relative, resolved)
    resolved = _resolve_rc6_5(relative, resolved)
    resolved = _resolve_rc6_6(relative, resolved)
    return _resolve_rc6_freeze(relative, resolved)


def test_rc6_1_record_binds_the_implementation_and_regression_test():
    from qualification.s27_forward import assert_forward_binding
    assert_forward_binding(Path(__file__).resolve().parents[1], 'tests/test_rc6_1_plane_diagnostic_error_record.py::test_rc6_1_record_binds_the_implementation_and_regression_test')


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
    from qualification.s27_forward import assert_forward_binding
    assert_forward_binding(Path(__file__).resolve().parents[1], 'tests/test_rc6_1_plane_diagnostic_error_record.py::test_rc6_1_registers_forward_only_historical_successors')


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
