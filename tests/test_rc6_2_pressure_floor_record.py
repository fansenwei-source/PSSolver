"""Closure-record checks for the RC6-2 Channel pressure contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


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
    return resolve_s03_successor(relative, historical)


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
