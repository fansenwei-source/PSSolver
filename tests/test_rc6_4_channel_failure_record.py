"""Closure-record checks for RC6-4 public execution failures."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


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
V022_RELEASE_RECORD = ROOT / "notes/pssolver_v0_2_2.json"


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
    return resolve_s03_successor(relative, historical)


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
    registration = json.loads(V022_RELEASE_RECORD.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if registration is not None and registration["historical_sha256"] == historical:
        historical = registration["current_sha256"]
    return resolve_s03_successor(relative, historical)


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
