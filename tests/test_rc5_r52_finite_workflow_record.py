"""Static closure record for the R5-2 finite production-state repair."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_2_finite_workflow_state.json"
R5_3_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_3_channel_checkpoint_dynamics.json"
)
R5_5_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_5_tf32_state_restore.json"
R5_6_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_6_public_error_boundary.json"
R5_7_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_7_device_normalization.json"
R5_8_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"
R5_9_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_9_pressure_postcondition_errors.json"
R5_10_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_10_channel_checkpoint_observation.json"
)
R5_11_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc5_r5_11_periodic_metadata_strict_contract.json"
)
R5_12_RECORD = ROOT / (
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


def _latest_successor(relative: str, historical: str) -> str:
    resolved = historical
    for record_path in (
        R5_3_RECORD,
        R5_5_RECORD,
        R5_6_RECORD,
        R5_7_RECORD,
        R5_8_RECORD,
        R5_9_RECORD,
        R5_10_RECORD,
        R5_11_RECORD,
        R5_12_RECORD,
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
        FINAL_RELEASE_RECORD,
        V021_N1_RECORD,
        V021_RELEASE_RECORD,
        V022_V17_RECORD,
        V022_RELEASE_RECORD,
    ):
        registration = json.loads(record_path.read_text(encoding="utf-8"))[
            "historical_successor_sources"
        ].get(relative)
        if (
            registration is not None
            and registration["historical_sha256"] == resolved
        ):
            resolved = registration["current_sha256"]
    return resolve_s03_successor(relative, resolved)


def test_r52_record_binds_implementation_regressions_and_successors():
    record = _record()
    assert record["classification"] == (
        "PASS_R5_2_FINITE_PRODUCTION_WORKFLOW_STATE_CPU_CLOSURE"
    )
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(relative) == _latest_successor(relative, expected)
    for relative, registration in record[
        "historical_successor_sources"
    ].items():
        assert registration["historical_sha256"] != registration[
            "current_sha256"
        ]
        assert _sha256(relative) == _latest_successor(
            relative,
            registration["current_sha256"],
        )


def test_r52_contract_fails_closed_without_a_hot_loop_sync():
    contract = _record()["contract"]
    assert contract["public_error_base"] == "SimulationExecutionError"
    assert contract["nonfinite_error"] == "SimulationStateError"
    assert contract["machine_code"] == "simulation_state_error"
    assert contract["checked_boundaries"] == [
        "observation",
        "diagnostic",
        "checkpoint",
        "final_observation",
    ]
    assert contract["complete_written_after_nonfinite_state"] is False
    assert contract["nonfinite_arrays_published"] is False
    assert contract["per_timestep_host_synchronization_added"] is False


def test_r52_source_and_installed_wheel_cpu_closures_are_exact():
    validation = _record()["validation"]
    assert validation["source_full_cpu"] == (
        "3454 passed, 7 skipped, 8 subtests passed, 0 failed"
    )
    assert validation["installed_wheel_targeted_cpu"] == (
        "73 passed, 0 failed"
    )
    assert validation["installed_wheel_applicable_full_cpu"] == (
        "3447 passed, 7 skipped, 8 subtests passed, 0 failed"
    )
    evidence = _record()["installed_wheel_evidence"]
    assert evidence["source_shadow_import"] is False
    assert evidence["pip_check"] == "PASS"
    assert evidence["nematics3d_modified"] is False


def test_r52_authorizes_only_the_next_local_rc5_must_item():
    record = _record()
    scope = record["scope"]
    assert scope["scientific_equations_changed"] is False
    assert scope["production_default_changed"] is False
    assert scope["PSSolver_Control_modified"] is False
    assert scope["nematics3d_modified"] is False
    assert scope["H100_required"] is False
    assert scope["H100_executed"] is False
    assert scope["push_executed"] is False
    assert record["authorization"] == {
        "r5_2_complete": True,
        "eligible_for_r5_3": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }
