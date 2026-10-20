"""Static closure record for R5-9 pressure postcondition contracts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_9_pressure_postcondition_errors.json"
R5_8_RECORD = ROOT / "notes/PSSolver_v0_2_0rc5_r5_8_checkpoint_integrity.json"
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


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _r510_successor(relative: str, historical: str) -> str:
    resolved = historical
    for path in (
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


def test_r59_record_binds_implementation_and_exact_sources():
    record = _record()
    assert record["git_identity"]["branch"] == "fix/v0.2.0rc5"
    assert record["git_identity"]["implementation_commit"] == (
        "d9672ea1a3b05340d444d4acb5d7525ab9e8d419"
    )
    assert record["git_identity"]["record_commit_is_qualified_source"] is False
    assert record["git_identity"]["pushed"] is False
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(ROOT / relative) == _r510_successor(
                relative, expected
            )


def test_r59_registers_exact_historical_successors():
    successors = _record()["historical_successor_sources"]
    assert successors
    for relative, registration in successors.items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert _sha256(ROOT / relative) == _r510_successor(
            relative, registration["current_sha256"]
        )
        assert registration["introduced_by_phase"] == "R5-9"
        assert registration["reason"]


def test_r59_pressure_contract_is_truthful_and_machine_readable():
    contract = _record()["contract"]
    for key in (
        "sub_floor_tolerance_mode_rejected_before_pressure_solve",
        "fixed_work_sub_floor_diagnostic_tolerance_preserved",
        "recurrence_residual_not_claimed_below_supported_roundoff_floor",
        "first_public_pressure_failure_is_channel_pressure_convergence_error",
        "first_functional_pressure_failure_is_functional_convergence_error",
        "functional_convergence_error_is_channel_pressure_convergence_error",
        "machine_readable_failure_details_preserved",
        "legacy_generic_static_model_validation_wrapper_preserved_for_unrelated_errors",
    ):
        assert contract[key] is True
    assert contract["tolerance_mode_floor_definition"] == "2*finfo(real_dtype).eps"
    assert contract["pressure_operator_call_added_to_hot_path"] is False


def test_r59_preserves_r58_and_claims_no_out_of_scope_work():
    record = _record()
    assert _sha256(R5_8_RECORD) == (
        "a66d8ec0f88d6e173282b25f008a21f2bc98950b56e1fc581caf36dabbe4f1a1"
    )
    assert set(record["external_reproducers"].values()) == {
        "b402b4274d9888ea47b20ec6d8697a643ccfbbf8c29467830dc60034da3a985c",
        "98edad60674f7d65c8dd47c4d830989ed882750c50f44ff7bb031ac7b55bdcb8",
        "f37dc79a159a43f38796c1af9085070aad062dfdd95a069860f79589d0e8825f",
    }
    scope = record["scope"]
    assert scope["pressure_stopping_contract_changed"] is True
    assert scope["public_exception_contract_changed"] is True
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


def test_r59_final_status_requires_completed_cpu_closure():
    record = _record()
    completed = record["status"] == "implementation_complete_cpu_verified"
    if completed:
        assert record["classification"] == (
            "PASS_R5_9_PRESSURE_POSTCONDITION_ERROR_CPU_CLOSURE"
        )
        assert record["git_identity"]["qualified_source_commit"] == (
            record["git_identity"]["implementation_commit"]
        )
        assert record["installed_wheel_evidence"] is not None
        assert record["authorization"]["r5_9_cpu_closure_complete"] is True
        assert record["authorization"]["eligible_for_next_work_item"] is True
    else:
        assert record["status"] == "implementation_complete_validation_pending"
        assert record["classification"].startswith("PENDING_")
        assert record["git_identity"]["qualified_source_commit"] is None
        assert record["authorization"]["r5_9_cpu_closure_complete"] is False
