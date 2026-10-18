"""R5-11 evidence record and fail-closed scope."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / (
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def _r512_successor(relative: str, historical: str) -> str:
    resolved = historical
    for path in (
        R5_12_RECORD,
        R5_13_RECORD,
        G1_RECORD,
        G1_GLIBC_RECORD,
        RC6_R1_RECORD,
        RC6_R2_RECORD,
        RC6_R3_RECORD,
        RC6_R4_RECORD,
        RC6_R5_RECORD,
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


def test_r511_record_binds_implementation_and_regression_sources():
    record = _record()
    for section in ("implementation_files", "regression_tests"):
        for relative, expected in record[section].items():
            assert _sha256(ROOT / relative) == _r512_successor(
                relative, expected
            ), relative


def test_r511_projection_descriptions_are_exact_and_storage_specific():
    contract = _record()["n6_contract"]
    assert contract["full_complex"] == (
        "complete_conjugate_pair_projection_each_step.v2"
    )
    assert contract["hermitian_half"] == (
        "self_conjugate_stored_planes_projection_each_step.v2"
    )
    assert contract["pre_r5_11_current_checkpoint_alias_normalized"] is True
    assert contract["released_rc1_rc2_rc3_identity_literal_frozen"] == (
        "self_conjugate_planes_each_step"
    )


def test_r511_strict_contract_does_not_invent_an_unregistered_wall_trace():
    contract = _record()["b20_contract_adjudication"]
    assert contract["historical_record_rewritten"] is False
    assert contract["disposition"] == "clarified_without_invented_wall_trace"
    assert contract["strict_effective_contract"] == {
        "formula": "r_conditioned = r_raw",
        "validates_shape_dtype_device_and_finiteness": True,
        "mutates_requested_initial_samples": False,
        "invents_finite_difference_wall_extrapolation": False,
        "automatic_conditioning": False,
        "restart_reconditioning": False,
    }
    assert contract["external_uniform_reproducer_acceptance_is_expected"] is True


def test_r511_successor_registry_is_forward_only_and_exact():
    for relative, registration in _record()[
        "historical_successor_sources"
    ].items():
        assert registration["historical_sha256"] != registration["current_sha256"]
        assert _sha256(ROOT / relative) == _r512_successor(
            relative, registration["current_sha256"]
        )
        assert registration["introduced_by_phase"] == "R5-11"


def test_r511_scope_and_completion_are_fail_closed():
    record = _record()
    scope = record["scope"]
    assert scope["scientific_equations_changed"] is False
    assert scope["projection_algorithm_changed"] is False
    assert scope["checkpoint_dynamics_identity_changed"] is False
    assert scope["fresh_conditioning_arithmetic_changed"] is False
    assert scope["production_default_changed"] is False
    assert scope["PSSolver_Control_modified"] is False
    assert scope["nematics3d_modified"] is False
    assert scope["H100_required"] is False
    assert scope["H100_executed"] is False
    assert scope["push_executed"] is False
    completed = record["status"] == "implementation_complete_cpu_verified"
    if completed:
        assert record["classification"] == (
            "PASS_R5_11_PERIODIC_METADATA_AND_STRICT_"
            "FRESH_STATE_CONTRACT_CPU_CLOSURE"
        )
        evidence = record["installed_wheel_evidence"]
        assert evidence["source_shadow_import"] is False
        assert evidence["pip_check"] == "PASS"
        assert evidence["nematics3d_modified"] is False
        assert record["authorization"]["r5_11_cpu_candidate_complete"] is True
        assert record["authorization"]["eligible_for_next_work_item"] is True
    else:
        assert record["installed_wheel_evidence"] is None
        assert record["authorization"]["r5_11_cpu_candidate_complete"] is False
