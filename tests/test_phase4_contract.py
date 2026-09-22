"""Phase 4.0 design, scope, ownership, and compatibility gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_4_contract.json"
)


def _contract() -> dict[str, object]:
    return json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_phase4_authorization_is_separate_from_phase3_closure():
    contract = _contract()
    assert contract["schema_version"] == 1
    assert contract["status"] == (
        "P4_0_DESIGN_FROZEN_IMPLEMENTATION_NOT_STARTED"
    )
    assert contract["authorization"] == {
        "phase_3_closure_classification": (
            "PASS_PHASE3_CLOSURE_NON_REGRESSION_WITH_AUTHORIZED_"
            "COMPOSITE_RECOVERY_V7"
        ),
        "phase_3_record_commit": (
            "e1fcc5d6df5ea6cad6b61cc770a85d19c04f9953"
        ),
        "eligible_for_phase_4_planning": True,
        "phase_4_authorized": True,
        "authorization_kind": (
            "explicit_user_decision_after_phase3_closure"
        ),
        "phase_5_authorized": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
    }


def test_phase4_freezes_sbdf2_without_expanding_integrator_scope():
    contract = _contract()
    assert contract["scope"] == {
        "second_integrator": "constant_step_sbdf2",
        "startup_integrator": "projected_semi_implicit_euler",
        "qualified_block_size": 2,
        "qualifying_model": (
            "two_component_periodic_reaction_diffusion"
        ),
        "plane_connection": False,
        "public_api_promotion": False,
    }
    sbdf2 = contract["sbdf2"]
    assert sbdf2["formal_order"] == 2
    assert sbdf2["constant_dt_only"] is True
    assert sbdf2["startup_steps"] == 1
    assert sbdf2["persistent_history"] == [
        "previous_evolved_native_spectrum",
        "previous_explicit_native_spectral_rhs",
    ]
    assert sbdf2["commit_history_last"] is True
    assert sbdf2["failed_step_preserves_last_complete_history"] is True
    assert sbdf2["step_order"][-2:] == [
        "commit_progress",
        "commit_integrator_history",
    ]


def test_phase4_owns_history_separately_from_workspace():
    ownership = _contract()["ownership"]
    state = set(ownership["runtime_state"])
    workspace = set(ownership["workspace"])
    assert "previous_evolved_native_spectrum" in state
    assert "previous_explicit_native_spectral_rhs" in state
    assert "previous_evolved_native_spectrum" not in workspace
    assert "previous_explicit_native_spectral_rhs" not in workspace
    assert "current_explicit_rhs" in workspace
    assert "assembled_implicit_rhs" in workspace
    assert state.isdisjoint(workspace)


def test_phase4_modal_block_scope_is_bounded_and_fail_closed():
    block = _contract()["modal_block"]
    assert block["qualified_block_size"] == 2
    assert block["physical_coefficient_policy"] == "constant"
    assert block["mode_coupling"] is False
    assert block["spatially_variable_coefficients"] is False
    assert block["arbitrary_block_size"] is False
    assert block["bounded_workspace_required"] is True
    assert set(block["construction_rejections"]) == {
        "component_order_mismatch",
        "shape_mismatch",
        "dtype_mismatch",
        "device_mismatch",
        "nonfinite_coefficients",
        "singular_or_rejected_modal_matrix",
        "unsupported_block_size",
        "geometry_or_basis_capability_mismatch",
    }


def test_phase4_does_not_change_plane_checkpoint_v1():
    checkpoint = _contract()["checkpoint"]
    assert checkpoint["plane_format_v1_unchanged"] is True
    assert checkpoint["phase_4_reference_schema"] == (
        "provisional_generic_multistep_v1"
    )
    assert checkpoint["history_is_persistent"] is True
    assert checkpoint["cross_integrator_restart_supported"] is False
    assert checkpoint["dt_change_after_history_supported"] is False
    assert set(checkpoint["reject_before_advance_or_output"]) == {
        "missing_history",
        "tampered_history",
        "changed_dt",
        "different_integrator_identity",
        "unsupported_format_version",
        "incompatible_model_or_discretization_identity",
    }


def test_phase4_freezes_convergence_and_h100_closure_gates():
    qualification = _contract()["qualification"]
    assert qualification["temporal_step_ratio_ladder"] == [
        1.0,
        0.5,
        0.25,
        0.125,
    ]
    assert qualification["minimum_euler_l2_order"] == 0.9
    assert qualification["minimum_sbdf2_l2_order"] == 1.8
    assert qualification["same_backend_restart_byte_identity"] is True
    assert qualification["finite_required"] is True
    assert qualification["h100_required_for_closure"] is True
    assert qualification["implicit_fallback_allowed"] is False


def test_phase4_explicitly_defers_plane_and_large_scope_features():
    deferred = set(_contract()["deferred"])
    assert {
        "cnab2",
        "adaptive_or_variable_timestep",
        "arbitrary_block_size",
        "variable_coefficient_implicit_operator",
        "symbolic_dsl",
        "newton_krylov_petsc",
        "plane_beris_edwards_migration",
        "new_geometry_or_physical_boundary",
        "strong_anchoring",
        "optimal_control",
        "production_default_promotion",
        "phase_5",
    } <= deferred


def test_phase4_p40_is_documentation_only_against_frozen_sources():
    source_oracle = _contract()["p4_0_source_oracle"]
    assert source_oracle
    for relative, expected in source_oracle.items():
        assert _sha256(PROJECT_ROOT / relative) == expected

