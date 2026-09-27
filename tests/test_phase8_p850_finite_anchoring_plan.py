"""Static contracts for planning-only P8.5 finite anchoring."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
PLAN_PATH = NOTES / "phase_8_p850_finite_anchoring_plan.json"


def _plan() -> dict[str, object]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_at_commit(commit: str, relative: str) -> str:
    payload = subprocess.run(
        ["git", "show", f"{commit}:{relative}"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    return hashlib.sha256(payload).hexdigest()


def test_p850_binds_completed_p845_and_only_authorizes_next_declaration_slice():
    plan = _plan()
    baseline = plan["baseline"]
    authorization = plan["authorization"]

    assert plan["phase"] == "P8.5.0"
    assert plan["status"] == (
        "P8_5_0_PLANNING_FROZEN_IMPLEMENTATION_NOT_AUTHORIZED"
    )
    assert plan["classification"] == "PASS_P8_5_0_PLANNING"
    assert baseline["commit"] == "104258d5787921e527c61b180b6eb828013d184b"
    assert baseline["p8_4_5_complete"] is True
    assert baseline["eligible_for_p8_5_planning"] is True
    assert _sha256(ROOT / baseline["p8_4_5_record"]) == (
        baseline["p8_4_5_record_sha256"]
    )
    assert authorization["p8_5_0_planning_complete"] is True
    assert authorization["eligible_for_p8_5_1"] is True
    assert authorization["p8_5_1_implementation_authorized"] is False
    assert authorization["p8_5_runtime_implementation_authorized"] is False
    assert authorization["h100_authorized"] is False
    assert authorization["phase_9_authorized"] is False
    assert authorization["production_default_changed"] is False


def test_p850_generic_robin_law_is_field_neutral_and_oriented():
    law = _plan()["generic_robin_law"]

    assert law["field_symbol"] == "phi"
    assert law["canonical_form"] == (
        "alpha*phi+beta*(n_dot_grad_phi)=gamma"
    )
    assert law["normal_orientation"] == "outward_unit_normal"
    assert law["lower_plane_normal_derivative"] == "-partial_z"
    assert law["upper_plane_normal_derivative"] == "+partial_z"
    assert law["per_component_per_face"] is True
    assert law["raw_coefficients_preserved_in_identity"] is True
    assert law["operator_normalization_is_separate_metadata"] is True
    assert law["declaration_names_transform_or_solver"] is False
    assert law["field_neutral"] is True


def test_p850_q_law_follows_variation_and_keeps_strong_anchoring_distinct():
    anchoring = _plan()["finite_q_anchoring"]

    assert anchoring["natural_boundary_law"] == (
        "K_Q*n_k*partial_k(Q_ij)+W*(Q_ij-Qstar_ij)=0"
    )
    assert anchoring["robin_alpha"] == "W"
    assert anchoring["robin_beta"] == "K_Q"
    assert anchoring["robin_gamma"] == "W*Qstar_component"
    assert anchoring["extrapolation_length"] == "K_Q/W"
    assert anchoring["w_zero_limit"] == (
        "homogeneous_neumann_with_existing_nullspace_contract"
    )
    assert anchoring["w_infinity_runtime_value_allowed"] is False
    assert anchoring["strong_dirichlet_remains_distinct"] is True
    assert anchoring[
        "ambiguous_k_l1_frank_alias_inference_allowed"
    ] is False
    assert anchoring["full_symmetric_traceless_variation"] is True
    assert anchoring["five_component_metric_consistency_required"] is True
    assert anchoring[
        "five_components_are_independent_scalar_order_parameters"
    ] is False


def test_p850_separates_generic_ownership_from_q_specialization():
    ownership = _plan()["ownership"]

    assert "tensor_free_robin_condition_identity" in ownership[
        "pssolver.core.boundary"
    ]
    assert "field_neutral_registered_field_robin_policy" in ownership[
        "pssolver.boundaries"
    ]
    assert "oriented_faces_outward_normals_wall_axis_grid_placement_and_metrics" in (
        ownership["pssolver.geometries"]
    )
    assert "discrete_robin_boundary_residual" in ownership[
        "pssolver.operators"
    ]
    assert "quadratic_finite_q_anchoring_adapter" in ownership[
        "pssolver.models.active_nematics.boundaries"
    ]
    assert "five_component_metric_validation" in ownership[
        "pssolver.models.active_nematics.boundaries"
    ]


def test_p850_does_not_disguise_arbitrary_robin_as_dct_dst_or_tau_physics():
    decision = _plan()["operator_decision"]

    assert decision["periodic_axes"] == "fourier_spectral"
    assert decision[
        "generic_dct_or_dst_is_valid_for_arbitrary_finite_robin"
    ] is False
    assert decision["full_dct_dst_then_wall_patch_allowed"] is False
    assert decision["public_boundary_name"] == "robin"
    assert decision["tau_is_public_physical_boundary_name"] is False
    assert set(decision["p8_5_2_candidates"]) == {
        "qualified_robin_eigenbasis",
        "chebyshev_or_ultraspherical_tau_or_bordered_operator",
    }
    assert decision["method_must_be_frozen_by_operator_adr"] is True
    assert decision["runtime_before_method_decision_allowed"] is False


def test_p850_first_slice_and_out_of_scope_are_fail_closed():
    plan = _plan()
    first = plan["first_qualified_slice"]
    rejected = set(plan["explicit_rejections_first_slice"])
    out_of_scope = set(plan["out_of_scope"])

    assert first["model"] == "complete_stress_beris_edwards"
    assert first["geometry"] == "plane_slab"
    assert first["q_elasticity"] == "one_constant"
    assert first["q_surface_energy"] == (
        "static_quadratic_finite_anchoring"
    )
    assert first["wall_strength"] == "constant_per_face"
    assert first["target_q"] == "constant_per_face"
    assert first["runtime_fallback_allowed"] is False
    assert {
        "multiple_elastic_constants",
        "nonlinear_or_degenerate_planar_surface_energy",
        "curved_surface_or_channel_finite_anchoring",
        "ambiguous_k_l1_or_frank_constant_inference",
    } <= rejected
    assert "prescribed_nonzero_neumann_flux" in out_of_scope
    assert "degenerate_planar_fournier_galatola_surface_energy" in out_of_scope
    assert "phase_9" in out_of_scope


def test_p850_manufactured_restart_and_performance_gates_are_explicit():
    plan = _plan()
    oracle = plan["manufactured_oracles"]
    checkpoint = plan["checkpoint_contract"]
    performance = plan["performance_contract"]

    assert oracle["wall_residual"] == (
        "alpha*phi+beta*(n_dot_grad_phi)-gamma"
    )
    assert oracle["lower_and_upper_signs_tested_independently"] is True
    assert oracle["minimum_first_discretization_convergence_rate"] == 1.9
    assert oracle[
        "q_oracle_checks_full_tensor_and_five_component_residuals"
    ] is True
    assert checkpoint["plan_rebuilt_and_hash_verified_before_target_mutation"] is True
    assert checkpoint[
        "same_runtime_continuous_restart_byte_identity_required"
    ] is True
    assert performance["hot_loop_policy_dispatch"] is False
    assert performance["hot_loop_root_solve_or_factorization"] is False
    assert performance["hot_loop_plan_allocation"] is False


def test_p850_slice_order_requires_separate_authorization_and_ends_on_h100():
    slices = _plan()["slices"]

    assert [value["id"] for value in slices] == [
        "P8.5.0",
        "P8.5.1",
        "P8.5.2",
        "P8.5.3",
        "P8.5.4",
        "P8.5.5",
        "P8.5.6",
    ]
    assert slices[0]["authorized"] is True
    assert slices[0]["implemented"] is True
    assert all(value["authorized"] is False for value in slices[1:])
    assert all(value["implemented"] is False for value in slices[1:])
    assert slices[-1]["requires_h100"] is True
    assert all(value["requires_h100"] is False for value in slices[:-1])


def test_p850_reviewed_sources_are_content_addressed_at_the_frozen_baseline():
    plan = _plan()
    baseline = plan["baseline"]["commit"]
    for relative, expected in plan["reviewed_source_sha256"].items():
        assert _sha256_at_commit(baseline, relative) == expected


def test_p850_records_local_planning_verification_without_h100_claim():
    plan = _plan()

    assert plan["local_verification"] == {
        "targeted": "41 passed",
        "full": "2326 passed, 8 subtests passed",
        "git_diff_check": "pass",
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "deselected": 0,
    }
    assert plan["authorization"]["h100_authorized"] is False


def test_future_verbatim_archive_records_all_p845_and_p850_sources_without_pdf():
    source_path = NOTES / "build_verbatim_archive_pdf.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    source_order = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "SOURCE_ORDER"
            for target in node.targets
        ):
            source_order = ast.literal_eval(node.value)
            break
    assert source_order is not None
    for name in (
        "phase_8_p845_memory_recovery.md",
        "phase_8_p845_fused_reconstruction_recovery.md",
        "phase_8_p845_corrected_closure_plan.md",
        "phase_8_p845_fused_reconstruction_h100_result.md",
        "phase_8_p845_h100_qualification.md",
        "phase_8_p850_finite_anchoring_plan.md",
        "phase_8_p850_finite_anchoring_plan.json",
    ):
        assert source_order.count(name) == 1

    discovered = {
        path.relative_to(NOTES).as_posix()
        for pattern in ("*.md", "*.json")
        for path in NOTES.rglob(pattern)
    }
    assert discovered == set(source_order)
