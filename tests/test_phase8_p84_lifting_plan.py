"""Static contracts for planning-only P8.4 Dirichlet lifting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

import pssolver.boundaries as public_boundaries
import pssolver.core.boundary as core_boundary


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"
PLAN_PATH = NOTES / "phase_8_p84_lifting_plan.json"


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


def test_p84_planning_binds_completed_p83_without_authorizing_implementation():
    plan = _plan()
    baseline = plan["baseline"]
    authorization = plan["authorization"]

    assert plan["phase"] == "P8.4.0"
    assert plan["status"] == (
        "P8_4_PLANNING_FROZEN_IMPLEMENTATION_NOT_AUTHORIZED"
    )
    assert plan["classification"] == "PASS_P8_4_PLANNING"
    assert baseline["commit"] == "9c0cdc25ad875a964cffafa73a12eda3e45c2244"
    assert baseline["p8_3_complete"] is True
    assert baseline["eligible_for_p8_4_planning"] is True
    assert _sha256(ROOT / baseline["p8_3_record"]) == (
        baseline["p8_3_record_sha256"]
    )
    assert authorization == {
        "p8_4_planning_authorized": True,
        "p8_4_planning_complete": True,
        "p8_4_1_implementation_authorized": False,
        "p8_4_implementation_authorized": False,
        "h100_authorized": False,
        "p8_5_authorized": False,
        "phase_9_authorized": False,
        "production_default_changed": False,
    }


def test_p84_lifting_core_is_field_neutral_and_q_is_only_a_specialization():
    plan = _plan()
    decision = plan["decision"]
    ownership = plan["ownership"]
    q_specialization = plan["q_specialization"]

    assert decision["generic_mathematical_field"] == "phi"
    assert decision["lifting_core_is_field_neutral"] is True
    assert decision["q_is_first_scientific_specialization"] is True
    assert decision["generic_core_knows_director_or_scalar_order"] is False
    assert decision["boundary_declaration_names_transform_or_solver"] is False
    assert "strong_planar_q_convenience" in ownership[
        "pssolver.models.active_nematics.boundaries"
    ]
    assert q_specialization["q_convention_id"] == (
        "de_gennes_S_lambda_max_v1"
    )
    assert q_specialization["component_order"] == [
        "Qxx",
        "Qxy",
        "Qxz",
        "Qyy",
        "Qyz",
    ]


def test_p84_freezes_lifted_equation_and_representation_contracts():
    plan = _plan()
    equation = plan["equation_contract"]
    state = plan["initial_observation_checkpoint_contract"]

    assert equation["physical_field"] == "phi_homogeneous+phi_lift"
    assert equation["evolved_field"] == "phi_homogeneous"
    assert equation["split_rhs"] == (
        "L(phi_homogeneous)+L(phi_lift)+N(phi_homogeneous+phi_lift)"
    )
    assert equation["linear_lift_correction_explicit_even_if_zero"] is True
    assert equation["output_only_lift_addition_is_valid"] is False
    assert state["initial_condition_representation"] == "physical_field"
    assert state["integrator_state_representation"] == "homogeneous_remainder"
    assert state["observation_representation"] == "physical_field"
    assert state["checkpoint_representation"] == "homogeneous_remainder"
    assert state["silent_wall_overwrite_allowed"] is False


def test_p84_first_slice_is_narrow_and_future_capabilities_fail_closed():
    plan = _plan()
    first = plan["first_qualified_slice"]
    rejected = set(plan["explicit_rejections_first_slice"])
    out_of_scope = set(plan["out_of_scope"])

    assert first == {
        "model": "complete_stress_beris_edwards",
        "geometry": "plane_slab",
        "evolved_field": "Q",
        "bounded_axis": 2,
        "q_boundary": "static_prescribed_dirichlet_both_faces",
        "velocity_boundary": "free_slip_velocity",
        "pressure_boundary": "existing_plane_compatibility",
        "wall_data": "constant_per_face",
        "lower_and_upper_values_may_differ": True,
        "homogeneous_remainder_basis": ["fft", "fft", "dst"],
        "runtime_fallback_allowed": False,
    }
    assert {
        "time_dependent_boundary_data",
        "python_callable_boundary_data",
        "trainable_boundary_data",
        "prescribed_data_on_periodic_face",
        "spatially_varying_wall_data",
        "robin_or_surface_energy_anchoring",
        "channel_strong_anchoring",
    } <= rejected
    assert "finite_robin_or_tau_anchoring" in out_of_scope
    assert "phase_9" in out_of_scope


def test_p84_slice_order_and_h100_boundary_are_explicit():
    slices = _plan()["slices"]

    assert [value["id"] for value in slices] == [
        "P8.4.0",
        "P8.4.1",
        "P8.4.2",
        "P8.4.3",
        "P8.4.4",
        "P8.4.5",
    ]
    assert slices[0]["authorized"] is True
    assert slices[0]["implemented"] is True
    assert all(value["authorized"] is False for value in slices[1:])
    assert all(value["implemented"] is False for value in slices[1:])
    assert slices[-1]["requires_h100"] is True
    assert all(value["requires_h100"] is False for value in slices[:-1])


def test_p84_reviewed_sources_are_content_addressed():
    plan = _plan()
    baseline = plan["baseline"]["commit"]
    for relative, expected in plan["reviewed_source_sha256"].items():
        assert _sha256_at_commit(baseline, relative) == expected


def test_p84_records_local_planning_verification_without_h100_claim():
    plan = _plan()

    assert plan["local_verification"] == {
        "targeted": "18 passed",
        "full": "2252 passed, 8 subtests passed",
        "git_diff_check": "pass",
        "failed": 0,
        "skipped": 0,
        "xfailed": 0,
        "deselected": 0,
    }
    assert plan["authorization"]["h100_authorized"] is False


def test_p84_historical_plan_did_not_authorize_lifting_execution():
    plan = _plan()

    assert plan["authorization"]["p8_4_1_implementation_authorized"] is False
    assert plan["authorization"]["p8_4_implementation_authorized"] is False
    assert hasattr(core_boundary, "PrescribedDirichletBC")
    assert hasattr(public_boundaries, "prescribed_dirichlet")
    assert not hasattr(public_boundaries, "strong_planar_q")
    assert not hasattr(public_boundaries, "strong_homeotropic_q")
    with pytest.raises(ValueError, match="homogeneous boundary contracts only"):
        core_boundary.BoundaryCondition(
            kind=core_boundary.BoundaryKind.DIRICHLET,
            is_homogeneous=False,
        )


def test_future_verbatim_archive_lists_p84_without_regenerating_pdf():
    source = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "phase_8_p84_lifting_plan.md",
        "phase_8_p84_lifting_plan.json",
    ):
        assert source.count(f'"{name}"') == 1
