"""P8.5.4 quadratic finite-Q anchoring specialization."""

from __future__ import annotations

from dataclasses import replace
import ast
import json
from pathlib import Path

import numpy as np
import pytest

import pssolver
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.finite_q_anchoring import (
    PlaneFiniteQAnchoringLoweringPlan,
    apply_plane_finite_q_anchoring_pilot,
    lower_plane_finite_q_anchoring_pilot,
)
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.configuration.simulation_lowering import (
    SimulationLoweringError,
    lower_simulation_spec,
)
from pssolver.core.boundary import BoundarySide
from pssolver.models.active_nematics import (
    Q_COMPONENTS,
    Q_COMPONENT_METRIC,
    QUADRATIC_FINITE_Q_SURFACE_LAW_ID,
    QuadraticFiniteQAnchoring,
    finite_homeotropic_q_anchoring,
    finite_planar_q_anchoring,
    finite_q_variational_residual,
    q_component_metric,
    quadratic_finite_q_anchoring,
    uniaxial_Q,
)


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def _base_simulation(tmp_path: Path):
    run_spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=tmp_path / "unused_p854_output",
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
        nx=12,
        ny=10,
        nz=8,
        steps=2,
        save_start_step=0,
        save_interval=1,
        diagnostic_interval=1,
        defect_min_separation=2.0,
        defect_core_radius=0.5,
        twist_modes=(1, 2, 3),
    )
    return compose_plane_beris_edwards_simulation(
        decompose_plane_beris_edwards_run_spec(run_spec)
    )


def _normals():
    return {
        (2, "lower"): (0.0, 0.0, -1.0),
        (2, "upper"): (0.0, 0.0, 1.0),
    }


def _anchoring(simulation) -> QuadraticFiniteQAnchoring:
    return quadratic_finite_q_anchoring(
        k_q=simulation.equation_system.parameters["ldg_l1"],
        wall_strengths={(2, "lower"): 0.04, (2, "upper"): 0.07},
        target_q={
            (2, "lower"): uniaxial_Q(np.asarray([0.0, 0.0, 1.0]), 0.6),
            (2, "upper"): uniaxial_Q(np.asarray([1.0, 0.0, 0.0]), 0.5),
        },
    )


def _law_lookup(anchoring: QuadraticFiniteQAnchoring):
    return {
        value.key: value.coefficients
        for value in anchoring.to_robin_policy().face_laws
    }


def test_q_component_metric_is_the_full_tensor_gram_matrix_and_invertible():
    expected = np.asarray(
        (
            (2.0, 0.0, 0.0, 1.0, 0.0),
            (0.0, 2.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 2.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 2.0, 0.0),
            (0.0, 0.0, 0.0, 0.0, 2.0),
        )
    )
    assert Q_COMPONENT_METRIC == q_component_metric()
    np.testing.assert_array_equal(np.asarray(q_component_metric()), expected)
    assert np.linalg.det(expected) > 0.0


def test_full_tensor_variation_and_five_component_variation_are_identical():
    q = np.asarray((0.13, -0.04, 0.07, -0.09, 0.02))
    target = np.asarray((-0.2, 0.03, 0.05, 0.11, -0.08))
    derivative = np.asarray((0.2, -0.1, 0.05, 0.09, -0.03))
    result = finite_q_variational_residual(
        q=q,
        normal_derivative_q=derivative,
        target_q=target,
        k_q=0.031,
        wall_strength=0.17,
    )
    np.testing.assert_allclose(
        result["coordinate_variation"],
        result["projected_full_variation"],
        rtol=2.0e-15,
        atol=2.0e-15,
    )
    solved_derivative = -(0.17 / 0.031) * (q - target)
    zero = finite_q_variational_residual(
        q=q,
        normal_derivative_q=solved_derivative,
        target_q=target,
        k_q=0.031,
        wall_strength=0.17,
    )
    np.testing.assert_allclose(zero["component_law"], 0.0, atol=2.0e-17)
    np.testing.assert_allclose(
        zero["full_tensor_residual"], 0.0, atol=2.0e-17
    )


def test_quadratic_surface_adapter_emits_exact_generic_robin_coefficients(
    tmp_path,
):
    simulation = _base_simulation(tmp_path)
    anchoring = _anchoring(simulation)
    laws = _law_lookup(anchoring)
    lower = anchoring.faces[0]
    upper = anchoring.faces[1]
    for component_index, component in enumerate(Q_COMPONENTS):
        lower_law = laws[(component, 2, BoundarySide.LOWER)]
        assert lower_law.alpha.value == lower.wall_strength
        assert lower_law.beta.value == anchoring.k_q
        assert lower_law.gamma.value == pytest.approx(
            lower.wall_strength * lower.target_components[component_index]
        )
        upper_law = laws[(component, 2, BoundarySide.UPPER)]
        assert upper_law.alpha.value == upper.wall_strength
        assert upper_law.beta.value == anchoring.k_q
        assert upper_law.gamma.value == pytest.approx(
            upper.wall_strength * upper.target_components[component_index]
        )
    metadata = anchoring.to_metadata()
    assert metadata["surface_law_id"] == QUADRATIC_FINITE_Q_SURFACE_LAW_ID
    assert metadata["q_convention"]["id"] == "de_gennes_S_lambda_max_v1"
    assert metadata["component_order"] == list(Q_COMPONENTS)
    assert metadata["metric_cancellation"].startswith("same_invertible")
    json.dumps(metadata, allow_nan=False, sort_keys=True)


def test_homeotropic_and_planar_conveniences_use_canonical_q_convention():
    homeotropic = finite_homeotropic_q_anchoring(
        k_q=0.02,
        wall_strengths={(2, "lower"): 0.1, (2, "upper"): 0.2},
        scalar_order=0.6,
        face_normals=_normals(),
    )
    for face in homeotropic.faces:
        assert face.target_components[0] == pytest.approx(-0.3)
        assert face.target_components[3] == pytest.approx(-0.3)
    planar = finite_planar_q_anchoring(
        k_q=0.02,
        wall_strengths={(2, "lower"): 0.1, (2, "upper"): 0.2},
        scalar_order=0.6,
        face_directors={
            (2, "lower"): (1.0, 0.0, 0.0),
            (2, "upper"): (0.0, 1.0, 0.0),
        },
        face_normals=_normals(),
    )
    assert planar.faces[0].target_components[0] == pytest.approx(0.6)
    assert planar.faces[1].target_components[3] == pytest.approx(0.6)


def test_w_zero_is_neumann_but_target_provenance_remains_distinct():
    first = quadratic_finite_q_anchoring(
        k_q=0.03,
        wall_strengths={(2, "lower"): 0.0, (2, "upper"): 0.0},
        target_q={(2, "lower"): np.zeros(5), (2, "upper"): np.zeros(5)},
    )
    second = quadratic_finite_q_anchoring(
        k_q=0.03,
        wall_strengths={(2, "lower"): 0.0, (2, "upper"): 0.0},
        target_q={
            (2, "lower"): np.asarray((0.2, 0.0, 0.0, -0.1, 0.0)),
            (2, "upper"): np.asarray((-0.1, 0.0, 0.0, 0.2, 0.0)),
        },
    )
    assert first.to_robin_policy() == second.to_robin_policy()
    assert first.canonical_sha256() != second.canonical_sha256()
    for law in first.to_robin_policy().face_laws:
        assert law.coefficients.alpha.value == 0.0
        assert law.coefficients.beta.value == 0.03
        assert law.coefficients.gamma.value == 0.0
    assert all(
        face["extrapolation_length"] is None
        for face in first.to_metadata()["faces"]
    )


@pytest.mark.parametrize("k_q", [0.0, -0.1, float("nan"), float("inf")])
def test_adapter_rejects_unqualified_elastic_coefficients(k_q):
    with pytest.raises(ValueError, match="positive and finite"):
        quadratic_finite_q_anchoring(
            k_q=k_q,
            wall_strengths={(2, "lower"): 0.1},
            target_q={(2, "lower"): np.zeros(5)},
        )


def test_adapter_rejects_invalid_strength_target_and_planar_data():
    with pytest.raises(ValueError, match="non-negative"):
        quadratic_finite_q_anchoring(
            k_q=0.02,
            wall_strengths={(2, "lower"): -0.1},
            target_q={(2, "lower"): np.zeros(5)},
        )
    invalid = np.eye(3)
    with pytest.raises(ValueError, match="traceless"):
        quadratic_finite_q_anchoring(
            k_q=0.02,
            wall_strengths={(2, "lower"): 0.1},
            target_q={(2, "lower"): invalid},
        )
    with pytest.raises(ValueError, match="tangent"):
        finite_planar_q_anchoring(
            k_q=0.02,
            wall_strengths={(2, "lower"): 0.1},
            scalar_order=0.6,
            face_directors={(2, "lower"): (0.0, 0.0, 1.0)},
            face_normals={(2, "lower"): (0.0, 0.0, 1.0)},
        )


def test_first_plane_application_lowers_all_five_q_components(tmp_path):
    base = _base_simulation(tmp_path)
    anchoring = _anchoring(base)
    simulation = apply_plane_finite_q_anchoring_pilot(base, anchoring)
    plan = lower_plane_finite_q_anchoring_pilot(simulation, anchoring)
    assert isinstance(plan, PlaneFiniteQAnchoringLoweringPlan)
    assert tuple(value.component for value in plan.component_plans) == Q_COMPONENTS
    assert plan.anchoring_sha256 == anchoring.canonical_sha256()
    assert plan.anchoring_metadata["surface_law_id"] == (
        QUADRATIC_FINITE_Q_SURFACE_LAW_ID
    )
    laws = _law_lookup(anchoring)
    for component in Q_COMPONENTS:
        component_plan = plan.for_component(component)
        for side, robin_plan in (
            (BoundarySide.LOWER, component_plan.robin_plan.lower),
            (BoundarySide.UPPER, component_plan.robin_plan.upper),
        ):
            expected = laws[(component, 2, side)]
            assert robin_plan == expected
    metadata = plan.to_metadata()
    assert metadata["complete_q_timestep_connected"] is False
    assert metadata["public_runner_connected"] is False


def test_first_application_requires_explicit_matching_ldg_l1_and_identity(
    tmp_path,
):
    base = _base_simulation(tmp_path)
    anchoring = _anchoring(base)
    simulation = apply_plane_finite_q_anchoring_pilot(base, anchoring)
    wrong_k = quadratic_finite_q_anchoring(
        k_q=anchoring.k_q * 2.0,
        wall_strengths={(2, "lower"): 0.04, (2, "upper"): 0.07},
        target_q={
            (2, "lower"): anchoring.faces[0].target_full,
            (2, "upper"): anchoring.faces[1].target_full,
        },
    )
    with pytest.raises(ValueError, match="exactly equal every qualified ldg_l1"):
        apply_plane_finite_q_anchoring_pilot(base, wrong_k)
    with pytest.raises(ValueError, match="requires legacy_production"):
        apply_plane_finite_q_anchoring_pilot(
            replace(
                base,
                execution=replace(base.execution, runtime_path="compiled_v2"),
            ),
            anchoring,
        )
    with pytest.raises(ValueError, match="provenance is missing"):
        lower_plane_finite_q_anchoring_pilot(
            replace(simulation, compatibility_metadata={}),
            anchoring,
        )
    with pytest.raises(ValueError, match="boundaries do not match"):
        lower_plane_finite_q_anchoring_pilot(
            replace(simulation, boundaries=base.boundaries),
            anchoring,
        )


def test_production_lowering_and_public_runner_remain_disconnected(tmp_path):
    base = _base_simulation(tmp_path)
    anchoring = _anchoring(base)
    simulation = apply_plane_finite_q_anchoring_pilot(base, anchoring)
    with pytest.raises(SimulationLoweringError) as error:
        lower_simulation_spec(simulation)
    assert error.value.rejection.code.value == "unsupported_robin_boundary"
    assert not hasattr(pssolver, "quadratic_finite_q_anchoring")
    assert not hasattr(pssolver, "lower_plane_finite_q_anchoring_pilot")


def test_generic_robin_layers_remain_free_of_active_nematic_imports():
    for relative in (
        "pssolver/boundaries/robin.py",
        "pssolver/planning/robin.py",
        "pssolver/operators/robin.py",
        "pssolver/runtime/robin_scalar.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(value.name for value in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.append(node.module)
        assert not any(
            value == "pssolver.models"
            or value.startswith("pssolver.models.")
            for value in imports
        )


def test_p854_record_and_future_archive_are_present():
    record = json.loads(
        (NOTES / "phase_8_p854_finite_q_anchoring.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["phase"] == "P8.5.4"
    assert record["classification"] == (
        "PASS_P8_5_4_QUADRATIC_FINITE_Q_ANCHORING_SPECIALIZATION"
    )
    assert record["complete_q_timestep_connected"] is False
    assert record["h100_used"] is False
    assert record["authorization"]["eligible_for_p8_5_5"] is True
    assert record["authorization"]["p8_5_5_implemented"] is False
    archive = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "phase_8_p854_finite_q_anchoring.md",
        "phase_8_p854_finite_q_anchoring.json",
    ):
        assert archive.count(f'"{name}"') == 1
