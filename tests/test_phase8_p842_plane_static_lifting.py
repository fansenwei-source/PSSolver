"""P8.4.2 field-neutral Plane static-lifting tests and manufactured oracle."""

from __future__ import annotations

import ast
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import pytest
import torch

from pssolver.boundaries import (
    HomogeneousBoundaryPolicy,
    assign_boundaries,
    prescribed_dirichlet,
)
from pssolver.configuration.lifting import build_plane_static_lifting_plan
from pssolver.core.boundary import BoundarySemantic
from pssolver.core.domain import DomainSpec, GridPlacement
from pssolver.core.fields import FieldRole
from pssolver.geometries import PlaneSlab, RectangularChannel
from pssolver.operators import materialize_plane_static_lifting
from pssolver.planning import (
    STATIC_LIFTING_PLAN_SCHEMA_VERSION,
    StaticLiftExtension,
    StaticLiftingPlan,
)
from pssolver.systems.algebraic import AlgebraicSystemSpec
from pssolver.systems.equations import (
    EquationFieldSpec,
    EquationSystemSpec,
    EquationTermSpec,
)


ROOT = Path(__file__).resolve().parents[1]
RECORD_PATH = ROOT / (
    "notes/architecture_v0_2/phase_8_p842_plane_static_lifting.json"
)
IMPLEMENTATION_SOURCES = (
    "pssolver/configuration/lifting.py",
    "pssolver/planning/lifting.py",
    "pssolver/operators/lifting.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _generic_system() -> EquationSystemSpec:
    return EquationSystemSpec(
        name="generic_transport",
        variant="generic_scalar_transport",
        fields=(
            EquationFieldSpec(
                "concentration",
                FieldRole.EVOLVED,
                ("c",),
            ),
            EquationFieldSpec("flux", FieldRole.TRANSIENT, ("j",)),
            EquationFieldSpec(
                "multiplier",
                FieldRole.ALGEBRAIC,
                ("lambda_value",),
            ),
        ),
        evolution_laws=(
            EquationTermSpec(
                "evolve_concentration",
                "generic_transport_rhs",
                ("c",),
                ("c", "j"),
            ),
        ),
        constitutive_laws=(
            EquationTermSpec(
                "construct_flux",
                "generic_flux",
                ("j",),
                ("c",),
            ),
        ),
        algebraic_systems=(
            AlgebraicSystemSpec(
                "solve_multiplier",
                "generic_constraint",
                ("lambda_value",),
                ("c", "j"),
            ),
        ),
        parameters={"diffusivity": 0.2, "reaction_rate": -0.4},
    )


def _geometry(
    shape: tuple[int, ...] = (8, 6, 16),
    *,
    grid_placement: GridPlacement = GridPlacement.CELL_CENTERED,
) -> PlaneSlab:
    lengths = (4.0, 3.0, 2.0)[-len(shape) :]
    return PlaneSlab(
        DomainSpec(
            shape,
            lengths,
            grid_placement=grid_placement,
        )
    )


def _boundaries(model: EquationSystemSpec, geometry):
    wall_axis = geometry.bounded_axes[-1]
    return assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "concentration": prescribed_dirichlet(
                "concentration",
                {
                    ("c", wall_axis, "lower"): 1.25,
                    ("c", wall_axis, "upper"): -0.5,
                },
            ),
            "multiplier": HomogeneousBoundaryPolicy(
                "multiplier",
                BoundarySemantic.ALGEBRAIC_COMPATIBILITY,
                "neumann",
            ),
        },
    )


def _plan(shape: tuple[int, ...] = (8, 6, 16)) -> StaticLiftingPlan:
    model = _generic_system()
    geometry = _geometry(shape)
    return build_plane_static_lifting_plan(
        equation_system=model,
        geometry=geometry,
        boundaries=_boundaries(model, geometry),
    )


def test_plane_lifting_plan_is_field_neutral_stable_and_content_addressed():
    first = _plan()
    second = _plan()
    component = first.for_component("c")

    assert first == second
    assert first.canonical_sha256() == second.canonical_sha256()
    assert len(first.canonical_sha256()) == 64
    assert first.extension is StaticLiftExtension.AFFINE_WALL_NORMAL
    assert first.wall_normal_axis == 2
    assert first.component_order == ("c",)
    assert component.field_name == "concentration"
    assert component.value_at_fraction(0.0) == 1.25
    assert component.value_at_fraction(1.0) == -0.5
    assert component.value_at_fraction(0.5) == 0.375
    assert first.to_metadata()["schema_version"] == (
        STATIC_LIFTING_PLAN_SCHEMA_VERSION
    )
    json.dumps(first.to_metadata(), allow_nan=False, sort_keys=True)


def test_plane_lifting_plan_rejects_unregistered_geometry_grid_and_empty_data():
    model = _generic_system()
    channel = RectangularChannel(
        DomainSpec((8, 6, 4), (4.0, 3.0, 2.0))
    )
    channel_boundaries = _boundaries(model, _geometry((8, 6, 4)))
    with pytest.raises(ValueError, match="one-bounded-axis plane_slab"):
        build_plane_static_lifting_plan(
            equation_system=model,
            geometry=channel,
            boundaries=channel_boundaries,
        )

    node_geometry = _geometry(grid_placement=GridPlacement.NODE_CENTERED)
    with pytest.raises(ValueError, match="cell-centered"):
        build_plane_static_lifting_plan(
            equation_system=model,
            geometry=node_geometry,
            boundaries=_boundaries(model, node_geometry),
        )

    homogeneous = assign_boundaries(
        model=model,
        geometry=_geometry(),
        policies={
            "concentration": HomogeneousBoundaryPolicy(
                "concentration",
                BoundarySemantic.PHYSICAL,
                "neumann",
            ),
            "multiplier": HomogeneousBoundaryPolicy(
                "multiplier",
                BoundarySemantic.ALGEBRAIC_COMPATIBILITY,
                "neumann",
            ),
        },
    )
    with pytest.raises(ValueError, match="no prescribed Dirichlet"):
        build_plane_static_lifting_plan(
            equation_system=model,
            geometry=_geometry(),
            boundaries=homogeneous,
        )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_materialized_affine_lift_uses_cell_centers_and_exact_wall_contract(dtype):
    plan = _plan()
    operator = materialize_plane_static_lifting(
        plan,
        dtype=dtype,
        device="cpu",
    )
    lift = operator.lift("c")
    component = plan.for_component("c")
    count = plan.domain_shape[plan.wall_normal_axis]

    assert lift.shape == plan.domain_shape
    assert lift.dtype is dtype
    assert lift.is_contiguous()
    assert torch.count_nonzero(operator.affine_laplacian("c")) == 0
    expected_lower_cell = component.value_at_fraction(0.5 / count)
    expected_upper_cell = component.value_at_fraction((count - 0.5) / count)
    torch.testing.assert_close(
        lift[..., 0],
        torch.full_like(lift[..., 0], expected_lower_cell),
        rtol=1.0e-6 if dtype is torch.float32 else 0.0,
        atol=1.0e-6 if dtype is torch.float32 else 0.0,
    )
    torch.testing.assert_close(
        lift[..., -1],
        torch.full_like(lift[..., -1], expected_upper_cell),
        rtol=1.0e-6 if dtype is torch.float32 else 0.0,
        atol=1.0e-6 if dtype is torch.float32 else 0.0,
    )
    assert component.value_at_fraction(0.0) == 1.25
    assert component.value_at_fraction(1.0) == -0.5
    assert len(operator.materialized_sha256) == 64
    operator.verify_materialized_identity()


def test_physical_reconstruction_and_remainder_extraction_support_workspace():
    operator = materialize_plane_static_lifting(
        _plan(),
        dtype=torch.float64,
        device="cpu",
    )
    shape = operator.plan.domain_shape
    remainder = torch.linspace(
        -0.125,
        0.125,
        math.prod(shape),
        dtype=torch.float64,
    ).reshape(shape)
    physical_workspace = torch.empty_like(remainder)
    remainder_workspace = torch.empty_like(remainder)

    physical = operator.reconstruct_physical(
        "c",
        remainder,
        out=physical_workspace,
    )
    recovered = operator.extract_homogeneous_remainder(
        "c",
        physical,
        out=remainder_workspace,
    )

    assert physical is physical_workspace
    assert recovered is remainder_workspace
    torch.testing.assert_close(recovered, remainder, rtol=0.0, atol=1.2e-16)
    with pytest.raises(ValueError, match="must not alias"):
        operator.reconstruct_physical("c", remainder, out=remainder)
    with pytest.raises(ValueError, match="shape"):
        operator.extract_homogeneous_remainder(
            "c",
            torch.zeros((2, 2), dtype=torch.float64),
        )


def test_explicit_linear_lift_correction_is_model_supplied_and_one_time():
    operator = materialize_plane_static_lifting(
        _plan(),
        dtype=torch.float64,
        device="cpu",
    )
    correction = operator.materialize_linear_correction(
        "c",
        operator_name="linear_reaction",
        linear_operator=lambda value: -0.4 * value,
    )

    torch.testing.assert_close(
        correction.values,
        -0.4 * operator.lift("c"),
        rtol=0.0,
        atol=0.0,
    )
    assert correction.source_lift_sha256 == operator.component_lift_sha256("c")
    assert len(correction.values_sha256) == 64
    assert correction.to_metadata()["operator_name"] == "linear_reaction"
    operator.verify_materialized_identity()

    with pytest.raises(ValueError, match="shape"):
        operator.materialize_linear_correction(
            "c",
            operator_name="wrong_shape",
            linear_operator=lambda value: value[..., :-1],
        )
    with pytest.raises(ValueError, match="finite"):
        operator.materialize_linear_correction(
            "c",
            operator_name="nonfinite",
            linear_operator=lambda value: value.fill_(float("nan")),
        )


def test_materialized_lift_mutation_is_detected_before_provenance_use():
    operator = materialize_plane_static_lifting(
        _plan(),
        dtype=torch.float64,
        device="cpu",
    )

    operator.lift("c").add_(1.0)
    with pytest.raises(RuntimeError, match="identity mismatch"):
        operator.verify_materialized_identity()


def _homogeneous_cell_centered_second_derivative(
    value: torch.Tensor,
    spacing: float,
) -> torch.Tensor:
    result = torch.empty_like(value)
    result[..., 1:-1] = (
        value[..., 2:] - 2.0 * value[..., 1:-1] + value[..., :-2]
    ) / spacing**2
    lower_ghost = -value[..., 0]
    upper_ghost = -value[..., -1]
    result[..., 0] = (
        value[..., 1] - 2.0 * value[..., 0] + lower_ghost
    ) / spacing**2
    result[..., -1] = (
        upper_ghost - 2.0 * value[..., -1] + value[..., -2]
    ) / spacing**2
    return result


def _manufactured_laplacian_error(wall_points: int) -> float:
    plan = _plan((4, wall_points))
    operator = materialize_plane_static_lifting(
        plan,
        dtype=torch.float64,
        device="cpu",
    )
    length = plan.domain_lengths[plan.wall_normal_axis]
    spacing = length / wall_points
    z = (torch.arange(wall_points, dtype=torch.float64) + 0.5) * spacing
    remainder_line = torch.sin(math.pi * z / length)
    remainder = remainder_line.expand(plan.domain_shape).clone()
    physical = operator.reconstruct_physical("c", remainder)
    recovered = operator.extract_homogeneous_remainder("c", physical)
    numerical = _homogeneous_cell_centered_second_derivative(
        recovered,
        spacing,
    ) + operator.affine_laplacian("c")
    exact = -((math.pi / length) ** 2) * remainder
    return float(torch.sqrt(torch.mean((numerical - exact).square())).item())


def test_lifted_manufactured_diffusion_retains_second_order_convergence():
    errors = tuple(_manufactured_laplacian_error(size) for size in (16, 32, 64))
    rates = tuple(
        math.log(errors[index] / errors[index + 1], 2.0)
        for index in range(len(errors) - 1)
    )

    assert errors[0] > errors[1] > errors[2]
    assert all(rate > 1.95 for rate in rates)


def test_planning_is_tensor_free_and_operator_is_model_runtime_free():
    planning_path = ROOT / "pssolver/planning/lifting.py"
    operator_path = ROOT / "pssolver/operators/lifting.py"
    planning_tree = ast.parse(
        planning_path.read_text(encoding="utf-8"),
        filename=str(planning_path),
    )
    planning_imports = {
        node.module
        for node in ast.walk(planning_tree)
        if isinstance(node, ast.ImportFrom) and node.module
    }
    planning_imports.update(
        alias.name
        for node in ast.walk(planning_tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    )
    assert not any(
        name == "torch" or name.startswith("torch.")
        for name in planning_imports
    )

    operator_source = operator_path.read_text(encoding="utf-8").lower()
    for forbidden in (
        "pssolver.models",
        "pssolver.runtime",
        "pssolver.workflows",
        "director",
        "scalar_order",
        "nematic",
    ):
        assert forbidden not in operator_source


def test_p842_record_freezes_local_operator_scope_without_runtime_claim():
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))

    assert record["phase"] == "P8.4.2"
    assert record["classification"] == (
        "PASS_P8_4_2_PLANE_STATIC_LIFTING_AND_MANUFACTURED_ORACLE"
    )
    assert record["field_neutral"] is True
    assert record["q_conveniences_implemented"] is False
    assert record["production_runtime_connected"] is False
    assert record["nonhomogeneous_neumann_supported"] is False
    assert record["h100_used"] is False
    assert record["authorization"]["p8_4_3_eligible_for_planning"] is True
    assert record["authorization"]["p8_4_3_implementation_authorized"] is False
    assert set(record["source_sha256"]) == set(IMPLEMENTATION_SOURCES)
    superseded_by_p844 = {"pssolver/operators/lifting.py"}
    for relative, expected in record["source_sha256"].items():
        if relative in superseded_by_p844:
            continue
        assert _sha256(ROOT / relative) == expected


def test_future_verbatim_archive_lists_p842_without_regenerating_pdf():
    source = (
        ROOT / "notes/architecture_v0_2/build_verbatim_archive_pdf.py"
    ).read_text(encoding="utf-8")
    for name in (
        "phase_8_p842_plane_static_lifting.md",
        "phase_8_p842_plane_static_lifting.json",
    ):
        assert source.count(f'"{name}"') == 1
