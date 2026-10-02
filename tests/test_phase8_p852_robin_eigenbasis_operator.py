"""P8.5.2 cell-centered Robin eigenbasis and CPU manufactured oracle."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest
import torch

from pssolver.core.boundary import StaticRobinCoefficients
from pssolver.operators import (
    CellCenteredRobinEigenbasisOperator,
    materialize_cell_centered_robin_eigenbasis,
)
from pssolver.planning import (
    CellCenteredRobinEigenbasisPlan,
    build_cell_centered_robin_eigenbasis_plan,
)


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def _coefficients(
    alpha: float,
    beta: float,
    gamma: float,
) -> StaticRobinCoefficients:
    return StaticRobinCoefficients(alpha, beta, gamma)


def _operator(
    size: int = 24,
    *,
    length: float = 2.3,
    lower: tuple[float, float, float] = (1.3, 0.8, 0.4),
    upper: tuple[float, float, float] = (0.7, 1.1, -0.2),
) -> CellCenteredRobinEigenbasisOperator:
    plan = build_cell_centered_robin_eigenbasis_plan(
        size=size,
        length=length,
        lower=_coefficients(*lower),
        upper=_coefficients(*upper),
    )
    return materialize_cell_centered_robin_eigenbasis(plan)


def test_robin_plan_uses_the_continuum_characteristic_and_stable_identity():
    plan = _operator().plan
    lower = (
        plan.length * plan.lower.alpha.value / plan.lower.beta.value
    )
    upper = (
        plan.length * plan.upper.alpha.value / plan.upper.beta.value
    )
    for root in plan.dimensionless_roots:
        residual = (
            (lower * upper - root * root) * math.sin(root)
            + root * (lower + upper) * math.cos(root)
        )
        scale = max(1.0, root * root, lower * upper, root * (lower + upper))
        assert abs(residual) / scale < 2.0e-13
    rebuilt = build_cell_centered_robin_eigenbasis_plan(
        size=plan.size,
        length=plan.length,
        lower=plan.lower,
        upper=plan.upper,
    )
    assert rebuilt == plan
    assert rebuilt.canonical_sha256() == plan.canonical_sha256()
    metadata = plan.to_metadata()
    assert metadata["operator_kind"] == "cell_centered_robin_eigenbasis"
    assert metadata["root_solve_location"] == "plan_construction_only"
    assert metadata["physical_reconstruction"] == (
        "homogeneous_remainder_plus_affine_lift"
    )
    json.dumps(metadata, allow_nan=False, sort_keys=True)


def test_raw_scaling_changes_provenance_but_not_normalized_eigenvalues():
    first = _operator().plan
    scaled = build_cell_centered_robin_eigenbasis_plan(
        size=first.size,
        length=first.length,
        lower=_coefficients(2.6, 1.6, 0.8),
        upper=_coefficients(2.1, 3.3, -0.6),
    )
    assert scaled.dimensionless_roots == pytest.approx(
        first.dimensionless_roots,
        rel=0.0,
        abs=2.0e-14,
    )
    assert scaled.canonical_sha256() != first.canonical_sha256()


def test_first_slice_rejects_unqualified_coefficients_and_nonzero_neumann_flux():
    for lower, match in (
        ((1.0, 0.0, 0.0), "beta must be positive"),
        ((-1.0, 2.0, 0.0), "alpha must be non-negative"),
    ):
        with pytest.raises(ValueError, match=match):
            build_cell_centered_robin_eigenbasis_plan(
                size=8,
                length=2.0,
                lower=_coefficients(*lower),
                upper=_coefficients(1.0, 2.0, 0.0),
            )
    plan = build_cell_centered_robin_eigenbasis_plan(
        size=8,
        length=2.0,
        lower=_coefficients(0.0, 1.0, 1.0),
        upper=_coefficients(0.0, 1.0, -1.0),
    )
    with pytest.raises(ValueError, match="nonzero Neumann flux"):
        materialize_cell_centered_robin_eigenbasis(plan)


def test_modal_roundtrip_batching_and_oriented_wall_residual_are_roundoff():
    operator = _operator(size=20)
    generator = torch.Generator().manual_seed(52)
    coefficients = torch.randn(
        (3, 2, operator.plan.size),
        dtype=torch.float64,
        generator=generator,
    )
    homogeneous = operator.from_modal(coefficients)
    physical = operator.reconstruct_physical(homogeneous)
    recovered = operator.to_modal(operator.homogeneous_remainder(physical))
    assert torch.max(torch.abs(recovered - coefficients)).item() < 5.0e-15
    lower, upper = operator.boundary_residual(physical)
    assert torch.max(torch.abs(lower)).item() < 2.0e-13
    assert torch.max(torch.abs(upper)).item() < 2.0e-11
    assert operator.to_metadata()["runtime_connected"] is False
    assert operator.to_metadata()["timestep_root_solve"] is False


def _manufactured_error(size: int) -> tuple[float, float, float, float]:
    length = 2.3
    mass = 1.7

    def exact(z: torch.Tensor) -> torch.Tensor:
        return (
            torch.exp(0.7 * z / length)
            + 0.15 * torch.cos(2.2 * z / length)
            + 0.03 * z**3
        )

    def derivative(z: torch.Tensor) -> torch.Tensor:
        return (
            0.7 / length * torch.exp(0.7 * z / length)
            - 0.15 * 2.2 / length * torch.sin(2.2 * z / length)
            + 0.09 * z**2
        )

    def second_derivative(z: torch.Tensor) -> torch.Tensor:
        return (
            (0.7 / length) ** 2 * torch.exp(0.7 * z / length)
            - 0.15 * (2.2 / length) ** 2 * torch.cos(2.2 * z / length)
            + 0.18 * z
        )

    lower_alpha, lower_beta = 1.3, 0.8
    upper_alpha, upper_beta = 0.7, 1.1
    zero = torch.tensor(0.0, dtype=torch.float64)
    end = torch.tensor(length, dtype=torch.float64)
    lower_gamma = lower_alpha * exact(zero) - lower_beta * derivative(zero)
    upper_gamma = upper_alpha * exact(end) + upper_beta * derivative(end)
    operator = _operator(
        size,
        length=length,
        lower=(lower_alpha, lower_beta, float(lower_gamma.item())),
        upper=(upper_alpha, upper_beta, float(upper_gamma.item())),
    )
    expected = exact(operator.coordinates)
    forcing = mass * expected - second_derivative(operator.coordinates)
    actual = operator.solve_helmholtz(forcing, mass=mass)
    relative_error = float(
        (
            torch.linalg.vector_norm(actual - expected)
            / torch.linalg.vector_norm(expected)
        ).item()
    )
    interior_residual = float(
        (
            torch.linalg.vector_norm(
                operator.apply_helmholtz(actual, mass=mass) - forcing
            )
            / torch.linalg.vector_norm(forcing)
        ).item()
    )
    lower_residual, upper_residual = operator.boundary_residual(actual)
    return (
        relative_error,
        interior_residual,
        abs(float(lower_residual.item())),
        abs(float(upper_residual.item())),
    )


def test_manufactured_helmholtz_solution_is_second_order_or_better():
    results = [_manufactured_error(size) for size in (8, 16, 32)]
    errors = [value[0] for value in results]
    observed_orders = [
        math.log(left / right, 2.0)
        for left, right in zip(errors, errors[1:])
    ]
    assert min(observed_orders) > 1.9
    assert errors[-1] < 2.1e-5
    assert max(value[1] for value in results) < 2.0e-12
    assert max(value[2] for value in results) < 2.0e-13
    assert max(value[3] for value in results) < 2.0e-12


def test_pure_neumann_limit_matches_cosine_mode_and_rejects_null_solve():
    length = 2.0
    plan = build_cell_centered_robin_eigenbasis_plan(
        size=16,
        length=length,
        lower=_coefficients(0.0, 1.0, 0.0),
        upper=_coefficients(0.0, 1.0, 0.0),
    )
    operator = materialize_cell_centered_robin_eigenbasis(plan)
    assert plan.is_pure_neumann is True
    assert plan.dimensionless_roots == pytest.approx(
        tuple(index * math.pi for index in range(plan.size)),
        rel=0.0,
        abs=0.0,
    )
    mass = 2.0
    expected = torch.cos(math.pi * operator.coordinates / length)
    forcing = (mass + (math.pi / length) ** 2) * expected
    actual = operator.solve_helmholtz(forcing, mass=mass)
    assert torch.equal(actual, expected) or torch.max(
        torch.abs(actual - expected)
    ).item() < 5.0e-15
    lower, upper = operator.boundary_residual(actual)
    assert abs(float(lower.item())) < 2.0e-14
    assert abs(float(upper.item())) < 2.0e-14
    with pytest.raises(ValueError, match="Neumann null mode"):
        operator.solve_helmholtz(
            torch.ones(plan.size, dtype=torch.float64),
            mass=0.0,
        )


@pytest.mark.parametrize("impedance", (1.0e-16, 1.0e-14, 1.0e-12))
def test_tiny_finite_impedance_continuously_approaches_neumann_roots(
    impedance,
):
    plan = build_cell_centered_robin_eigenbasis_plan(
        size=12,
        length=1.0,
        lower=_coefficients(impedance, 1.0, 0.0),
        upper=_coefficients(impedance, 1.0, 0.0),
    )

    assert plan.is_pure_neumann is False
    assert plan.dimensionless_roots[0] > 0.0
    assert plan.dimensionless_roots[0] == pytest.approx(
        math.sqrt(2.0 * impedance + impedance * impedance),
        rel=5.0e-8,
        abs=0.0,
    )
    for index, root in enumerate(plan.dimensionless_roots[1:], start=1):
        assert root == pytest.approx(index * math.pi, abs=2.0e-12)
        residual = (
            (impedance * impedance - root * root) * math.sin(root)
            + root * (2.0 * impedance) * math.cos(root)
        )
        scale = max(1.0, root * root, root * 2.0 * impedance)
        assert abs(residual) / scale < 3.0e-14


def test_increasing_finite_impedance_approaches_strong_dirichlet_control():
    length = 2.0
    size = 64
    mass = 1.3
    relative_errors = []
    for impedance in (10.0, 100.0, 1000.0):
        operator = _operator(
            size,
            length=length,
            lower=(impedance, 1.0, 0.0),
            upper=(impedance, 1.0, 0.0),
        )
        expected = torch.sin(math.pi * operator.coordinates / length)
        forcing = (mass + (math.pi / length) ** 2) * expected
        actual = operator.solve_helmholtz(forcing, mass=mass)
        relative_errors.append(
            float(
                (
                    torch.linalg.vector_norm(actual - expected)
                    / torch.linalg.vector_norm(expected)
                ).item()
            )
        )
    assert relative_errors[1] < 0.12 * relative_errors[0]
    assert relative_errors[2] < 0.12 * relative_errors[1]
    assert relative_errors[2] < 0.002


def test_operator_is_cpu_float64_only_and_does_not_claim_runtime_support():
    operator = _operator(size=8)
    with pytest.raises(ValueError, match="torch.float64"):
        operator.solve_helmholtz(
            torch.ones(8, dtype=torch.float32),
            mass=1.0,
        )
    with pytest.raises(ValueError, match="non-negative"):
        operator.solve_helmholtz(
            torch.ones(8, dtype=torch.float64),
            mass=-1.0,
        )
    assert isinstance(operator.plan, CellCenteredRobinEigenbasisPlan)
    assert operator.condition_number < 1.1


def test_planning_module_remains_tensor_free_and_model_neutral():
    path = ROOT / "pssolver" / "planning" / "robin.py"
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    assert "torch" not in imports
    assert "numpy" not in imports
    assert not any(value.startswith("pssolver.models") for value in imports)
    lowered = source.lower()
    assert "nematic" not in lowered
    assert "director" not in lowered
    assert "scalar_order" not in lowered


def test_p852_method_record_and_archive_are_present():
    record_path = NOTES / "phase_8_p852_robin_eigenbasis_operator.json"
    record = json.loads(record_path.read_text(encoding="utf-8"))
    assert record["phase"] == "P8.5.2"
    assert record["classification"] == (
        "PASS_P8_5_2_CELL_CENTERED_ROBIN_EIGENBASIS_CPU_ORACLE"
    )
    assert record["decision"]["selected"] == (
        "cell_centered_robin_eigenbasis"
    )
    assert record["decision"]["chebyshev_tau_selected"] is False
    assert record["runtime_connected"] is False
    assert record["h100_used"] is False
    assert record["authorization"]["eligible_for_p8_5_3"] is True
    assert record["authorization"]["p8_5_3_implemented"] is False
    archive_source = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "adr/0012-cell-centered-robin-eigenbasis.md",
        "phase_8_p852_robin_eigenbasis_operator.md",
        "phase_8_p852_robin_eigenbasis_operator.json",
    ):
        assert archive_source.count(f'"{name}"') == 1
