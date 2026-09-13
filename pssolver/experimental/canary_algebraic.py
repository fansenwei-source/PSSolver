"""Geometry-specific algebraic solvers used only by the Stage F canary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

import torch

from pssolver.execution import (
    AlgebraicSolverContext,
    AlgebraicSystemSpec,
    GeometrySolverRegistry,
)
from pssolver.geometries import PeriodicBox, PlaneSlab


_CAPABILITY = "scalar_helmholtz"


def _positive_parameter(system: AlgebraicSystemSpec, name: str) -> float:
    try:
        value = system.parameters[name]
    except KeyError as exc:
        raise ValueError(
            f"{system.name!r} is missing parameter {name!r}"
        ) from exc
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class _DiagonalHelmholtzSolver:
    capability: str
    implementation_name: str
    output_components: tuple[str, ...]
    dependency: str
    context: AlgebraicSolverContext
    denominator: torch.Tensor

    def solve_spectral(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        if set(state) != {self.dependency}:
            raise ValueError(
                "Helmholtz solver state must contain its exact dependency"
            )
        source_hat = self.context.forward_projected(
            self.dependency,
            state[self.dependency],
        )
        return {
            self.output_components[0]: source_hat / self.denominator,
        }


def _build_diagonal_helmholtz(
    context: AlgebraicSolverContext,
    system: AlgebraicSystemSpec,
    *,
    expected_geometry: str,
    implementation_name: str,
) -> _DiagonalHelmholtzSolver:
    if context.geometry_name != expected_geometry:
        raise ValueError(
            f"{implementation_name} requires geometry {expected_geometry!r}"
        )
    if system.capability != _CAPABILITY:
        raise ValueError(
            f"{implementation_name} cannot provide {system.capability!r}"
        )
    if len(system.output_components) != 1 or len(system.dependencies) != 1:
        raise ValueError("scalar Helmholtz systems need one input and one output")
    expected_parameters = {
        "helmholtz_length_sq",
        "helmholtz_shift",
    }
    if set(system.parameters) != expected_parameters:
        raise ValueError(
            "scalar Helmholtz parameters must be exactly "
            f"{tuple(sorted(expected_parameters))!r}"
        )
    shift = _positive_parameter(system, "helmholtz_shift")
    length_sq = _positive_parameter(system, "helmholtz_length_sq")
    output = system.output_components[0]
    dependency = system.dependencies[0]
    if context.boundary_conditions(output) != context.boundary_conditions(
        dependency
    ):
        raise ValueError(
            "scalar Helmholtz input and output must share one spectral basis"
        )
    denominator = shift - length_sq * context.laplacian_eigenvalues(output)
    if not bool(torch.isfinite(denominator).all().item()):
        raise ValueError("Helmholtz denominator contains non-finite values")
    if not bool(
        (denominator.abs() > torch.finfo(context.real_dtype).eps).all().item()
    ):
        raise ValueError("Helmholtz denominator contains a zero mode")
    return _DiagonalHelmholtzSolver(
        capability=_CAPABILITY,
        implementation_name=implementation_name,
        output_components=system.output_components,
        dependency=dependency,
        context=context,
        denominator=denominator,
    )


def _periodic_helmholtz_factory(
    context: AlgebraicSolverContext,
    system: AlgebraicSystemSpec,
) -> _DiagonalHelmholtzSolver:
    return _build_diagonal_helmholtz(
        context,
        system,
        expected_geometry="periodic_box",
        implementation_name="periodic_diagonal_helmholtz",
    )


def _plane_helmholtz_factory(
    context: AlgebraicSolverContext,
    system: AlgebraicSystemSpec,
) -> _DiagonalHelmholtzSolver:
    return _build_diagonal_helmholtz(
        context,
        system,
        expected_geometry="plane_slab",
        implementation_name="plane_mixed_basis_diagonal_helmholtz",
    )


def create_canary_geometry_solver_registry() -> GeometrySolverRegistry:
    """Return explicit PeriodicBox and PlaneSlab canary registrations."""

    registry = GeometrySolverRegistry()
    registry.register(
        geometry_type=PeriodicBox,
        geometry_name="periodic_box",
        capability=_CAPABILITY,
        implementation_name="periodic_diagonal_helmholtz",
        factory=_periodic_helmholtz_factory,
    )
    registry.register(
        geometry_type=PlaneSlab,
        geometry_name="plane_slab",
        capability=_CAPABILITY,
        implementation_name="plane_mixed_basis_diagonal_helmholtz",
        factory=_plane_helmholtz_factory,
    )
    return registry
