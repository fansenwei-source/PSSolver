"""Opt-in legacy Plane and Channel adapters for the Stage G Stokes contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real
from types import SimpleNamespace

import torch

from pssolver.channel import ModalSaddleStokesCompute
from pssolver.execution import (
    AlgebraicSolverContext,
    AlgebraicSystemSpec,
    GeometrySolverRegistry,
    INCOMPRESSIBLE_STOKES_CAPABILITY,
    IncompressibleStokesSystemSpec,
    PressureGauge,
    TangentialZeroModePolicy,
)
from pssolver.geometries import PlaneSlab, RectangularChannel
from pssolver.transforms import FreeSlipModalStokesSolver

from .model_execution import LegacyAlgebraicSolverContext


_PLANE_TANGENTIAL_BCS = ("periodic", "periodic", "neumann")
_PLANE_NORMAL_BCS = ("periodic", "periodic", "dirichlet")
_PLANE_PRESSURE_BCS = ("periodic", "periodic", "neumann")
_CHANNEL_VELOCITY_BCS = ("periodic", "dirichlet", "dirichlet")
_CHANNEL_PRESSURE_BCS = ("periodic", "neumann", "neumann")


def _finite_positive(value: object, description: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{description} must be positive and finite")
    return float(value)


@dataclass(frozen=True, slots=True)
class PlaneStokesSolverOptions:
    """Runtime-only diagnostics configuration for the direct Plane solve."""

    pressure_diagnostics: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.pressure_diagnostics, bool):
            raise TypeError("pressure_diagnostics must be a bool")


@dataclass(frozen=True, slots=True)
class ChannelStokesSolverOptions:
    """Runtime-only convergence controls for the Channel pressure solve."""

    pressure_relative_tolerance: float = 1.0e-6
    pressure_max_iterations: int = 80
    pressure_fixed_iterations: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pressure_relative_tolerance",
            _finite_positive(
                self.pressure_relative_tolerance,
                "pressure_relative_tolerance",
            ),
        )
        if (
            not isinstance(self.pressure_max_iterations, int)
            or isinstance(self.pressure_max_iterations, bool)
            or self.pressure_max_iterations <= 0
        ):
            raise ValueError("pressure_max_iterations must be positive")
        fixed = self.pressure_fixed_iterations
        if fixed is not None and (
            not isinstance(fixed, int)
            or isinstance(fixed, bool)
            or fixed <= 0
        ):
            raise ValueError("pressure_fixed_iterations must be positive or None")


def _legacy_context(
    context: AlgebraicSolverContext,
) -> LegacyAlgebraicSolverContext:
    if not isinstance(context, LegacyAlgebraicSolverContext):
        raise TypeError(
            "Stage G legacy Stokes adapters require the opt-in legacy context"
        )
    if len(context.physical_shape) != 3:
        raise ValueError("Stage G Stokes adapters require three dimensions")
    return context


def _require_boundaries(
    context: LegacyAlgebraicSolverContext,
    spec: IncompressibleStokesSystemSpec,
    *,
    velocity_boundaries: tuple[tuple[str, ...], ...],
    pressure_boundaries: tuple[str, ...],
) -> None:
    for dependency, output, expected in zip(
        spec.force_components,
        spec.velocity_components,
        velocity_boundaries,
    ):
        if context.boundary_conditions(dependency) != expected:
            raise ValueError(
                f"force component {dependency!r} has the wrong boundary space"
            )
        if context.boundary_conditions(output) != expected:
            raise ValueError(
                f"velocity component {output!r} has the wrong boundary space"
            )
    if context.boundary_conditions(spec.pressure_component) != pressure_boundaries:
        raise ValueError("pressure component has the wrong boundary space")


class _LegacyStokesAlgebraicSolver:
    capability = INCOMPRESSIBLE_STOKES_CAPABILITY

    def __init__(
        self,
        *,
        context: LegacyAlgebraicSolverContext,
        spec: IncompressibleStokesSystemSpec,
        implementation_name: str,
        lower_solver,
        pressure_diagnostics_enabled: bool,
        warm_start_supported: bool,
        pressure_algorithm: str,
        solver_options: Mapping[str, object],
    ) -> None:
        self.context = context
        self.spec = spec
        self.implementation_name = implementation_name
        self.output_components = (
            *spec.velocity_components,
            spec.pressure_component,
        )
        self.lower_solver = lower_solver
        self._pressure_diagnostics_enabled = pressure_diagnostics_enabled
        self._warm_start_supported = warm_start_supported
        self._pressure_algorithm = pressure_algorithm
        self._solver_options = dict(solver_options)
        self._solve_count = 0
        self._last_warm_start_used = False

    @property
    def physical_dependencies(self) -> tuple[str, ...]:
        if self.context.lazy_physical_materialization:
            return ()
        return self.spec.force_components

    def solve_spectral(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        if set(state) != set(self.spec.force_components):
            raise ValueError("Stokes state must contain exactly three forces")
        force_hats = tuple(
            self.context.spectral_dependency(state, name)
            for name in self.spec.force_components
        )
        self._last_warm_start_used = bool(
            self._warm_start_supported
            and self.lower_solver.pressure_guess is not None
        )
        solution = self.lower_solver.solve_force_hats(*force_hats)
        self._solve_count += 1
        return dict(zip(self.output_components, solution, strict=True))

    def observability_metadata(self) -> Mapping[str, object]:
        return {
            "diagnostics": {
                "kind": "out_of_state_snapshot",
                "pressure_residuals_enabled": (
                    self._pressure_diagnostics_enabled
                ),
            },
            "pressure_algorithm": self._pressure_algorithm,
            "solver_options": self._solver_options,
            "restart": {
                "kind": (
                    "warm_start" if self._warm_start_supported else "stateless"
                ),
                "state_keys": (
                    ["pressure_guess"] if self._warm_start_supported else []
                ),
            },
        }

    def diagnostic_snapshot(self) -> Mapping[str, object]:
        residual = self.lower_solver.last_pressure_residual
        relative = self.lower_solver.last_pressure_relative_residual
        return {
            "last_pressure_iterations": int(
                self.lower_solver.last_pressure_iterations
            ),
            "last_pressure_residual": (
                float(residual) if math.isfinite(float(residual)) else None
            ),
            "last_pressure_relative_residual": (
                float(relative) if math.isfinite(float(relative)) else None
            ),
            "last_warm_start_used": self._last_warm_start_used,
            "pressure_diagnostics_enabled": (
                self._pressure_diagnostics_enabled
            ),
            "solve_count": self._solve_count,
        }

    def capture_restart_state(self) -> Mapping[str, torch.Tensor]:
        if not self._warm_start_supported:
            return {}
        pressure_guess = self.lower_solver.pressure_guess
        if pressure_guess is None:
            return {}
        return {"pressure_guess": pressure_guess.detach().clone()}

    def restore_restart_state(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("restart state must be a mapping")
        state = dict(state)
        if not self._warm_start_supported:
            if state:
                raise ValueError("stateless Plane solve cannot restore state")
            return
        if not state:
            self.lower_solver.pressure_guess = None
            self._last_warm_start_used = False
            return
        if set(state) != {"pressure_guess"}:
            raise ValueError("Channel restart state requires pressure_guess")
        pressure_guess = state["pressure_guess"]
        expected_shape = (
            self.context.batch_size,
            *self.context.spectral_shape,
        )
        if not isinstance(pressure_guess, torch.Tensor):
            raise TypeError("pressure_guess must be a tensor")
        if pressure_guess.shape != expected_shape:
            raise ValueError("pressure_guess shape does not match this runtime")
        if pressure_guess.dtype != self.context.spectral_dtype:
            raise ValueError("pressure_guess dtype does not match this runtime")
        if pressure_guess.device != self.context.device:
            raise ValueError("pressure_guess device does not match this runtime")
        if not bool(torch.isfinite(pressure_guess).all().item()):
            raise ValueError("pressure_guess must be finite")
        self.lower_solver.pressure_guess = (
            self.lower_solver._project_pressure_gauge(pressure_guess)
            .detach()
            .clone()
        )
        self._last_warm_start_used = False


def _plane_factory(
    context: AlgebraicSolverContext,
    system: AlgebraicSystemSpec,
    *,
    options: PlaneStokesSolverOptions,
) -> _LegacyStokesAlgebraicSolver:
    context = _legacy_context(context)
    spec = IncompressibleStokesSystemSpec.from_algebraic_system_spec(system)
    if context.geometry_name != "plane_slab":
        raise ValueError("Plane Stokes implementation requires PlaneSlab")
    if spec.pressure_gauge is not PressureGauge.ZERO_MEAN:
        raise ValueError("Plane Stokes currently requires zero-mean pressure")
    if spec.tangential_zero_mode_policy not in {
        TangentialZeroModePolicy.ZERO_MEAN,
        TangentialZeroModePolicy.FRICTION,
    }:
        raise ValueError(
            "Plane Stokes requires zero_mean or friction tangential policy"
        )
    velocity_boundaries = (
        _PLANE_TANGENTIAL_BCS,
        _PLANE_TANGENTIAL_BCS,
        _PLANE_NORMAL_BCS,
    )
    _require_boundaries(
        context,
        spec,
        velocity_boundaries=velocity_boundaries,
        pressure_boundaries=_PLANE_PRESSURE_BCS,
    )
    lower = FreeSlipModalStokesSolver(
        context.legacy_transform_backend,
        tangential_boundary_conditions=_PLANE_TANGENTIAL_BCS,
        normal_boundary_conditions=_PLANE_NORMAL_BCS,
        pressure_boundary_conditions=_PLANE_PRESSURE_BCS,
        friction=spec.friction,
        viscosity=spec.viscosity,
        zero_mode_policy=spec.tangential_zero_mode_policy.value,
        pressure_diagnostics=options.pressure_diagnostics,
    )
    return _LegacyStokesAlgebraicSolver(
        context=context,
        spec=spec,
        implementation_name="plane_free_slip_modal_stokes",
        lower_solver=lower,
        pressure_diagnostics_enabled=options.pressure_diagnostics,
        warm_start_supported=False,
        pressure_algorithm="diagonal_schur",
        solver_options={
            "pressure_diagnostics": options.pressure_diagnostics,
        },
    )


def _channel_factory(
    context: AlgebraicSolverContext,
    system: AlgebraicSystemSpec,
    *,
    options: ChannelStokesSolverOptions,
) -> _LegacyStokesAlgebraicSolver:
    context = _legacy_context(context)
    spec = IncompressibleStokesSystemSpec.from_algebraic_system_spec(system)
    if context.geometry_name != "rectangular_channel":
        raise ValueError(
            "Channel Stokes implementation requires RectangularChannel"
        )
    if spec.pressure_gauge is not PressureGauge.ZERO_MEAN:
        raise ValueError("Channel Stokes currently requires zero-mean pressure")
    if (
        spec.tangential_zero_mode_policy
        is not TangentialZeroModePolicy.NOT_APPLICABLE
    ):
        raise ValueError(
            "no-slip Channel has no uniform tangential null mode; use "
            "not_applicable"
        )
    velocity_boundaries = (_CHANNEL_VELOCITY_BCS,) * 3
    _require_boundaries(
        context,
        spec,
        velocity_boundaries=velocity_boundaries,
        pressure_boundaries=_CHANNEL_PRESSURE_BCS,
    )
    backend = context.legacy_transform_backend
    legacy_solver_view = SimpleNamespace(
        transform_backend=backend,
        qx=SimpleNamespace(device=backend.device),
    )
    lower = ModalSaddleStokesCompute(
        legacy_solver_view,
        beta=0.0,
        friction=spec.friction,
        viscosity=spec.viscosity,
        pressure_rel_tol=options.pressure_relative_tolerance,
        pressure_max_iter=options.pressure_max_iterations,
        pressure_fixed_iterations=options.pressure_fixed_iterations,
    )
    return _LegacyStokesAlgebraicSolver(
        context=context,
        spec=spec,
        implementation_name="channel_no_slip_modal_stokes",
        lower_solver=lower,
        pressure_diagnostics_enabled=True,
        warm_start_supported=True,
        pressure_algorithm="preconditioned_conjugate_gradient",
        solver_options={
            "pressure_fixed_iterations": options.pressure_fixed_iterations,
            "pressure_max_iterations": options.pressure_max_iterations,
            "pressure_relative_tolerance": (
                options.pressure_relative_tolerance
            ),
        },
    )


def create_stokes_geometry_solver_registry(
    *,
    plane_options: PlaneStokesSolverOptions | None = None,
    channel_options: ChannelStokesSolverOptions | None = None,
) -> GeometrySolverRegistry:
    """Register distinct Plane and Channel Stokes implementations."""

    if plane_options is None:
        plane_options = PlaneStokesSolverOptions()
    if channel_options is None:
        channel_options = ChannelStokesSolverOptions()
    if not isinstance(plane_options, PlaneStokesSolverOptions):
        raise TypeError("plane_options must be PlaneStokesSolverOptions")
    if not isinstance(channel_options, ChannelStokesSolverOptions):
        raise TypeError("channel_options must be ChannelStokesSolverOptions")
    registry = GeometrySolverRegistry()
    registry.register(
        geometry_type=PlaneSlab,
        geometry_name="plane_slab",
        capability=INCOMPRESSIBLE_STOKES_CAPABILITY,
        implementation_name="plane_free_slip_modal_stokes",
        factory=lambda context, system: _plane_factory(
            context,
            system,
            options=plane_options,
        ),
    )
    registry.register(
        geometry_type=RectangularChannel,
        geometry_name="rectangular_channel",
        capability=INCOMPRESSIBLE_STOKES_CAPABILITY,
        implementation_name="channel_no_slip_modal_stokes",
        factory=lambda context, system: _channel_factory(
            context,
            system,
            options=channel_options,
        ),
    )
    return registry
