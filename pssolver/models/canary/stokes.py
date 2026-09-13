"""Model-independent body-force canary for Stage G Stokes execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

import torch

from pssolver.core import (
    BoundaryKind,
    BoundarySet,
    FieldComponentSpec,
    FieldRole,
    FieldSpec,
)
from pssolver.execution import (
    AlgebraicSystemSpec,
    IncompressibleStokesSystemSpec,
    ModelExecutionContext,
)


def _finite(value: object, description: str, *, positive: bool) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{description} must be {qualifier}")
    return float(value)


def _initial_mode(
    context: ModelExecutionContext,
    boundaries: BoundarySet,
    modes: tuple[int, int, int],
    amplitude: float,
) -> torch.Tensor:
    value = torch.full(
        context.physical_shape,
        amplitude,
        dtype=context.real_dtype,
        device=context.device,
    )
    for axis, (coordinate, length, mode, condition) in enumerate(
        zip(
            context.axis_coordinates,
            context.lengths,
            modes,
            boundaries.axes,
        )
    ):
        if condition.kind is BoundaryKind.PERIODIC:
            factor = torch.cos(2.0 * math.pi * mode * coordinate / length)
        elif condition.kind is BoundaryKind.NEUMANN:
            factor = torch.cos(math.pi * mode * coordinate / length)
        elif condition.kind is BoundaryKind.DIRICHLET:
            factor = torch.sin(math.pi * mode * coordinate / length)
        else:  # pragma: no cover - BoundaryKind is exhaustive.
            raise ValueError(f"unsupported boundary kind {condition.kind!r}")
        shape = [1, 1, 1]
        shape[axis] = coordinate.numel()
        value = value * factor.reshape(shape)
    return value


@dataclass(frozen=True, slots=True)
class BodyForceStokesCanaryModel:
    """Diffusing body force coupled to instantaneous incompressible flow.

    The force fields are deliberately evolved so successive Stokes solves see
    changing inputs.  Velocity and pressure are algebraic fields.  No solver
    diagnostic is declared as a physical field.
    """

    force_boundaries: tuple[BoundarySet, BoundarySet, BoundarySet]
    velocity_boundaries: tuple[BoundarySet, BoundarySet, BoundarySet]
    pressure_boundaries: BoundarySet
    stokes_system: IncompressibleStokesSystemSpec
    force_diffusivity: float
    initial_amplitudes: tuple[float, float, float]
    initial_modes: tuple[
        tuple[int, int, int],
        tuple[int, int, int],
        tuple[int, int, int],
    ]
    name: str = "body_force_stokes_canary"

    def __post_init__(self) -> None:
        force_boundaries = tuple(self.force_boundaries)
        velocity_boundaries = tuple(self.velocity_boundaries)
        for values, description in (
            (force_boundaries, "force_boundaries"),
            (velocity_boundaries, "velocity_boundaries"),
        ):
            if len(tuple(values)) != 3 or not all(
                isinstance(boundaries, BoundarySet)
                and boundaries.ndim == 3
                for boundaries in values
            ):
                raise ValueError(
                    f"{description} must contain three 3D BoundarySet objects"
                )
        if (
            not isinstance(self.pressure_boundaries, BoundarySet)
            or self.pressure_boundaries.ndim != 3
        ):
            raise TypeError("pressure_boundaries must be a 3D BoundarySet")
        if not isinstance(
            self.stokes_system,
            IncompressibleStokesSystemSpec,
        ):
            raise TypeError(
                "stokes_system must be an IncompressibleStokesSystemSpec"
            )
        object.__setattr__(
            self,
            "force_diffusivity",
            _finite(
                self.force_diffusivity,
                "force_diffusivity",
                positive=True,
            ),
        )
        amplitudes = tuple(
            _finite(value, "initial amplitude", positive=False)
            for value in self.initial_amplitudes
        )
        if len(amplitudes) != 3:
            raise ValueError("initial_amplitudes must contain three values")
        modes = tuple(tuple(component) for component in self.initial_modes)
        if len(modes) != 3 or any(len(component) != 3 for component in modes):
            raise ValueError("initial_modes must contain three 3D mode tuples")
        for component_modes, boundaries in zip(modes, force_boundaries):
            for mode, condition in zip(component_modes, boundaries.axes):
                if (
                    not isinstance(mode, int)
                    or isinstance(mode, bool)
                    or mode < 0
                    or (
                        condition.kind is BoundaryKind.DIRICHLET
                        and mode == 0
                    )
                ):
                    raise ValueError(
                        "initial mode is incompatible with its boundary"
                    )
        if self.stokes_system.force_components != (
            "force_x",
            "force_y",
            "force_z",
        ):
            raise ValueError("canary force components must use coordinate names")
        if self.stokes_system.velocity_components != ("ux", "uy", "uz"):
            raise ValueError(
                "canary velocity components must use coordinate names"
            )
        if self.stokes_system.pressure_component != "p":
            raise ValueError("canary pressure component must be named 'p'")
        object.__setattr__(self, "force_boundaries", force_boundaries)
        object.__setattr__(
            self,
            "velocity_boundaries",
            velocity_boundaries,
        )
        object.__setattr__(self, "initial_amplitudes", amplitudes)
        object.__setattr__(self, "initial_modes", modes)
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

    def field_specs(self) -> tuple[FieldSpec, ...]:
        return (
            FieldSpec(
                "body_force",
                FieldRole.EVOLVED,
                tuple(
                    FieldComponentSpec(name, boundaries)
                    for name, boundaries in zip(
                        self.stokes_system.force_components,
                        self.force_boundaries,
                    )
                ),
            ),
            FieldSpec(
                "velocity",
                FieldRole.ALGEBRAIC,
                tuple(
                    FieldComponentSpec(name, boundaries)
                    for name, boundaries in zip(
                        self.stokes_system.velocity_components,
                        self.velocity_boundaries,
                    )
                ),
            ),
            FieldSpec.scalar(
                self.stokes_system.pressure_component,
                FieldRole.ALGEBRAIC,
                self.pressure_boundaries,
            ),
        )

    def parameter_metadata(self) -> Mapping[str, object]:
        return {
            "force_diffusivity": self.force_diffusivity,
            "initial_amplitudes": list(self.initial_amplitudes),
            "initial_modes": [list(modes) for modes in self.initial_modes],
            "stokes": self.stokes_system.to_metadata(),
        }

    def algebraic_system_specs(self) -> tuple[AlgebraicSystemSpec, ...]:
        return (self.stokes_system.to_algebraic_system_spec(),)

    def initial_values(
        self,
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        return {
            name: _initial_mode(context, boundaries, modes, amplitude)
            for name, boundaries, modes, amplitude in zip(
                self.stokes_system.force_components,
                self.force_boundaries,
                self.initial_modes,
                self.initial_amplitudes,
            )
        }

    def linear_operators(
        self,
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        return {
            name: self.force_diffusivity
            * context.laplacian_eigenvalues(name)
            for name in self.stokes_system.force_components
        }

    def explicit_rhs(
        self,
        state: Mapping[str, torch.Tensor],
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        del context
        return {
            name: torch.zeros_like(state[name])
            for name in self.stokes_system.force_components
        }
