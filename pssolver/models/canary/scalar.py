"""Simple scalar PDEs for execution-contract qualification.

These models are intentionally independent of the current ``SpectralSolver``
and ``Fields`` classes.  They request a mathematical Laplacian from an
execution context and return physical-space explicit terms.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

import torch

from pssolver.core import (
    BoundaryKind,
    BoundarySet,
    FieldRole,
    FieldSpec,
    HomogeneousDirichletBC,
    HomogeneousNeumannBC,
)
from pssolver.execution import AlgebraicSystemSpec, ModelExecutionContext


def _finite_real(value: object, description: str, *, positive: bool) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{description} must be {qualifier}")
    return float(value)


def _normalize_modes(
    modes: tuple[int, ...],
    boundaries: BoundarySet,
) -> tuple[int, ...]:
    try:
        modes = tuple(modes)
    except TypeError as exc:
        raise TypeError("initial_modes must be iterable") from exc
    if len(modes) != boundaries.ndim:
        raise ValueError("initial_modes must match the boundary dimension")
    for mode, condition in zip(modes, boundaries.axes):
        if not isinstance(mode, int) or isinstance(mode, bool) or mode < 0:
            raise ValueError("initial modes must be non-negative integers")
        if condition.kind is BoundaryKind.DIRICHLET and mode == 0:
            raise ValueError("a Dirichlet eigenmode index must be positive")
    return modes


def _validate_context(
    context: ModelExecutionContext,
    boundaries: BoundarySet,
) -> None:
    if not isinstance(context, ModelExecutionContext):
        raise TypeError("context must implement ModelExecutionContext")
    if len(context.physical_shape) != boundaries.ndim:
        raise ValueError("execution context dimension does not match boundaries")
    if len(context.axis_coordinates) != boundaries.ndim:
        raise ValueError("coordinate count does not match boundaries")


def _initial_eigenmode(
    context: ModelExecutionContext,
    boundaries: BoundarySet,
    modes: tuple[int, ...],
    amplitude: float,
) -> torch.Tensor:
    _validate_context(context, boundaries)
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
        broadcast_shape = [1] * boundaries.ndim
        broadcast_shape[axis] = coordinate.numel()
        value = value * factor.reshape(broadcast_shape)
    return value


@dataclass(frozen=True, slots=True)
class ScalarDiffusionModel:
    """Scalar diffusion, ``d_t phi = diffusivity * Laplacian(phi)``."""

    boundaries: BoundarySet
    diffusivity: float
    initial_amplitude: float = 1.0
    initial_modes: tuple[int, ...] = (1,)
    name: str = "scalar_diffusion"

    def __post_init__(self) -> None:
        if not isinstance(self.boundaries, BoundarySet):
            raise TypeError("boundaries must be a BoundarySet")
        object.__setattr__(
            self,
            "diffusivity",
            _finite_real(self.diffusivity, "diffusivity", positive=True),
        )
        object.__setattr__(
            self,
            "initial_amplitude",
            _finite_real(
                self.initial_amplitude,
                "initial_amplitude",
                positive=False,
            ),
        )
        object.__setattr__(
            self,
            "initial_modes",
            _normalize_modes(self.initial_modes, self.boundaries),
        )
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

    def field_specs(self) -> tuple[FieldSpec, ...]:
        return (
            FieldSpec.scalar("phi", FieldRole.EVOLVED, self.boundaries),
        )

    def parameter_metadata(self) -> Mapping[str, object]:
        return {
            "diffusivity": self.diffusivity,
            "initial_amplitude": self.initial_amplitude,
            "initial_modes": list(self.initial_modes),
        }

    def initial_values(
        self,
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        return {
            "phi": _initial_eigenmode(
                context,
                self.boundaries,
                self.initial_modes,
                self.initial_amplitude,
            )
        }

    def linear_operators(
        self,
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        return {
            "phi": self.diffusivity
            * context.laplacian_eigenvalues("phi")
        }

    def explicit_rhs(
        self,
        state: Mapping[str, torch.Tensor],
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        del context
        return {"phi": torch.zeros_like(state["phi"])}


@dataclass(frozen=True, slots=True)
class AllenCahnModel:
    """Allen--Cahn reaction diffusion with a cubic local potential."""

    boundaries: BoundarySet
    diffusivity: float
    linear_reaction: float
    cubic_reaction: float
    initial_amplitude: float = 0.1
    initial_modes: tuple[int, ...] = (1,)
    name: str = "allen_cahn"

    def __post_init__(self) -> None:
        if not isinstance(self.boundaries, BoundarySet):
            raise TypeError("boundaries must be a BoundarySet")
        object.__setattr__(
            self,
            "diffusivity",
            _finite_real(self.diffusivity, "diffusivity", positive=True),
        )
        for attribute in (
            "linear_reaction",
            "cubic_reaction",
            "initial_amplitude",
        ):
            object.__setattr__(
                self,
                attribute,
                _finite_real(getattr(self, attribute), attribute, positive=False),
            )
        object.__setattr__(
            self,
            "initial_modes",
            _normalize_modes(self.initial_modes, self.boundaries),
        )
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

    def field_specs(self) -> tuple[FieldSpec, ...]:
        return (
            FieldSpec.scalar("phi", FieldRole.EVOLVED, self.boundaries),
        )

    def parameter_metadata(self) -> Mapping[str, object]:
        return {
            "diffusivity": self.diffusivity,
            "linear_reaction": self.linear_reaction,
            "cubic_reaction": self.cubic_reaction,
            "initial_amplitude": self.initial_amplitude,
            "initial_modes": list(self.initial_modes),
        }

    def initial_values(
        self,
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        return {
            "phi": _initial_eigenmode(
                context,
                self.boundaries,
                self.initial_modes,
                self.initial_amplitude,
            )
        }

    def linear_operators(
        self,
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        return {
            "phi": self.diffusivity
            * context.laplacian_eigenvalues("phi")
        }

    def explicit_rhs(
        self,
        state: Mapping[str, torch.Tensor],
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        del context
        phi = state["phi"]
        return {
            "phi": self.linear_reaction * phi
            - self.cubic_reaction * phi.pow(3)
        }


@dataclass(frozen=True, slots=True)
class DiffusionHelmholtzCouplingModel:
    """Diffusion coupled to an instantaneous screened response.

    The equations are

    ``d_t phi = diffusivity * Laplacian(phi) + coupling * response``

    and

    ``(helmholtz_shift - helmholtz_length_sq * Laplacian) response = phi``.
    """

    boundaries: BoundarySet
    diffusivity: float
    coupling: float
    helmholtz_shift: float
    helmholtz_length_sq: float
    initial_amplitude: float = 0.1
    initial_modes: tuple[int, ...] = (1,)
    name: str = "diffusion_helmholtz_coupling"

    def __post_init__(self) -> None:
        if not isinstance(self.boundaries, BoundarySet):
            raise TypeError("boundaries must be a BoundarySet")
        for attribute in (
            "diffusivity",
            "helmholtz_shift",
            "helmholtz_length_sq",
        ):
            object.__setattr__(
                self,
                attribute,
                _finite_real(getattr(self, attribute), attribute, positive=True),
            )
        for attribute in ("coupling", "initial_amplitude"):
            object.__setattr__(
                self,
                attribute,
                _finite_real(getattr(self, attribute), attribute, positive=False),
            )
        object.__setattr__(
            self,
            "initial_modes",
            _normalize_modes(self.initial_modes, self.boundaries),
        )
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

    def field_specs(self) -> tuple[FieldSpec, ...]:
        return (
            FieldSpec.scalar("phi", FieldRole.EVOLVED, self.boundaries),
            FieldSpec.scalar(
                "response",
                FieldRole.ALGEBRAIC,
                self.boundaries,
            ),
        )

    def parameter_metadata(self) -> Mapping[str, object]:
        return {
            "coupling": self.coupling,
            "diffusivity": self.diffusivity,
            "helmholtz_length_sq": self.helmholtz_length_sq,
            "helmholtz_shift": self.helmholtz_shift,
            "initial_amplitude": self.initial_amplitude,
            "initial_modes": list(self.initial_modes),
        }

    def algebraic_system_specs(self) -> tuple[AlgebraicSystemSpec, ...]:
        return (
            AlgebraicSystemSpec(
                name="screened_response",
                capability="scalar_helmholtz",
                output_components=("response",),
                dependencies=("phi",),
                parameters={
                    "helmholtz_length_sq": self.helmholtz_length_sq,
                    "helmholtz_shift": self.helmholtz_shift,
                },
            ),
        )

    def initial_values(
        self,
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        return {
            "phi": _initial_eigenmode(
                context,
                self.boundaries,
                self.initial_modes,
                self.initial_amplitude,
            )
        }

    def linear_operators(
        self,
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        return {
            "phi": self.diffusivity
            * context.laplacian_eigenvalues("phi")
        }

    def explicit_rhs(
        self,
        state: Mapping[str, torch.Tensor],
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        del context
        return {"phi": self.coupling * state["response"]}


@dataclass(frozen=True, slots=True)
class DifferentialAlgebraicChainModel:
    """Canary for an evolved -> transient -> stored algebraic DAG.

    A physical derivative is cached transiently, a Helmholtz system consumes
    it, and the explicit RHS consumes both products.  The declarations are
    intentionally reverse-topological so construction must freeze a real DAG
    rather than rely on tuple order.
    """

    boundaries: BoundarySet
    gradient_boundaries: BoundarySet
    gradient_axis: int
    diffusivity: float
    coupling: float
    helmholtz_shift: float
    helmholtz_length_sq: float
    initial_amplitude: float = 0.1
    initial_modes: tuple[int, ...] = (1,)
    name: str = "differential_algebraic_chain"

    def __post_init__(self) -> None:
        for value, description in (
            (self.boundaries, "boundaries"),
            (self.gradient_boundaries, "gradient_boundaries"),
        ):
            if not isinstance(value, BoundarySet):
                raise TypeError(f"{description} must be a BoundarySet")
        if self.boundaries.ndim != self.gradient_boundaries.ndim:
            raise ValueError("boundary dimensions must match")
        if (
            not isinstance(self.gradient_axis, int)
            or isinstance(self.gradient_axis, bool)
            or self.gradient_axis < 0
            or self.gradient_axis >= self.boundaries.ndim
        ):
            raise ValueError("gradient_axis is out of range")
        expected = list(self.boundaries.axes)
        source_kind = expected[self.gradient_axis].kind
        if source_kind is BoundaryKind.NEUMANN:
            expected[self.gradient_axis] = HomogeneousDirichletBC()
        elif source_kind is BoundaryKind.DIRICHLET:
            expected[self.gradient_axis] = HomogeneousNeumannBC()
        if tuple(expected) != self.gradient_boundaries.axes:
            raise ValueError(
                "gradient_boundaries do not match the mathematical derivative"
            )
        for attribute in (
            "diffusivity",
            "helmholtz_shift",
            "helmholtz_length_sq",
        ):
            object.__setattr__(
                self,
                attribute,
                _finite_real(getattr(self, attribute), attribute, positive=True),
            )
        for attribute in ("coupling", "initial_amplitude"):
            object.__setattr__(
                self,
                attribute,
                _finite_real(getattr(self, attribute), attribute, positive=False),
            )
        object.__setattr__(
            self,
            "initial_modes",
            _normalize_modes(self.initial_modes, self.boundaries),
        )
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

    def field_specs(self) -> tuple[FieldSpec, ...]:
        return (
            FieldSpec.scalar("phi", FieldRole.EVOLVED, self.boundaries),
            FieldSpec.scalar(
                "gradient_phi",
                FieldRole.TRANSIENT,
                self.gradient_boundaries,
            ),
            FieldSpec.scalar(
                "response",
                FieldRole.ALGEBRAIC,
                self.gradient_boundaries,
            ),
        )

    def parameter_metadata(self) -> Mapping[str, object]:
        return {
            "coupling": self.coupling,
            "diffusivity": self.diffusivity,
            "gradient_axis": self.gradient_axis,
            "helmholtz_length_sq": self.helmholtz_length_sq,
            "helmholtz_shift": self.helmholtz_shift,
            "initial_amplitude": self.initial_amplitude,
            "initial_modes": list(self.initial_modes),
        }

    def algebraic_system_specs(self) -> tuple[AlgebraicSystemSpec, ...]:
        return (
            AlgebraicSystemSpec(
                name="screened_gradient",
                capability="scalar_helmholtz",
                output_components=("response",),
                dependencies=("gradient_phi",),
                parameters={
                    "helmholtz_length_sq": self.helmholtz_length_sq,
                    "helmholtz_shift": self.helmholtz_shift,
                },
            ),
            AlgebraicSystemSpec(
                name="wall_normal_gradient",
                capability="component_gradient",
                output_components=("gradient_phi",),
                dependencies=("phi",),
                parameters={"axis": self.gradient_axis},
            ),
        )

    def initial_values(
        self,
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        return {
            "phi": _initial_eigenmode(
                context,
                self.boundaries,
                self.initial_modes,
                self.initial_amplitude,
            )
        }

    def linear_operators(
        self,
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        return {
            "phi": self.diffusivity
            * context.laplacian_eigenvalues("phi")
        }

    def explicit_rhs(
        self,
        state: Mapping[str, torch.Tensor],
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        del context
        return {
            "phi": self.coupling
            * state["gradient_phi"]
            * state["response"]
        }
