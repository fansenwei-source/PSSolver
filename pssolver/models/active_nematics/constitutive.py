"""Declarative Beris--Edwards constitutive chain for architecture migration.

The physical declarations in this module contain no transforms, legacy field
containers, or concrete Stokes implementation.  Stage I uses them to express
the dependency chain

``Q -> (H, grad(Q)) -> stress -> force -> (u, p)``.

The final explicit Q right-hand side is deliberately zero in this qualification
model.  Migrating advection, flow alignment, and Q relaxation belongs to the
next coupled-model stage; this class qualifies the complete constitutive force
without silently changing the production benchmark driver.
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
    FieldComponentSpec,
    FieldRole,
    FieldSpec,
    HomogeneousDirichletBC,
    HomogeneousNeumannBC,
)
from pssolver.execution import (
    AlgebraicSystemSpec,
    IncompressibleStokesSystemSpec,
    ModelExecutionContext,
)

from .beris_edwards import STRESS_COMPONENTS
from .fields import Q_COMPONENTS


BERIS_EDWARDS_MOLECULAR_FIELD_CAPABILITY = (
    "beris_edwards_molecular_field"
)
BERIS_EDWARDS_Q_GRADIENT_CAPABILITY = "beris_edwards_q_gradient"
BERIS_EDWARDS_STRESS_CAPABILITY = "beris_edwards_stress"
BERIS_EDWARDS_FORCE_CAPABILITY = "beris_edwards_force"

H_COMPONENTS = ("Hxx", "Hxy", "Hxz", "Hyy", "Hyz")
CARTESIAN_AXES = ("x", "y", "z")
Q_GRADIENT_COMPONENTS = tuple(
    f"d{component}_d{axis}"
    for axis in CARTESIAN_AXES
    for component in Q_COMPONENTS
)
ALGEBRAIC_STRESS_COMPONENTS = tuple(
    f"stress_algebraic_{component}" for component in STRESS_COMPONENTS
)
DISTORTION_STRESS_COMPONENTS = tuple(
    f"stress_distortion_{component}" for component in STRESS_COMPONENTS
)
NEMATIC_FORCE_COMPONENTS = ("force_x", "force_y", "force_z")
VELOCITY_COMPONENTS = ("ux", "uy", "uz")


def _finite(value: object, description: str, *, positive: bool = False) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{description} must be {qualifier}")
    return float(value)


def _three_dimensional(boundaries: BoundarySet, description: str) -> BoundarySet:
    if not isinstance(boundaries, BoundarySet) or boundaries.ndim != 3:
        raise TypeError(f"{description} must be a three-dimensional BoundarySet")
    return boundaries


def _derivative_boundaries(
    boundaries: BoundarySet,
    axis: int,
) -> BoundarySet:
    conditions = list(boundaries.axes)
    kind = conditions[axis].kind
    if kind is BoundaryKind.NEUMANN:
        conditions[axis] = HomogeneousDirichletBC()
    elif kind is BoundaryKind.DIRICHLET:
        conditions[axis] = HomogeneousNeumannBC()
    return BoundarySet(tuple(conditions))


def _initial_q_mode(
    context: ModelExecutionContext,
    component_index: int,
    amplitude: float,
) -> torch.Tensor:
    """Return one deterministic nontrivial Plane-compatible Q component."""

    if len(context.physical_shape) != 3:
        raise ValueError("the Stage I Beris--Edwards canary requires 3D")
    x, y, z = context.axis_coordinates
    lx, ly, lz = context.lengths
    phase = 0.17 * (component_index + 1)
    kx = 1 + component_index % 2
    ky = 1 + (component_index // 2) % 2
    kz = 1 + component_index % 3
    x_factor = torch.cos(2.0 * math.pi * kx * x / lx + phase)
    y_factor = torch.sin(2.0 * math.pi * ky * y / ly - phase)
    z_factor = torch.cos(math.pi * kz * z / lz)
    return (
        amplitude
        * (component_index + 1)
        * x_factor.reshape(-1, 1, 1)
        * y_factor.reshape(1, -1, 1)
        * z_factor.reshape(1, 1, -1)
    )


@dataclass(frozen=True, slots=True)
class BerisEdwardsConstitutiveParameters:
    """Physical coefficients used by the complete one-constant force."""

    ldg_a: float
    ldg_b: float
    ldg_c: float
    ldg_l1: float
    flow_alignment: float
    active_prefactor: float

    def __post_init__(self) -> None:
        for name in (
            "ldg_a",
            "ldg_b",
            "ldg_c",
            "flow_alignment",
            "active_prefactor",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        object.__setattr__(
            self,
            "ldg_l1",
            _finite(self.ldg_l1, "ldg_l1", positive=True),
        )

    def to_metadata(self) -> dict[str, float]:
        return {
            "active_prefactor": self.active_prefactor,
            "flow_alignment": self.flow_alignment,
            "ldg_a": self.ldg_a,
            "ldg_b": self.ldg_b,
            "ldg_c": self.ldg_c,
            "ldg_l1": self.ldg_l1,
        }


@dataclass(frozen=True, slots=True)
class BerisEdwardsConstitutiveStokesCanaryModel:
    """Static-Q qualification model for the complete constitutive/Stokes DAG.

    Q is the only evolved field.  H, all Q gradients, both stress parts, and
    the projected force are transient.  Velocity and pressure are stored
    algebraic fields so they can be observed and checkpoint reconstruction can
    recompute the transient chain.

    Free Q anchoring does not automatically make the mean active tangential
    force vanish.  Removing a uniform tangential mode is therefore an explicit
    modeling choice, not a pressure gauge.  The supplied Stokes specification
    owns that choice through its zero-mode policy and friction coefficient.
    """

    q_boundaries: BoundarySet
    tangential_boundaries: BoundarySet
    normal_boundaries: BoundarySet
    pressure_boundaries: BoundarySet
    parameters: BerisEdwardsConstitutiveParameters
    stokes_system: IncompressibleStokesSystemSpec
    initial_amplitude: float = 0.02
    name: str = "beris_edwards_constitutive_stokes_canary"

    def __post_init__(self) -> None:
        q_boundaries = _three_dimensional(self.q_boundaries, "q_boundaries")
        tangential = _three_dimensional(
            self.tangential_boundaries,
            "tangential_boundaries",
        )
        normal = _three_dimensional(self.normal_boundaries, "normal_boundaries")
        pressure = _three_dimensional(
            self.pressure_boundaries,
            "pressure_boundaries",
        )
        if not isinstance(self.parameters, BerisEdwardsConstitutiveParameters):
            raise TypeError(
                "parameters must be BerisEdwardsConstitutiveParameters"
            )
        if not isinstance(self.stokes_system, IncompressibleStokesSystemSpec):
            raise TypeError("stokes_system must be IncompressibleStokesSystemSpec")
        if self.stokes_system.force_components != NEMATIC_FORCE_COMPONENTS:
            raise ValueError("Stokes force components must use the canonical names")
        if self.stokes_system.velocity_components != VELOCITY_COMPONENTS:
            raise ValueError(
                "Stokes velocity components must use the canonical names"
            )
        if self.stokes_system.pressure_component != "p":
            raise ValueError("Stokes pressure component must be named 'p'")
        if q_boundaries != tangential or q_boundaries != pressure:
            raise ValueError(
                "Stage I Plane Q, tangential velocity, and pressure must share "
                "one boundary space"
            )
        expected_normal = _derivative_boundaries(q_boundaries, 2)
        if normal != expected_normal:
            raise ValueError(
                "Stage I Plane normal velocity must use the wall-odd space"
            )
        object.__setattr__(
            self,
            "initial_amplitude",
            _finite(self.initial_amplitude, "initial_amplitude"),
        )
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("name must be a Python identifier")

    @property
    def q_gradient_boundaries(self) -> tuple[BoundarySet, ...]:
        return tuple(
            _derivative_boundaries(self.q_boundaries, axis)
            for axis in range(3)
            for _ in Q_COMPONENTS
        )

    @property
    def distortion_stress_boundaries(self) -> tuple[BoundarySet, ...]:
        # -L1 partial_i Q:partial_j Q is wall-even unless exactly one of
        # i,j is the wall-normal coordinate.
        return tuple(
            self.normal_boundaries
            if component in {"xz", "yz", "zx", "zy"}
            else self.q_boundaries
            for component in STRESS_COMPONENTS
        )

    def field_specs(self) -> tuple[FieldSpec, ...]:
        return (
            FieldSpec(
                "Q",
                FieldRole.EVOLVED,
                tuple(
                    FieldComponentSpec(name, self.q_boundaries)
                    for name in Q_COMPONENTS
                ),
            ),
            FieldSpec(
                "molecular_field",
                FieldRole.TRANSIENT,
                tuple(
                    FieldComponentSpec(name, self.q_boundaries)
                    for name in H_COMPONENTS
                ),
            ),
            FieldSpec(
                "q_gradient",
                FieldRole.TRANSIENT,
                tuple(
                    FieldComponentSpec(name, boundaries)
                    for name, boundaries in zip(
                        Q_GRADIENT_COMPONENTS,
                        self.q_gradient_boundaries,
                        strict=True,
                    )
                ),
            ),
            FieldSpec(
                "algebraic_stress",
                FieldRole.TRANSIENT,
                tuple(
                    FieldComponentSpec(name, self.q_boundaries)
                    for name in ALGEBRAIC_STRESS_COMPONENTS
                ),
            ),
            FieldSpec(
                "distortion_stress",
                FieldRole.TRANSIENT,
                tuple(
                    FieldComponentSpec(name, boundaries)
                    for name, boundaries in zip(
                        DISTORTION_STRESS_COMPONENTS,
                        self.distortion_stress_boundaries,
                        strict=True,
                    )
                ),
            ),
            FieldSpec(
                "nematic_force",
                FieldRole.TRANSIENT,
                tuple(
                    FieldComponentSpec(name, boundaries)
                    for name, boundaries in zip(
                        NEMATIC_FORCE_COMPONENTS,
                        (
                            self.tangential_boundaries,
                            self.tangential_boundaries,
                            self.normal_boundaries,
                        ),
                        strict=True,
                    )
                ),
            ),
            FieldSpec(
                "velocity",
                FieldRole.ALGEBRAIC,
                tuple(
                    FieldComponentSpec(name, boundaries)
                    for name, boundaries in zip(
                        VELOCITY_COMPONENTS,
                        (
                            self.tangential_boundaries,
                            self.tangential_boundaries,
                            self.normal_boundaries,
                        ),
                        strict=True,
                    )
                ),
            ),
            FieldSpec.scalar("p", FieldRole.ALGEBRAIC, self.pressure_boundaries),
        )

    def parameter_metadata(self) -> Mapping[str, object]:
        return {
            "constitutive": self.parameters.to_metadata(),
            "initial_amplitude": self.initial_amplitude,
            "qualification_scope": "static_q_constitutive_chain",
            "stokes": self.stokes_system.to_metadata(),
        }

    def algebraic_system_specs(self) -> tuple[AlgebraicSystemSpec, ...]:
        physical = self.parameters.to_metadata()
        # Reverse declaration is intentional: the Stage H planner, rather than
        # tuple position, must establish the constitutive execution order.
        return (
            self.stokes_system.to_algebraic_system_spec(),
            AlgebraicSystemSpec(
                name="nematic_force",
                capability=BERIS_EDWARDS_FORCE_CAPABILITY,
                output_components=NEMATIC_FORCE_COMPONENTS,
                dependencies=(
                    *ALGEBRAIC_STRESS_COMPONENTS,
                    *DISTORTION_STRESS_COMPONENTS,
                ),
            ),
            AlgebraicSystemSpec(
                name="nematic_stress",
                capability=BERIS_EDWARDS_STRESS_CAPABILITY,
                output_components=(
                    *ALGEBRAIC_STRESS_COMPONENTS,
                    *DISTORTION_STRESS_COMPONENTS,
                ),
                dependencies=(
                    *Q_COMPONENTS,
                    *H_COMPONENTS,
                    *Q_GRADIENT_COMPONENTS,
                ),
                parameters={
                    "active_prefactor": physical["active_prefactor"],
                    "flow_alignment": physical["flow_alignment"],
                    "ldg_l1": physical["ldg_l1"],
                },
            ),
            AlgebraicSystemSpec(
                name="molecular_field",
                capability=BERIS_EDWARDS_MOLECULAR_FIELD_CAPABILITY,
                output_components=H_COMPONENTS,
                dependencies=Q_COMPONENTS,
                parameters={
                    name: physical[name]
                    for name in ("ldg_a", "ldg_b", "ldg_c", "ldg_l1")
                },
            ),
            AlgebraicSystemSpec(
                name="q_gradient",
                capability=BERIS_EDWARDS_Q_GRADIENT_CAPABILITY,
                output_components=Q_GRADIENT_COMPONENTS,
                dependencies=Q_COMPONENTS,
            ),
        )

    def initial_values(
        self,
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        return {
            component: _initial_q_mode(
                context,
                index,
                self.initial_amplitude,
            )
            for index, component in enumerate(Q_COMPONENTS)
        }

    def linear_operators(
        self,
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        return {
            component: torch.zeros_like(
                context.laplacian_eigenvalues(component)
            )
            for component in Q_COMPONENTS
        }

    def explicit_rhs(
        self,
        state: Mapping[str, torch.Tensor],
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        del context
        return {
            component: torch.zeros_like(state[component])
            for component in Q_COMPONENTS
        }


__all__ = [
    "ALGEBRAIC_STRESS_COMPONENTS",
    "BERIS_EDWARDS_FORCE_CAPABILITY",
    "BERIS_EDWARDS_MOLECULAR_FIELD_CAPABILITY",
    "BERIS_EDWARDS_Q_GRADIENT_CAPABILITY",
    "BERIS_EDWARDS_STRESS_CAPABILITY",
    "BerisEdwardsConstitutiveParameters",
    "BerisEdwardsConstitutiveStokesCanaryModel",
    "CARTESIAN_AXES",
    "DISTORTION_STRESS_COMPONENTS",
    "H_COMPONENTS",
    "NEMATIC_FORCE_COMPONENTS",
    "Q_GRADIENT_COMPONENTS",
    "VELOCITY_COMPONENTS",
]
