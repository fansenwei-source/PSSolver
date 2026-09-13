"""Opt-in Plane executors for the Stage I Beris--Edwards constitutive DAG.

The pointwise constitutive equations are imported from the reusable physical
model package.  This module owns only numerical realization: native projected
derivatives, Plane parity splitting, force projection, and exact geometry
dispatch.  Production drivers do not import this module.
"""

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
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics.beris_edwards import (
    BerisEdwardsPointwiseKernels,
    POINTWISE_EXECUTION_MODES,
    beris_edwards_molecular_field_components,
)
from pssolver.models.active_nematics.constitutive import (
    ALGEBRAIC_STRESS_COMPONENTS,
    BERIS_EDWARDS_FORCE_CAPABILITY,
    BERIS_EDWARDS_MOLECULAR_FIELD_CAPABILITY,
    BERIS_EDWARDS_Q_GRADIENT_CAPABILITY,
    BERIS_EDWARDS_STRESS_CAPABILITY,
    BERIS_EDWARDS_VELOCITY_GRADIENT_CAPABILITY,
    DISTORTION_STRESS_COMPONENTS,
    H_COMPONENTS,
    NEMATIC_FORCE_COMPONENTS,
    Q_GRADIENT_COMPONENTS,
    VELOCITY_COMPONENTS,
    VELOCITY_GRADIENT_COMPONENTS,
)
from pssolver.models.active_nematics.fields import Q_COMPONENTS
from pssolver.transforms import (
    projected_common_basis_stress_divergence,
    projected_distortion_stress_divergence,
)

from .model_execution import LegacyAlgebraicSolverContext
from .stokes import (
    ChannelStokesSolverOptions,
    PlaneStokesSolverOptions,
    create_stokes_geometry_solver_registry,
)


_PLANE_EVEN_BOUNDARIES = ("periodic", "periodic", "neumann")
_PLANE_ODD_BOUNDARIES = ("periodic", "periodic", "dirichlet")
_MOLECULAR_FIELD_SPACES = ("physical", "spectral")
_STRESS_DIVERGENCE_SUM_SPACES = ("physical", "spectral")


def _finite_parameter(
    system: AlgebraicSystemSpec,
    name: str,
    *,
    positive: bool = False,
) -> float:
    try:
        value = system.parameters[name]
    except KeyError as exc:
        raise ValueError(f"{system.name!r} is missing parameter {name!r}") from exc
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{name} must be {qualifier}")
    return float(value)


def _legacy_plane_context(
    context: AlgebraicSolverContext,
) -> LegacyAlgebraicSolverContext:
    if not isinstance(context, LegacyAlgebraicSolverContext):
        raise TypeError(
            "Stage I Beris--Edwards executors require the opt-in legacy context"
        )
    if context.geometry_name != "plane_slab":
        raise ValueError("Stage I Beris--Edwards executors require PlaneSlab")
    if len(context.physical_shape) != 3:
        raise ValueError("Stage I Beris--Edwards executors require 3D")
    return context


class _StatelessConstitutiveSolver:
    """Shared observability contract for deterministic stateless executors."""

    numerical_policy: Mapping[str, object]

    def observability_metadata(self) -> Mapping[str, object]:
        return {
            "diagnostics": {"kind": "none"},
            "numerical_policy": dict(self.numerical_policy),
            "restart": {"kind": "stateless", "state_keys": []},
        }

    def diagnostic_snapshot(self) -> Mapping[str, object]:
        return {}

    def capture_restart_state(self) -> Mapping[str, torch.Tensor]:
        return {}

    def restore_restart_state(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("restart state must be a mapping")
        if state:
            raise ValueError("stateless constitutive solvers have no restart state")


@dataclass(slots=True)
class _MolecularFieldSolver(_StatelessConstitutiveSolver):
    capability: str
    implementation_name: str
    output_components: tuple[str, ...]
    dependencies: tuple[str, ...]
    context: LegacyAlgebraicSolverContext
    ldg_a: float
    ldg_b: float
    ldg_c: float
    ldg_l1: float
    linear_space: str
    kernels: BerisEdwardsPointwiseKernels
    numerical_policy: Mapping[str, object]

    def solve_spectral(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        if set(state) != set(self.dependencies):
            raise ValueError("molecular-field state has the wrong components")
        q_components = tuple(state[name] for name in self.dependencies)
        if self.linear_space == "physical":
            laplacians = tuple(
                self.context.laplacian(name, state[name])
                for name in self.dependencies
            )
            h_components = beris_edwards_molecular_field_components(
                q_components,
                laplacians,
                ldg_a=self.ldg_a,
                ldg_b=self.ldg_b,
                ldg_c=self.ldg_c,
                ldg_l1=self.ldg_l1,
            )
            return {
                output: self.context.forward_projected(output, value)
                for output, value in zip(
                    self.output_components,
                    h_components,
                    strict=True,
                )
            }

        bulk = self.kernels.bulk_molecular_field_components(
            q_components,
            ldg_a=self.ldg_a,
            ldg_b=self.ldg_b,
            ldg_c=self.ldg_c,
        )
        result = {}
        for q_name, h_name, bulk_value in zip(
            self.dependencies,
            self.output_components,
            bulk,
            strict=True,
        ):
            bulk_hat = self.context.forward_projected(h_name, bulk_value)
            q_hat = self.context.forward_projected(q_name, state[q_name])
            result[h_name] = bulk_hat + (
                self.ldg_l1
                * self.context.laplacian_eigenvalues(q_name)
                * q_hat
            )
        return result


@dataclass(slots=True)
class _TensorGradientSolver(_StatelessConstitutiveSolver):
    capability: str
    implementation_name: str
    output_components: tuple[str, ...]
    dependencies: tuple[str, ...]
    context: LegacyAlgebraicSolverContext
    numerical_policy: Mapping[str, object]

    def solve_spectral(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        if set(state) != set(self.dependencies):
            raise ValueError("tensor-gradient state has the wrong components")
        result = {}
        output_index = 0
        for axis in range(3):
            for source_name in self.dependencies:
                output = self.output_components[output_index]
                source_hat = self.context.forward_projected(
                    source_name,
                    state[source_name],
                )
                gradient_hat, gradient_boundaries = (
                    self.context.legacy_transform_backend.gradient_hat(
                        source_hat,
                        self.context.boundary_conditions(source_name),
                        axis,
                    )
                )
                if tuple(gradient_boundaries) != self.context.boundary_conditions(
                    output
                ):
                    raise ValueError(
                        "tensor-gradient output "
                        f"{output!r} has the wrong boundary space"
                    )
                result[output] = gradient_hat
                output_index += 1
        return result


@dataclass(slots=True)
class _StressSolver(_StatelessConstitutiveSolver):
    capability: str
    implementation_name: str
    output_components: tuple[str, ...]
    dependencies: tuple[str, ...]
    context: LegacyAlgebraicSolverContext
    flow_alignment: float
    active_prefactor: float
    ldg_l1: float
    kernels: BerisEdwardsPointwiseKernels
    numerical_policy: Mapping[str, object]

    def solve_spectral(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        if set(state) != set(self.dependencies):
            raise ValueError("stress state has the wrong components")
        q_components = tuple(state[name] for name in Q_COMPONENTS)
        h_components = tuple(state[name] for name in H_COMPONENTS)
        flat_gradients = tuple(state[name] for name in Q_GRADIENT_COMPONENTS)
        q_gradients = tuple(
            flat_gradients[axis * len(Q_COMPONENTS) : (axis + 1) * len(Q_COMPONENTS)]
            for axis in range(3)
        )
        algebraic = self.kernels.algebraic_stress_components(
            q_components,
            h_components,
            flow_alignment=self.flow_alignment,
            active_prefactor=self.active_prefactor,
        )
        distortion = self.kernels.distortion_stress_components(
            q_gradients,
            ldg_l1=self.ldg_l1,
        )
        values = (*algebraic, *distortion)
        return {
            output: self.context.forward_projected(output, value)
            for output, value in zip(
                self.output_components,
                values,
                strict=True,
            )
        }


@dataclass(slots=True)
class _ForceSolver(_StatelessConstitutiveSolver):
    capability: str
    implementation_name: str
    output_components: tuple[str, ...]
    dependencies: tuple[str, ...]
    context: LegacyAlgebraicSolverContext
    sum_space: str
    numerical_policy: Mapping[str, object]

    def solve_spectral(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        if set(state) != set(self.dependencies):
            raise ValueError("nematic-force state has the wrong components")
        algebraic = tuple(state[name] for name in ALGEBRAIC_STRESS_COMPONENTS)
        distortion = tuple(state[name] for name in DISTORTION_STRESS_COMPONENTS)
        backend = self.context.legacy_transform_backend
        projector = self.context.legacy_projector
        algebraic_force = projected_common_basis_stress_divergence(
            backend,
            algebraic,
            _PLANE_EVEN_BOUNDARIES,
            projector=projector,
            sum_space=self.sum_space,
        )
        distortion_force = projected_distortion_stress_divergence(
            backend,
            distortion,
            _PLANE_EVEN_BOUNDARIES,
            _PLANE_ODD_BOUNDARIES,
            projector=projector,
            sum_space=self.sum_space,
        )
        force = algebraic_force + distortion_force
        # The raw row divergence combines wall-even and wall-odd terms.  The
        # force field is defined as its projection into the corresponding
        # velocity space, exactly as in the qualified production Plane path.
        return {
            output: self.context.forward_projected(output, force[index])
            for index, output in enumerate(self.output_components)
        }


@dataclass(frozen=True, slots=True)
class PlaneBerisEdwardsSolverOptions:
    """Numerical policy for the opt-in Stage I Plane executors."""

    molecular_field_linear_space: str = "spectral"
    stress_divergence_sum_space: str = "spectral"
    pointwise_execution: str = "eager"

    def __post_init__(self) -> None:
        if self.molecular_field_linear_space not in _MOLECULAR_FIELD_SPACES:
            raise ValueError(
                "molecular_field_linear_space must be 'physical' or 'spectral'"
            )
        if self.stress_divergence_sum_space not in (
            _STRESS_DIVERGENCE_SUM_SPACES
        ):
            raise ValueError(
                "stress_divergence_sum_space must be 'physical' or 'spectral'"
            )
        if self.pointwise_execution not in POINTWISE_EXECUTION_MODES:
            raise ValueError("pointwise_execution must be 'eager' or 'compile'")


def _molecular_field_factory(
    context: AlgebraicSolverContext,
    system: AlgebraicSystemSpec,
    *,
    options: PlaneBerisEdwardsSolverOptions,
    kernels: BerisEdwardsPointwiseKernels,
) -> _MolecularFieldSolver:
    context = _legacy_plane_context(context)
    expected_parameters = {"ldg_a", "ldg_b", "ldg_c", "ldg_l1"}
    if system.capability != BERIS_EDWARDS_MOLECULAR_FIELD_CAPABILITY:
        raise ValueError("wrong molecular-field capability")
    if system.dependencies != Q_COMPONENTS or system.output_components != H_COMPONENTS:
        raise ValueError("molecular-field components must use canonical ordering")
    if set(system.parameters) != expected_parameters:
        raise ValueError("molecular-field parameters are incomplete")
    for q_name, h_name in zip(Q_COMPONENTS, H_COMPONENTS, strict=True):
        if context.boundary_conditions(q_name) != _PLANE_EVEN_BOUNDARIES:
            raise ValueError("Stage I Q components require Plane Neumann parity")
        if context.boundary_conditions(h_name) != _PLANE_EVEN_BOUNDARIES:
            raise ValueError("Stage I H components require Q's boundary space")
    return _MolecularFieldSolver(
        capability=system.capability,
        implementation_name="plane_projected_beris_edwards_molecular_field",
        output_components=system.output_components,
        dependencies=system.dependencies,
        context=context,
        ldg_a=_finite_parameter(system, "ldg_a"),
        ldg_b=_finite_parameter(system, "ldg_b"),
        ldg_c=_finite_parameter(system, "ldg_c"),
        ldg_l1=_finite_parameter(system, "ldg_l1", positive=True),
        linear_space=options.molecular_field_linear_space,
        kernels=kernels,
        numerical_policy={
            "linear_space": options.molecular_field_linear_space,
            "pointwise_execution": options.pointwise_execution,
            "projection": "complete_molecular_field",
        },
    )


def _tensor_gradient_solver(
    context: AlgebraicSolverContext,
    system: AlgebraicSystemSpec,
    *,
    dependencies: tuple[str, ...],
    outputs: tuple[str, ...],
    implementation_name: str,
) -> _TensorGradientSolver:
    if system.dependencies != dependencies:
        raise ValueError("gradient dependencies must use canonical ordering")
    if system.output_components != outputs:
        raise ValueError("gradient outputs must use canonical ordering")
    if system.parameters:
        raise ValueError("gradient systems have no physical parameters")
    for axis in range(3):
        for index, source in enumerate(dependencies):
            output = outputs[axis * len(dependencies) + index]
            expected = list(context.boundary_conditions(source))
            if expected[axis] == "neumann":
                expected[axis] = "dirichlet"
            elif expected[axis] == "dirichlet":
                expected[axis] = "neumann"
            if tuple(expected) != context.boundary_conditions(output):
                raise ValueError(
                    f"gradient output {output!r} has the wrong boundary space"
                )
    return _TensorGradientSolver(
        capability=system.capability,
        implementation_name=implementation_name,
        output_components=system.output_components,
        dependencies=system.dependencies,
        context=context,
        numerical_policy={"evaluation_space": "native_spectral"},
    )


def _q_gradient_factory(
    context: AlgebraicSolverContext,
    system: AlgebraicSystemSpec,
) -> _TensorGradientSolver:
    context = _legacy_plane_context(context)
    if system.capability != BERIS_EDWARDS_Q_GRADIENT_CAPABILITY:
        raise ValueError("wrong Q-gradient capability")
    return _tensor_gradient_solver(
        context,
        system,
        dependencies=Q_COMPONENTS,
        outputs=Q_GRADIENT_COMPONENTS,
        implementation_name="plane_native_projected_q_gradient",
    )


def _velocity_gradient_factory(
    context: AlgebraicSolverContext,
    system: AlgebraicSystemSpec,
) -> _TensorGradientSolver:
    context = _legacy_plane_context(context)
    if system.capability != BERIS_EDWARDS_VELOCITY_GRADIENT_CAPABILITY:
        raise ValueError("wrong velocity-gradient capability")
    return _tensor_gradient_solver(
        context,
        system,
        dependencies=VELOCITY_COMPONENTS,
        outputs=VELOCITY_GRADIENT_COMPONENTS,
        implementation_name="plane_native_projected_velocity_gradient",
    )


def _stress_factory(
    context: AlgebraicSolverContext,
    system: AlgebraicSystemSpec,
    *,
    options: PlaneBerisEdwardsSolverOptions,
    kernels: BerisEdwardsPointwiseKernels,
) -> _StressSolver:
    context = _legacy_plane_context(context)
    expected_outputs = (*ALGEBRAIC_STRESS_COMPONENTS, *DISTORTION_STRESS_COMPONENTS)
    expected_dependencies = (*Q_COMPONENTS, *H_COMPONENTS, *Q_GRADIENT_COMPONENTS)
    if system.capability != BERIS_EDWARDS_STRESS_CAPABILITY:
        raise ValueError("wrong stress capability")
    if system.output_components != expected_outputs:
        raise ValueError("stress outputs must use canonical ordering")
    if system.dependencies != expected_dependencies:
        raise ValueError("stress dependencies must use canonical ordering")
    if set(system.parameters) != {"active_prefactor", "flow_alignment", "ldg_l1"}:
        raise ValueError("stress parameters are incomplete")
    return _StressSolver(
        capability=system.capability,
        implementation_name="plane_projected_complete_beris_edwards_stress",
        output_components=system.output_components,
        dependencies=system.dependencies,
        context=context,
        flow_alignment=_finite_parameter(system, "flow_alignment"),
        active_prefactor=_finite_parameter(system, "active_prefactor"),
        ldg_l1=_finite_parameter(system, "ldg_l1", positive=True),
        kernels=kernels,
        numerical_policy={
            "pointwise_execution": options.pointwise_execution,
            "projection": "complete_stress_components",
        },
    )


def _force_factory(
    context: AlgebraicSolverContext,
    system: AlgebraicSystemSpec,
    *,
    options: PlaneBerisEdwardsSolverOptions,
) -> _ForceSolver:
    context = _legacy_plane_context(context)
    expected_dependencies = (
        *ALGEBRAIC_STRESS_COMPONENTS,
        *DISTORTION_STRESS_COMPONENTS,
    )
    if system.capability != BERIS_EDWARDS_FORCE_CAPABILITY:
        raise ValueError("wrong nematic-force capability")
    if system.output_components != NEMATIC_FORCE_COMPONENTS:
        raise ValueError("nematic force must use canonical ordering")
    if system.dependencies != expected_dependencies:
        raise ValueError("nematic-force dependencies must be complete stress")
    if system.parameters:
        raise ValueError("force system has no physical parameters")
    expected_force_boundaries = (
        _PLANE_EVEN_BOUNDARIES,
        _PLANE_EVEN_BOUNDARIES,
        _PLANE_ODD_BOUNDARIES,
    )
    for name, boundaries in zip(
        system.output_components,
        expected_force_boundaries,
        strict=True,
    ):
        if context.boundary_conditions(name) != boundaries:
            raise ValueError("nematic force has the wrong velocity space")
    return _ForceSolver(
        capability=system.capability,
        implementation_name="plane_projected_complete_stress_divergence",
        output_components=system.output_components,
        dependencies=system.dependencies,
        context=context,
        sum_space=options.stress_divergence_sum_space,
        numerical_policy={
            "projection": "complete_force_into_velocity_spaces",
            "stress_divergence_sum_space": (
                options.stress_divergence_sum_space
            ),
        },
    )


def create_beris_edwards_plane_geometry_solver_registry(
    *,
    constitutive_options: PlaneBerisEdwardsSolverOptions | None = None,
    plane_stokes_options: PlaneStokesSolverOptions | None = None,
    channel_stokes_options: ChannelStokesSolverOptions | None = None,
) -> GeometrySolverRegistry:
    """Return the Stage I Plane constitutive plus Stage G Stokes registry."""

    if constitutive_options is None:
        constitutive_options = PlaneBerisEdwardsSolverOptions()
    if not isinstance(constitutive_options, PlaneBerisEdwardsSolverOptions):
        raise TypeError("constitutive_options has the wrong type")
    kernels = BerisEdwardsPointwiseKernels(
        constitutive_options.pointwise_execution
    )
    registry = create_stokes_geometry_solver_registry(
        plane_options=plane_stokes_options,
        channel_options=channel_stokes_options,
    )
    registry.register(
        geometry_type=PlaneSlab,
        geometry_name="plane_slab",
        capability=BERIS_EDWARDS_MOLECULAR_FIELD_CAPABILITY,
        implementation_name="plane_projected_beris_edwards_molecular_field",
        factory=lambda context, system: _molecular_field_factory(
            context,
            system,
            options=constitutive_options,
            kernels=kernels,
        ),
    )
    registry.register(
        geometry_type=PlaneSlab,
        geometry_name="plane_slab",
        capability=BERIS_EDWARDS_Q_GRADIENT_CAPABILITY,
        implementation_name="plane_native_projected_q_gradient",
        factory=_q_gradient_factory,
    )
    registry.register(
        geometry_type=PlaneSlab,
        geometry_name="plane_slab",
        capability=BERIS_EDWARDS_VELOCITY_GRADIENT_CAPABILITY,
        implementation_name="plane_native_projected_velocity_gradient",
        factory=_velocity_gradient_factory,
    )
    registry.register(
        geometry_type=PlaneSlab,
        geometry_name="plane_slab",
        capability=BERIS_EDWARDS_STRESS_CAPABILITY,
        implementation_name="plane_projected_complete_beris_edwards_stress",
        factory=lambda context, system: _stress_factory(
            context,
            system,
            options=constitutive_options,
            kernels=kernels,
        ),
    )
    registry.register(
        geometry_type=PlaneSlab,
        geometry_name="plane_slab",
        capability=BERIS_EDWARDS_FORCE_CAPABILITY,
        implementation_name="plane_projected_complete_stress_divergence",
        factory=lambda context, system: _force_factory(
            context,
            system,
            options=constitutive_options,
        ),
    )
    return registry


__all__ = [
    "PlaneBerisEdwardsSolverOptions",
    "create_beris_edwards_plane_geometry_solver_registry",
]
