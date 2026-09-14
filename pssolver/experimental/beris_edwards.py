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

    @property
    def physical_dependencies(self) -> tuple[str, ...]:
        return self.dependencies

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
        bulk_hats = self.context.forward_projected_many(
            self.output_components,
            tuple(bulk),
        )
        result = {}
        for q_name, h_name, bulk_hat in zip(
            self.dependencies,
            self.output_components,
            (bulk_hats[name] for name in self.output_components),
            strict=True,
        ):
            q_hat = self.context.spectral_dependency(state, q_name)
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

    @property
    def physical_dependencies(self) -> tuple[str, ...]:
        return ()

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
                source_hat = self.context.spectral_dependency(
                    state,
                    source_name,
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

    @property
    def physical_dependencies(self) -> tuple[str, ...]:
        return self.dependencies

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
        return self.context.forward_projected_many(
            self.output_components,
            values,
        )


def _inverse_gradient_from_spectral(
    context: LegacyAlgebraicSolverContext,
    spectral: torch.Tensor,
    boundaries: tuple[str, ...],
    axis: int,
) -> tuple[torch.Tensor, tuple[str, ...]]:
    derivative_hat, derivative_boundaries = (
        context.legacy_transform_backend.gradient_hat(
            spectral,
            boundaries,
            axis,
        )
    )
    physical = context.legacy_projector.inverse_transform(
        derivative_hat,
        derivative_boundaries,
    )
    return physical, tuple(derivative_boundaries)


def _common_stress_divergence_from_spectral(
    context: LegacyAlgebraicSolverContext,
    stress_hats: tuple[torch.Tensor, ...],
    *,
    sum_space: str,
) -> torch.Tensor:
    """Match the qualified mixed-parity divergence without re-forwarding."""

    if len(stress_hats) != 9:
        raise ValueError("a three-dimensional stress requires nine spectra")
    packed = torch.stack(stress_hats)
    if sum_space == "spectral":
        derivative_x_hat, x_boundaries = (
            context.legacy_transform_backend.gradient_hat(
                packed[[0, 3, 6]],
                _PLANE_EVEN_BOUNDARIES,
                0,
            )
        )
        derivative_y_hat, y_boundaries = (
            context.legacy_transform_backend.gradient_hat(
                packed[[1, 4, 7]],
                _PLANE_EVEN_BOUNDARIES,
                1,
            )
        )
        if tuple(x_boundaries) != tuple(y_boundaries):
            raise RuntimeError("x/y stress derivatives must share one basis")
        derivative_xy = context.legacy_projector.inverse_transform(
            derivative_x_hat + derivative_y_hat,
            x_boundaries,
        )
        derivative_z, _ = _inverse_gradient_from_spectral(
            context,
            packed[[2, 5, 8]],
            _PLANE_EVEN_BOUNDARIES,
            2,
        )
        return derivative_xy + derivative_z
    if sum_space != "physical":
        raise ValueError("sum_space must be 'physical' or 'spectral'")
    derivative_x, x_boundaries = _inverse_gradient_from_spectral(
        context,
        packed[[0, 3, 6]],
        _PLANE_EVEN_BOUNDARIES,
        0,
    )
    derivative_y, y_boundaries = _inverse_gradient_from_spectral(
        context,
        packed[[1, 4, 7]],
        _PLANE_EVEN_BOUNDARIES,
        1,
    )
    derivative_z, _ = _inverse_gradient_from_spectral(
        context,
        packed[[2, 5, 8]],
        _PLANE_EVEN_BOUNDARIES,
        2,
    )
    if x_boundaries != y_boundaries:
        raise RuntimeError("x/y stress derivatives must share one basis")
    return derivative_x + derivative_y + derivative_z


def _distortion_stress_divergence_from_spectral(
    context: LegacyAlgebraicSolverContext,
    stress_hats: tuple[torch.Tensor, ...],
    *,
    sum_space: str,
) -> torch.Tensor:
    """Differentiate the even/odd distortion spectra without round trips."""

    if len(stress_hats) != 9:
        raise ValueError("a three-dimensional stress requires nine spectra")
    even_hat = torch.stack(
        tuple(stress_hats[index] for index in (0, 1, 3, 4, 8))
    )
    odd_hat = torch.stack(
        tuple(stress_hats[index] for index in (2, 5, 6, 7))
    )
    if sum_space == "spectral":
        backend = context.legacy_transform_backend
        even_x_hat, even_x_boundaries = backend.gradient_hat(
            even_hat[[0, 2]], _PLANE_EVEN_BOUNDARIES, 0
        )
        even_y_hat, even_y_boundaries = backend.gradient_hat(
            even_hat[[1, 3]], _PLANE_EVEN_BOUNDARIES, 1
        )
        odd_z_hat, odd_z_boundaries = backend.gradient_hat(
            odd_hat[[0, 1]], _PLANE_ODD_BOUNDARIES, 2
        )
        if not (
            tuple(even_x_boundaries)
            == tuple(even_y_boundaries)
            == tuple(odd_z_boundaries)
        ):
            raise RuntimeError(
                "tangential distortion-force terms must share one basis"
            )
        tangential = context.legacy_projector.inverse_transform(
            even_x_hat + even_y_hat + odd_z_hat,
            even_x_boundaries,
        )
        odd_x_hat, odd_x_boundaries = backend.gradient_hat(
            odd_hat[2], _PLANE_ODD_BOUNDARIES, 0
        )
        odd_y_hat, odd_y_boundaries = backend.gradient_hat(
            odd_hat[3], _PLANE_ODD_BOUNDARIES, 1
        )
        even_z_hat, even_z_boundaries = backend.gradient_hat(
            even_hat[4], _PLANE_EVEN_BOUNDARIES, 2
        )
        if not (
            tuple(odd_x_boundaries)
            == tuple(odd_y_boundaries)
            == tuple(even_z_boundaries)
        ):
            raise RuntimeError(
                "normal distortion-force terms must share one basis"
            )
        normal = context.legacy_projector.inverse_transform(
            odd_x_hat + odd_y_hat + even_z_hat,
            odd_x_boundaries,
        )
        return torch.stack((tangential[0], tangential[1], normal))
    if sum_space != "physical":
        raise ValueError("sum_space must be 'physical' or 'spectral'")
    even_x, _ = _inverse_gradient_from_spectral(
        context,
        even_hat[[0, 2]],
        _PLANE_EVEN_BOUNDARIES,
        0,
    )
    even_y, _ = _inverse_gradient_from_spectral(
        context,
        even_hat[[1, 3]],
        _PLANE_EVEN_BOUNDARIES,
        1,
    )
    even_z, _ = _inverse_gradient_from_spectral(
        context,
        even_hat[4],
        _PLANE_EVEN_BOUNDARIES,
        2,
    )
    odd_x, _ = _inverse_gradient_from_spectral(
        context,
        odd_hat[2],
        _PLANE_ODD_BOUNDARIES,
        0,
    )
    odd_y, _ = _inverse_gradient_from_spectral(
        context,
        odd_hat[3],
        _PLANE_ODD_BOUNDARIES,
        1,
    )
    odd_z, _ = _inverse_gradient_from_spectral(
        context,
        odd_hat[[0, 1]],
        _PLANE_ODD_BOUNDARIES,
        2,
    )
    return torch.stack(
        (
            even_x[0] + even_y[0] + odd_z[0],
            even_x[1] + even_y[1] + odd_z[1],
            odd_x + odd_y + even_z,
        )
    )


@dataclass(slots=True)
class _ForceSolver(_StatelessConstitutiveSolver):
    capability: str
    implementation_name: str
    output_components: tuple[str, ...]
    dependencies: tuple[str, ...]
    context: LegacyAlgebraicSolverContext
    sum_space: str
    numerical_policy: Mapping[str, object]

    @property
    def physical_dependencies(self) -> tuple[str, ...]:
        if self.context.lazy_physical_materialization:
            return ()
        return self.dependencies

    def solve_spectral(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        if set(state) != set(self.dependencies):
            raise ValueError("nematic-force state has the wrong components")
        if self.context.lazy_physical_materialization:
            algebraic_hats = tuple(
                self.context.spectral_dependency(state, name)
                for name in ALGEBRAIC_STRESS_COMPONENTS
            )
            distortion_hats = tuple(
                self.context.spectral_dependency(state, name)
                for name in DISTORTION_STRESS_COMPONENTS
            )
            force = _common_stress_divergence_from_spectral(
                self.context,
                algebraic_hats,
                sum_space=self.sum_space,
            ) + _distortion_stress_divergence_from_spectral(
                self.context,
                distortion_hats,
                sum_space=self.sum_space,
            )
            return self.context.forward_projected_many(
                self.output_components,
                tuple(
                    force[index]
                    for index in range(len(self.output_components))
                ),
            )
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
        return self.context.forward_projected_many(
            self.output_components,
            tuple(
                force[index] for index in range(len(self.output_components))
            ),
        )


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
