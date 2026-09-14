"""Opt-in adapter from executable models to the qualified legacy runtime.

This is a migration seam, not a production entry point.  Physical models see
only :class:`pssolver.execution.ModelExecutionContext`; every dependency on
``SpectralSolver``, ``Fields``, or the spectral projector remains here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import hashlib
import json
import math
from types import MappingProxyType

import torch

from pssolver.adapters import (
    boundary_set_to_legacy,
    compare_spectral_plan_to_runtime,
)
from pssolver.core import FieldRole, GeometrySpec, NumericsConfig, ProblemSpec
from pssolver.execution import (
    AlgebraicPhysicalDependenciesProtocol,
    AlgebraicExecutionPolicy,
    AlgebraicExecutionPlan,
    AlgebraicExecutableModelProtocol,
    AlgebraicRuntimeRestartState,
    AlgebraicSolverContext,
    AlgebraicSolverProtocol,
    AlgebraicSolverRegistration,
    AlgebraicSystemRestartState,
    AlgebraicSystemSpec,
    AlgebraicUpdatePhase,
    ExecutableModelProtocol,
    ExplicitRHSPhysicalDependenciesProtocol,
    GeometrySolverRegistry,
    InspectableAlgebraicSolverProtocol,
    ModelExecutionContext,
    build_algebraic_execution_plan,
    resolve_algebraic_execution_policy,
)
from pssolver.planning import SpectralPlan, assemble_spectral_plan
from pssolver.solver import SpectralSolver
from pssolver.transforms import BasisAwareSpectralProjector

from .legacy_assembly import (
    LegacyAssemblySpec,
    create_legacy_projector,
    create_legacy_solver,
    declare_legacy_fields,
    materialize_legacy_assembly,
)
from .integrators import (
    InstrumentedProjectedSemiImplicitEulerIntegrator,
    ProjectedSemiImplicitEulerIntegrator,
)
from .performance import (
    RuntimePerformanceRecorder,
    instrument_transform_backend,
)
from .representations import (
    AlgebraicGenerationState,
    AlgebraicPhysicalStateView,
    AlgebraicRepresentationCache,
)
from .projected_scheduler import (
    AlgebraicPhysicalIslandScheduler,
    BoundarySignatureTransformScheduler,
    ProjectedTransformDirection,
)


@dataclass(frozen=True, slots=True)
class LegacyModelExecutionContext:
    """Mathematical model context copied from a legacy runtime assembly."""

    physical_shape: tuple[int, ...]
    spectral_shape: tuple[int, ...]
    lengths: tuple[float, ...]
    axis_coordinates: tuple[torch.Tensor, ...]
    real_dtype: torch.dtype
    device: torch.device
    batch_size: int
    _laplacians: Mapping[str, torch.Tensor]
    _boundary_conditions: Mapping[str, tuple[str, ...]]
    _projector: BasisAwareSpectralProjector
    _spectral_dtype: torch.dtype
    _performance_recorder: RuntimePerformanceRecorder | None = None

    def __post_init__(self) -> None:
        physical_shape = tuple(self.physical_shape)
        spectral_shape = tuple(self.spectral_shape)
        lengths = tuple(self.lengths)
        coordinates = tuple(self.axis_coordinates)
        if not physical_shape or any(
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            for size in physical_shape
        ):
            raise ValueError("physical_shape must contain positive integers")
        if len(spectral_shape) != len(physical_shape) or any(
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            for size in spectral_shape
        ):
            raise ValueError("spectral_shape must match the physical dimension")
        if len(lengths) != len(physical_shape) or any(
            not isinstance(length, (int, float))
            or isinstance(length, bool)
            or not math.isfinite(float(length))
            or float(length) <= 0.0
            for length in lengths
        ):
            raise ValueError("lengths must be positive and match the dimension")
        if not isinstance(self.real_dtype, torch.dtype):
            raise TypeError("real_dtype must be a torch.dtype")
        if not isinstance(self.device, torch.device):
            raise TypeError("device must be a torch.device")
        if (
            not isinstance(self.batch_size, int)
            or isinstance(self.batch_size, bool)
            or self.batch_size <= 0
        ):
            raise ValueError("batch_size must be a positive integer")
        if len(coordinates) != len(physical_shape):
            raise ValueError("axis coordinates must match the dimension")
        for axis, coordinate in enumerate(coordinates):
            if not isinstance(coordinate, torch.Tensor):
                raise TypeError("axis coordinates must be tensors")
            if coordinate.shape != (physical_shape[axis],):
                raise ValueError("axis-coordinate shape is inconsistent")
            if coordinate.dtype != self.real_dtype:
                raise ValueError("axis-coordinate dtype is inconsistent")
            if coordinate.device != self.device:
                raise ValueError("axis-coordinate device is inconsistent")
        if not isinstance(self._laplacians, Mapping):
            raise TypeError("_laplacians must be a mapping")
        laplacians = dict(self._laplacians)
        if any(not isinstance(name, str) for name in laplacians):
            raise TypeError("Laplacian component names must be strings")
        for laplacian in laplacians.values():
            if not isinstance(laplacian, torch.Tensor):
                raise TypeError("Laplacian eigenvalues must be tensors")
            if laplacian.shape != spectral_shape:
                raise ValueError("Laplacian eigenvalue shape is inconsistent")
            if laplacian.dtype != self.real_dtype:
                raise ValueError("Laplacian eigenvalue dtype is inconsistent")
            if laplacian.device != self.device:
                raise ValueError("Laplacian eigenvalue device is inconsistent")
        if not isinstance(self._boundary_conditions, Mapping):
            raise TypeError("_boundary_conditions must be a mapping")
        boundaries = dict(self._boundary_conditions)
        if set(boundaries) != set(laplacians):
            raise ValueError(
                "model operator boundaries must match planned components"
            )
        if not all(
            isinstance(name, str)
            and name.isidentifier()
            and isinstance(value, tuple)
            and len(value) == len(physical_shape)
            and all(
                condition in ("periodic", "dirichlet", "neumann")
                for condition in value
            )
            for name, value in boundaries.items()
        ):
            raise ValueError("model operator boundaries are invalid")
        if not isinstance(self._projector, BasisAwareSpectralProjector):
            raise TypeError("_projector must be a BasisAwareSpectralProjector")
        if (
            tuple(self._projector.physical_shape) != physical_shape
            or tuple(self._projector.shape) != spectral_shape
            or self._projector.real_dtype != self.real_dtype
            or torch.device(self._projector.device) != self.device
        ):
            raise ValueError("model operator projector is inconsistent")
        if not isinstance(self._spectral_dtype, torch.dtype):
            raise TypeError("_spectral_dtype must be a torch.dtype")
        if self._spectral_dtype not in (torch.complex64, torch.complex128):
            raise ValueError("_spectral_dtype must be a complex tensor dtype")
        if self._performance_recorder is not None and not isinstance(
            self._performance_recorder,
            RuntimePerformanceRecorder,
        ):
            raise TypeError(
                "_performance_recorder must be a RuntimePerformanceRecorder "
                "or None"
            )
        if (
            self._performance_recorder is not None
            and self._performance_recorder.device != self.device
        ):
            raise ValueError("performance-recorder device is inconsistent")
        object.__setattr__(self, "physical_shape", physical_shape)
        object.__setattr__(self, "spectral_shape", spectral_shape)
        object.__setattr__(
            self,
            "lengths",
            tuple(float(length) for length in lengths),
        )
        object.__setattr__(self, "axis_coordinates", coordinates)
        object.__setattr__(
            self,
            "_laplacians",
            MappingProxyType(laplacians),
        )
        object.__setattr__(
            self,
            "_boundary_conditions",
            MappingProxyType(boundaries),
        )

    def laplacian_eigenvalues(self, component_name: str) -> torch.Tensor:
        if not isinstance(component_name, str):
            raise TypeError("component_name must be a string")
        try:
            return self._laplacians[component_name]
        except KeyError as exc:
            raise KeyError(
                f"unknown planned component {component_name!r}"
            ) from exc

    def _boundaries(self, component_name: str) -> tuple[str, ...]:
        try:
            return self._boundary_conditions[component_name]
        except KeyError as exc:
            raise KeyError(
                f"unknown planned component {component_name!r}"
            ) from exc

    def _forward_projected(
        self,
        component_name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        boundaries = self._boundaries(component_name)
        if not isinstance(value, torch.Tensor):
            raise TypeError("operator input must be a tensor")
        expected_shape = (self.batch_size, *self.physical_shape)
        if value.shape != expected_shape:
            raise ValueError(
                f"physical value shape must be {expected_shape!r}, "
                f"got {tuple(value.shape)!r}"
            )
        if value.dtype != self.real_dtype or value.device != self.device:
            raise ValueError(
                "physical value dtype and device must match the model context"
            )
        if self._performance_recorder is None:
            return self._projector.forward_transform(value, boundaries)
        with self._performance_recorder.region("operators.forward_projected"):
            return self._projector.forward_transform(value, boundaries)

    def _inverse_projected(
        self,
        component_name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        boundaries = self._boundaries(component_name)
        if not isinstance(value, torch.Tensor):
            raise TypeError("operator spectral value must be a tensor")
        expected_shape = (self.batch_size, *self.spectral_shape)
        if value.shape != expected_shape:
            raise ValueError(
                f"spectral value shape must be {expected_shape!r}, "
                f"got {tuple(value.shape)!r}"
            )
        if value.dtype != self._spectral_dtype or value.device != self.device:
            raise ValueError(
                "spectral value dtype and device must match the model context"
            )
        if self._performance_recorder is None:
            return self._projector.inverse_transform(value, boundaries)
        with self._performance_recorder.region("operators.inverse_projected"):
            return self._projector.inverse_transform(value, boundaries)

    def _transform_projected_packed(
        self,
        boundary_conditions: tuple[str, ...],
        value: torch.Tensor,
        *,
        direction: ProjectedTransformDirection,
    ) -> torch.Tensor:
        """Transform a scheduler-owned, boundary-compatible packed batch."""

        boundaries = tuple(boundary_conditions)
        if len(boundaries) != len(self.physical_shape) or any(
            condition not in ("periodic", "dirichlet", "neumann")
            for condition in boundaries
        ):
            raise ValueError("packed transform boundaries are invalid")
        if not isinstance(direction, ProjectedTransformDirection):
            raise TypeError("direction must be a ProjectedTransformDirection")
        if not isinstance(value, torch.Tensor):
            raise TypeError("packed projected value must be a tensor")
        inverse = direction is ProjectedTransformDirection.INVERSE
        expected_tail = self.spectral_shape if inverse else self.physical_shape
        expected_dtype = self._spectral_dtype if inverse else self.real_dtype
        if (
            value.ndim != len(expected_tail) + 1
            or tuple(value.shape[1:]) != tuple(expected_tail)
            or value.shape[0] <= 0
            or value.shape[0] % self.batch_size != 0
        ):
            raise ValueError("packed projected value shape is incompatible")
        if value.dtype != expected_dtype or value.device != self.device:
            raise ValueError("packed projected value dtype and device must match")
        region = (
            "operators.inverse_projected"
            if inverse
            else "operators.forward_projected"
        )

        if self._performance_recorder is None:
            return (
                self._projector.inverse_transform(value, boundaries)
                if inverse
                else self._projector.forward_transform(value, boundaries)
            )
        with self._performance_recorder.region(region):
            return (
                self._projector.inverse_transform(value, boundaries)
                if inverse
                else self._projector.forward_transform(value, boundaries)
            )

    def gradient(
        self,
        source_component: str,
        output_component: str,
        value: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        if (
            not isinstance(axis, int)
            or isinstance(axis, bool)
            or axis < 0
            or axis >= len(self.physical_shape)
        ):
            raise ValueError("gradient axis is out of range")
        source_boundaries = self._boundaries(source_component)
        output_boundaries = self._boundaries(output_component)
        source_hat = self._forward_projected(source_component, value)
        derivative_hat, derivative_boundaries = (
            self._projector.transform_backend.gradient_hat(
                source_hat,
                source_boundaries,
                axis,
            )
        )
        if tuple(derivative_boundaries) != output_boundaries:
            raise ValueError(
                "gradient output boundary space does not match the declared "
                f"component {output_component!r}"
            )
        return self._inverse_projected(output_component, derivative_hat)

    def laplacian(
        self,
        component_name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        boundaries = self._boundaries(component_name)
        value_hat = self._forward_projected(component_name, value)
        laplacian_hat = self._projector.transform_backend.laplacian_hat(
            value_hat,
            boundaries,
        )
        return self._inverse_projected(component_name, laplacian_hat)

    def divergence(
        self,
        source_components: tuple[str, ...],
        output_component: str,
        values: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        try:
            source_components = tuple(source_components)
            values = tuple(values)
        except TypeError as exc:
            raise TypeError(
                "divergence sources and values must be iterable"
            ) from exc
        ndim = len(self.physical_shape)
        if len(source_components) != ndim or len(values) != ndim:
            raise ValueError(
                "divergence requires one source component and value per axis"
            )
        output_boundaries = self._boundaries(output_component)
        divergence_hat = None
        for axis, (source_component, value) in enumerate(
            zip(source_components, values, strict=True)
        ):
            source_boundaries = self._boundaries(source_component)
            source_hat = self._forward_projected(source_component, value)
            derivative_hat, derivative_boundaries = (
                self._projector.transform_backend.gradient_hat(
                    source_hat,
                    source_boundaries,
                    axis,
                )
            )
            if tuple(derivative_boundaries) != output_boundaries:
                raise ValueError(
                    "divergence term boundary space does not match the "
                    f"declared component {output_component!r}"
                )
            divergence_hat = (
                derivative_hat
                if divergence_hat is None
                else divergence_hat + derivative_hat
            )
        return self._inverse_projected(output_component, divergence_hat)


@dataclass(frozen=True, slots=True)
class LegacyAlgebraicSolverContext:
    """Restricted spectral operations for dispatched algebraic solvers."""

    geometry_name: str
    model_context: LegacyModelExecutionContext
    spectral_dtype: torch.dtype
    execution_policy: AlgebraicExecutionPolicy
    representation_cache: AlgebraicRepresentationCache | None = None
    _transform_scheduler: BoundarySignatureTransformScheduler = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.geometry_name, str) or not self.geometry_name:
            raise ValueError("geometry_name must be a non-empty string")
        if not isinstance(self.model_context, LegacyModelExecutionContext):
            raise TypeError(
                "model_context must be a LegacyModelExecutionContext"
            )
        if not isinstance(self.spectral_dtype, torch.dtype):
            raise TypeError("spectral_dtype must be a torch.dtype")
        if self.spectral_dtype != self.model_context._spectral_dtype:
            raise ValueError("algebraic and model spectral dtypes must match")
        if not isinstance(self.execution_policy, AlgebraicExecutionPolicy):
            raise TypeError(
                "execution_policy must be an AlgebraicExecutionPolicy"
            )
        if self.representation_cache is not None and not isinstance(
            self.representation_cache,
            AlgebraicRepresentationCache,
        ):
            raise TypeError(
                "representation_cache must be an AlgebraicRepresentationCache "
                "or None"
            )
        if self.execution_policy.representation_reuse != (
            self.representation_cache is not None
        ):
            raise ValueError(
                "representation cache does not match the execution policy"
            )
        object.__setattr__(
            self,
            "_transform_scheduler",
            BoundarySignatureTransformScheduler(self),
        )

    @property
    def lazy_physical_materialization(self) -> bool:
        """Compatibility view derived from the unified policy."""

        return self.execution_policy.lazy_physical_materialization

    @property
    def batched_physical_islands(self) -> bool:
        """Compatibility view derived from the unified policy."""

        return self.execution_policy.batched_physical_islands

    @property
    def transform_scheduler(self) -> BoundarySignatureTransformScheduler:
        """Runtime-local scheduler reused by every algebraic generation."""

        return self._transform_scheduler

    @property
    def physical_shape(self) -> tuple[int, ...]:
        return self.model_context.physical_shape

    @property
    def spectral_shape(self) -> tuple[int, ...]:
        return self.model_context.spectral_shape

    @property
    def lengths(self) -> tuple[float, ...]:
        return self.model_context.lengths

    @property
    def axis_coordinates(self) -> tuple[torch.Tensor, ...]:
        return self.model_context.axis_coordinates

    @property
    def real_dtype(self) -> torch.dtype:
        return self.model_context.real_dtype

    @property
    def device(self) -> torch.device:
        return self.model_context.device

    @property
    def batch_size(self) -> int:
        return self.model_context.batch_size

    @property
    def legacy_transform_backend(self):
        """Current backend exposed only to opt-in legacy solver adapters."""

        return self.model_context._projector.transform_backend

    @property
    def legacy_projector(self):
        """Current projector exposed only to opt-in legacy solver adapters."""

        return self.model_context._projector

    def laplacian_eigenvalues(self, component_name: str) -> torch.Tensor:
        return self.model_context.laplacian_eigenvalues(component_name)

    def forward_projected(
        self,
        component_name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if self.representation_cache is not None:
            cached = self.representation_cache.spectral_for(
                component_name,
                value,
            )
            if cached is not None:
                return cached
        return self.model_context._forward_projected(component_name, value)

    def forward_projected_many(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
    ) -> Mapping[str, torch.Tensor]:
        """Compatibility facade over the centralized transform scheduler."""

        component_names = tuple(component_names)
        values = tuple(values)
        if not self.execution_policy.batched_physical_islands:
            return {
                name: self.forward_projected(name, value)
                for name, value in zip(component_names, values, strict=True)
            }
        transformed = self._transform_scheduler.forward_values_many(
            component_names,
            values,
        )
        return dict(zip(component_names, transformed, strict=True))

    def inverse_projected(
        self,
        component_name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        if self.representation_cache is not None:
            cached = self.representation_cache.physical_for(
                component_name,
                value,
            )
            if cached is not None:
                return cached
        return self.model_context._inverse_projected(component_name, value)

    def transform_projected_packed(
        self,
        boundary_conditions: tuple[str, ...],
        value: torch.Tensor,
        *,
        direction: ProjectedTransformDirection,
    ) -> torch.Tensor:
        return self.model_context._transform_projected_packed(
            boundary_conditions,
            value,
            direction=direction,
        )

    def spectral_dependency(
        self,
        state: Mapping[str, torch.Tensor],
        component_name: str,
    ) -> torch.Tensor:
        """Read a dependency in native spectral form when it is available."""

        if not isinstance(state, Mapping):
            raise TypeError("algebraic state must be a mapping")
        if isinstance(state, AlgebraicPhysicalStateView):
            return state.spectral(component_name)
        if component_name not in state:
            raise KeyError(component_name)
        return self.forward_projected(component_name, state[component_name])

    def materialize_dependency(
        self,
        component_name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        """Invert one spectrum and register its exact generation-local pair."""

        physical = self.inverse_projected(component_name, value)
        if (
            self.representation_cache is not None
            and self.representation_cache.active
        ):
            self.register_representation_pair(
                component_name,
                physical,
                value,
            )
        return physical

    def begin_representation_generation(
        self,
        generation: int,
        physical: dict[str, torch.Tensor],
        spectral: dict[str, torch.Tensor],
    ) -> None:
        if self.representation_cache is not None:
            self.representation_cache.begin(generation, physical, spectral)

    def register_representation_pair(
        self,
        component_name: str,
        physical: torch.Tensor,
        spectral: torch.Tensor,
    ) -> None:
        if self.representation_cache is not None:
            self.representation_cache.register(
                component_name,
                physical,
                spectral,
            )

    def end_representation_generation(self) -> Mapping[str, object] | None:
        if self.representation_cache is None:
            return None
        return self.representation_cache.end()

    def abort_representation_generation(self) -> None:
        if self.representation_cache is not None:
            self.representation_cache.abort()

    def representation_reuse_snapshot(self) -> Mapping[str, object] | None:
        if self.representation_cache is None:
            return None
        return self.representation_cache.last_snapshot()

    def gradient(
        self,
        source_component: str,
        output_component: str,
        value: torch.Tensor,
        axis: int,
    ) -> torch.Tensor:
        if (
            not isinstance(axis, int)
            or isinstance(axis, bool)
            or axis < 0
            or axis >= len(self.physical_shape)
        ):
            raise ValueError("gradient axis is out of range")
        source_boundaries = self.boundary_conditions(source_component)
        output_boundaries = self.boundary_conditions(output_component)
        source_hat = self.forward_projected(source_component, value)
        derivative_hat, derivative_boundaries = (
            self.legacy_transform_backend.gradient_hat(
                source_hat,
                source_boundaries,
                axis,
            )
        )
        if tuple(derivative_boundaries) != output_boundaries:
            raise ValueError(
                "gradient output boundary space does not match the declared "
                f"component {output_component!r}"
            )
        return self.inverse_projected(output_component, derivative_hat)

    def laplacian(
        self,
        component_name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        boundaries = self.boundary_conditions(component_name)
        value_hat = self.forward_projected(component_name, value)
        laplacian_hat = self.legacy_transform_backend.laplacian_hat(
            value_hat,
            boundaries,
        )
        return self.inverse_projected(component_name, laplacian_hat)

    def divergence(
        self,
        source_components: tuple[str, ...],
        output_component: str,
        values: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        try:
            source_components = tuple(source_components)
            values = tuple(values)
        except TypeError as exc:
            raise TypeError(
                "divergence sources and values must be iterable"
            ) from exc
        ndim = len(self.physical_shape)
        if len(source_components) != ndim or len(values) != ndim:
            raise ValueError(
                "divergence requires one source component and value per axis"
            )
        output_boundaries = self.boundary_conditions(output_component)
        divergence_hat = None
        for axis, (source_component, value) in enumerate(
            zip(source_components, values, strict=True)
        ):
            source_boundaries = self.boundary_conditions(source_component)
            source_hat = self.forward_projected(source_component, value)
            derivative_hat, derivative_boundaries = (
                self.legacy_transform_backend.gradient_hat(
                    source_hat,
                    source_boundaries,
                    axis,
                )
            )
            if tuple(derivative_boundaries) != output_boundaries:
                raise ValueError(
                    "divergence term boundary space does not match the "
                    f"declared component {output_component!r}"
                )
            divergence_hat = (
                derivative_hat
                if divergence_hat is None
                else divergence_hat + derivative_hat
            )
        return self.inverse_projected(output_component, divergence_hat)

    def boundary_conditions(self, component_name: str) -> tuple[str, ...]:
        return self.model_context._boundaries(component_name)


@dataclass(frozen=True, slots=True)
class ResolvedAlgebraicSystem:
    """Auditable result of one exact geometry/capability dispatch."""

    system: AlgebraicSystemSpec
    registration: AlgebraicSolverRegistration
    solver: AlgebraicSolverProtocol

    def __post_init__(self) -> None:
        if not isinstance(self.system, AlgebraicSystemSpec):
            raise TypeError("system must be an AlgebraicSystemSpec")
        if not isinstance(self.registration, AlgebraicSolverRegistration):
            raise TypeError(
                "registration must be an AlgebraicSolverRegistration"
            )
        if not isinstance(self.solver, AlgebraicSolverProtocol):
            raise TypeError("solver must implement AlgebraicSolverProtocol")
        if self.registration.capability != self.system.capability:
            raise ValueError(
                "resolved registration capability does not match request"
            )
        if self.solver.capability != self.system.capability:
            raise ValueError("resolved solver capability does not match request")
        if self.solver.implementation_name != (
            self.registration.implementation_name
        ):
            raise ValueError(
                "resolved solver implementation does not match registration"
            )
        if tuple(self.solver.output_components) != (
            self.system.output_components
        ):
            raise ValueError("resolved solver outputs do not match request")

    def to_metadata(self) -> dict[str, object]:
        solver_observability = {
            "diagnostics": "unavailable",
            "restart": "stateless",
        }
        if isinstance(self.solver, InspectableAlgebraicSolverProtocol):
            solver_observability = dict(
                self.solver.observability_metadata()
            )
            try:
                json.dumps(
                    solver_observability,
                    allow_nan=False,
                    sort_keys=True,
                )
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "algebraic observability metadata must be finite and "
                    "JSON-compatible"
                ) from exc
        return {
            "system": self.system.to_metadata(),
            "dispatch": self.registration.to_metadata(),
            "observability": solver_observability,
        }


class LegacyAlgebraicFieldsAdapter(torch.nn.Module):
    """Evaluate a frozen algebraic DAG and publish its transient cache."""

    def __init__(
        self,
        assembly: LegacyAssemblySpec,
        execution_plan: AlgebraicExecutionPlan,
        resolved_systems: tuple[ResolvedAlgebraicSystem, ...],
        context: LegacyAlgebraicSolverContext,
        *,
        spectral_dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
        physical_island_scheduler: (
            AlgebraicPhysicalIslandScheduler | None
        ) = None,
        performance_recorder: RuntimePerformanceRecorder | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(execution_plan, AlgebraicExecutionPlan):
            raise TypeError("execution_plan must be an AlgebraicExecutionPlan")
        if not isinstance(context, LegacyAlgebraicSolverContext):
            raise TypeError("context must be a LegacyAlgebraicSolverContext")
        if tuple(
            resolved.system.name for resolved in resolved_systems
        ) != execution_plan.execution_order:
            raise ValueError(
                "resolved algebraic systems must follow the frozen execution "
                "order"
            )
        self._execution_plan = execution_plan
        self._resolved_systems = tuple(resolved_systems)
        self._component_names = tuple(
            field.name for field in assembly.algebraic_fields
        )
        self._spectral_shape = assembly.source_plan.spectral_shape
        self._spectral_dtype = spectral_dtype
        self._device = device
        self._batch_size = batch_size
        self._context = context
        if physical_island_scheduler is not None and not isinstance(
            physical_island_scheduler,
            AlgebraicPhysicalIslandScheduler,
        ):
            raise TypeError(
                "physical_island_scheduler must be an "
                "AlgebraicPhysicalIslandScheduler or None"
            )
        if context.batched_physical_islands != (
            physical_island_scheduler is not None
        ):
            raise ValueError(
                "physical-island scheduler does not match execution policy"
            )
        self._physical_island_scheduler = physical_island_scheduler
        self._performance_recorder = performance_recorder
        self._lazy_physical_materialization = (
            context.lazy_physical_materialization
        )
        self._batched_physical_islands = context.batched_physical_islands
        self._transient_names = execution_plan.transient_components
        produced_dependencies = {
            dependency
            for system in execution_plan.declared_systems
            for dependency in system.dependencies
            if dependency in execution_plan.output_components
        }
        self._physical_cache_names = frozenset(
            produced_dependencies | set(self._transient_names)
        )
        self._transient_cache: Mapping[str, torch.Tensor] | None = None
        self._generation_state: AlgebraicGenerationState | None = None
        self._last_invalidated_materialization: Mapping[str, object] | None = None
        self._cache_generation = 0
        self._last_representation_reuse: Mapping[str, object] | None = None

    @property
    def cache_generation(self) -> int:
        return self._cache_generation

    @property
    def representation_reuse_enabled(self) -> bool:
        return self._context.representation_cache is not None

    @property
    def lazy_physical_materialization_enabled(self) -> bool:
        return self._lazy_physical_materialization

    @property
    def batched_physical_islands_enabled(self) -> bool:
        return self._batched_physical_islands

    def clear_cached_outputs(self) -> None:
        """Invalidate non-stored outputs without touching physical state."""

        if self._generation_state is not None:
            self._last_invalidated_materialization = (
                self._generation_state.invalidate()
            )
            self._generation_state = None
        self._transient_cache = None
        self._last_representation_reuse = None
        self._context.abort_representation_generation()

    def representation_reuse_snapshot(self) -> Mapping[str, object] | None:
        """Return counters from the most recent completed DAG generation."""

        return self._last_representation_reuse

    def physical_materialization_snapshot(
        self,
    ) -> Mapping[str, object] | None:
        """Return current lazy-state counters without exposing its tensors."""

        if not self._lazy_physical_materialization:
            return None
        if self._generation_state is not None:
            return self._generation_state.snapshot()
        return self._last_invalidated_materialization

    def transient_state(self) -> Mapping[str, torch.Tensor]:
        """Return the current read-only transient cache."""

        if not self._transient_names:
            return MappingProxyType({})
        if self._transient_cache is None:
            raise RuntimeError(
                "transient algebraic outputs are not synchronized"
            )
        return self._transient_cache

    def prefetch_physical_components(
        self,
        component_names: tuple[str, ...],
    ) -> None:
        """Prepare one explicit-RHS physical island without extending lifetime."""

        if not self._batched_physical_islands:
            raise RuntimeError("batched physical islands are not enabled")
        component_names = tuple(component_names)
        if not set(component_names).issubset(self._transient_names):
            raise ValueError(
                "explicit physical island must contain transient components"
            )
        if self._generation_state is None:
            raise RuntimeError("algebraic outputs are not synchronized")
        self._generation_state.prefetch_physical(component_names)

    def forward(self, fields, parameters):
        del parameters
        if self._generation_state is not None:
            self._last_invalidated_materialization = (
                self._generation_state.invalidate()
            )
            self._generation_state = None
            self._transient_cache = None
        outputs: dict[str, torch.Tensor] = {}
        physical_state = {
            name: fields[name]
            for name in self._execution_plan.initial_components
        }
        spectral_state = {
            name: fields[f"{name}.hat"]
            for name in self._execution_plan.initial_components
        }
        transient_cache: dict[str, torch.Tensor] = {}
        generation = self._cache_generation + 1
        self._last_representation_reuse = None
        self._context.begin_representation_generation(
            generation,
            physical_state,
            spectral_state,
        )
        generation_state = None
        if self._lazy_physical_materialization:
            generation_state = AlgebraicGenerationState(
                generation,
                physical_state,
                spectral_state,
                materialize=self._context.materialize_dependency,
                materialize_many=(
                    self._physical_island_scheduler.materialize_dependencies
                    if self._physical_island_scheduler is not None
                    else None
                ),
            )
        try:
            for resolved in self._resolved_systems:
                if generation_state is None:
                    dependencies = MappingProxyType(
                        {
                            name: physical_state[name]
                            for name in resolved.system.dependencies
                        }
                    )
                else:
                    dependencies = generation_state.view(
                        resolved.system.dependencies
                    )
                    if self._batched_physical_islands:
                        physical_dependencies = (
                            resolved.solver.physical_dependencies
                            if isinstance(
                                resolved.solver,
                                AlgebraicPhysicalDependenciesProtocol,
                            )
                            else resolved.system.dependencies
                        )
                        physical_dependencies = tuple(physical_dependencies)
                        if not set(physical_dependencies).issubset(
                            resolved.system.dependencies
                        ):
                            raise ValueError(
                                "solver physical dependencies exceed its "
                                "declared algebraic dependencies"
                            )
                        dependencies.prefetch_physical(physical_dependencies)
                if self._performance_recorder is None:
                    solution = resolved.solver.solve_spectral(dependencies)
                else:
                    with self._performance_recorder.region(
                        f"algebraic.{resolved.system.name}.solve"
                    ):
                        solution = resolved.solver.solve_spectral(dependencies)
                if not isinstance(solution, Mapping):
                    raise TypeError("algebraic solve must return a mapping")
                solution = dict(solution)
                if any(not isinstance(name, str) for name in solution):
                    raise TypeError("algebraic solution keys must be strings")
                if set(solution) != set(resolved.system.output_components):
                    raise ValueError(
                        f"algebraic solution keys do not match system "
                        f"{resolved.system.name!r}"
                    )
                for name in resolved.system.output_components:
                    value = solution[name]
                    expected_shape = (
                        self._batch_size,
                        *self._spectral_shape,
                    )
                    if not isinstance(value, torch.Tensor):
                        raise TypeError(
                            f"algebraic solution for {name!r} must be a tensor"
                        )
                    if value.shape != expected_shape:
                        raise ValueError(
                            f"algebraic solution for {name!r} has shape "
                            f"{tuple(value.shape)!r}; expected {expected_shape!r}"
                        )
                    if value.dtype != self._spectral_dtype:
                        raise ValueError(
                            f"algebraic solution for {name!r} has dtype "
                            f"{value.dtype}; expected {self._spectral_dtype}"
                        )
                    if value.device != self._device:
                        raise ValueError(
                            f"algebraic solution for {name!r} is on "
                            f"{value.device}; expected {self._device}"
                        )
                    outputs[name] = value
                    spectral_state[name] = value
                    if generation_state is not None:
                        generation_state.publish_spectral(name, value)
                    elif name in self._physical_cache_names:
                        if self._performance_recorder is None:
                            physical = self._context.inverse_projected(name, value)
                        else:
                            with self._performance_recorder.region(
                                "algebraic.materialize_physical"
                            ):
                                physical = self._context.inverse_projected(
                                    name,
                                    value,
                                )
                        physical_state[name] = physical
                        self._context.register_representation_pair(
                            name,
                            physical,
                            value,
                        )
                        if name in self._transient_names:
                            transient_cache[name] = physical
        except BaseException:
            if generation_state is not None:
                generation_state.invalidate()
            self._context.abort_representation_generation()
            raise
        values = [outputs[name] for name in self._component_names]
        if self._performance_recorder is None:
            packed = torch.stack(values)
        else:
            with self._performance_recorder.region(
                "algebraic.publish_spectral"
            ):
                packed = torch.stack(values)
        if generation_state is not None:
            for index, name in enumerate(self._component_names):
                generation_state.replace_spectral(name, packed[index])
            transient_mapping: Mapping[str, torch.Tensor] = (
                generation_state.view(self._transient_names)
            )
        else:
            transient_mapping = MappingProxyType(transient_cache)
        self._last_representation_reuse = (
            self._context.end_representation_generation()
        )
        if set(transient_mapping) != set(self._transient_names):
            raise RuntimeError("transient algebraic cache is incomplete")
        self._generation_state = generation_state
        self._transient_cache = transient_mapping
        self._cache_generation += 1
        return packed


class _CombinedPhysicalState(Mapping[str, torch.Tensor]):
    """Read evolved/static fields plus lazily materialized transients."""

    def __init__(
        self,
        primary: Mapping[str, torch.Tensor],
        transient: Mapping[str, torch.Tensor],
    ) -> None:
        if not isinstance(primary, Mapping) or not isinstance(transient, Mapping):
            raise TypeError("combined physical state inputs must be mappings")
        overlap = set(primary) & set(transient)
        if overlap:
            raise ValueError(
                "primary and transient physical states must be disjoint"
            )
        self._primary = primary
        self._transient = transient
        self._names = (*tuple(primary), *tuple(transient))

    def __getitem__(self, name: str) -> torch.Tensor:
        if name in self._primary:
            return self._primary[name]
        return self._transient[name]

    def __iter__(self):
        return iter(self._names)

    def __len__(self) -> int:
        return len(self._names)


class LegacyExplicitRHSAdapter(torch.nn.Module):
    """Translate a physical-space model RHS to native spectral storage."""

    def __init__(
        self,
        model: ExecutableModelProtocol,
        assembly: LegacyAssemblySpec,
        projector: BasisAwareSpectralProjector,
        context: ModelExecutionContext,
        algebraic_fields_adapter: LegacyAlgebraicFieldsAdapter | None = None,
        physical_island_scheduler: (
            AlgebraicPhysicalIslandScheduler | None
        ) = None,
        performance_recorder: RuntimePerformanceRecorder | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._state_names = tuple(field.name for field in assembly.fields)
        self._component_names = tuple(
            field.name for field in assembly.evolved_fields
        )
        self._boundary_conditions = tuple(
            field.boundary_conditions for field in assembly.evolved_fields
        )
        self._projector = projector
        self._context = context
        self._algebraic_fields_adapter = algebraic_fields_adapter
        if physical_island_scheduler is not None and not isinstance(
            physical_island_scheduler,
            AlgebraicPhysicalIslandScheduler,
        ):
            raise TypeError(
                "physical_island_scheduler must be an "
                "AlgebraicPhysicalIslandScheduler or None"
            )
        self._physical_island_scheduler = physical_island_scheduler
        self._performance_recorder = performance_recorder
        self._transient_names = assembly.omitted_transient_components
        self._batched_physical_islands = bool(
            algebraic_fields_adapter is not None
            and algebraic_fields_adapter.batched_physical_islands_enabled
        )
        if self._batched_physical_islands != (
            physical_island_scheduler is not None
        ):
            raise ValueError(
                "explicit-RHS scheduler does not match execution policy"
            )
        self._explicit_physical_dependencies = (
            tuple(model.explicit_rhs_physical_dependencies)
            if isinstance(model, ExplicitRHSPhysicalDependenciesProtocol)
            else ()
        )
        if self._batched_physical_islands and not isinstance(
            model,
            ExplicitRHSPhysicalDependenciesProtocol,
        ):
            raise TypeError(
                "batched physical islands require explicit RHS dependency "
                "metadata"
            )
        available_state = {*self._state_names, *self._transient_names}
        if not set(self._explicit_physical_dependencies).issubset(
            available_state
        ):
            raise ValueError(
                "explicit RHS physical dependencies exceed declared state"
            )
        if self._transient_names and algebraic_fields_adapter is None:
            raise ValueError(
                "transient fields require an algebraic cache provider"
            )

    def forward(self, fields, parameters):
        del parameters
        state_values = {name: fields[name] for name in self._state_names}
        state: Mapping[str, torch.Tensor]
        if self._algebraic_fields_adapter is not None:
            if self._batched_physical_islands:
                required_transients = tuple(
                    name
                    for name in self._explicit_physical_dependencies
                    if name in self._transient_names
                )
                self._algebraic_fields_adapter.prefetch_physical_components(
                    required_transients
                )
            transient = self._algebraic_fields_adapter.transient_state()
            if set(transient) != set(self._transient_names):
                raise RuntimeError("transient algebraic state is incomplete")
            state = _CombinedPhysicalState(
                MappingProxyType(state_values),
                transient,
            )
        else:
            state = MappingProxyType(state_values)
        if self._performance_recorder is None:
            explicit = self._model.explicit_rhs(state, self._context)
        else:
            with self._performance_recorder.region("explicit_rhs.model"):
                explicit = self._model.explicit_rhs(state, self._context)
        if not isinstance(explicit, Mapping):
            raise TypeError("model explicit_rhs() must return a mapping")
        explicit = dict(explicit)
        if any(not isinstance(name, str) for name in explicit):
            raise TypeError("explicit_rhs keys must be strings")
        actual = set(explicit)
        expected = set(self._component_names)
        if actual != expected:
            missing = tuple(sorted(expected - actual))
            unexpected = tuple(sorted(actual - expected))
            raise ValueError(
                "explicit_rhs keys do not match evolved components; "
                f"missing={missing!r}, unexpected={unexpected!r}"
            )

        values = []
        for name, boundary_conditions in zip(
            self._component_names,
            self._boundary_conditions,
        ):
            value = explicit[name]
            reference = state[name]
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"explicit RHS for {name!r} must be a tensor")
            if value.shape != reference.shape:
                raise ValueError(
                    f"explicit RHS for {name!r} has shape {tuple(value.shape)!r}; "
                    f"expected {tuple(reference.shape)!r}"
                )
            if value.dtype != reference.dtype:
                raise ValueError(
                    f"explicit RHS for {name!r} has dtype {value.dtype}; "
                    f"expected {reference.dtype}"
                )
            if value.device != reference.device:
                raise ValueError(
                    f"explicit RHS for {name!r} is on {value.device}; "
                    f"expected {reference.device}"
                )
            values.append(value)
        if self._batched_physical_islands:
            spectral = list(
                self._physical_island_scheduler.project_output_values(
                    self._component_names,
                    tuple(values),
                )
            )
        else:
            spectral = []
            for value, boundary_conditions in zip(
                values,
                self._boundary_conditions,
                strict=True,
            ):
                if self._performance_recorder is None:
                    spectral.append(
                        self._projector.forward_transform(
                            value,
                            boundary_conditions,
                        )
                    )
                else:
                    with self._performance_recorder.region(
                        "explicit_rhs.forward_projected"
                    ):
                        spectral.append(
                            self._projector.forward_transform(
                                value,
                                boundary_conditions,
                            )
                        )
        if self._performance_recorder is None:
            return torch.stack(spectral)
        with self._performance_recorder.region(
            "explicit_rhs.publish_spectral"
        ):
            return torch.stack(spectral)


@dataclass(frozen=True, slots=True)
class ExperimentalModelRuntime:
    """Objects created by one opt-in executable-model assembly."""

    problem: ProblemSpec
    plan: SpectralPlan
    assembly: LegacyAssemblySpec
    context: ModelExecutionContext
    algebraic_execution_policy: AlgebraicExecutionPolicy
    solver: SpectralSolver
    projector: BasisAwareSpectralProjector
    explicit_rhs_adapter: LegacyExplicitRHSAdapter
    algebraic_fields_adapter: LegacyAlgebraicFieldsAdapter | None
    algebraic_execution_plan: AlgebraicExecutionPlan | None
    resolved_algebraic_systems: tuple[ResolvedAlgebraicSystem, ...]
    performance_recorder: RuntimePerformanceRecorder | None = None

    def to_metadata(self) -> dict[str, object]:
        has_algebraic_fields = bool(self.resolved_algebraic_systems)
        return {
            "problem": self.problem.to_metadata(),
            "spectral_plan": self.plan.to_metadata(),
            "legacy_assembly": self.assembly.to_metadata(),
            "execution_adapter": "legacy_explicit_rhs",
            "algebraic_execution_policy": (
                self.algebraic_execution_policy.to_metadata()
            ),
            "time_integration": {
                "scheme": "semi_implicit_euler",
                "dynamic_spectral_projection": self.projector.enabled,
                "integrator": type(self.solver.integrator).__name__,
            },
            "algebraic_lifecycle": {
                "update_phase": AlgebraicUpdatePhase.PRE_EXPLICIT_RHS.value,
                "has_algebraic_fields": has_algebraic_fields,
                "initial_state_synchronized": (
                    True if has_algebraic_fields else None
                ),
                "post_step_state": (
                    "stale_until_next_pre_rhs_or_sync"
                    if has_algebraic_fields
                    else "not_applicable"
                ),
                "systems": [
                    resolved.to_metadata()
                    for resolved in self.resolved_algebraic_systems
                ],
                "execution_plan": (
                    self.algebraic_execution_plan.to_metadata()
                    if self.algebraic_execution_plan is not None
                    else None
                ),
                "stored_components": [
                    field.name for field in self.assembly.algebraic_fields
                ],
                "transient_components": list(
                    self.assembly.omitted_transient_components
                ),
                "transient_cache": {
                    "lifetime": "one_synchronized_pre_rhs_state",
                    "stored_in_legacy_fields": False,
                    "checkpointed": False,
                },
                "representation_reuse": {
                    "enabled": (
                        self.algebraic_fields_adapter is not None
                        and self.algebraic_fields_adapter.representation_reuse_enabled
                    ),
                    "scope": "one_pre_explicit_rhs_generation",
                    "identity_guard": "tensor_object_and_in_place_version",
                    "cross_generation_reuse": False,
                    "checkpointed": False,
                },
                "physical_materialization": {
                    "mode": (
                        "lazy_generation_local"
                        if (
                            self.algebraic_fields_adapter is not None
                            and self.algebraic_fields_adapter
                            .lazy_physical_materialization_enabled
                        )
                        else "eager"
                    ),
                    "spectral_dependencies_may_bypass_physical": bool(
                        self.algebraic_fields_adapter is not None
                        and self.algebraic_fields_adapter
                        .lazy_physical_materialization_enabled
                    ),
                    "lifetime": "one_synchronized_pre_rhs_state",
                    "cross_timestep_reuse": False,
                    "checkpointed": False,
                },
                "physical_islands": {
                    "mode": (
                        "boundary_signature_batched"
                        if (
                            self.algebraic_fields_adapter is not None
                            and self.algebraic_fields_adapter
                            .batched_physical_islands_enabled
                        )
                        else "on_demand_componentwise"
                    ),
                    "solver_dependencies_declared": bool(
                        self.algebraic_fields_adapter is not None
                        and self.algebraic_fields_adapter
                        .batched_physical_islands_enabled
                    ),
                    "explicit_rhs_dependencies_declared": bool(
                        self.algebraic_fields_adapter is not None
                        and self.algebraic_fields_adapter
                        .batched_physical_islands_enabled
                    ),
                    "grouping_key": "boundary_signature",
                    "cross_timestep_reuse": False,
                    "checkpointed": False,
                },
                "materialization_scheduler": {
                    "implementation": (
                        "boundary_signature_transform_scheduler"
                        if self.algebraic_execution_policy
                        .batched_physical_islands
                        else "componentwise_adapter"
                    ),
                    "grouping_key_fields": [
                        "boundary_signature",
                        "direction",
                        "physical_shape",
                        "spectral_shape",
                        "batch_size",
                        "real_dtype",
                        "spectral_dtype",
                        "device",
                    ],
                    "call_scope": "one_physical_computation_island",
                    "lifecycle_owner": "algebraic_generation_state",
                    "plan_compilation": "runtime_local_cached",
                    "plan_cache_key": [
                        "direction",
                        "ordered_component_names",
                    ],
                    "compiled_plans_retain_tensors": False,
                },
            },
        }

    def algebraic_representation_reuse_diagnostics(
        self,
    ) -> Mapping[str, object] | None:
        """Return last-generation reuse counters without retaining tensors."""

        if self.algebraic_fields_adapter is None:
            return None
        return self.algebraic_fields_adapter.representation_reuse_snapshot()

    def algebraic_physical_materialization_diagnostics(
        self,
    ) -> Mapping[str, object] | None:
        """Return Stage N.2 lazy-materialization counters, if enabled."""

        if self.algebraic_fields_adapter is None:
            return None
        return self.algebraic_fields_adapter.physical_materialization_snapshot()

    def synchronize_algebraic_for_observation(self) -> None:
        """Refresh algebraic fields against the current evolved state.

        This does not advance time or alter the spectral-refresh clock.  The
        next timestep may recompute the same algebraic state before its RHS.
        """

        if self.algebraic_fields_adapter is not None:
            self.solver.refresh_static_fields()

    def transient_algebraic_state(self) -> Mapping[str, torch.Tensor]:
        """Return synchronized transient outputs without promoting storage."""

        if self.algebraic_fields_adapter is None:
            return MappingProxyType({})
        return self.algebraic_fields_adapter.transient_state()

    def algebraic_diagnostics(self) -> dict[str, object]:
        """Return solver diagnostics without adding diagnostic state fields."""

        diagnostics: dict[str, object] = {}
        for resolved in self.resolved_algebraic_systems:
            if not isinstance(
                resolved.solver,
                InspectableAlgebraicSolverProtocol,
            ):
                continue
            value = dict(resolved.solver.diagnostic_snapshot())
            try:
                json.dumps(value, allow_nan=False, sort_keys=True)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "algebraic diagnostics must be finite and JSON-compatible"
                ) from exc
            diagnostics[resolved.system.name] = value
        return diagnostics

    def capture_algebraic_restart_state(
        self,
    ) -> AlgebraicRuntimeRestartState:
        """Capture implementation warm-start state with dispatch identity."""

        systems = []
        for resolved in self.resolved_algebraic_systems:
            tensors: Mapping[str, torch.Tensor] = {}
            if isinstance(
                resolved.solver,
                InspectableAlgebraicSolverProtocol,
            ):
                tensors = resolved.solver.capture_restart_state()
            systems.append(
                AlgebraicSystemRestartState(
                    system_name=resolved.system.name,
                    capability=resolved.system.capability,
                    implementation_name=(
                        resolved.registration.implementation_name
                    ),
                    provenance_sha256=self._algebraic_restart_identity(
                        resolved
                    ),
                    tensors=tensors,
                )
            )
        return AlgebraicRuntimeRestartState(1, tuple(systems))

    def _algebraic_restart_identity(
        self,
        resolved: ResolvedAlgebraicSystem,
    ) -> str:
        payload = {
            "dispatch": resolved.registration.to_metadata(),
            "geometry": self.problem.geometry.to_metadata(),
            "numerics": self.problem.numerics.to_metadata(),
            "real_dtype": str(self.context.real_dtype),
            "spectral_dtype": str(
                self.solver.transform_backend.spectral_dtype
            ),
            "system": resolved.system.to_metadata(),
        }
        canonical = json.dumps(
            payload,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(canonical).hexdigest()

    def restore_algebraic_restart_state(
        self,
        restart: AlgebraicRuntimeRestartState,
    ) -> None:
        """Restore warm-start state only after exact identity validation."""

        if not isinstance(restart, AlgebraicRuntimeRestartState):
            raise TypeError(
                "restart must be an AlgebraicRuntimeRestartState"
            )
        expected = {
            resolved.system.name: resolved
            for resolved in self.resolved_algebraic_systems
        }
        actual = {system.system_name: system for system in restart.systems}
        if set(actual) != set(expected):
            raise ValueError(
                "algebraic restart systems do not match this runtime"
            )
        for name, resolved in expected.items():
            saved = actual[name]
            identity = (
                resolved.system.capability,
                resolved.registration.implementation_name,
            )
            if (saved.capability, saved.implementation_name) != identity:
                raise ValueError(
                    f"algebraic restart dispatch mismatch for {name!r}"
                )
            if saved.provenance_sha256 != self._algebraic_restart_identity(
                resolved
            ):
                raise ValueError(
                    f"algebraic restart provenance mismatch for {name!r}"
                )
            if isinstance(
                resolved.solver,
                InspectableAlgebraicSolverProtocol,
            ):
                resolved.solver.restore_restart_state(saved.tensors)
            elif saved.tensors:
                raise ValueError(
                    f"stateless algebraic solver {name!r} received state"
                )

    def reset(
        self,
        initial_values: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        """Reset evolved fields and restore a current algebraic state."""

        if initial_values is not None:
            if not isinstance(initial_values, Mapping):
                raise TypeError("initial_values must be a mapping or None")
            initial_values = dict(initial_values)
            evolved_names = {
                field.name for field in self.assembly.evolved_fields
            }
            if any(not isinstance(name, str) for name in initial_values):
                raise TypeError("initial_values keys must be strings")
            unexpected = tuple(sorted(set(initial_values) - evolved_names))
            if unexpected:
                raise ValueError(
                    "reset initial_values may contain evolved components only; "
                    f"unexpected={unexpected!r}"
                )
            allowed_shapes = {
                self.assembly.backend.shape,
                (
                    self.solver.batchsize,
                    *self.assembly.backend.shape,
                ),
            }
            for name, value in initial_values.items():
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"reset value for {name!r} must be a tensor"
                    )
                if tuple(value.shape) not in allowed_shapes:
                    raise ValueError(
                        f"reset value for {name!r} has invalid shape "
                        f"{tuple(value.shape)!r}"
                    )
        self.solver.model.nlmodel = None
        self.solver.model.static_model = None
        if self.algebraic_fields_adapter is not None:
            self.algebraic_fields_adapter.clear_cached_outputs()
        self.solver.reset(initial_values)
        if self.projector.enabled:
            self.projector.project_dynamic_fields(
                self.solver.fields,
                sync_spatial=True,
            )
        if self.algebraic_fields_adapter is not None:
            for resolved in self.resolved_algebraic_systems:
                if isinstance(
                    resolved.solver,
                    InspectableAlgebraicSolverProtocol,
                ):
                    resolved.solver.restore_restart_state({})
            self.solver.model.set_static_compute_model(
                self.algebraic_fields_adapter
            )
            self.solver.model.set_static_inverse_transform(
                self.projector.inverse_transform
            )
            self.solver.refresh_static_fields()
        self.solver.model.set_nonlinear_model(self.explicit_rhs_adapter)
        self.solver.integrator.restore_progress(
            0,
            static_fields_are_current=(
                self.algebraic_fields_adapter is not None
            ),
        )


def _create_execution_context(
    solver: SpectralSolver,
    assembly: LegacyAssemblySpec,
    projector: BasisAwareSpectralProjector,
    performance_recorder: RuntimePerformanceRecorder | None,
) -> LegacyModelExecutionContext:
    boundaries = {
        component.component_name: boundary_set_to_legacy(
            component.boundaries
        )
        for component in assembly.source_plan.components
        if component.role is not FieldRole.DIAGNOSTIC
    }
    laplacians = {
        name: solver.get_laplacian_eigs(component_boundaries)
        for name, component_boundaries in boundaries.items()
    }
    return LegacyModelExecutionContext(
        physical_shape=assembly.backend.shape,
        spectral_shape=assembly.source_plan.spectral_shape,
        lengths=assembly.backend.lengths,
        axis_coordinates=tuple(solver.transform_backend.axes),
        real_dtype=solver.dtype,
        device=torch.device(solver.device),
        batch_size=solver.batchsize,
        _laplacians=laplacians,
        _boundary_conditions=boundaries,
        _projector=projector,
        _spectral_dtype=solver.transform_backend.spectral_dtype,
        _performance_recorder=performance_recorder,
    )


def _validate_algebraic_system_specs(
    model: ExecutableModelProtocol,
    plan: SpectralPlan,
    assembly: LegacyAssemblySpec,
) -> AlgebraicExecutionPlan | None:
    stored_algebraic_names = tuple(
        field.name for field in assembly.algebraic_fields
    )
    transient_names = tuple(
        component.component_name for component in plan.transient_components
    )
    algebraic_names = (*stored_algebraic_names, *transient_names)
    if not algebraic_names:
        if isinstance(model, AlgebraicExecutableModelProtocol):
            systems = tuple(model.algebraic_system_specs())
            if systems:
                raise ValueError(
                    "model declares algebraic systems without algebraic fields"
                )
        return None
    if transient_names and not stored_algebraic_names:
        raise ValueError(
            "the legacy execution adapter requires at least one stored "
            "algebraic output to drive transient pre-RHS evaluation"
        )
    if not isinstance(model, AlgebraicExecutableModelProtocol):
        raise TypeError(
            "models with algebraic fields must implement "
            "AlgebraicExecutableModelProtocol"
        )
    try:
        systems = tuple(model.algebraic_system_specs())
    except TypeError as exc:
        raise TypeError(
            "model.algebraic_system_specs() must return an iterable"
        ) from exc
    if not systems or not all(
        isinstance(system, AlgebraicSystemSpec) for system in systems
    ):
        raise TypeError(
            "algebraic_system_specs must contain AlgebraicSystemSpec objects"
        )
    evolved_names = tuple(
        field.name for field in assembly.evolved_fields
    )
    for system in systems:
        if system.update_phase is not AlgebraicUpdatePhase.PRE_EXPLICIT_RHS:
            raise ValueError("the current adapter supports pre-RHS updates only")
    return build_algebraic_execution_plan(
        systems,
        initial_components=evolved_names,
        output_components=algebraic_names,
        transient_components=transient_names,
    )


def _resolve_algebraic_systems(
    execution_plan: AlgebraicExecutionPlan | None,
    geometry: GeometrySpec,
    registry: GeometrySolverRegistry | None,
    context: LegacyAlgebraicSolverContext,
) -> tuple[ResolvedAlgebraicSystem, ...]:
    if execution_plan is None:
        return ()
    if not isinstance(registry, GeometrySolverRegistry):
        raise TypeError(
            "algebraic execution requires an explicit GeometrySolverRegistry"
        )
    resolved = []
    for system in execution_plan.ordered_systems:
        registration = registry.resolve(geometry, system)
        solver = registration.factory(context, system)
        resolved.append(
            ResolvedAlgebraicSystem(system, registration, solver)
        )
    return tuple(resolved)


def _require_explicit_output_shape(
    output: object,
    assembly: LegacyAssemblySpec,
    batch_size: int,
) -> None:
    if not isinstance(output, torch.Tensor):
        raise TypeError("explicit RHS adapter must return a tensor")
    expected = (
        len(assembly.evolved_fields),
        batch_size,
        *assembly.source_plan.spectral_shape,
    )
    if output.shape != expected:
        raise ValueError(
            f"explicit RHS adapter returned {tuple(output.shape)!r}; "
            f"expected {expected!r}"
        )


def build_experimental_model_runtime(
    model: ExecutableModelProtocol,
    geometry: GeometrySpec,
    numerics: NumericsConfig,
    *,
    dt: float,
    device: object = "cpu",
    batch_size: int = 1,
    allow_unstored_diagnostics: bool = False,
    geometry_solver_registry: GeometrySolverRegistry | None = None,
    enable_performance_instrumentation: bool = False,
    algebraic_execution_policy: AlgebraicExecutionPolicy | None = None,
    enable_algebraic_representation_reuse: bool | None = None,
    enable_lazy_algebraic_materialization: bool | None = None,
    enable_batched_physical_islands: bool | None = None,
) -> ExperimentalModelRuntime:
    """Build a canary through the frozen plan and legacy runtime adapter."""

    if not isinstance(model, ExecutableModelProtocol):
        raise TypeError("model must implement ExecutableModelProtocol")
    if not isinstance(enable_performance_instrumentation, bool):
        raise TypeError("enable_performance_instrumentation must be a bool")
    execution_policy = resolve_algebraic_execution_policy(
        algebraic_execution_policy,
        enable_algebraic_representation_reuse=(
            enable_algebraic_representation_reuse
        ),
        enable_lazy_algebraic_materialization=(
            enable_lazy_algebraic_materialization
        ),
        enable_batched_physical_islands=enable_batched_physical_islands,
    )
    if execution_policy.batched_physical_islands and not isinstance(
        model,
        ExplicitRHSPhysicalDependenciesProtocol,
    ):
        raise TypeError(
            "batched physical islands require explicit RHS dependency metadata"
        )
    problem = ProblemSpec(model, geometry, numerics)
    plan = assemble_spectral_plan(problem)
    assembly = materialize_legacy_assembly(
        plan,
        allow_unstored_diagnostics=allow_unstored_diagnostics,
    )
    algebraic_execution_plan = _validate_algebraic_system_specs(
        model,
        plan,
        assembly,
    )
    if (
        execution_policy.batched_physical_islands
        and algebraic_execution_plan is None
    ):
        raise ValueError("batched physical islands require algebraic systems")

    solver = create_legacy_solver(
        assembly,
        dt=dt,
        device=device,
        batchsize=batch_size,
    )
    performance_recorder = (
        RuntimePerformanceRecorder(torch.device(solver.device))
        if enable_performance_instrumentation
        else None
    )
    if performance_recorder is not None:
        instrument_transform_backend(
            solver.transform_backend,
            performance_recorder,
        )
    projector = create_legacy_projector(solver, assembly)
    solver.model.spectral_projector = projector
    if projector.enabled:
        solver.integrator_cl = (
            InstrumentedProjectedSemiImplicitEulerIntegrator
            if performance_recorder is not None
            else ProjectedSemiImplicitEulerIntegrator
        )
    elif performance_recorder is not None:
        raise ValueError(
            "performance instrumentation requires projected integration"
        )
    context = _create_execution_context(
        solver,
        assembly,
        projector,
        performance_recorder,
    )
    representation_cache = (
        AlgebraicRepresentationCache()
        if execution_policy.representation_reuse
        else None
    )
    algebraic_context = LegacyAlgebraicSolverContext(
        geometry_name=geometry.name,
        model_context=context,
        spectral_dtype=solver.transform_backend.spectral_dtype,
        execution_policy=execution_policy,
        representation_cache=representation_cache,
    )
    physical_island_scheduler = None
    if execution_policy.batched_physical_islands:
        if representation_cache is None:
            raise RuntimeError("batched policy lacks a representation cache")
        physical_island_scheduler = AlgebraicPhysicalIslandScheduler(
            algebraic_context.transform_scheduler,
            representation_cache,
        )
    resolved_algebraic_systems = _resolve_algebraic_systems(
        algebraic_execution_plan,
        geometry,
        geometry_solver_registry,
        algebraic_context,
    )
    initial_values = model.initial_values(context)
    linear_operators = model.linear_operators(context)
    declare_legacy_fields(
        solver,
        assembly,
        initial_values=initial_values,
        linear_operators=linear_operators,
    )
    algebraic_fields_adapter = None
    if resolved_algebraic_systems:
        solver.build()
        if projector.enabled:
            projector.project_dynamic_fields(
                solver.fields,
                sync_spatial=True,
            )
        algebraic_fields_adapter = LegacyAlgebraicFieldsAdapter(
            assembly,
            algebraic_execution_plan,
            resolved_algebraic_systems,
            algebraic_context,
            spectral_dtype=solver.transform_backend.spectral_dtype,
            device=torch.device(solver.device),
            batch_size=solver.batchsize,
            physical_island_scheduler=physical_island_scheduler,
            performance_recorder=performance_recorder,
        )
        solver.model.set_static_compute_model(algebraic_fields_adapter)
        solver.model.set_static_inverse_transform(
            projector.inverse_transform
        )
        solver.refresh_static_fields()
        explicit_rhs_adapter = LegacyExplicitRHSAdapter(
            model,
            assembly,
            projector,
            context,
            algebraic_fields_adapter,
            physical_island_scheduler,
            performance_recorder,
        )
        explicit_output = explicit_rhs_adapter(
            solver.fields,
            solver.parameters,
        )
        _require_explicit_output_shape(
            explicit_output,
            assembly,
            solver.batchsize,
        )
        solver.model.set_nonlinear_model(explicit_rhs_adapter)
        solver.integrator.restore_progress(
            0,
            static_fields_are_current=True,
        )
    else:
        explicit_rhs_adapter = LegacyExplicitRHSAdapter(
            model,
            assembly,
            projector,
            context,
            performance_recorder=performance_recorder,
        )
        solver.model.set_nonlinear_model(explicit_rhs_adapter)
        solver.build()
        if projector.enabled:
            projector.project_dynamic_fields(
                solver.fields,
                sync_spatial=True,
            )
    compare_spectral_plan_to_runtime(
        plan,
        solver.transform_backend,
        fields=solver.fields,
        projector=projector,
    ).require_match()
    if performance_recorder is not None:
        solver.integrator.performance_recorder = performance_recorder
    return ExperimentalModelRuntime(
        problem=problem,
        plan=plan,
        assembly=assembly,
        context=context,
        algebraic_execution_policy=execution_policy,
        solver=solver,
        projector=projector,
        explicit_rhs_adapter=explicit_rhs_adapter,
        algebraic_fields_adapter=algebraic_fields_adapter,
        algebraic_execution_plan=algebraic_execution_plan,
        resolved_algebraic_systems=resolved_algebraic_systems,
        performance_recorder=performance_recorder,
    )
