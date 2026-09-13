"""Opt-in adapter from executable models to the qualified legacy runtime.

This is a migration seam, not a production entry point.  Physical models see
only :class:`pssolver.execution.ModelExecutionContext`; every dependency on
``SpectralSolver``, ``Fields``, or the spectral projector remains here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from types import MappingProxyType

import torch

from pssolver.adapters import compare_spectral_plan_to_runtime
from pssolver.core import GeometrySpec, NumericsConfig, ProblemSpec
from pssolver.execution import (
    AlgebraicExecutableModelProtocol,
    AlgebraicSolverContext,
    AlgebraicSolverProtocol,
    AlgebraicSolverRegistration,
    AlgebraicSystemSpec,
    AlgebraicUpdatePhase,
    ExecutableModelProtocol,
    GeometrySolverRegistry,
    ModelExecutionContext,
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

    def laplacian_eigenvalues(self, component_name: str) -> torch.Tensor:
        if not isinstance(component_name, str):
            raise TypeError("component_name must be a string")
        try:
            return self._laplacians[component_name]
        except KeyError as exc:
            raise KeyError(
                f"unknown planned component {component_name!r}"
            ) from exc


@dataclass(frozen=True, slots=True)
class LegacyAlgebraicSolverContext:
    """Restricted spectral operations for dispatched algebraic solvers."""

    geometry_name: str
    model_context: LegacyModelExecutionContext
    spectral_dtype: torch.dtype
    _boundary_conditions: Mapping[str, tuple[str, ...]]
    _projector: BasisAwareSpectralProjector

    def __post_init__(self) -> None:
        if not isinstance(self.geometry_name, str) or not self.geometry_name:
            raise ValueError("geometry_name must be a non-empty string")
        if not isinstance(self.model_context, LegacyModelExecutionContext):
            raise TypeError(
                "model_context must be a LegacyModelExecutionContext"
            )
        if not isinstance(self.spectral_dtype, torch.dtype):
            raise TypeError("spectral_dtype must be a torch.dtype")
        if not isinstance(self._boundary_conditions, Mapping):
            raise TypeError("_boundary_conditions must be a mapping")
        boundaries = dict(self._boundary_conditions)
        expected = set(self.model_context._laplacians)
        if set(boundaries) != expected:
            raise ValueError(
                "algebraic solver boundaries must match planned components"
            )
        object.__setattr__(
            self,
            "_boundary_conditions",
            MappingProxyType(boundaries),
        )
        if not isinstance(self._projector, BasisAwareSpectralProjector):
            raise TypeError("_projector must be a BasisAwareSpectralProjector")

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

    def laplacian_eigenvalues(self, component_name: str) -> torch.Tensor:
        return self.model_context.laplacian_eigenvalues(component_name)

    def forward_projected(
        self,
        component_name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        try:
            boundary_conditions = self._boundary_conditions[component_name]
        except KeyError as exc:
            raise KeyError(
                f"unknown planned component {component_name!r}"
            ) from exc
        if not isinstance(value, torch.Tensor):
            raise TypeError("forward_projected value must be a tensor")
        expected_shape = (self.batch_size, *self.physical_shape)
        if value.shape != expected_shape:
            raise ValueError(
                f"physical value shape must be {expected_shape!r}, "
                f"got {tuple(value.shape)!r}"
            )
        if value.dtype != self.real_dtype or value.device != self.device:
            raise ValueError(
                "physical value dtype and device must match the solver context"
            )
        return self._projector.forward_transform(
            value,
            boundary_conditions,
        )

    def boundary_conditions(self, component_name: str) -> tuple[str, ...]:
        try:
            return self._boundary_conditions[component_name]
        except KeyError as exc:
            raise KeyError(
                f"unknown planned component {component_name!r}"
            ) from exc


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
        return {
            "system": self.system.to_metadata(),
            "dispatch": self.registration.to_metadata(),
        }


class LegacyAlgebraicFieldsAdapter(torch.nn.Module):
    """Evaluate resolved algebraic systems in native spectral storage."""

    def __init__(
        self,
        assembly: LegacyAssemblySpec,
        resolved_systems: tuple[ResolvedAlgebraicSystem, ...],
        *,
        spectral_dtype: torch.dtype,
        device: torch.device,
        batch_size: int,
    ) -> None:
        super().__init__()
        self._resolved_systems = tuple(resolved_systems)
        self._component_names = tuple(
            field.name for field in assembly.algebraic_fields
        )
        self._spectral_shape = assembly.source_plan.spectral_shape
        self._spectral_dtype = spectral_dtype
        self._device = device
        self._batch_size = batch_size

    def forward(self, fields, parameters):
        del parameters
        outputs: dict[str, torch.Tensor] = {}
        for resolved in self._resolved_systems:
            dependencies = MappingProxyType(
                {
                    name: fields[name]
                    for name in resolved.system.dependencies
                }
            )
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
        return torch.stack([outputs[name] for name in self._component_names])


class LegacyExplicitRHSAdapter(torch.nn.Module):
    """Translate a physical-space model RHS to native spectral storage."""

    def __init__(
        self,
        model: ExecutableModelProtocol,
        assembly: LegacyAssemblySpec,
        projector: BasisAwareSpectralProjector,
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

    def forward(self, fields, parameters):
        del parameters
        state = MappingProxyType(
            {name: fields[name] for name in self._state_names}
        )
        explicit = self._model.explicit_rhs(state)
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

        spectral = []
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
            spectral.append(
                self._projector.forward_transform(
                    value,
                    boundary_conditions,
                )
            )
        return torch.stack(spectral)


@dataclass(frozen=True, slots=True)
class ExperimentalModelRuntime:
    """Objects created by one explicit Stage E/F model assembly."""

    problem: ProblemSpec
    plan: SpectralPlan
    assembly: LegacyAssemblySpec
    context: ModelExecutionContext
    solver: SpectralSolver
    projector: BasisAwareSpectralProjector
    explicit_rhs_adapter: LegacyExplicitRHSAdapter
    algebraic_fields_adapter: LegacyAlgebraicFieldsAdapter | None
    resolved_algebraic_systems: tuple[ResolvedAlgebraicSystem, ...]

    def to_metadata(self) -> dict[str, object]:
        has_algebraic_fields = bool(self.resolved_algebraic_systems)
        return {
            "problem": self.problem.to_metadata(),
            "spectral_plan": self.plan.to_metadata(),
            "legacy_assembly": self.assembly.to_metadata(),
            "execution_adapter": "legacy_explicit_rhs",
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
            },
        }

    def synchronize_algebraic_for_observation(self) -> None:
        """Refresh algebraic fields against the current evolved state.

        This does not advance time or alter the spectral-refresh clock.  The
        next timestep may recompute the same algebraic state before its RHS.
        """

        if self.algebraic_fields_adapter is not None:
            self.solver.refresh_static_fields()

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
        self.solver.reset(initial_values)
        if self.algebraic_fields_adapter is not None:
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
) -> LegacyModelExecutionContext:
    laplacians = {
        field.name: solver.get_laplacian_eigs(field.boundary_conditions)
        for field in assembly.fields
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
    )


def _validate_algebraic_system_specs(
    model: ExecutableModelProtocol,
    assembly: LegacyAssemblySpec,
) -> tuple[AlgebraicSystemSpec, ...]:
    algebraic_names = tuple(
        field.name for field in assembly.algebraic_fields
    )
    if not algebraic_names:
        if isinstance(model, AlgebraicExecutableModelProtocol):
            systems = tuple(model.algebraic_system_specs())
            if systems:
                raise ValueError(
                    "model declares algebraic systems without algebraic fields"
                )
        return ()
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
    names = tuple(system.name for system in systems)
    if len(set(names)) != len(names):
        raise ValueError("algebraic system names must be unique")
    output_names = tuple(
        output
        for system in systems
        for output in system.output_components
    )
    if len(set(output_names)) != len(output_names):
        raise ValueError("each algebraic component must have one owner")
    if set(output_names) != set(algebraic_names):
        missing = tuple(sorted(set(algebraic_names) - set(output_names)))
        unexpected = tuple(sorted(set(output_names) - set(algebraic_names)))
        raise ValueError(
            "algebraic system outputs do not match algebraic fields; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )
    evolved_names = {
        field.name for field in assembly.evolved_fields
    }
    for system in systems:
        if system.update_phase is not AlgebraicUpdatePhase.PRE_EXPLICIT_RHS:
            raise ValueError("Stage F supports pre-RHS algebraic updates only")
        invalid_dependencies = tuple(
            sorted(set(system.dependencies) - evolved_names)
        )
        if invalid_dependencies:
            raise ValueError(
                "Stage F algebraic dependencies must be evolved components; "
                f"invalid={invalid_dependencies!r}"
            )
    return systems


def _resolve_algebraic_systems(
    systems: tuple[AlgebraicSystemSpec, ...],
    geometry: GeometrySpec,
    registry: GeometrySolverRegistry | None,
    context: LegacyAlgebraicSolverContext,
) -> tuple[ResolvedAlgebraicSystem, ...]:
    if not systems:
        return ()
    if not isinstance(registry, GeometrySolverRegistry):
        raise TypeError(
            "algebraic execution requires an explicit GeometrySolverRegistry"
        )
    resolved = []
    for system in systems:
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
) -> ExperimentalModelRuntime:
    """Build a Stage E/F canary through the frozen plan and legacy adapter."""

    if not isinstance(model, ExecutableModelProtocol):
        raise TypeError("model must implement ExecutableModelProtocol")
    problem = ProblemSpec(model, geometry, numerics)
    plan = assemble_spectral_plan(problem)
    assembly = materialize_legacy_assembly(
        plan,
        allow_unstored_diagnostics=allow_unstored_diagnostics,
    )
    algebraic_systems = _validate_algebraic_system_specs(model, assembly)

    solver = create_legacy_solver(
        assembly,
        dt=dt,
        device=device,
        batchsize=batch_size,
    )
    projector = create_legacy_projector(solver, assembly)
    context = _create_execution_context(solver, assembly)
    algebraic_context = LegacyAlgebraicSolverContext(
        geometry_name=geometry.name,
        model_context=context,
        spectral_dtype=solver.transform_backend.spectral_dtype,
        _boundary_conditions={
            field.name: field.boundary_conditions
            for field in assembly.fields
        },
        _projector=projector,
    )
    resolved_algebraic_systems = _resolve_algebraic_systems(
        algebraic_systems,
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
    explicit_rhs_adapter = LegacyExplicitRHSAdapter(
        model,
        assembly,
        projector,
    )
    algebraic_fields_adapter = None
    if resolved_algebraic_systems:
        solver.build()
        algebraic_fields_adapter = LegacyAlgebraicFieldsAdapter(
            assembly,
            resolved_algebraic_systems,
            spectral_dtype=solver.transform_backend.spectral_dtype,
            device=torch.device(solver.device),
            batch_size=solver.batchsize,
        )
        solver.model.set_static_compute_model(algebraic_fields_adapter)
        solver.model.set_static_inverse_transform(
            projector.inverse_transform
        )
        solver.refresh_static_fields()
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
        solver.model.set_nonlinear_model(explicit_rhs_adapter)
        solver.build()
    compare_spectral_plan_to_runtime(
        plan,
        solver.transform_backend,
        fields=solver.fields,
        projector=projector,
    ).require_match()
    return ExperimentalModelRuntime(
        problem=problem,
        plan=plan,
        assembly=assembly,
        context=context,
        solver=solver,
        projector=projector,
        explicit_rhs_adapter=explicit_rhs_adapter,
        algebraic_fields_adapter=algebraic_fields_adapter,
        resolved_algebraic_systems=resolved_algebraic_systems,
    )
