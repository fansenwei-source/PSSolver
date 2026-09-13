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
from pssolver.execution import ExecutableModelProtocol, ModelExecutionContext
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
                f"unknown evolved component {component_name!r}"
            ) from exc


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
            {name: fields[name] for name in self._component_names}
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
    """Objects created by one explicit Stage E model assembly."""

    problem: ProblemSpec
    plan: SpectralPlan
    assembly: LegacyAssemblySpec
    context: ModelExecutionContext
    solver: SpectralSolver
    projector: BasisAwareSpectralProjector
    explicit_rhs_adapter: LegacyExplicitRHSAdapter

    def to_metadata(self) -> dict[str, object]:
        return {
            "problem": self.problem.to_metadata(),
            "spectral_plan": self.plan.to_metadata(),
            "legacy_assembly": self.assembly.to_metadata(),
            "execution_adapter": "legacy_explicit_rhs",
        }


def _create_execution_context(
    solver: SpectralSolver,
    assembly: LegacyAssemblySpec,
) -> LegacyModelExecutionContext:
    laplacians = {
        field.name: solver.get_laplacian_eigs(field.boundary_conditions)
        for field in assembly.evolved_fields
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


def build_experimental_model_runtime(
    model: ExecutableModelProtocol,
    geometry: GeometrySpec,
    numerics: NumericsConfig,
    *,
    dt: float,
    device: object = "cpu",
    batch_size: int = 1,
    allow_unstored_diagnostics: bool = False,
) -> ExperimentalModelRuntime:
    """Build a Stage E canary through the frozen plan and legacy adapter."""

    if not isinstance(model, ExecutableModelProtocol):
        raise TypeError("model must implement ExecutableModelProtocol")
    problem = ProblemSpec(model, geometry, numerics)
    plan = assemble_spectral_plan(problem)
    assembly = materialize_legacy_assembly(
        plan,
        allow_unstored_diagnostics=allow_unstored_diagnostics,
    )
    if assembly.algebraic_fields:
        raise NotImplementedError(
            "Stage E execution supports evolved components only; algebraic "
            "field execution remains in the geometry-specific runtime"
        )

    solver = create_legacy_solver(
        assembly,
        dt=dt,
        device=device,
        batchsize=batch_size,
    )
    projector = create_legacy_projector(solver, assembly)
    context = _create_execution_context(solver, assembly)
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
    )
