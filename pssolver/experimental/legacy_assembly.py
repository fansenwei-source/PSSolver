"""Opt-in compatibility assembly from a frozen plan to the current runtime.

This module is the only Stage D bridge that constructs legacy runtime objects.
Production drivers do not import it.  It does not invent initial conditions,
linear operators, or executable physical models.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from numbers import Real

import torch

from pssolver.adapters import (
    boundary_set_to_legacy,
    compare_spectral_plan_to_runtime,
)
from pssolver.core.fields import FieldRole
from pssolver.core.numerics import (
    DealiasRule,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.planning import SpectralPlan
from pssolver.solver import SpectralSolver
from pssolver.transforms import BasisAwareSpectralProjector


@dataclass(frozen=True, slots=True)
class LegacyBackendSpec:
    """Tensor-free constructor inputs for the current transform backend."""

    shape: tuple[int, ...]
    lengths: tuple[float, ...]
    precision: Precision
    transform_execution_order: TransformExecutionOrder
    spectral_storage: SpectralStorage
    hermitian_axis: int | None

    def __post_init__(self) -> None:
        shape = tuple(self.shape)
        lengths = tuple(self.lengths)
        if not shape or any(
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            for size in shape
        ):
            raise ValueError("backend shape must contain positive integers")
        if len(lengths) != len(shape) or any(
            not isinstance(length, Real)
            or isinstance(length, bool)
            or not math.isfinite(float(length))
            or float(length) <= 0.0
            for length in lengths
        ):
            raise ValueError("backend lengths must be positive and match shape")
        if not isinstance(self.precision, Precision):
            raise TypeError("precision must be a Precision")
        if not isinstance(
            self.transform_execution_order,
            TransformExecutionOrder,
        ):
            raise TypeError(
                "transform_execution_order must be a TransformExecutionOrder"
            )
        if not isinstance(self.spectral_storage, SpectralStorage):
            raise TypeError("spectral_storage must be a SpectralStorage")
        if self.spectral_storage is SpectralStorage.HERMITIAN_HALF:
            if (
                self.transform_execution_order
                is not TransformExecutionOrder.REAL_FIRST
            ):
                raise ValueError(
                    "Hermitian backend requires real-first execution"
                )
            if (
                not isinstance(self.hermitian_axis, int)
                or isinstance(self.hermitian_axis, bool)
                or self.hermitian_axis < 0
                or self.hermitian_axis >= len(shape)
            ):
                raise ValueError("Hermitian backend axis is invalid")
        elif self.hermitian_axis is not None:
            raise ValueError("full-complex backend cannot have a Hermitian axis")
        object.__setattr__(self, "shape", shape)
        object.__setattr__(
            self,
            "lengths",
            tuple(float(length) for length in lengths),
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "shape": list(self.shape),
            "lengths": list(self.lengths),
            "precision": self.precision.value,
            "transform_execution_order": self.transform_execution_order.value,
            "spectral_storage": self.spectral_storage.value,
            "hermitian_axis": self.hermitian_axis,
        }


@dataclass(frozen=True, slots=True)
class LegacyProjectorSpec:
    """Constructor inputs for the current dealiasing projector."""

    dealias_rule: DealiasRule
    transform_execution: ProjectedTransformExecution

    def __post_init__(self) -> None:
        if not isinstance(self.dealias_rule, DealiasRule):
            raise TypeError("dealias_rule must be a DealiasRule")
        if not isinstance(
            self.transform_execution,
            ProjectedTransformExecution,
        ):
            raise TypeError(
                "transform_execution must be a ProjectedTransformExecution"
            )
        if (
            self.dealias_rule is DealiasRule.NONE
            and self.transform_execution
            is ProjectedTransformExecution.TRUNCATED
        ):
            raise ValueError("truncated projector execution requires dealiasing")

    def to_metadata(self) -> dict[str, str]:
        return {
            "dealias_rule": self.dealias_rule.value,
            "transform_execution": self.transform_execution.value,
        }


@dataclass(frozen=True, slots=True)
class LegacyFieldDeclaration:
    """One flattened field declaration accepted by the current PDEModel."""

    name: str
    role: FieldRole
    boundary_conditions: tuple[str, ...]
    storage_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("legacy field name must be a Python identifier")
        if self.role not in (FieldRole.EVOLVED, FieldRole.ALGEBRAIC):
            raise ValueError("legacy fields must be evolved or algebraic")
        boundary_conditions = tuple(self.boundary_conditions)
        if not boundary_conditions or any(
            condition not in ("periodic", "dirichlet", "neumann")
            for condition in boundary_conditions
        ):
            raise ValueError("legacy boundary-condition tuple is invalid")
        if (
            not isinstance(self.storage_index, int)
            or isinstance(self.storage_index, bool)
            or self.storage_index < 0
        ):
            raise ValueError("storage_index must be non-negative")
        object.__setattr__(
            self,
            "boundary_conditions",
            boundary_conditions,
        )

    @property
    def legacy_role(self) -> str:
        return "dynamic" if self.role is FieldRole.EVOLVED else "static"

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "role": self.role.value,
            "legacy_role": self.legacy_role,
            "boundary_conditions": list(self.boundary_conditions),
            "storage_index": self.storage_index,
        }


@dataclass(frozen=True, slots=True)
class LegacyAssemblySpec:
    """Complete opt-in description of one legacy runtime assembly."""

    source_plan: SpectralPlan
    backend: LegacyBackendSpec
    projector: LegacyProjectorSpec
    fields: tuple[LegacyFieldDeclaration, ...]
    omitted_diagnostic_components: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source_plan, SpectralPlan):
            raise TypeError("source_plan must be a SpectralPlan")
        if not isinstance(self.backend, LegacyBackendSpec):
            raise TypeError("backend must be a LegacyBackendSpec")
        if not isinstance(self.projector, LegacyProjectorSpec):
            raise TypeError("projector must be a LegacyProjectorSpec")
        fields = tuple(self.fields)
        if not fields or not all(
            isinstance(field, LegacyFieldDeclaration) for field in fields
        ):
            raise TypeError("fields must contain LegacyFieldDeclaration objects")
        if tuple(field.storage_index for field in fields) != tuple(
            range(len(fields))
        ):
            raise ValueError("legacy field storage indices must be ordered")
        expected_names = tuple(
            component.component_name
            for component in self.source_plan.stored_components
        )
        if tuple(field.name for field in fields) != expected_names:
            raise ValueError("legacy fields do not match source plan storage")
        if any(
            len(field.boundary_conditions) != len(self.backend.shape)
            for field in fields
        ):
            raise ValueError("legacy field dimensions do not match backend")
        for field, component in zip(
            fields,
            self.source_plan.stored_components,
        ):
            if field.role is not component.role:
                raise ValueError("legacy field roles do not match source plan")
            if field.storage_index != component.storage_index:
                raise ValueError(
                    "legacy field indices do not match source plan"
                )
            if field.boundary_conditions != boundary_set_to_legacy(
                component.boundaries
            ):
                raise ValueError(
                    "legacy field boundaries do not match source plan"
                )
        diagnostics = tuple(self.omitted_diagnostic_components)
        expected_diagnostics = tuple(
            component.component_name
            for component in self.source_plan.components
            if component.role is FieldRole.DIAGNOSTIC
        )
        if diagnostics != expected_diagnostics:
            raise ValueError("omitted diagnostics do not match source plan")
        numerics = self.source_plan.numerics
        expected_backend = (
            self.source_plan.physical_shape,
            self.source_plan.lengths,
            numerics.precision,
            numerics.transform_execution_order,
            numerics.spectral_storage,
            numerics.hermitian_axis,
        )
        observed_backend = (
            self.backend.shape,
            self.backend.lengths,
            self.backend.precision,
            self.backend.transform_execution_order,
            self.backend.spectral_storage,
            self.backend.hermitian_axis,
        )
        if observed_backend != expected_backend:
            raise ValueError("legacy backend does not match source plan")
        if (
            self.projector.dealias_rule,
            self.projector.transform_execution,
        ) != (
            numerics.dealias_rule,
            numerics.projected_transform_execution,
        ):
            raise ValueError("legacy projector does not match source plan")
        object.__setattr__(self, "fields", fields)
        object.__setattr__(
            self,
            "omitted_diagnostic_components",
            diagnostics,
        )

    @property
    def evolved_fields(self) -> tuple[LegacyFieldDeclaration, ...]:
        return tuple(
            field for field in self.fields if field.role is FieldRole.EVOLVED
        )

    @property
    def algebraic_fields(self) -> tuple[LegacyFieldDeclaration, ...]:
        return tuple(
            field for field in self.fields if field.role is FieldRole.ALGEBRAIC
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "source_plan_schema_version": self.source_plan.to_metadata()[
                "schema_version"
            ],
            "backend": self.backend.to_metadata(),
            "projector": self.projector.to_metadata(),
            "fields": [field.to_metadata() for field in self.fields],
            "omitted_diagnostic_components": list(
                self.omitted_diagnostic_components
            ),
        }


def materialize_legacy_assembly(
    plan: SpectralPlan,
    *,
    allow_unstored_diagnostics: bool = False,
) -> LegacyAssemblySpec:
    """Create legacy constructor declarations without creating a solver."""

    if not isinstance(plan, SpectralPlan):
        raise TypeError("plan must be a SpectralPlan")
    if not isinstance(allow_unstored_diagnostics, bool):
        raise TypeError("allow_unstored_diagnostics must be a bool")
    diagnostics = tuple(
        component.component_name
        for component in plan.components
        if component.role is FieldRole.DIAGNOSTIC
    )
    if diagnostics and not allow_unstored_diagnostics:
        raise ValueError(
            "the current runtime cannot store diagnostic components; "
            "set allow_unstored_diagnostics=True to record their omission"
        )
    if not plan.stored_components:
        raise ValueError("the current runtime requires stored components")
    if not any(
        component.role is FieldRole.EVOLVED
        for component in plan.stored_components
    ):
        raise ValueError("the current runtime requires an evolved component")

    backend = LegacyBackendSpec(
        shape=plan.physical_shape,
        lengths=plan.lengths,
        precision=plan.numerics.precision,
        transform_execution_order=(
            plan.numerics.transform_execution_order
        ),
        spectral_storage=plan.numerics.spectral_storage,
        hermitian_axis=plan.numerics.hermitian_axis,
    )
    projector = LegacyProjectorSpec(
        dealias_rule=plan.numerics.dealias_rule,
        transform_execution=plan.numerics.projected_transform_execution,
    )
    fields = tuple(
        LegacyFieldDeclaration(
            name=component.component_name,
            role=component.role,
            boundary_conditions=boundary_set_to_legacy(
                component.boundaries
            ),
            storage_index=component.storage_index,
        )
        for component in plan.stored_components
    )
    return LegacyAssemblySpec(
        source_plan=plan,
        backend=backend,
        projector=projector,
        fields=fields,
        omitted_diagnostic_components=diagnostics,
    )


def _validate_positive_real(value: object, description: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{description} must be positive and finite")
    return float(value)


def create_legacy_solver(
    assembly: LegacyAssemblySpec,
    *,
    dt: float,
    device: object = "cpu",
    batchsize: int = 1,
) -> SpectralSolver:
    """Create an empty current solver using an explicit assembly spec."""

    if not isinstance(assembly, LegacyAssemblySpec):
        raise TypeError("assembly must be a LegacyAssemblySpec")
    dt = _validate_positive_real(dt, "dt")
    if (
        not isinstance(batchsize, int)
        or isinstance(batchsize, bool)
        or batchsize <= 0
    ):
        raise ValueError("batchsize must be a positive integer")
    dtype = {
        Precision.FLOAT32: torch.float32,
        Precision.FLOAT64: torch.float64,
    }[assembly.backend.precision]
    solver = SpectralSolver(
        shape=assembly.backend.shape,
        L=assembly.backend.lengths,
        dt=dt,
        batchsize=batchsize,
        device=device,
        dtype=dtype,
        transform_execution_order=(
            assembly.backend.transform_execution_order.value
        ),
        spectral_storage=assembly.backend.spectral_storage.value,
        hermitian_axis=assembly.backend.hermitian_axis,
    )
    compare_spectral_plan_to_runtime(
        assembly.source_plan,
        solver.transform_backend,
    ).require_match()
    return solver


def create_legacy_projector(
    solver: SpectralSolver,
    assembly: LegacyAssemblySpec,
) -> BasisAwareSpectralProjector:
    """Create and shadow-check the current projector for an assembly."""

    if not isinstance(solver, SpectralSolver):
        raise TypeError("solver must be a SpectralSolver")
    if not isinstance(assembly, LegacyAssemblySpec):
        raise TypeError("assembly must be a LegacyAssemblySpec")
    projector = BasisAwareSpectralProjector(
        solver,
        rule=assembly.projector.dealias_rule.value,
        transform_execution=assembly.projector.transform_execution.value,
    )
    compare_spectral_plan_to_runtime(
        assembly.source_plan,
        solver.transform_backend,
        projector=projector,
    ).require_match()
    return projector


def _require_exact_mapping_keys(
    values: Mapping[str, object],
    expected: tuple[str, ...],
    description: str,
) -> None:
    if any(not isinstance(name, str) for name in values):
        raise TypeError(f"{description} keys must be strings")
    actual = set(values)
    expected_set = set(expected)
    if actual != expected_set:
        missing = tuple(sorted(expected_set - actual))
        unexpected = tuple(sorted(actual - expected_set))
        raise ValueError(
            f"{description} keys do not match evolved fields; "
            f"missing={missing!r}, unexpected={unexpected!r}"
        )


def _require_tensor_shape(
    value: object,
    allowed_shapes: tuple[tuple[int, ...], ...],
    description: str,
) -> None:
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"{description} must be a torch.Tensor")
    if tuple(value.shape) not in allowed_shapes:
        raise ValueError(
            f"{description} shape must be one of {allowed_shapes!r}, "
            f"got {tuple(value.shape)!r}"
        )


def declare_legacy_fields(
    solver: SpectralSolver,
    assembly: LegacyAssemblySpec,
    *,
    initial_values: Mapping[str, torch.Tensor],
    linear_operators: Mapping[str, torch.Tensor],
) -> None:
    """Declare all planned fields on a new, otherwise empty current solver."""

    if not isinstance(solver, SpectralSolver):
        raise TypeError("solver must be a SpectralSolver")
    if not isinstance(assembly, LegacyAssemblySpec):
        raise TypeError("assembly must be a LegacyAssemblySpec")
    if not isinstance(initial_values, Mapping):
        raise TypeError("initial_values must be a mapping")
    if not isinstance(linear_operators, Mapping):
        raise TypeError("linear_operators must be a mapping")
    initial_values = dict(initial_values)
    linear_operators = dict(linear_operators)
    if solver.model.dyn_fields or solver.model.stat_fields:
        raise RuntimeError("solver already contains field declarations")
    compare_spectral_plan_to_runtime(
        assembly.source_plan,
        solver.transform_backend,
    ).require_match()

    evolved_names = tuple(field.name for field in assembly.evolved_fields)
    _require_exact_mapping_keys(
        initial_values,
        evolved_names,
        "initial_values",
    )
    _require_exact_mapping_keys(
        linear_operators,
        evolved_names,
        "linear_operators",
    )
    physical_shapes = (
        assembly.backend.shape,
        (solver.batchsize, *assembly.backend.shape),
    )
    spectral_shapes = (
        assembly.source_plan.spectral_shape,
        (solver.batchsize, *assembly.source_plan.spectral_shape),
    )
    for field in assembly.evolved_fields:
        _require_tensor_shape(
            initial_values[field.name],
            physical_shapes,
            f"initial value for {field.name!r}",
        )
        _require_tensor_shape(
            linear_operators[field.name],
            spectral_shapes,
            f"linear operator for {field.name!r}",
        )

    for field in assembly.fields:
        if field.role is FieldRole.EVOLVED:
            solver.model.add_dynamic_field(
                field.name,
                initial_values[field.name],
                linear_operators[field.name],
                boundary_conditions=field.boundary_conditions,
            )
        else:
            solver.model.add_static_field(
                field.name,
                boundary_conditions=field.boundary_conditions,
            )
