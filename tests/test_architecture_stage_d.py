"""Characterization tests for opt-in Stage D legacy assembly."""

from __future__ import annotations

from dataclasses import dataclass
import torch

import pssolver
from pssolver import SpectralSolver
from pssolver.core import DealiasRule, FieldRole, FieldSpec, NumericsConfig, Precision, ProblemSpec, ProjectedTransformExecution, SpectralStorage, TransformExecutionOrder


@dataclass(frozen=True)
class ScalarRuntimeModel:
    specs: tuple[FieldSpec, ...]
    name: str = "scalar_runtime_model"

    def field_specs(self):
        return self.specs

    def parameter_metadata(self):
        return {"diffusivity": 0.2}


def _numerics(
    *,
    storage=SpectralStorage.FULL_COMPLEX,
    hermitian_axis=None,
    dealias_rule=DealiasRule.CUBIC_HALF,
    execution=ProjectedTransformExecution.TRUNCATED,
):
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=dealias_rule,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=execution,
        spectral_storage=storage,
        hermitian_axis=hermitian_axis,
    )


def _scalar_problem(
    geometry,
    boundaries,
    numerics,
    *,
    include_algebraic=True,
    include_diagnostic=False,
):
    specs = [FieldSpec.scalar("phi", FieldRole.EVOLVED, boundaries)]
    if include_algebraic:
        specs.append(
            FieldSpec.scalar("response", FieldRole.ALGEBRAIC, boundaries)
        )
    if include_diagnostic:
        specs.append(
            FieldSpec.scalar("energy", FieldRole.DIAGNOSTIC, boundaries)
        )
    return ProblemSpec(
        ScalarRuntimeModel(tuple(specs)),
        geometry,
        numerics,
    )


def _initial_values(shape):
    size = 1
    for extent in shape:
        size *= extent
    return torch.linspace(-0.3, 0.7, size, dtype=torch.float64).reshape(shape)


def _manual_solver(problem, boundary_conditions, *, dt, initial):
    dtype = {
        Precision.FLOAT32: torch.float32,
        Precision.FLOAT64: torch.float64,
    }[problem.numerics.precision]
    solver = SpectralSolver(
        shape=problem.geometry.domain.shape,
        L=problem.geometry.domain.lengths,
        dt=dt,
        device="cpu",
        dtype=dtype,
        transform_execution_order=(
            problem.numerics.transform_execution_order.value
        ),
        spectral_storage=problem.numerics.spectral_storage.value,
        hermitian_axis=problem.numerics.hermitian_axis,
    )
    linear = -0.2 * solver.get_q2(boundary_conditions)
    solver.model.add_dynamic_field(
        "phi",
        initial.clone(),
        linear,
        boundary_conditions=boundary_conditions,
    )
    for field in problem.field_specs:
        if field.role is not FieldRole.ALGEBRAIC:
            continue
        component = field.components[0]
        solver.model.add_static_field(
            component.name,
            boundary_conditions=boundary_conditions,
        )
    solver.build()
    return solver


def test_legacy_assembly_is_explicit_and_not_exported_by_production_api():
    assert not hasattr(pssolver, "materialize_legacy_assembly")
    assert not hasattr(pssolver, "create_legacy_solver")
