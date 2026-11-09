"""Characterization tests for Stage F algebraic lifecycle and dispatch."""

from __future__ import annotations

import json
import math
from dataclasses import FrozenInstanceError, dataclass

import pytest
import torch

import pssolver
from pssolver.core import (
    BoundarySet,
    DealiasRule,
    DomainSpec,
    HomogeneousNeumannBC,
    NumericsConfig,
    PeriodicBC,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.execution import (
    AlgebraicSystemSpec,
    GeometrySolverRegistry,
)
from pssolver.geometries import PeriodicBox, PlaneSlab
from pssolver.models.canary import (
    DiffusionHelmholtzCouplingModel,
)


def _numerics(
    *,
    storage=SpectralStorage.FULL_COMPLEX,
    hermitian_axis=None,
    dealias_rule=DealiasRule.CUBIC_HALF,
    projected_execution=ProjectedTransformExecution.TRUNCATED,
):
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=dealias_rule,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=projected_execution,
        spectral_storage=storage,
        hermitian_axis=hermitian_axis,
    )


def _coupled_cases():
    periodic_boundaries = BoundarySet((PeriodicBC(),))
    yield (
        DiffusionHelmholtzCouplingModel(
            periodic_boundaries,
            diffusivity=0.12,
            coupling=0.35,
            helmholtz_shift=1.1,
            helmholtz_length_sq=0.25,
            initial_amplitude=0.3,
            initial_modes=(2,),
        ),
        PeriodicBox(DomainSpec((9,), (5.0,))),
        _numerics(
            dealias_rule=DealiasRule.NONE,
            projected_execution=ProjectedTransformExecution.FULL,
        ),
    )

    plane_boundaries = BoundarySet((PeriodicBC(), HomogeneousNeumannBC()))
    yield (
        DiffusionHelmholtzCouplingModel(
            plane_boundaries,
            diffusivity=0.08,
            coupling=-0.2,
            helmholtz_shift=0.9,
            helmholtz_length_sq=0.4,
            initial_amplitude=0.25,
            initial_modes=(1, 2),
        ),
        PlaneSlab(DomainSpec((8, 7), (4.0, 3.5))),
        _numerics(
            storage=SpectralStorage.HERMITIAN_HALF,
            hermitian_axis=0,
        ),
    )


def _legacy_boundaries(model):
    return tuple(condition.kind.value for condition in model.boundaries.axes)


def _manual_initial(solver, model):
    value = torch.full(
        solver.shape,
        model.initial_amplitude,
        dtype=solver.dtype,
        device=solver.device,
    )
    for axis, (coordinate, length, mode, condition) in enumerate(
        zip(
            solver.transform_backend.axes,
            solver.L,
            model.initial_modes,
            model.boundaries.axes,
            strict=False,
        )
    ):
        if condition.kind.value == "periodic":
            factor = torch.cos(2.0 * math.pi * mode * coordinate / length)
        elif condition.kind.value == "neumann":
            factor = torch.cos(math.pi * mode * coordinate / length)
        else:
            factor = torch.sin(math.pi * mode * coordinate / length)
        shape = [1] * len(solver.shape)
        shape[axis] = coordinate.numel()
        value = value * factor.reshape(shape)
    return value


class _ManualHelmholtzStatic(torch.nn.Module):
    def __init__(self, model, solver, projector, boundaries):
        super().__init__()
        self.projector = projector
        self.boundaries = boundaries
        laplacian = solver.get_laplacian_eigs(boundaries)
        self.denominator = model.helmholtz_shift - model.helmholtz_length_sq * laplacian

    def forward(self, fields, parameters):
        del parameters
        source_hat = self.projector.forward_transform(
            fields["phi"],
            self.boundaries,
        )
        return (source_hat / self.denominator).unsqueeze(0)


class _ManualCoupledRHS(torch.nn.Module):
    def __init__(self, model, projector, boundaries):
        super().__init__()
        self.model = model
        self.projector = projector
        self.boundaries = boundaries

    def forward(self, fields, parameters):
        del parameters
        rhs = self.model.coupling * fields["response"]
        return self.projector.forward_transform(
            rhs,
            self.boundaries,
        ).unsqueeze(0)


def test_registry_rejects_duplicate_geometry_capability_registration():
    registry = GeometrySolverRegistry()

    def factory(context, system):
        del context, system
        raise AssertionError("factory should not run")

    registry.register(
        geometry_type=PeriodicBox,
        geometry_name="periodic_box",
        capability="scalar_helmholtz",
        implementation_name="first",
        factory=factory,
    )
    with pytest.raises(ValueError, match="duplicate"):
        registry.register(
            geometry_type=PeriodicBox,
            geometry_name="periodic_box",
            capability="scalar_helmholtz",
            implementation_name="second",
            factory=factory,
        )


def test_algebraic_system_spec_is_immutable_and_finite():
    system = AlgebraicSystemSpec(
        name="response_system",
        capability="scalar_helmholtz",
        output_components=("response",),
        dependencies=("phi",),
        parameters={"shift": 1.0},
    )
    with pytest.raises(FrozenInstanceError):
        system.name = "changed"
    with pytest.raises(TypeError):
        system.parameters["shift"] = 2.0
    json.dumps(system.to_metadata(), allow_nan=False)

    with pytest.raises(ValueError, match="finite"):
        AlgebraicSystemSpec(
            name="bad",
            capability="scalar_helmholtz",
            output_components=("response",),
            dependencies=("phi",),
            parameters={"shift": float("nan")},
        )


@dataclass(frozen=True)
class _BadDependencyModel(DiffusionHelmholtzCouplingModel):
    def algebraic_system_specs(self):
        system = super().algebraic_system_specs()[0]
        return (
            AlgebraicSystemSpec(
                name=system.name,
                capability=system.capability,
                output_components=system.output_components,
                dependencies=("missing",),
                parameters=system.parameters,
            ),
        )


def test_stage_f_remains_outside_production_paths():
    assert not hasattr(pssolver, "GeometrySolverRegistry")
    assert not hasattr(pssolver, "AlgebraicSystemSpec")
    assert not hasattr(pssolver, "create_canary_geometry_solver_registry")

    for path in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/plane.py",
        "pssolver/models/active_nematics/stokes.py",
    ):
        with open(path, encoding="utf-8") as source:
            text = source.read()
        assert "GeometrySolverRegistry" not in text
        assert "AlgebraicSystemSpec" not in text
        assert "pssolver.experimental" not in text
