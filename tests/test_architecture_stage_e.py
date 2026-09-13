"""Characterization tests for the Stage E executable-model canaries."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math

import pytest
import torch

import pssolver
from pssolver import SpectralSolver
from pssolver.core import (
    BoundarySet,
    DealiasRule,
    DomainSpec,
    FieldRole,
    FieldSpec,
    HomogeneousNeumannBC,
    NumericsConfig,
    PeriodicBC,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.execution import ExecutableModelProtocol, ModelExecutionContext
from pssolver.experimental import build_experimental_model_runtime
from pssolver.geometries import PeriodicBox, PlaneSlab
from pssolver.models.canary import AllenCahnModel, ScalarDiffusionModel
from pssolver.transforms import BasisAwareSpectralProjector


def _numerics(
    *,
    dealias_rule=DealiasRule.CUBIC_HALF,
    projected_execution=ProjectedTransformExecution.TRUNCATED,
    storage=SpectralStorage.FULL_COMPLEX,
    hermitian_axis=None,
):
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=dealias_rule,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=projected_execution,
        spectral_storage=storage,
        hermitian_axis=hermitian_axis,
    )


def _legacy_boundaries(boundaries):
    return tuple(condition.kind.value for condition in boundaries.axes)


def _manual_initial(solver, boundaries, modes, amplitude):
    value = torch.full(
        solver.shape,
        amplitude,
        dtype=solver.dtype,
        device=solver.device,
    )
    for axis, (coordinate, length, mode, condition) in enumerate(
        zip(
            solver.transform_backend.axes,
            solver.L,
            modes,
            boundaries.axes,
        )
    ):
        if condition.kind.value == "periodic":
            factor = torch.cos(2.0 * math.pi * mode * coordinate / length)
        elif condition.kind.value == "neumann":
            factor = torch.cos(math.pi * mode * coordinate / length)
        else:
            factor = torch.sin(math.pi * mode * coordinate / length)
        broadcast_shape = [1] * len(solver.shape)
        broadcast_shape[axis] = coordinate.numel()
        value = value * factor.reshape(broadcast_shape)
    return value


class _ManualScalarRHS(torch.nn.Module):
    def __init__(self, model, projector, boundaries):
        super().__init__()
        self.model = model
        self.projector = projector
        self.boundaries = boundaries

    def forward(self, fields, parameters):
        del parameters
        phi = fields["phi"]
        if isinstance(self.model, ScalarDiffusionModel):
            rhs = torch.zeros_like(phi)
        else:
            rhs = (
                self.model.linear_reaction * phi
                - self.model.cubic_reaction * phi.pow(3)
            )
        return self.projector.forward_transform(
            rhs,
            self.boundaries,
        ).unsqueeze(0)


def _manual_runtime(model, geometry, numerics, *, dt, batch_size):
    boundaries = _legacy_boundaries(model.boundaries)
    solver = SpectralSolver(
        geometry.domain.shape,
        L=geometry.domain.lengths,
        dt=dt,
        batchsize=batch_size,
        device="cpu",
        dtype=torch.float64,
        transform_execution_order=numerics.transform_execution_order.value,
        spectral_storage=numerics.spectral_storage.value,
        hermitian_axis=numerics.hermitian_axis,
    )
    projector = BasisAwareSpectralProjector(
        solver,
        rule=numerics.dealias_rule.value,
        transform_execution=numerics.projected_transform_execution.value,
    )
    initial = _manual_initial(
        solver,
        model.boundaries,
        model.initial_modes,
        model.initial_amplitude,
    )
    linear = model.diffusivity * solver.get_laplacian_eigs(boundaries)
    solver.model.add_dynamic_field(
        "phi",
        initial,
        linear,
        boundary_conditions=boundaries,
    )
    solver.model.set_nonlinear_model(
        _ManualScalarRHS(model, projector, boundaries)
    )
    solver.build()
    return solver


def _canary_cases():
    periodic_boundaries = BoundarySet((PeriodicBC(),))
    diffusion = ScalarDiffusionModel(
        periodic_boundaries,
        diffusivity=0.2,
        initial_amplitude=0.75,
        initial_modes=(2,),
    )
    yield (
        diffusion,
        PeriodicBox(DomainSpec((9,), (6.0,))),
        _numerics(
            dealias_rule=DealiasRule.NONE,
            projected_execution=ProjectedTransformExecution.FULL,
        ),
    )

    plane_boundaries = BoundarySet(
        (PeriodicBC(), HomogeneousNeumannBC())
    )
    allen_cahn = AllenCahnModel(
        plane_boundaries,
        diffusivity=0.15,
        linear_reaction=0.7,
        cubic_reaction=1.1,
        initial_amplitude=0.2,
        initial_modes=(1, 2),
    )
    yield (
        allen_cahn,
        PlaneSlab(DomainSpec((8, 7), (5.0, 4.0))),
        _numerics(
            storage=SpectralStorage.HERMITIAN_HALF,
            hermitian_axis=0,
        ),
    )


@pytest.mark.parametrize(
    "model,geometry,numerics",
    tuple(_canary_cases()),
    ids=("periodic_diffusion", "plane_allen_cahn"),
)
def test_canary_models_match_independent_manual_legacy_trajectories(
    model,
    geometry,
    numerics,
):
    manual = _manual_runtime(
        model,
        geometry,
        numerics,
        dt=0.01,
        batch_size=2,
    )
    experimental = build_experimental_model_runtime(
        model,
        geometry,
        numerics,
        dt=0.01,
        device="cpu",
        batch_size=2,
    )

    assert torch.equal(manual.fields.spatial, experimental.solver.fields.spatial)
    assert torch.equal(manual.fields.spectral, experimental.solver.fields.spectral)
    assert torch.equal(manual.fields.L_hat, experimental.solver.fields.L_hat)

    for _ in range(8):
        manual.integrator.step()
        experimental.solver.integrator.step()
        assert torch.equal(
            manual.fields.spatial,
            experimental.solver.fields.spatial,
        )
        assert torch.equal(
            manual.fields.spectral,
            experimental.solver.fields.spectral,
        )


def test_canaries_satisfy_the_execution_protocol_and_equations():
    boundaries = BoundarySet((PeriodicBC(), HomogeneousNeumannBC()))
    model = AllenCahnModel(
        boundaries,
        diffusivity=0.3,
        linear_reaction=0.8,
        cubic_reaction=1.2,
        initial_amplitude=0.1,
        initial_modes=(1, 1),
    )
    runtime = build_experimental_model_runtime(
        model,
        PlaneSlab(DomainSpec((6, 5), (3.0, 2.0))),
        _numerics(),
        dt=0.02,
        device="cpu",
    )

    assert isinstance(model, ExecutableModelProtocol)
    assert isinstance(runtime.context, ModelExecutionContext)
    expected_linear = 0.3 * runtime.solver.get_laplacian_eigs(
        ("periodic", "neumann")
    )
    assert torch.equal(runtime.solver.fields.L_hat[0, 0], expected_linear)

    phi = torch.linspace(
        -0.4,
        0.5,
        30,
        dtype=torch.float64,
    ).reshape(1, 6, 5)
    rhs = model.explicit_rhs({"phi": phi})["phi"]
    assert torch.equal(rhs, 0.8 * phi - 1.2 * phi.pow(3))
    assert json.loads(runtime.plan.model_parameters_json) == {
        "cubic_reaction": 1.2,
        "diffusivity": 0.3,
        "initial_amplitude": 0.1,
        "initial_modes": [1, 1],
        "linear_reaction": 0.8,
    }
    json.dumps(runtime.to_metadata(), allow_nan=False)


def test_model_context_does_not_expose_legacy_runtime_objects():
    boundaries = BoundarySet((PeriodicBC(),))
    runtime = build_experimental_model_runtime(
        ScalarDiffusionModel(boundaries, diffusivity=0.2),
        PeriodicBox(DomainSpec((8,), (4.0,))),
        _numerics(
            dealias_rule=DealiasRule.NONE,
            projected_execution=ProjectedTransformExecution.FULL,
        ),
        dt=0.01,
    )

    assert not hasattr(runtime.context, "solver")
    assert not hasattr(runtime.context, "fields")
    assert not hasattr(runtime.context, "projector")
    assert tuple(
        coordinate.shape for coordinate in runtime.context.axis_coordinates
    ) == ((8,),)
    with pytest.raises(TypeError):
        runtime.context._laplacians["extra"] = torch.zeros(8)
    with pytest.raises(KeyError, match="unknown planned"):
        runtime.context.laplacian_eigenvalues("missing")


@dataclass(frozen=True)
class _AlgebraicCanary(ScalarDiffusionModel):
    def field_specs(self):
        return (
            *super().field_specs(),
            FieldSpec.scalar("response", FieldRole.ALGEBRAIC, self.boundaries),
        )


def test_algebraic_execution_requires_the_extended_protocol():
    boundaries = BoundarySet((PeriodicBC(),))
    model = _AlgebraicCanary(boundaries, diffusivity=0.2)

    with pytest.raises(
        TypeError,
        match="AlgebraicExecutableModelProtocol",
    ):
        build_experimental_model_runtime(
            model,
            PeriodicBox(DomainSpec((8,), (4.0,))),
            _numerics(
                dealias_rule=DealiasRule.NONE,
                projected_execution=ProjectedTransformExecution.FULL,
            ),
            dt=0.01,
        )


@dataclass(frozen=True)
class _MissingRHSCanary(ScalarDiffusionModel):
    def explicit_rhs(
        self,
        state: Mapping[str, torch.Tensor],
    ) -> Mapping[str, torch.Tensor]:
        del state
        return {}


def test_invalid_explicit_rhs_is_rejected_during_build():
    boundaries = BoundarySet((PeriodicBC(),))
    model = _MissingRHSCanary(boundaries, diffusivity=0.2)

    with pytest.raises(RuntimeError, match="nonlinear model validation") as error:
        build_experimental_model_runtime(
            model,
            PeriodicBox(DomainSpec((8,), (4.0,))),
            _numerics(
                dealias_rule=DealiasRule.NONE,
                projected_execution=ProjectedTransformExecution.FULL,
            ),
            dt=0.01,
        )
    assert isinstance(error.value.__cause__, ValueError)
    assert "explicit_rhs keys" in str(error.value.__cause__)


def test_stage_e_api_remains_opt_in_and_production_drivers_are_unchanged():
    assert not hasattr(pssolver, "build_experimental_model_runtime")
    assert not hasattr(pssolver, "ExecutableModelProtocol")

    for path in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        with open(path, encoding="utf-8") as source:
            assert "pssolver.experimental" not in source.read()
