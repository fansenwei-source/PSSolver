"""Characterization tests for the Stage E executable-model canaries."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import torch

import pssolver
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
from pssolver.execution import ModelExecutionContext
from pssolver.geometries import PeriodicBox, PlaneSlab
from pssolver.models.canary import AllenCahnModel, ScalarDiffusionModel


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
            strict=False,
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
            rhs = self.model.linear_reaction * phi - self.model.cubic_reaction * phi.pow(3)
        return self.projector.forward_transform(
            rhs,
            self.boundaries,
        ).unsqueeze(0)


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

    plane_boundaries = BoundarySet((PeriodicBC(), HomogeneousNeumannBC()))
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


@dataclass(frozen=True)
class _AlgebraicCanary(ScalarDiffusionModel):
    def field_specs(self):
        return (
            *super().field_specs(),
            FieldSpec.scalar("response", FieldRole.ALGEBRAIC, self.boundaries),
        )


@dataclass(frozen=True)
class _MissingRHSCanary(ScalarDiffusionModel):
    def explicit_rhs(
        self,
        state: Mapping[str, torch.Tensor],
        context: ModelExecutionContext,
    ) -> Mapping[str, torch.Tensor]:
        del state, context
        return {}


def test_stage_e_api_remains_opt_in_and_production_drivers_are_unchanged():
    assert not hasattr(pssolver, "build_experimental_model_runtime")
    assert not hasattr(pssolver, "ExecutableModelProtocol")

    for path in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/plane.py",
    ):
        with open(path, encoding="utf-8") as source:
            assert "pssolver.experimental" not in source.read()
