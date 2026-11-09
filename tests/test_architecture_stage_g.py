"""Characterization tests for Stage G geometry-specific Stokes execution."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest
import torch

import pssolver
from pssolver.core import (
    BoundarySet,
    DealiasRule,
    HomogeneousDirichletBC,
    HomogeneousNeumannBC,
    NumericsConfig,
    PeriodicBC,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.execution import (
    INCOMPRESSIBLE_STOKES_CAPABILITY,
    IncompressibleStokesSystemSpec,
    PressureGauge,
    TangentialZeroModePolicy,
)
from pssolver.models.canary import BodyForceStokesCanaryModel


P = PeriodicBC()


N = HomogeneousNeumannBC()


D = HomogeneousDirichletBC()


PLANE_TANGENTIAL = BoundarySet((P, P, N))


PLANE_NORMAL = BoundarySet((P, P, D))


PLANE_PRESSURE = BoundarySet((P, P, N))


CHANNEL_VELOCITY = BoundarySet((P, D, D))


CHANNEL_PRESSURE = BoundarySet((P, N, N))


def _numerics():
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=DealiasRule.NONE,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=ProjectedTransformExecution.FULL,
        spectral_storage=SpectralStorage.FULL_COMPLEX,
    )


def _stokes_spec(*, channel=False, friction=0.0):
    policy = (
        TangentialZeroModePolicy.NOT_APPLICABLE
        if channel
        else (
            TangentialZeroModePolicy.FRICTION
            if friction > 0.0
            else TangentialZeroModePolicy.ZERO_MEAN
        )
    )
    return IncompressibleStokesSystemSpec(
        name="flow",
        force_components=("force_x", "force_y", "force_z"),
        velocity_components=("ux", "uy", "uz"),
        pressure_component="p",
        viscosity=0.73,
        friction=friction,
        pressure_gauge=PressureGauge.ZERO_MEAN,
        tangential_zero_mode_policy=policy,
    )


def _plane_model(*, friction=0.0, amplitudes=(0.7, -0.35, 0.2)):
    return BodyForceStokesCanaryModel(
        force_boundaries=(
            PLANE_TANGENTIAL,
            PLANE_TANGENTIAL,
            PLANE_NORMAL,
        ),
        velocity_boundaries=(
            PLANE_TANGENTIAL,
            PLANE_TANGENTIAL,
            PLANE_NORMAL,
        ),
        pressure_boundaries=PLANE_PRESSURE,
        stokes_system=_stokes_spec(friction=friction),
        force_diffusivity=0.025,
        initial_amplitudes=amplitudes,
        initial_modes=((1, 1, 1), (2, 1, 2), (1, 2, 1)),
    )


def _channel_model():
    return BodyForceStokesCanaryModel(
        force_boundaries=(CHANNEL_VELOCITY,) * 3,
        velocity_boundaries=(CHANNEL_VELOCITY,) * 3,
        pressure_boundaries=CHANNEL_PRESSURE,
        stokes_system=_stokes_spec(channel=True),
        force_diffusivity=0.018,
        initial_amplitudes=(0.55, -0.25, 0.3),
        initial_modes=((1, 1, 1), (2, 2, 1), (1, 1, 2)),
    )


def _direct_solution(runtime, lower):
    system = runtime.resolved_algebraic_systems[0].system
    force_hats = tuple(
        runtime.projector.forward_transform(
            runtime.solver.fields[name],
            runtime.solver.fields[f"{name}.bc"],
        )
        for name in system.dependencies
    )
    return lower.solve_force_hats(*force_hats)


def _assert_solution_is_stored(runtime, solution):
    outputs = runtime.resolved_algebraic_systems[0].system.output_components
    for name, expected_hat in zip(outputs, solution, strict=True):
        torch.testing.assert_close(
            runtime.solver.fields[f"{name}.hat"],
            expected_hat,
            rtol=0.0,
            atol=0.0,
        )
        expected = runtime.projector.inverse_transform(
            expected_hat,
            runtime.solver.fields[f"{name}.bc"],
        )
        torch.testing.assert_close(
            runtime.solver.fields[name],
            expected,
            rtol=0.0,
            atol=0.0,
        )


def test_stokes_contract_is_typed_immutable_and_round_trips():
    value = _stokes_spec()
    generic = value.to_algebraic_system_spec()
    recovered = IncompressibleStokesSystemSpec.from_algebraic_system_spec(generic)

    assert generic.capability == INCOMPRESSIBLE_STOKES_CAPABILITY
    assert recovered == value
    with pytest.raises(FrozenInstanceError):
        value.viscosity = 2.0
    with pytest.raises(ValueError, match="friction == 0"):
        IncompressibleStokesSystemSpec(
            name="bad",
            force_components=("fx", "fy", "fz"),
            velocity_components=("ux", "uy", "uz"),
            pressure_component="p",
            viscosity=1.0,
            friction=0.1,
            tangential_zero_mode_policy=(TangentialZeroModePolicy.ZERO_MEAN),
        )


def test_stage_g_stays_opt_in_and_benchmark_driver_is_unchanged():
    assert not hasattr(pssolver, "IncompressibleStokesSystemSpec")
    assert not hasattr(pssolver, "create_stokes_geometry_solver_registry")

    with open("Plane_beris_edwards_stokes.py", encoding="utf-8") as source:
        text = source.read()
    assert "pssolver.experimental" not in text
    assert "IncompressibleStokesSystemSpec" not in text
    assert "create_stokes_geometry_solver_registry" not in text
