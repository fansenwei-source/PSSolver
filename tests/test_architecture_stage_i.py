"""Qualification tests for the Stage I Beris--Edwards constitutive DAG."""

from __future__ import annotations

import json
import math

import pytest

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
    IncompressibleStokesSystemSpec,
    TangentialZeroModePolicy,
)
from pssolver.models.active_nematics import (
    NEMATIC_FORCE_COMPONENTS,
    BerisEdwardsConstitutiveParameters,
    BerisEdwardsConstitutiveStokesCanaryModel,
)


P = PeriodicBC()


N = HomogeneousNeumannBC()


D = HomogeneousDirichletBC()


EVEN = BoundarySet((P, P, N))


ODD = BoundarySet((P, P, D))


EVEN_SIGNATURE = ("periodic", "periodic", "neumann")


ODD_SIGNATURE = ("periodic", "periodic", "dirichlet")


def _parameters(**overrides):
    values = {
        "ldg_a": 0.07,
        "ldg_b": -0.3,
        "ldg_c": 0.3,
        "ldg_l1": 0.04,
        "flow_alignment": 0.31,
        "active_prefactor": -0.18,
    }
    values.update(overrides)
    return BerisEdwardsConstitutiveParameters(**values)


def _model(*, parameters=None, friction=0.0, zero_mode="zero_mean"):
    if parameters is None:
        parameters = _parameters()
    policy = TangentialZeroModePolicy(zero_mode)
    return BerisEdwardsConstitutiveStokesCanaryModel(
        q_boundaries=EVEN,
        tangential_boundaries=EVEN,
        normal_boundaries=ODD,
        pressure_boundaries=EVEN,
        parameters=parameters,
        stokes_system=IncompressibleStokesSystemSpec(
            name="flow",
            force_components=NEMATIC_FORCE_COMPONENTS,
            velocity_components=("ux", "uy", "uz"),
            pressure_component="p",
            viscosity=2.0 / 3.0,
            friction=friction,
            tangential_zero_mode_policy=policy,
        ),
    )


def _numerics(*, half_spectrum=False):
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=(DealiasRule.CUBIC_HALF if half_spectrum else DealiasRule.NONE),
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=(
            ProjectedTransformExecution.TRUNCATED
            if half_spectrum
            else ProjectedTransformExecution.FULL
        ),
        spectral_storage=(
            SpectralStorage.HERMITIAN_HALF if half_spectrum else SpectralStorage.FULL_COMPLEX
        ),
        hermitian_axis=1 if half_spectrum else None,
    )


def _projected_physical(runtime, component, value):
    return runtime.context._inverse_projected(
        component,
        runtime.context._forward_projected(component, value),
    )


def test_constitutive_parameters_are_finite_and_metadata_is_explicit():
    parameters = _parameters()
    assert parameters.to_metadata() == {
        "active_prefactor": -0.18,
        "flow_alignment": 0.31,
        "ldg_a": 0.07,
        "ldg_b": -0.3,
        "ldg_c": 0.3,
        "ldg_l1": 0.04,
    }
    json.dumps(parameters.to_metadata(), allow_nan=False)
    with pytest.raises(ValueError, match="ldg_l1"):
        _parameters(ldg_l1=0.0)
    with pytest.raises(ValueError, match="active_prefactor"):
        _parameters(active_prefactor=math.nan)


def test_stage_i_is_opt_in_and_does_not_enter_production_drivers():
    assert not hasattr(pssolver, "BerisEdwardsConstitutiveStokesCanaryModel")
    for path in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/plane.py",
        "pssolver/models/active_nematics/stokes.py",
    ):
        with open(path, encoding="utf-8") as source:
            text = source.read()
        assert "BerisEdwardsConstitutiveStokesCanaryModel" not in text
        assert "create_beris_edwards_plane_geometry_solver_registry" not in text
