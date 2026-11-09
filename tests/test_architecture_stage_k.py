"""Qualification tests for the Stage K Plane shadow-run boundary."""

from __future__ import annotations

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
from pssolver.execution import IncompressibleStokesSystemSpec
from pssolver.models.active_nematics import (
    NEMATIC_FORCE_COMPONENTS,
    Q_COMPONENTS,
    VELOCITY_COMPONENTS,
    BerisEdwardsConstitutiveParameters,
    BerisEdwardsConstitutiveStokesCanaryModel,
    BerisEdwardsPlaneCoupledModel,
)


P = PeriodicBC()


N = HomogeneousNeumannBC()


D = HomogeneousDirichletBC()


EVEN = BoundarySet((P, P, N))


ODD = BoundarySet((P, P, D))


SHAPE = (10, 8, 7)


LENGTHS = (5.0, 4.0, 3.5)


DT = 0.005


def _model(*, active_prefactor=-0.18):
    constitutive = BerisEdwardsConstitutiveStokesCanaryModel(
        q_boundaries=EVEN,
        tangential_boundaries=EVEN,
        normal_boundaries=ODD,
        pressure_boundaries=EVEN,
        parameters=BerisEdwardsConstitutiveParameters(
            ldg_a=0.0,
            ldg_b=-0.3,
            ldg_c=0.3,
            ldg_l1=0.04,
            flow_alignment=0.31,
            active_prefactor=active_prefactor,
        ),
        stokes_system=IncompressibleStokesSystemSpec(
            name="flow",
            force_components=NEMATIC_FORCE_COMPONENTS,
            velocity_components=VELOCITY_COMPONENTS,
            pressure_component="p",
            viscosity=2.0 / 3.0,
        ),
        initial_amplitude=0.02,
    )
    return BerisEdwardsPlaneCoupledModel(
        constitutive_model=constitutive,
        rotational_viscosity=2.94,
    )


def _numerics(*, half_spectrum):
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=DealiasRule.CUBIC_HALF,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=ProjectedTransformExecution.TRUNCATED,
        spectral_storage=(
            SpectralStorage.HERMITIAN_HALF if half_spectrum else SpectralStorage.FULL_COMPLEX
        ),
        hermitian_axis=1 if half_spectrum else None,
    )


def _initial_values():
    generator = torch.Generator().manual_seed(2468)
    return {
        name: 0.04
        * torch.randn(
            SHAPE,
            dtype=torch.float64,
            generator=generator,
        )
        for name in Q_COMPONENTS
    }


def _assert_runtime_states_close(left, right, *, exact):
    left.synchronize_algebraic_for_observation()
    right.synchronize_algebraic_for_observation()
    for name in (*Q_COMPONENTS, *VELOCITY_COMPONENTS, "p"):
        if exact:
            assert torch.equal(
                left.solver.fields[name],
                right.solver.fields[name],
            )
            assert torch.equal(
                left.solver.fields[f"{name}.hat"],
                right.solver.fields[f"{name}.hat"],
            )
        else:
            torch.testing.assert_close(
                left.solver.fields[name],
                right.solver.fields[name],
                rtol=2.0e-12,
                atol=2.0e-13,
            )


def test_stage_k_remains_opt_in_and_production_drivers_are_unchanged():
    assert not hasattr(pssolver, "ExperimentalPlaneShadowRun")
    for path in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/plane.py",
        "pssolver/solver.py",
    ):
        with open(path, encoding="utf-8") as source:
            text = source.read()
        assert "ExperimentalPlaneShadowRun" not in text
        assert "ShadowRunCheckpoint" not in text
