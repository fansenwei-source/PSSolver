"""Qualification tests for the Stage J coupled Beris--Edwards Plane model."""

from __future__ import annotations

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


EVEN_SIGNATURE = ("periodic", "periodic", "neumann")


ODD_SIGNATURE = ("periodic", "periodic", "dirichlet")


SHAPE = (10, 8, 7)


LENGTHS = (5.0, 4.0, 3.5)


DT = 0.005


def _parameters():
    return BerisEdwardsConstitutiveParameters(
        ldg_a=0.04,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=0.04,
        flow_alignment=0.31,
        active_prefactor=-0.18,
    )


def _coupled_model():
    constitutive = BerisEdwardsConstitutiveStokesCanaryModel(
        q_boundaries=EVEN,
        tangential_boundaries=EVEN,
        normal_boundaries=ODD,
        pressure_boundaries=EVEN,
        parameters=_parameters(),
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


def _state(runtime):
    values = {
        name: runtime.solver.fields[name] for name in (*Q_COMPONENTS, *VELOCITY_COMPONENTS, "p")
    }
    values.update(runtime.transient_algebraic_state())
    return values


def test_stage_j_remains_opt_in_and_does_not_enter_production_drivers():
    assert not hasattr(pssolver, "BerisEdwardsPlaneCoupledModel")
    assert not hasattr(pssolver, "ProjectedSemiImplicitEulerIntegrator")
    for path in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/plane.py",
        "pssolver/models/active_nematics/stokes.py",
    ):
        with open(path, encoding="utf-8") as source:
            text = source.read()
        assert "BerisEdwardsPlaneCoupledModel" not in text
        assert "ProjectedSemiImplicitEulerIntegrator" not in text
