"""CPU qualification for Stage Q.1 execution-owned explicit RHS kernels."""

from __future__ import annotations

from pathlib import Path

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
    Q_GRADIENT_COMPONENTS,
    VELOCITY_COMPONENTS,
    VELOCITY_GRADIENT_COMPONENTS,
    BerisEdwardsConstitutiveParameters,
    BerisEdwardsConstitutiveStokesCanaryModel,
    BerisEdwardsPlaneCoupledModel,
)


P = PeriodicBC()


N = HomogeneousNeumannBC()


D = HomogeneousDirichletBC()


EVEN = BoundarySet((P, P, N))


ODD = BoundarySet((P, P, D))


PROJECT_ROOT = Path(__file__).parents[1]


def _model() -> BerisEdwardsPlaneCoupledModel:
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
            active_prefactor=-0.18,
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


def _numerics() -> NumericsConfig:
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=DealiasRule.CUBIC_HALF,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=ProjectedTransformExecution.TRUNCATED,
        spectral_storage=SpectralStorage.HERMITIAN_HALF,
        hermitian_axis=1,
    )


def _complete_state(runtime):
    state = {
        name: runtime.solver.fields[name]
        for name in (*Q_COMPONENTS, *VELOCITY_COMPONENTS, "p")
    }
    state.update(runtime.transient_algebraic_state())
    return state


def _random_complete_state():
    generator = torch.Generator().manual_seed(1701)
    return {
        name: torch.randn(
            (1, 4, 3, 2),
            dtype=torch.float64,
            generator=generator,
        )
        for name in (
            *Q_COMPONENTS,
            *VELOCITY_COMPONENTS,
            *Q_GRADIENT_COMPONENTS,
            *VELOCITY_GRADIENT_COMPONENTS,
        )
    }


def test_stage_q1_keeps_compile_policy_out_of_physical_models_and_production():
    constitutive = (
        PROJECT_ROOT
        / "pssolver/models/active_nematics/constitutive.py"
    ).read_text(encoding="utf-8")
    assert "torch.compile" not in constitutive
    assert not hasattr(pssolver, "BerisEdwardsPlaneExplicitRHSExecutor")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/plane.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "BerisEdwardsPlaneExplicitRHSExecutor" not in source
