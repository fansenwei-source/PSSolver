"""CPU qualification for Stage Q.1 execution-owned explicit RHS kernels."""

from __future__ import annotations

from pathlib import Path

import torch

import pssolver
from pssolver.core import (
    BoundarySet,
    DealiasRule,
    DomainSpec,
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
from pssolver.experimental import (
    BerisEdwardsPlaneExplicitRHSExecutor,
    PlaneBerisEdwardsSolverOptions,
    build_experimental_model_runtime,
    create_beris_edwards_plane_geometry_solver_registry,
)
from pssolver.geometries import PeriodicBox, PlaneSlab
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
from pssolver.models.canary import ScalarDiffusionModel


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


def _runtime():
    return build_experimental_model_runtime(
        _model(),
        PlaneSlab(DomainSpec((8, 6, 5), (5.0, 4.0, 3.5))),
        _numerics(),
        dt=0.005,
        batch_size=1,
        geometry_solver_registry=(
            create_beris_edwards_plane_geometry_solver_registry(
                constitutive_options=PlaneBerisEdwardsSolverOptions(
                    molecular_field_linear_space="spectral",
                    stress_divergence_sum_space="spectral",
                    pointwise_execution="eager",
                )
            )
        ),
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


def test_plane_registry_dispatches_exact_execution_owned_rhs():
    model = _model()
    geometry = PlaneSlab(DomainSpec((8, 6, 5), (5.0, 4.0, 3.5)))
    registry = create_beris_edwards_plane_geometry_solver_registry(
        constitutive_options=PlaneBerisEdwardsSolverOptions(
            pointwise_execution="eager"
        )
    )
    registration = registry.find_explicit_rhs(geometry, model)

    assert registration is not None
    assert registration.implementation_name == (
        "plane_beris_edwards_pointwise_explicit_rhs"
    )
    assert registry.explicit_rhs_metadata() == [registration.to_metadata()]
    executor = registration.factory(object(), model)
    assert isinstance(executor, BerisEdwardsPlaneExplicitRHSExecutor)


def test_execution_owned_eager_rhs_exactly_matches_physical_model():
    runtime = _runtime()
    runtime.synchronize_algebraic_for_observation()
    state = _complete_state(runtime)
    executor = runtime.explicit_rhs_executor

    assert isinstance(executor, BerisEdwardsPlaneExplicitRHSExecutor)
    expected = runtime.problem.model.explicit_rhs(state, runtime.context)
    observed = executor.evaluate(state, runtime.context)
    assert tuple(observed) == Q_COMPONENTS
    for name in Q_COMPONENTS:
        assert torch.equal(observed[name], expected[name])

    metadata = runtime.to_metadata()["explicit_rhs_execution"]
    assert metadata["owner"] == "geometry_executor"
    assert metadata["implementation_name"] == (
        "plane_beris_edwards_pointwise_explicit_rhs"
    )
    assert metadata["observability"]["fallback_to_model"] is False
    assert metadata["observability"]["pointwise_kernels"]["requested"] == (
        "eager"
    )


def test_compile_policy_wraps_the_shared_physical_rhs_kernel(monkeypatch):
    compiled = []
    executed = []

    def fake_compile(function, **kwargs):
        compiled.append((function.__name__, dict(kwargs)))

        def wrapper(*args, **call_kwargs):
            executed.append(function.__name__)
            return function(*args, **call_kwargs)

        return wrapper

    monkeypatch.setattr(torch, "compile", fake_compile)
    model = _model()
    geometry = PlaneSlab(DomainSpec((8, 6, 5), (5.0, 4.0, 3.5)))
    registry = create_beris_edwards_plane_geometry_solver_registry(
        constitutive_options=PlaneBerisEdwardsSolverOptions(
            pointwise_execution="compile"
        )
    )
    registration = registry.find_explicit_rhs(geometry, model)
    executor = registration.factory(object(), model)
    state = _random_complete_state()

    expected = model.explicit_rhs(state, object())
    observed = executor.evaluate(state, object())
    for name in Q_COMPONENTS:
        assert torch.equal(observed[name], expected[name])

    assert "beris_edwards_q_nonlinear_components" in executed
    compile_options = dict(compiled)[
        "beris_edwards_q_nonlinear_components"
    ]
    assert compile_options["dynamic"] is False
    assert compile_options["fullgraph"] is True


def test_generic_models_retain_the_physical_model_fallback():
    boundaries = BoundarySet((P,))
    runtime = build_experimental_model_runtime(
        ScalarDiffusionModel(
            boundaries=boundaries,
            diffusivity=0.2,
            initial_amplitude=0.1,
            initial_modes=(1,),
        ),
        PeriodicBox(DomainSpec((8,), (4.0,))),
        NumericsConfig(
            precision=Precision.FLOAT64,
            dealias_rule=DealiasRule.NONE,
            transform_execution_order=TransformExecutionOrder.LEGACY,
            projected_transform_execution=ProjectedTransformExecution.FULL,
            spectral_storage=SpectralStorage.FULL_COMPLEX,
        ),
        dt=0.01,
    )

    assert runtime.explicit_rhs_executor is None
    metadata = runtime.to_metadata()["explicit_rhs_execution"]
    assert metadata == {
        "owner": "physical_model_fallback",
        "implementation_name": "model_explicit_rhs",
        "registration": None,
        "observability": {"fallback_to_model": True},
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
        "pssolver/channel.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "BerisEdwardsPlaneExplicitRHSExecutor" not in source
