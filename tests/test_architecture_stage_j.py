"""Qualification tests for the Stage J coupled Beris--Edwards Plane model."""

from __future__ import annotations

import json

import pytest
import torch

import pssolver
from pssolver import BasisAwareSpectralProjector, SpectralSolver
from pssolver.core import (
    BoundarySet,
    DealiasRule,
    DomainSpec,
    FieldRole,
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
    PlaneBerisEdwardsSolverOptions,
    ProjectedSemiImplicitEulerIntegrator,
    build_experimental_model_runtime,
    create_beris_edwards_plane_geometry_solver_registry,
)
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics import (
    NEMATIC_FORCE_COMPONENTS,
    Q_COMPONENTS,
    Q_GRADIENT_COMPONENTS,
    VELOCITY_COMPONENTS,
    VELOCITY_GRADIENT_COMPONENTS,
    BerisEdwardsConstitutiveParameters,
    BerisEdwardsConstitutiveStokesCanaryModel,
    BerisEdwardsFreeSlipStokes,
    BerisEdwardsPlaneCoupledModel,
    BerisEdwardsPointwiseKernels,
    BerisEdwardsQGradientCache,
    BerisEdwardsQNonlinearModel,
    beris_edwards_linear_operator,
    beris_edwards_q_nonlinear_components,
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
            SpectralStorage.HERMITIAN_HALF
            if half_spectrum
            else SpectralStorage.FULL_COMPLEX
        ),
        hermitian_axis=1 if half_spectrum else None,
    )


def _build(*, half_spectrum=False):
    return build_experimental_model_runtime(
        _coupled_model(),
        PlaneSlab(DomainSpec(SHAPE, LENGTHS)),
        _numerics(half_spectrum=half_spectrum),
        dt=DT,
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


def _manual_production_runtime(stage_j_runtime, *, half_spectrum):
    """Build the fixed production adapters without the new model/DAG layer."""

    model = _coupled_model()
    parameters = model.constitutive_model.parameters
    solver = SpectralSolver(
        SHAPE,
        L=LENGTHS,
        dt=DT,
        batchsize=1,
        device="cpu",
        dtype=torch.float64,
        transform_execution_order="real_first",
        spectral_storage=(
            "hermitian_half" if half_spectrum else "full_complex"
        ),
        hermitian_axis=1 if half_spectrum else None,
    )
    projector = BasisAwareSpectralProjector(
        solver,
        rule="cubic_half",
        transform_execution="truncated",
    )
    solver.model.spectral_projector = projector
    solver.model.set_static_inverse_transform(projector.inverse_transform)
    solver.integrator_cl = ProjectedSemiImplicitEulerIntegrator
    q_linear = beris_edwards_linear_operator(
        solver.get_q2(EVEN_SIGNATURE),
        ldg_a=parameters.ldg_a,
        ldg_l1=parameters.ldg_l1,
        rotational_viscosity=model.rotational_viscosity,
    )
    for name in Q_COMPONENTS:
        solver.model.add_dynamic_field(
            name,
            stage_j_runtime.solver.fields[name].detach().clone(),
            q_linear,
            boundary_conditions=EVEN_SIGNATURE,
        )
    solver.model.add_static_field("ux", boundary_conditions=EVEN_SIGNATURE)
    solver.model.add_static_field("uy", boundary_conditions=EVEN_SIGNATURE)
    solver.model.add_static_field("uz", boundary_conditions=ODD_SIGNATURE)
    solver.model.add_static_field("p", boundary_conditions=EVEN_SIGNATURE)

    kernels = BerisEdwardsPointwiseKernels("eager")
    gradient_cache = BerisEdwardsQGradientCache()
    solver.model.set_nonlinear_model(
        BerisEdwardsQNonlinearModel(
            projector,
            EVEN_SIGNATURE,
            ldg_b=parameters.ldg_b,
            ldg_c=parameters.ldg_c,
            rotational_viscosity=model.rotational_viscosity,
            flow_alignment=parameters.flow_alignment,
            q_gradient_cache=gradient_cache,
            pointwise_kernels=kernels,
        )
    )
    solver.model.set_static_compute_model(
        BerisEdwardsFreeSlipStokes(
            solver,
            projector,
            beta_value=-1.0,
            friction=0.0,
            viscosity=2.0 / 3.0,
            ldg_a=parameters.ldg_a,
            ldg_b=parameters.ldg_b,
            ldg_c=parameters.ldg_c,
            ldg_l1=parameters.ldg_l1,
            flow_alignment=parameters.flow_alignment,
            molecular_field_linear_space="spectral",
            stress_divergence_sum_space="spectral",
            q_gradient_cache=gradient_cache,
            pointwise_kernels=kernels,
            zero_mode_policy="zero_mean",
        )
    )
    solver.model.parameters.new_param(
        "alpha",
        torch.tensor(
            -parameters.active_prefactor,
            dtype=torch.float64,
        ),
    )
    solver.build()
    projector.project_dynamic_fields(solver.fields, sync_spatial=True)
    solver.refresh_static_fields()
    solver.integrator.restore_progress(0, static_fields_are_current=True)
    return solver


def _state(runtime):
    values = {
        name: runtime.solver.fields[name]
        for name in (*Q_COMPONENTS, *VELOCITY_COMPONENTS, "p")
    }
    values.update(runtime.transient_algebraic_state())
    return values


def test_complete_model_declares_velocity_gradient_after_stokes():
    model = _coupled_model()
    by_name = {field.name: field for field in model.field_specs()}
    assert by_name["velocity_gradient"].role is FieldRole.TRANSIENT
    assert (
        by_name["velocity_gradient"].component_names
        == VELOCITY_GRADIENT_COMPONENTS
    )
    runtime = _build()
    assert runtime.algebraic_execution_plan.execution_order == (
        "molecular_field",
        "q_gradient",
        "nematic_stress",
        "nematic_force",
        "flow",
        "velocity_gradient",
    )
    assert len(runtime.transient_algebraic_state()) == 50
    assert runtime.solver.fields.spatial.shape[0] == 9


def test_q_linear_operator_is_the_production_a_l1_imex_symbol():
    runtime = _build()
    model = _coupled_model()
    parameters = model.constitutive_model.parameters
    expected = beris_edwards_linear_operator(
        runtime.solver.get_q2(EVEN_SIGNATURE),
        ldg_a=parameters.ldg_a,
        ldg_l1=parameters.ldg_l1,
        rotational_viscosity=model.rotational_viscosity,
    )
    for index, name in enumerate(Q_COMPONENTS):
        assert runtime.solver.fields.name_to_idx[name] == index
        torch.testing.assert_close(
            runtime.solver.fields.L_hat[index, 0],
            expected,
            rtol=0.0,
            atol=0.0,
        )


@pytest.mark.parametrize("half_spectrum", (False, True))
def test_velocity_gradients_and_complete_explicit_rhs_match_equations(
    half_spectrum,
):
    runtime = _build(half_spectrum=half_spectrum)
    transient = runtime.transient_algebraic_state()
    velocity_gradients = []
    for axis in range(3):
        axis_values = []
        for index, velocity_name in enumerate(VELOCITY_COMPONENTS):
            gradient_name = VELOCITY_GRADIENT_COMPONENTS[
                axis * len(VELOCITY_COMPONENTS) + index
            ]
            expected = runtime.context.gradient(
                velocity_name,
                gradient_name,
                runtime.solver.fields[velocity_name],
                axis,
            )
            torch.testing.assert_close(
                transient[gradient_name],
                expected,
                rtol=3.0e-12,
                atol=3.0e-12,
            )
            axis_values.append(transient[gradient_name])
        velocity_gradients.append(tuple(axis_values))

    q_gradients_flat = tuple(
        transient[name] for name in Q_GRADIENT_COMPONENTS
    )
    q_gradients = tuple(
        q_gradients_flat[
            axis * len(Q_COMPONENTS) : (axis + 1) * len(Q_COMPONENTS)
        ]
        for axis in range(3)
    )
    model = _coupled_model()
    parameters = model.constitutive_model.parameters
    expected = beris_edwards_q_nonlinear_components(
        tuple(runtime.solver.fields[name] for name in Q_COMPONENTS),
        tuple(runtime.solver.fields[name] for name in VELOCITY_COMPONENTS),
        q_gradients,
        tuple(velocity_gradients),
        ldg_b_over_gamma=parameters.ldg_b / model.rotational_viscosity,
        ldg_c_over_gamma=parameters.ldg_c / model.rotational_viscosity,
        flow_alignment=parameters.flow_alignment,
    )
    observed = model.explicit_rhs(_state(runtime), runtime.context)
    for name, value in zip(Q_COMPONENTS, expected, strict=True):
        torch.testing.assert_close(
            observed[name],
            value,
            rtol=0.0,
            atol=0.0,
        )

    expected_hat = torch.stack(
        tuple(
            runtime.projector.forward_transform(value, EVEN_SIGNATURE)
            for value in expected
        )
    )
    observed_hat = runtime.explicit_rhs_adapter(
        runtime.solver.fields,
        runtime.solver.parameters,
    )
    torch.testing.assert_close(
        observed_hat,
        expected_hat,
        rtol=0.0,
        atol=0.0,
    )


def test_dealiased_runtime_projects_initial_reset_and_each_evolved_update():
    runtime = _build()
    assert isinstance(
        runtime.solver.integrator,
        ProjectedSemiImplicitEulerIntegrator,
    )
    mask = runtime.projector.mask(EVEN_SIGNATURE)
    for name in Q_COMPONENTS:
        spectral = runtime.solver.fields[f"{name}.hat"]
        assert torch.count_nonzero(spectral[..., ~mask]) == 0

    reset = {
        name: torch.randn(
            SHAPE,
            dtype=torch.float64,
            generator=torch.Generator().manual_seed(100 + index),
        )
        for index, name in enumerate(Q_COMPONENTS)
    }
    runtime.reset(reset)
    for name in Q_COMPONENTS:
        spectral = runtime.solver.fields[f"{name}.hat"]
        assert torch.count_nonzero(spectral[..., ~mask]) == 0
    runtime.solver.run(1)
    for name in Q_COMPONENTS:
        spectral = runtime.solver.fields[f"{name}.hat"]
        assert torch.count_nonzero(spectral[..., ~mask]) == 0


@pytest.mark.parametrize("half_spectrum", (False, True))
def test_single_and_multistep_trajectory_matches_fixed_production_adapters(
    half_spectrum,
):
    runtime = _build(half_spectrum=half_spectrum)
    production = _manual_production_runtime(
        runtime,
        half_spectrum=half_spectrum,
    )
    for name in Q_COMPONENTS:
        torch.testing.assert_close(
            runtime.solver.fields[name],
            production.fields[name],
            rtol=2.0e-13,
            atol=2.0e-13,
        )

    for step in range(6):
        runtime.solver.integrator.step()
        production.integrator.step()
        for name in Q_COMPONENTS:
            torch.testing.assert_close(
                runtime.solver.fields[name],
                production.fields[name],
                rtol=2.0e-11,
                atol=2.0e-12,
            )
        runtime.synchronize_algebraic_for_observation()
        production.refresh_static_fields()
        for name in (*VELOCITY_COMPONENTS, "p"):
            torch.testing.assert_close(
                runtime.solver.fields[name],
                production.fields[name],
                rtol=3.0e-10,
                atol=3.0e-11,
            )
        assert step + 1 == production.integrator.step_count
        assert step + 1 == runtime.solver.integrator.step_count


def test_coupled_metadata_records_imex_and_projected_integrator_semantics():
    runtime = _build(half_spectrum=True)
    metadata = runtime.to_metadata()
    physical = metadata["problem"]["model"]["parameters"]
    assert physical["qualification_scope"] == "complete_coupled_q_stokes"
    assert physical["rotational_viscosity"] == 2.94
    assert physical["q_imex_split"] == {
        "explicit": [
            "bulk_B_C",
            "material_advection",
            "flow_alignment",
            "co_rotation",
        ],
        "implicit": ["bulk_A", "one_constant_L1_laplacian"],
    }
    assert metadata["time_integration"] == {
        "scheme": "semi_implicit_euler",
        "dynamic_spectral_projection": True,
        "integrator": "ProjectedSemiImplicitEulerIntegrator",
    }
    json.dumps(metadata, allow_nan=False)


def test_stage_j_remains_opt_in_and_does_not_enter_production_drivers():
    assert not hasattr(pssolver, "BerisEdwardsPlaneCoupledModel")
    assert not hasattr(pssolver, "ProjectedSemiImplicitEulerIntegrator")
    for path in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
        "pssolver/models/active_nematics/stokes.py",
    ):
        with open(path, encoding="utf-8") as source:
            text = source.read()
        assert "BerisEdwardsPlaneCoupledModel" not in text
        assert "ProjectedSemiImplicitEulerIntegrator" not in text
