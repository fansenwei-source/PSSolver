"""Qualification tests for the Stage I Beris--Edwards constitutive DAG."""

from __future__ import annotations

import json
import math

import pytest
import torch

import pssolver
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
from pssolver.execution import (
    IncompressibleStokesSystemSpec,
    TangentialZeroModePolicy,
)
from pssolver.experimental import (
    PlaneBerisEdwardsSolverOptions,
    build_experimental_model_runtime,
    create_beris_edwards_plane_geometry_solver_registry,
)
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics import (
    ALGEBRAIC_STRESS_COMPONENTS,
    DISTORTION_STRESS_COMPONENTS,
    H_COMPONENTS,
    NEMATIC_FORCE_COMPONENTS,
    Q_COMPONENTS,
    Q_GRADIENT_COMPONENTS,
    BerisEdwardsConstitutiveParameters,
    BerisEdwardsConstitutiveStokesCanaryModel,
    BerisEdwardsFreeSlipStokes,
    BerisEdwardsPointwiseKernels,
    beris_edwards_algebraic_stress_components,
    beris_edwards_distortion_stress_components,
    beris_edwards_molecular_field_components,
)
from pssolver.transforms import (
    projected_common_basis_stress_divergence,
    projected_distortion_stress_divergence,
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
        dealias_rule=(
            DealiasRule.CUBIC_HALF if half_spectrum else DealiasRule.NONE
        ),
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=(
            ProjectedTransformExecution.TRUNCATED
            if half_spectrum
            else ProjectedTransformExecution.FULL
        ),
        spectral_storage=(
            SpectralStorage.HERMITIAN_HALF
            if half_spectrum
            else SpectralStorage.FULL_COMPLEX
        ),
        hermitian_axis=1 if half_spectrum else None,
    )


def _build(
    *,
    model=None,
    half_spectrum=False,
    linear_space="spectral",
    sum_space="spectral",
):
    if model is None:
        model = _model()
    return build_experimental_model_runtime(
        model,
        PlaneSlab(DomainSpec((10, 8, 7), (5.0, 4.0, 3.5))),
        _numerics(half_spectrum=half_spectrum),
        dt=0.01,
        batch_size=1,
        geometry_solver_registry=(
            create_beris_edwards_plane_geometry_solver_registry(
                constitutive_options=PlaneBerisEdwardsSolverOptions(
                    molecular_field_linear_space=linear_space,
                    stress_divergence_sum_space=sum_space,
                    pointwise_execution="eager",
                )
            )
        ),
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


def test_field_lifetimes_and_reverse_declaration_form_the_required_dag():
    model = _model()
    fields = model.field_specs()
    by_name = {field.name: field for field in fields}
    assert by_name["Q"].role is FieldRole.EVOLVED
    assert by_name["molecular_field"].role is FieldRole.TRANSIENT
    assert by_name["q_gradient"].role is FieldRole.TRANSIENT
    assert by_name["algebraic_stress"].role is FieldRole.TRANSIENT
    assert by_name["distortion_stress"].role is FieldRole.TRANSIENT
    assert by_name["nematic_force"].role is FieldRole.TRANSIENT
    assert by_name["velocity"].role is FieldRole.ALGEBRAIC
    assert by_name["p"].role is FieldRole.ALGEBRAIC
    assert by_name["Q"].component_names == Q_COMPONENTS
    assert by_name["molecular_field"].component_names == H_COMPONENTS
    assert by_name["q_gradient"].component_names == Q_GRADIENT_COMPONENTS

    systems = model.algebraic_system_specs()
    assert tuple(system.name for system in systems) == (
        "flow",
        "nematic_force",
        "nematic_stress",
        "molecular_field",
        "q_gradient",
    )
    runtime = _build(model=model)
    assert runtime.algebraic_execution_plan.execution_order == (
        "molecular_field",
        "q_gradient",
        "nematic_stress",
        "nematic_force",
        "flow",
    )
    assert runtime.assembly.omitted_transient_components == (
        *H_COMPONENTS,
        *Q_GRADIENT_COMPONENTS,
        *ALGEBRAIC_STRESS_COMPONENTS,
        *DISTORTION_STRESS_COMPONENTS,
        *NEMATIC_FORCE_COMPONENTS,
    )
    assert runtime.solver.fields.spatial.shape[0] == 9
    assert len(runtime.transient_algebraic_state()) == 41


@pytest.mark.parametrize("half_spectrum", (False, True))
@pytest.mark.parametrize("linear_space", ("physical", "spectral"))
def test_molecular_field_matches_qualified_physical_equation(
    half_spectrum,
    linear_space,
):
    runtime = _build(
        half_spectrum=half_spectrum,
        linear_space=linear_space,
    )
    q = tuple(runtime.solver.fields[name] for name in Q_COMPONENTS)
    laplacians = tuple(
        runtime.context.laplacian(name, runtime.solver.fields[name])
        for name in Q_COMPONENTS
    )
    parameters = _parameters()
    expected_raw = beris_edwards_molecular_field_components(
        q,
        laplacians,
        ldg_a=parameters.ldg_a,
        ldg_b=parameters.ldg_b,
        ldg_c=parameters.ldg_c,
        ldg_l1=parameters.ldg_l1,
    )
    actual = runtime.transient_algebraic_state()
    for name, expected in zip(H_COMPONENTS, expected_raw, strict=True):
        torch.testing.assert_close(
            actual[name],
            _projected_physical(runtime, name, expected),
            rtol=3.0e-12,
            atol=3.0e-12,
        )


@pytest.mark.parametrize("half_spectrum", (False, True))
def test_q_gradients_and_both_stress_parts_match_constitutive_helpers(
    half_spectrum,
):
    runtime = _build(half_spectrum=half_spectrum)
    transient = runtime.transient_algebraic_state()
    q = tuple(runtime.solver.fields[name] for name in Q_COMPONENTS)
    gradients = []
    for axis in range(3):
        axis_values = []
        for index, q_name in enumerate(Q_COMPONENTS):
            gradient_name = Q_GRADIENT_COMPONENTS[
                axis * len(Q_COMPONENTS) + index
            ]
            expected = runtime.context.gradient(
                q_name,
                gradient_name,
                runtime.solver.fields[q_name],
                axis,
            )
            torch.testing.assert_close(
                transient[gradient_name],
                expected,
                rtol=2.0e-13,
                atol=2.0e-13,
            )
            axis_values.append(transient[gradient_name])
        gradients.append(tuple(axis_values))

    h = tuple(transient[name] for name in H_COMPONENTS)
    parameters = _parameters()
    expected_algebraic = beris_edwards_algebraic_stress_components(
        q,
        h,
        flow_alignment=parameters.flow_alignment,
        active_prefactor=parameters.active_prefactor,
    )
    expected_distortion = beris_edwards_distortion_stress_components(
        tuple(gradients),
        ldg_l1=parameters.ldg_l1,
    )
    for names, expected_values in (
        (ALGEBRAIC_STRESS_COMPONENTS, expected_algebraic),
        (DISTORTION_STRESS_COMPONENTS, expected_distortion),
    ):
        for name, expected in zip(names, expected_values, strict=True):
            torch.testing.assert_close(
                transient[name],
                _projected_physical(runtime, name, expected),
                rtol=4.0e-12,
                atol=4.0e-12,
            )


@pytest.mark.parametrize("sum_space", ("physical", "spectral"))
def test_complete_force_uses_qualified_plane_parity_split_and_projection(
    sum_space,
):
    runtime = _build(sum_space=sum_space)
    transient = runtime.transient_algebraic_state()
    algebraic = tuple(
        transient[name] for name in ALGEBRAIC_STRESS_COMPONENTS
    )
    distortion = tuple(
        transient[name] for name in DISTORTION_STRESS_COMPONENTS
    )
    expected_raw = projected_common_basis_stress_divergence(
        runtime.solver.transform_backend,
        algebraic,
        EVEN_SIGNATURE,
        projector=runtime.projector,
        sum_space=sum_space,
    ) + projected_distortion_stress_divergence(
        runtime.solver.transform_backend,
        distortion,
        EVEN_SIGNATURE,
        ODD_SIGNATURE,
        projector=runtime.projector,
        sum_space=sum_space,
    )
    for index, name in enumerate(NEMATIC_FORCE_COMPONENTS):
        torch.testing.assert_close(
            transient[name],
            _projected_physical(runtime, name, expected_raw[index]),
            rtol=3.0e-12,
            atol=3.0e-12,
        )


def test_complete_chain_matches_legacy_production_force_and_stokes_oracle():
    runtime = _build()
    parameters = _parameters()
    legacy = BerisEdwardsFreeSlipStokes(
        runtime.solver,
        runtime.projector,
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
        pointwise_kernels=BerisEdwardsPointwiseKernels("eager"),
        zero_mode_policy="zero_mean",
    )
    alpha = torch.tensor(
        -parameters.active_prefactor,
        dtype=runtime.context.real_dtype,
        device=runtime.context.device,
    )
    legacy_force, _ = legacy.compute_nematic_force(runtime.solver.fields, alpha)
    transient = runtime.transient_algebraic_state()
    for index, name in enumerate(NEMATIC_FORCE_COMPONENTS):
        expected = _projected_physical(runtime, name, legacy_force[index])
        torch.testing.assert_close(
            transient[name],
            expected,
            rtol=8.0e-12,
            atol=8.0e-12,
        )

    legacy_solution_hat = legacy(runtime.solver.fields, {"alpha": alpha})
    stage_i_solution_hat = torch.stack(
        tuple(
            runtime.solver.fields[f"{name}.hat"]
            for name in ("ux", "uy", "uz", "p")
        )
    )
    torch.testing.assert_close(
        stage_i_solution_hat,
        legacy_solution_hat,
        rtol=1.0e-11,
        atol=1.0e-11,
    )


def test_free_q_active_force_mean_and_zero_mean_flow_are_distinct_choices():
    parameters = _parameters(
        ldg_a=0.0,
        ldg_b=0.0,
        ldg_c=0.0,
        ldg_l1=0.17,
        flow_alignment=0.0,
        active_prefactor=-0.13,
    )
    runtime = _build(model=_model(parameters=parameters))
    x, _, z = runtime.solver.transform_backend.spatial_grids
    zeros = torch.zeros_like(x + z)
    qxz = 0.19 * torch.cos(math.pi * z / runtime.context.lengths[2]) + zeros
    runtime.reset(
        {
            "Qxx": zeros,
            "Qxy": zeros,
            "Qxz": qxz,
            "Qyy": zeros,
            "Qyz": zeros,
        }
    )
    force_x = runtime.transient_algebraic_state()["force_x"]
    assert force_x.mean().abs() > 1.0e-3
    assert runtime.solver.fields["ux"].mean().abs() < 1.0e-12


def test_numerical_policies_are_observable_but_not_physical_parameters():
    runtime = _build(linear_space="spectral", sum_space="spectral")
    metadata = runtime.to_metadata()
    systems = {
        item["system"]["name"]: item
        for item in metadata["algebraic_lifecycle"]["systems"]
    }
    assert systems["molecular_field"]["observability"]["numerical_policy"] == {
        "linear_space": "spectral",
        "pointwise_execution": "eager",
        "projection": "complete_molecular_field",
    }
    assert systems["nematic_force"]["observability"]["numerical_policy"] == {
        "projection": "complete_force_into_velocity_spaces",
        "stress_divergence_sum_space": "spectral",
    }
    assert "linear_space" not in _model().parameter_metadata()["constitutive"]
    json.dumps(metadata, allow_nan=False)


def test_stage_i_is_opt_in_and_does_not_enter_production_drivers():
    assert not hasattr(pssolver, "BerisEdwardsConstitutiveStokesCanaryModel")
    for path in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
        "pssolver/models/active_nematics/stokes.py",
    ):
        with open(path, encoding="utf-8") as source:
            text = source.read()
        assert "BerisEdwardsConstitutiveStokesCanaryModel" not in text
        assert "create_beris_edwards_plane_geometry_solver_registry" not in text
