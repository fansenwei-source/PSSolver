"""Source/installed-wheel CPU gates for the RC4.3 lifting boundary."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pssolver import (
    GeneratedInitialCondition,
    Output,
    Simulation,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.configuration.public_simulation_runner import (
    PublicCompilationRejectionCode,
    PublicSimulationCompilationError,
)
from pssolver.core.fields import FieldRole
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics import (
    CompleteStressBerisEdwards,
    strong_homeotropic_q,
)
from pssolver.systems.algebraic import AlgebraicSystemSpec
from pssolver.systems.equations import (
    EquationFieldSpec,
    EquationSystemSpec,
    EquationTermSpec,
)


def _model() -> CompleteStressBerisEdwards:
    return CompleteStressBerisEdwards(
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=1.0 / 81.0,
        gamma=2.94,
        flow_alignment=0.3,
        activity=0.01,
        beta=-1.0,
        viscosity=2.0 / 3.0,
    )


def _public_simulation(
    tmp_path: Path,
    *,
    lifted: bool,
    runtime_path: str,
    linear_space: str,
) -> Simulation:
    model = _model()
    geometry = PlaneSlab(shape=(8, 8, 6), lengths=(100.0, 100.0, 20.0))
    q_policy = (
        strong_homeotropic_q(
            scalar_order=0.6,
            face_normals={
                (2, "lower"): (0.0, 0.0, -1.0),
                (2, "upper"): (0.0, 0.0, 1.0),
            },
        )
        if lifted
        else neumann_q()
    )
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": q_policy,
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="full_complex",
        ),
        time=TimeStepping(dt=0.005, refresh={"mode": "disabled"}),
        initial_condition=GeneratedInitialCondition(
            "extruded_defect_gas",
            parameters={
                "seed": 24,
                "num_defect_pairs": 1,
                "defect_min_separation": 10.0,
                "defect_core_radius": 1.5,
                "background_angle": 0.0,
                "twist_amplitude": 0.01,
                "twist_modes": [1, 2, 3],
                "initial_s": 1.0 / 3.0,
            },
        ),
        execution=TorchSpectralExecution(
            runtime_path=runtime_path,
            device="cpu",
            options={
                "tf32": "off",
                "fallback_allowed": False,
                "molecular_field_linear_space": linear_space,
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=tmp_path / f"{runtime_path}_{linear_space}_{lifted}",
            steps=1,
            save_start_step=0,
            save_interval=1,
            diagnostic_interval=1,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
    )


@pytest.mark.parametrize("runtime_path", ("legacy_production", "compiled_v2"))
@pytest.mark.parametrize("linear_space", ("physical", "spectral"))
def test_public_lifted_plane_remains_fail_closed_before_output(
    tmp_path,
    runtime_path,
    linear_space,
):
    simulation = _public_simulation(
        tmp_path,
        lifted=True,
        runtime_path=runtime_path,
        linear_space=linear_space,
    )
    with pytest.raises(PublicSimulationCompilationError) as caught:
        compile_simulation(simulation)
    assert caught.value.code is (
        PublicCompilationRejectionCode.UNSUPPORTED_LIFTED_WALL_MOLECULAR_FIELD
    )
    assert caught.value.context["supported_Q_boundary"] == "homogeneous_neumann"
    assert not Path(simulation.output.options["output_dir"]).exists()


@pytest.mark.parametrize("runtime_path", ("legacy_production", "compiled_v2"))
@pytest.mark.parametrize("linear_space", ("physical", "spectral"))
def test_public_homogeneous_plane_preserves_requested_effective_space(
    tmp_path,
    runtime_path,
    linear_space,
):
    compiled = compile_simulation(
        _public_simulation(
            tmp_path,
            lifted=False,
            runtime_path=runtime_path,
            linear_space=linear_space,
        )
    )
    assert compiled.lowering_plan.lifting_plan is None
    assert compiled.normalization["requested_molecular_field_linear_space"] == (
        linear_space
    )
    assert compiled.normalization["effective_molecular_field_linear_space"] == (
        linear_space
    )
    assert compiled.normalization["molecular_field_space_fallback_used"] is False


def _generic_system() -> EquationSystemSpec:
    return EquationSystemSpec(
        name="generic_transport",
        variant="generic_scalar_transport",
        fields=(
            EquationFieldSpec("concentration", FieldRole.EVOLVED, ("c",)),
            EquationFieldSpec("flux", FieldRole.TRANSIENT, ("j",)),
            EquationFieldSpec("multiplier", FieldRole.ALGEBRAIC, ("lambda_value",)),
        ),
        evolution_laws=(
            EquationTermSpec(
                "evolve_concentration", "generic_transport_rhs", ("c",), ("c", "j")
            ),
        ),
        constitutive_laws=(
            EquationTermSpec("construct_flux", "generic_flux", ("j",), ("c",)),
        ),
        algebraic_systems=(
            AlgebraicSystemSpec(
                "solve_multiplier", "generic_constraint", ("lambda_value",), ("c", "j")
            ),
        ),
        parameters={"diffusivity": 0.2, "reaction_rate": -0.4},
    )




def _homogeneous_second_derivative(value: torch.Tensor, spacing: float) -> torch.Tensor:
    result = torch.empty_like(value)
    result[..., 1:-1] = (
        value[..., 2:] - 2.0 * value[..., 1:-1] + value[..., :-2]
    ) / spacing**2
    result[..., 0] = (value[..., 1] - 3.0 * value[..., 0]) / spacing**2
    result[..., -1] = (value[..., -2] - 3.0 * value[..., -1]) / spacing**2
    return result
