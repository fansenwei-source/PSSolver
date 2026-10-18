"""RC6-2 truthful Channel pressure-floor and public compile regressions."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    SpectralSolver,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_channel_active_nematics_simulation,
)
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.public_simulation_runner import (
    PublicCompilationRejectionCode,
    PublicSimulationCompilationError,
)
from pssolver.functional import ChannelPressureTransposeOperator
from pssolver.geometries import RectangularChannel
from pssolver.linear_solvers.stokes import (
    ChannelNoSlipModalStokesSolver,
    channel_pressure_minimum_relative_tolerance,
)
from pssolver.linear_solvers.stokes.channel_no_slip import (
    CHANNEL_PRESSURE_NORMALIZED_RESIDUAL_DEFINITION,
    channel_pressure_active_rms,
)
from pssolver.models.active_nematics import CompleteStressBerisEdwards


@pytest.mark.parametrize(
    ("dtype", "complex_dtype", "precision"),
    (
        (torch.float32, torch.complex64, "float32"),
        (torch.float64, torch.complex128, "float64"),
    ),
)
def test_pressure_floor_is_eight_real_dtype_epsilons(
    dtype,
    complex_dtype,
    precision,
):
    expected = 8.0 * float(torch.finfo(dtype).eps)
    assert channel_pressure_minimum_relative_tolerance(dtype) == expected
    assert channel_pressure_minimum_relative_tolerance(complex_dtype) == expected
    assert channel_pressure_minimum_relative_tolerance(precision) == expected
    assert CHANNEL_PRESSURE_NORMALIZED_RESIDUAL_DEFINITION == (
        "active_rms(projected_pcg_recurrence_residual)/"
        "active_rms(projected_rhs)"
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("shape", ((8, 6, 4), (9, 8, 8)))
def test_supported_floor_bounds_recomputed_same_dtype_residual(dtype, shape):
    floor = channel_pressure_minimum_relative_tolerance(dtype)
    spectral = SpectralSolver(
        shape,
        L=tuple(1.0 + 0.37 * value for value in shape),
        device="cpu",
        dtype=dtype,
    )
    solver = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        friction=0.2,
        viscosity=0.73,
        pressure_relative_tolerance=floor,
        pressure_max_iterations=2000,
        pressure_fixed_iterations=None,
    )
    operator = ChannelPressureTransposeOperator(solver)
    complex_dtype = (
        torch.complex64 if dtype is torch.float32 else torch.complex128
    )
    for seed in (9, 10, 11):
        generator = torch.Generator().manual_seed(seed)
        real = torch.randn((1, *shape), generator=generator, dtype=dtype)
        imag = torch.randn((1, *shape), generator=generator, dtype=dtype)
        rhs = operator._project_gauge(torch.complex(real, imag).to(complex_dtype))
        solver.pressure_guess = None

        pressure = solver._solve_pressure(rhs.clone())
        residual = operator._project_gauge(
            rhs - operator.apply_pressure_operator(pressure)
        )
        recomputed = (
            channel_pressure_active_rms(
                residual,
                solver.pressure_inactive_mask,
            ).float64_value
            / channel_pressure_active_rms(
                rhs,
                solver.pressure_inactive_mask,
            ).float64_value
        )

        assert solver.last_pressure_termination_reason == (
            "relative_tolerance_met"
        )
        assert solver.last_pressure_relative_residual <= floor
        assert recomputed <= floor


def _legacy_channel_simulation(
    tmp_path: Path,
    *,
    tolerance: float,
    fixed_iterations: int | None,
) -> Simulation:
    request = ChannelActiveNematicRunSpec(
        shape=(16, 8, 8),
        lengths=(16.0, 8.0, 8.0),
        steps=2,
        save_interval=1,
        diagnostic_interval=1,
        generated_output_directory=tmp_path / "legacy",
        snapshot_output_directory=tmp_path / "legacy_snapshot",
        pressure_relative_tolerance=tolerance,
        pressure_max_iterations=80,
        pressure_fixed_iterations=fixed_iterations,
        device="cpu",
        dtype="float32",
        runtime_path="compiled_channel_v2",
    )
    source = compose_channel_active_nematics_simulation(request.components)
    return Simulation(
        model=source.equation_system,
        geometry=source.geometry,
        boundaries=source.boundaries,
        numerics=source.numerics,
        time=source.time_integration,
        discretization=source.discretization_parameters,
        initial_condition=source.initial_condition,
        execution=source.execution,
        output=source.workflow,
        invocation=source.invocation,
    )


def _complete_stress_channel_simulation(
    tmp_path: Path,
    *,
    tolerance: float,
    fixed_iterations: int | None,
) -> Simulation:
    model = CompleteStressBerisEdwards(
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
    geometry = RectangularChannel(
        shape=(8, 8, 8),
        lengths=(8.0, 8.0, 8.0),
    )
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": no_slip_velocity(),
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
        time=TimeStepping(dt=1.0e-3),
        initial_condition=SnapshotInitialCondition(
            tmp_path / "snapshot",
            step=0,
        ),
        execution=TorchSpectralExecution(
            runtime_path="channel_complete_stress",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "physical",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=tmp_path / "complete",
            steps=2,
            save_interval=2,
            save_start_step=0,
            diagnostic_interval=1,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        discretization={
            "pressure_solver": {
                "algorithm": "preconditioned_conjugate_gradient",
                "warm_start": True,
                "relative_tolerance": tolerance,
                "max_iterations": 80,
                "fixed_iterations": fixed_iterations,
            }
        },
    )


@pytest.mark.parametrize(
    ("factory", "precision", "output_name"),
    (
        (_legacy_channel_simulation, "float32", "legacy"),
        (_complete_stress_channel_simulation, "float64", "complete"),
    ),
)
def test_public_compilers_reject_sub_floor_tolerance_with_structured_error(
    tmp_path,
    factory,
    precision,
    output_name,
):
    floor = channel_pressure_minimum_relative_tolerance(precision)
    simulation = factory(
        tmp_path,
        tolerance=floor / 2.0,
        fixed_iterations=None,
    )

    with pytest.raises(PublicSimulationCompilationError) as captured:
        compile_simulation(simulation)

    assert captured.value.code is (
        PublicCompilationRejectionCode.DISCRETIZATION_CONTRACT
    )
    assert "below the supported tolerance-mode floor" in str(captured.value)
    assert not (tmp_path / output_name).exists()


@pytest.mark.parametrize(
    ("factory", "precision"),
    (
        (_legacy_channel_simulation, "float32"),
        (_complete_stress_channel_simulation, "float64"),
    ),
)
def test_public_compilers_accept_floor_and_preserve_fixed_work_exception(
    tmp_path,
    factory,
    precision,
):
    floor = channel_pressure_minimum_relative_tolerance(precision)

    tolerance_mode = compile_simulation(
        factory(
            tmp_path,
            tolerance=floor,
            fixed_iterations=None,
        )
    )
    fixed_work = compile_simulation(
        factory(
            tmp_path,
            tolerance=floor / 2.0,
            fixed_iterations=4,
        )
    )

    assert tolerance_mode.source.discretization["pressure_solver"][
        "relative_tolerance"
    ] == floor
    assert fixed_work.source.discretization["pressure_solver"] == {
        "algorithm": "preconditioned_conjugate_gradient",
        "fixed_iterations": 4,
        "max_iterations": 80,
        "relative_tolerance": floor / 2.0,
        "warm_start": True,
    }
