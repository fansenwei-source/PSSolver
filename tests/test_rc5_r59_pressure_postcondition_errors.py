"""R5-9 pressure postcondition and first-solve exception regressions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pssolver import (
    ChannelPressureConvergenceError,
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    SpectralSolver,
    TimeStepping,
    TorchSpectralExecution,
    run_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.functional import FunctionalConvergenceError
from pssolver.functional import api as functional_api
from pssolver.geometries import RectangularChannel
from pssolver.linear_solvers.stokes import (
    ChannelNoSlipModalStokesSolver,
    channel_pressure_minimum_relative_tolerance,
)
from pssolver.models.active_nematics import CompleteStressBerisEdwards


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_tolerance_mode_rejects_targets_below_the_dtype_floor(dtype):
    spectral = SpectralSolver(
        (4, 3, 2),
        L=(4.0, 3.0, 2.0),
        device="cpu",
        dtype=dtype,
    )
    floor = channel_pressure_minimum_relative_tolerance(dtype)

    with pytest.raises(ValueError, match="attainable.*tolerance-mode floor"):
        ChannelNoSlipModalStokesSolver(
            spectral.transform_backend,
            pressure_relative_tolerance=floor / 2.0,
            pressure_max_iterations=40,
            pressure_fixed_iterations=None,
        )

    accepted = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        pressure_relative_tolerance=floor,
        pressure_max_iterations=40,
        pressure_fixed_iterations=None,
    )
    assert accepted.pressure_rel_tol == floor


def test_fixed_work_keeps_sub_floor_tolerance_as_a_diagnostic_only_field():
    spectral = SpectralSolver(
        (4, 3, 2),
        L=(4.0, 3.0, 2.0),
        device="cpu",
        dtype=torch.float64,
    )
    requested = channel_pressure_minimum_relative_tolerance(torch.float64) / 2

    solver = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        pressure_relative_tolerance=requested,
        pressure_max_iterations=40,
        pressure_fixed_iterations=4,
    )

    assert solver.pressure_rel_tol == requested
    assert (
        solver.pressure_convergence_policy()["relative_tolerance_role"]
        == "diagnostic_only_not_early_exit"
    )


def _failing_first_solve_simulation(tmp_path: Path, name: str) -> Simulation:
    snapshot = tmp_path / f"{name}_initial"
    snapshot.mkdir()
    generator = np.random.default_rng(1)
    q = generator.normal(scale=0.02, size=(8, 8, 8, 5))
    q[..., 0] += 0.2
    q[..., 3] -= 0.1
    np.save(snapshot / "Q_0.npy", q)
    model = CompleteStressBerisEdwards(
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=1.0 / 81.0,
        gamma=2.94,
        flow_alignment=0.3,
        activity=0.05,
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
        initial_condition=SnapshotInitialCondition(snapshot, step=0),
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
            directory=tmp_path / name,
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
                "relative_tolerance": 1.0e-14,
                "max_iterations": 1,
                "fixed_iterations": None,
            }
        },
    )


def _assert_first_solve_details(error: ChannelPressureConvergenceError) -> None:
    assert error.details["termination_reason"] == (
        "iteration_limit_reached_without_postcondition"
    )
    assert error.details["achieved_iteration_count"] == 1
    assert error.details["postcondition_satisfied"] is False


def test_public_first_solve_failure_preserves_channel_contract_exception(tmp_path):
    simulation = _failing_first_solve_simulation(tmp_path, "public")

    with pytest.raises(ChannelPressureConvergenceError) as captured:
        run_simulation(simulation)

    _assert_first_solve_details(captured.value)
    assert not Path(simulation.output.options["output_dir"]).exists()


def test_functional_first_solve_failure_is_both_public_error_types(tmp_path):
    simulation = _failing_first_solve_simulation(tmp_path, "functional")
    request = functional_api.channel_activity_functional_request(
        simulation.specification
    )

    with pytest.raises(ChannelPressureConvergenceError) as captured:
        functional_api.build_channel_activity_functional_runtime(request)

    assert isinstance(captured.value, FunctionalConvergenceError)
    assert captured.value.operation == "build_channel_functional_runtime"
    _assert_first_solve_details(captured.value)
    assert not Path(simulation.output.options["output_dir"]).exists()
