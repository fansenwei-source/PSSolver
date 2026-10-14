"""R5-2 fail-closed production finite-state contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pssolver import (
    Output,
    Simulation,
    SimulationExecutionError,
    SimulationStateError,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    run_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.configuration.simulation import InvocationSpec
from pssolver.geometries import PeriodicBox
from pssolver.models.active_nematics import CompleteStressBerisEdwards
from pssolver.workflows.channel_beris_edwards import (
    ChannelBerisEdwardsDiagnostic,
    ChannelBerisEdwardsObservation,
)
from pssolver.workflows.channel_observation import (
    ChannelDiagnostic,
    ChannelObservation,
)
from pssolver.workflows.periodic_beris_edwards import (
    PeriodicDiagnostic,
    PeriodicObservation,
)
from pssolver.workflows.plane_observation import PlaneDiagnostic, PlaneObservation


def _arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros((2, 2, 2, 5), dtype=np.float64),
        np.zeros((2, 2, 2, 3), dtype=np.float64),
        np.zeros((2, 2, 2), dtype=np.float64),
    )


@pytest.mark.parametrize(
    ("factory", "family"),
    (
        (PeriodicObservation, "periodic_complete_stress"),
        (ChannelBerisEdwardsObservation, "channel_complete_stress"),
        (PlaneObservation, "plane_complete_stress"),
        (ChannelObservation, "legacy_channel_active_nematic"),
    ),
)
def test_production_observations_raise_the_public_state_error(factory, family):
    q, velocity, pressure = _arrays()
    q[0, 0, 0, 0] = np.nan

    with pytest.raises(SimulationStateError) as captured:
        factory(7, q, velocity, pressure)

    error = captured.value
    assert isinstance(error, SimulationExecutionError)
    assert isinstance(error, ValueError)
    assert error.application_family == family
    assert error.stage == "observation"
    assert error.step == 7
    assert error.to_metadata()["code"] == "simulation_state_error"


@pytest.mark.parametrize(
    ("factory", "arguments", "family"),
    (
        (
            PeriodicDiagnostic,
            (4, 0.0, 0.0, float("inf"), 0.0),
            "periodic_complete_stress",
        ),
        (
            ChannelBerisEdwardsDiagnostic,
            (4, 0.0, 0.0, 0.0, 1, float("inf")),
            "channel_complete_stress",
        ),
        (
            PlaneDiagnostic,
            (4, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, float("inf")),
            "plane_complete_stress",
        ),
        (
            ChannelDiagnostic,
            (4, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, float("inf")),
            "legacy_channel_active_nematic",
        ),
    ),
)
def test_production_diagnostics_raise_the_public_state_error(
    factory,
    arguments,
    family,
):
    with pytest.raises(SimulationStateError) as captured:
        factory(*arguments)

    assert captured.value.application_family == family
    assert captured.value.stage == "diagnostic"
    assert captured.value.step == 4


def _unstable_periodic_simulation(
    tmp_path: Path,
    *,
    diagnostics: bool,
    checkpoint_interval: int | None,
) -> Simulation:
    shape = (8, 8, 4)
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    values = 0.05 * np.random.default_rng(1).standard_normal((*shape, 5))
    np.save(snapshot / "Q_0.npy", values, allow_pickle=False)
    model = CompleteStressBerisEdwards(
        ldg_a=-1.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=1.0 / 81.0,
        gamma=2.94,
        flow_alignment=0.3,
        activity=5.0,
        beta=-1.0,
        viscosity=2.0 / 3.0,
    )
    geometry = PeriodicBox(shape=shape, lengths=(8.0, 8.0, 4.0))
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
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
            projected_transform_execution="truncated",
            spectral_storage="full_complex",
        ),
        time=TimeStepping(dt=20.0, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(snapshot, step=0),
        execution=TorchSpectralExecution(
            runtime_path="periodic_spectral",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=tmp_path / "out",
            steps=40,
            save_start_step=0,
            save_interval=40,
            diagnostic_interval=10,
            diagnostics=diagnostics,
            save_hydrodynamics=True,
            checkpoint_interval=checkpoint_interval,
        ),
        invocation=InvocationSpec({}),
    )


@pytest.mark.parametrize(
    ("diagnostics", "checkpoint_interval", "expected_stage"),
    (
        (True, None, "diagnostic"),
        (False, None, "observation"),
        (False, 10, "checkpoint"),
    ),
)
def test_unstable_periodic_run_fails_closed_without_complete_or_nonfinite_output(
    tmp_path,
    diagnostics,
    checkpoint_interval,
    expected_stage,
):
    simulation = _unstable_periodic_simulation(
        tmp_path,
        diagnostics=diagnostics,
        checkpoint_interval=checkpoint_interval,
    )

    with pytest.raises(SimulationStateError) as captured:
        run_simulation(simulation, progress=range(40))

    output = tmp_path / "out"
    assert captured.value.application_family == "periodic_complete_stress"
    assert captured.value.stage == expected_stage
    assert not (output / "COMPLETE").exists()
    assert not (output / "metadata.json").exists()
    arrays = tuple(output.glob("[Qup]_*.npy"))
    assert arrays
    assert all(np.isfinite(np.load(path, allow_pickle=False)).all() for path in arrays)
    assert not tuple(output.glob("checkpoint_*"))
