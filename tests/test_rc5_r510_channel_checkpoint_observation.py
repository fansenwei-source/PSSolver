"""R5-10 Channel checkpoint observation must not perturb live evolution."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    run_simulation,
)
from pssolver.applications.channel_active_nematics import (
    run_channel_active_nematics,
)
from pssolver.boundaries import (
    assign_boundaries,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.geometries import RectangularChannel
from pssolver.models.active_nematics import CompleteStressBerisEdwards
from pssolver.workflows.channel_checkpoint_observation import (
    capture_synchronized_pressure_guess,
)


class _CheckpointObservationAdapter:
    def __init__(self, *, fail: bool = False) -> None:
        self.pressure_guess = torch.tensor([1.0])
        self.fail = fail

    def capture_pressure_guess(self) -> torch.Tensor:
        return self.pressure_guess.detach().clone()

    def restore_pressure_guess(self, value: torch.Tensor) -> None:
        self.pressure_guess = value.detach().clone()

    def synchronize_for_observation(self) -> None:
        self.pressure_guess.add_(2.0)
        if self.fail:
            raise RuntimeError("sentinel observation failure")


def test_checkpoint_observation_keeps_synchronized_copy_and_restores_live_state():
    adapter = _CheckpointObservationAdapter()

    checkpoint_guess = capture_synchronized_pressure_guess(adapter)

    assert torch.equal(checkpoint_guess, torch.tensor([3.0]))
    assert torch.equal(adapter.pressure_guess, torch.tensor([1.0]))


def test_checkpoint_observation_restores_live_state_when_synchronization_fails():
    adapter = _CheckpointObservationAdapter(fail=True)

    with pytest.raises(RuntimeError, match="sentinel observation failure"):
        capture_synchronized_pressure_guess(adapter)

    assert torch.equal(adapter.pressure_guess, torch.tensor([1.0]))


def _assert_saved_state_is_byte_identical(
    left: Path,
    right: Path,
    *,
    step: int,
) -> None:
    for name in ("Q", "u", "p"):
        left_path = left / f"{name}_{step}.npy"
        right_path = right / f"{name}_{step}.npy"
        assert left_path.read_bytes() == right_path.read_bytes()


def _legacy_spec(
    directory: Path,
    *,
    runtime_path: str,
    checkpoint_interval: int | None,
) -> ChannelActiveNematicRunSpec:
    return ChannelActiveNematicRunSpec(
        shape=(7, 6, 5),
        lengths=(7.0, 6.0, 5.0),
        dt=1.0e-3,
        steps=4,
        save_interval=4,
        diagnostic_interval=4,
        checkpoint_interval=checkpoint_interval,
        generated_output_directory=directory,
        snapshot_output_directory=directory.parent / f"{directory.name}_snapshots",
        device="cpu",
        dtype="float32",
        activity=0.2,
        pressure_relative_tolerance=1.0e-8,
        pressure_max_iterations=100,
        pressure_fixed_iterations=12,
        runtime_path=runtime_path,
        seed=24,
    )


def test_p7_checkpoint_schedule_does_not_change_live_trajectory(tmp_path):
    for runtime_path in ("legacy_channel", "compiled_channel_v2"):
        uninterrupted = tmp_path / f"{runtime_path}_uninterrupted"
        checkpointed = tmp_path / f"{runtime_path}_checkpointed"
        run_channel_active_nematics(
            _legacy_spec(
                uninterrupted,
                runtime_path=runtime_path,
                checkpoint_interval=None,
            )
        )
        result = run_channel_active_nematics(
            _legacy_spec(
                checkpointed,
                runtime_path=runtime_path,
                checkpoint_interval=2,
            )
        )

        assert result.checkpoint_steps == (2, 4)
        _assert_saved_state_is_byte_identical(
            uninterrupted,
            checkpointed,
            step=4,
        )


def _complete_stress_simulation(
    tmp_path: Path,
    *,
    name: str,
    checkpoint_interval: int | None,
) -> Simulation:
    shape = (6, 6, 6)
    snapshot = tmp_path / "complete_stress_initial"
    snapshot.mkdir(exist_ok=True)
    initial = snapshot / "Q_0.npy"
    if not initial.exists():
        values = np.random.default_rng(510).normal(
            scale=1.0e-3,
            size=(*shape, 5),
        )
        values[..., 0] += 0.2
        values[..., 3] -= 0.1
        np.save(initial, values, allow_pickle=False)
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
    geometry = RectangularChannel(shape=shape, lengths=(6.0, 6.0, 6.0))
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
            steps=4,
            save_interval=4,
            save_start_step=0,
            diagnostic_interval=4,
            diagnostics=True,
            save_hydrodynamics=True,
            checkpoint_interval=checkpoint_interval,
        ),
        discretization={
            "pressure_solver": {
                "algorithm": "preconditioned_conjugate_gradient",
                "relative_tolerance": 1.0e-10,
                "max_iterations": 40,
                "fixed_iterations": 12,
                "warm_start": True,
            }
        },
    )


def test_complete_stress_checkpoint_schedule_does_not_change_live_trajectory(
    tmp_path,
):
    uninterrupted = run_simulation(
        _complete_stress_simulation(
            tmp_path,
            name="complete_stress_uninterrupted",
            checkpoint_interval=None,
        )
    )
    checkpointed = run_simulation(
        _complete_stress_simulation(
            tmp_path,
            name="complete_stress_checkpointed",
            checkpoint_interval=2,
        )
    )

    assert checkpointed.checkpoint_steps == (2, 4)
    _assert_saved_state_is_byte_identical(
        uninterrupted.output_directory,
        checkpointed.output_directory,
        step=4,
    )
