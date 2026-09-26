"""P8.3 public complete-stress Channel application and restart closure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pssolver import (
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
    compile_simulation,
    run_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.geometries import RectangularChannel
from pssolver.models.active_nematics import (
    BerisEdwardsQGradientCache,
    CompleteStressBerisEdwards,
)
from pssolver.planning.construction import RuntimeConstructionKind
from pssolver.runtime.channel_beris_edwards import BerisEdwardsChannelStokes


def _simulation(
    tmp_path: Path,
    *,
    output_name="run",
    steps=2,
    checkpoint_interval=None,
    restart_from=None,
    stress_sum="physical",
    diagnostics=True,
    diagnostic_interval=1,
    save_interval=10,
):
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
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir(exist_ok=True)
    path = snapshot / "Q_0.npy"
    if not path.exists():
        values = np.zeros((8, 8, 8, 5), dtype=np.float64)
        values[..., 0] = 0.2
        values[..., 3] = -0.1
        rng = np.random.default_rng(20260926)
        values += rng.normal(scale=1.0e-3, size=values.shape)
        np.save(path, values, allow_pickle=False)
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="full_complex",
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(snapshot, step=0),
        execution=TorchSpectralExecution(
            runtime_path="channel_complete_stress",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": stress_sum,
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=tmp_path / output_name,
            steps=steps,
            save_interval=save_interval,
            diagnostic_interval=diagnostic_interval,
            diagnostics=diagnostics,
            checkpoint_interval=checkpoint_interval,
            restart_from=restart_from,
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


def test_channel_complete_stress_compiler_records_two_axis_parity(tmp_path):
    compiled = compile_simulation(_simulation(tmp_path))

    assert compiled.application == "channel_complete_stress_beris_edwards"
    assert compiled.construction_plan.kind is (
        RuntimeConstructionKind.CHANNEL_COMPLETE_STRESS
    )
    assert compiled.lowering_plan.solver.family == (
        "channel_no_slip_modal_stokes_pcg"
    )
    assert compiled.lowering_plan.nullspace.tangential_policy == "not_applicable"
    bases = {
        value.component: tuple(axis.modal_kind.value for axis in value.axes)
        for value in compiled.lowering_plan.component_bases
    }
    assert bases["stress_distortion_xy"] == (
        "periodic",
        "dirichlet",
        "neumann",
    )
    assert bases["stress_distortion_xz"] == (
        "periodic",
        "neumann",
        "dirichlet",
    )
    assert bases["stress_distortion_yz"] == (
        "periodic",
        "dirichlet",
        "dirichlet",
    )
    assert bases["force_x"] == (
        "periodic",
        "dirichlet",
        "dirichlet",
    )


def test_channel_complete_stress_is_finite_and_restart_is_byte_identical(tmp_path):
    continuous = run_simulation(
        _simulation(tmp_path, output_name="continuous", steps=6)
    )
    first = run_simulation(
        _simulation(
            tmp_path,
            output_name="first",
            steps=3,
            checkpoint_interval=3,
        )
    )
    resumed = run_simulation(
        _simulation(
            tmp_path,
            output_name="resumed",
            steps=3,
            restart_from=first.output_directory / "checkpoint_3",
        )
    )

    assert continuous.final_step == resumed.final_step == 6
    for prefix in ("Q", "u", "p"):
        left = continuous.output_directory / f"{prefix}_6.npy"
        right = resumed.output_directory / f"{prefix}_6.npy"
        assert left.read_bytes() == right.read_bytes()
        assert np.isfinite(np.load(left, allow_pickle=False)).all()
    assert max(abs(value.pressure_mean) for value in resumed.diagnostics) < 1.0e-12
    assert max(
        value.pressure_relative_residual for value in resumed.diagnostics
    ) < 1.0


def test_channel_restart_rebuilds_q_gradient_cache_before_first_step(
    tmp_path,
    monkeypatch,
):
    first = run_simulation(
        _simulation(
            tmp_path,
            output_name="cache_source",
            steps=2,
            checkpoint_interval=2,
        )
    )
    original_take = BerisEdwardsQGradientCache.take
    cache_hits = []

    def recording_take(self, fields):
        gradients = original_take(self, fields)
        cache_hits.append(gradients is not None)
        return gradients

    monkeypatch.setattr(BerisEdwardsQGradientCache, "take", recording_take)
    run_simulation(
        _simulation(
            tmp_path,
            output_name="cache_resumed",
            steps=1,
            restart_from=first.output_directory / "checkpoint_2",
        )
    )

    # Runtime construction validates the nonlinear model before the checkpoint
    # is loaded.  The final take is the first resumed timestep and must consume
    # the cache reconstructed from the restored Q spectra.
    assert cache_hits[-1] is True


def test_channel_observation_schedule_does_not_change_trajectory(tmp_path):
    dense = run_simulation(
        _simulation(
            tmp_path,
            output_name="dense_observation",
            steps=6,
            diagnostics=True,
            diagnostic_interval=1,
            save_interval=1,
        )
    )
    sparse = run_simulation(
        _simulation(
            tmp_path,
            output_name="sparse_observation",
            steps=6,
            diagnostics=False,
            diagnostic_interval=6,
            save_interval=6,
        )
    )

    for prefix in ("Q", "u", "p"):
        left = dense.output_directory / f"{prefix}_6.npy"
        right = sparse.output_directory / f"{prefix}_6.npy"
        assert left.read_bytes() == right.read_bytes()


def test_channel_observation_sync_performs_one_static_solve_per_state(
    tmp_path,
    monkeypatch,
):
    calls = []
    original_forward = BerisEdwardsChannelStokes.forward

    def recording_forward(self, fields, params):
        calls.append(self)
        return original_forward(self, fields, params)

    monkeypatch.setattr(
        BerisEdwardsChannelStokes,
        "forward",
        recording_forward,
    )
    steps = 4
    run_simulation(
        _simulation(
            tmp_path,
            output_name="static_solve_count",
            steps=steps,
            diagnostics=True,
            diagnostic_interval=1,
            save_interval=1,
        )
    )

    # One validation call during runtime construction, one solve for each
    # pre-step Q state, and one final solve for Q at the completed step.
    assert len(calls) == steps + 2


def test_channel_complete_stress_rejects_spectral_component_sum(tmp_path):
    with pytest.raises(ValueError, match="physical-space"):
        compile_simulation(_simulation(tmp_path, stress_sum="spectral"))


def test_channel_complete_stress_restart_rejects_tamper(tmp_path):
    first = run_simulation(
        _simulation(
            tmp_path,
            output_name="tamper_source",
            steps=1,
            checkpoint_interval=1,
        )
    )
    checkpoint = first.output_directory / "checkpoint_1"
    tensor = checkpoint / "backend__pressure_guess.npy"
    payload = bytearray(tensor.read_bytes())
    payload[-1] ^= 1
    tensor.write_bytes(payload)

    with pytest.raises(ValueError, match="checksum mismatch"):
        run_simulation(
            _simulation(
                tmp_path,
                output_name="tamper_target",
                steps=1,
                restart_from=checkpoint,
            )
        )
    metadata = json.loads((checkpoint / "checkpoint.json").read_text())
    assert metadata["backend_restart"]["state_keys"] == ["pressure_guess"]
    assert metadata["backend_restart"]["reconstructed_state_keys"] == [
        "q_gradient_cache"
    ]
