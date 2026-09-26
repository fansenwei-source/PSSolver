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
from pssolver.models.active_nematics import CompleteStressBerisEdwards
from pssolver.planning.construction import RuntimeConstructionKind


def _simulation(
    tmp_path: Path,
    *,
    output_name="run",
    steps=2,
    checkpoint_interval=None,
    restart_from=None,
    stress_sum="physical",
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
            save_interval=10,
            diagnostic_interval=1,
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
        _simulation(tmp_path, output_name="continuous", steps=2)
    )
    first = run_simulation(
        _simulation(
            tmp_path,
            output_name="first",
            steps=1,
            checkpoint_interval=1,
        )
    )
    resumed = run_simulation(
        _simulation(
            tmp_path,
            output_name="resumed",
            steps=1,
            restart_from=first.output_directory / "checkpoint_1",
        )
    )

    assert continuous.final_step == resumed.final_step == 2
    for prefix in ("Q", "u", "p"):
        left = continuous.output_directory / f"{prefix}_2.npy"
        right = resumed.output_directory / f"{prefix}_2.npy"
        assert left.read_bytes() == right.read_bytes()
        assert np.isfinite(np.load(left, allow_pickle=False)).all()
    assert max(abs(value.pressure_mean) for value in resumed.diagnostics) < 1.0e-12
    assert max(
        value.pressure_relative_residual for value in resumed.diagnostics
    ) < 1.0


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
