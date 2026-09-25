"""P8.2 public periodic application, lowering, and restart closure."""

from __future__ import annotations

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
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.geometries import PeriodicBox
from pssolver.models.active_nematics import CompleteStressBerisEdwards
from pssolver.planning.construction import RuntimeConstructionKind


def _simulation(
    tmp_path: Path,
    *,
    output_name="run",
    steps=2,
    checkpoint_interval=None,
    restart_from=None,
    runtime_path="periodic_spectral",
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
    geometry = PeriodicBox(
        shape=(8, 8, 8),
        lengths=(8.0, 8.0, 8.0),
    )
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": free_slip_velocity(),
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
        rng = np.random.default_rng(20260925)
        values += rng.normal(scale=1.0e-3, size=values.shape)
        np.save(path, values, allow_pickle=False)
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            spectral_storage="hermitian_half",
            hermitian_axis=1,
        ),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(snapshot, step=0),
        execution=TorchSpectralExecution(
            runtime_path=runtime_path,
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
            directory=tmp_path / output_name,
            steps=steps,
            save_interval=10,
            diagnostic_interval=1,
            checkpoint_interval=checkpoint_interval,
            restart_from=restart_from,
        ),
    )


def test_periodic_public_compiler_records_fft_lowering_and_uniform_policy(tmp_path):
    compiled = compile_simulation(_simulation(tmp_path))

    assert compiled.application == "periodic_complete_stress_beris_edwards"
    assert compiled.construction_plan.kind is (
        RuntimeConstructionKind.PERIODIC_COMPLETE_STRESS
    )
    assert compiled.lowering_plan.solver.family == "periodic_modal_stokes"
    assert compiled.lowering_plan.nullspace.uniform_mode_action == (
        "remove_all_uniform_velocity_and_force"
    )
    assert compiled.lowering_plan.nullspace.tangential_velocity_components == (
        "ux",
        "uy",
        "uz",
    )
    assert {
        axis.transform_kind.value
        for basis in compiled.lowering_plan.component_bases
        for axis in basis.axes
    } == {"fft"}


def test_periodic_application_is_finite_and_restart_is_byte_identical(tmp_path):
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
    checkpoint = first.output_directory / "checkpoint_1"
    resumed = run_simulation(
        _simulation(
            tmp_path,
            output_name="resumed",
            steps=1,
            restart_from=checkpoint,
        )
    )

    assert continuous.final_step == resumed.final_step == 2
    assert (continuous.output_directory / "COMPLETE").is_file()
    assert (resumed.output_directory / "COMPLETE").is_file()
    for prefix in ("Q", "u", "p"):
        left = continuous.output_directory / f"{prefix}_2.npy"
        right = resumed.output_directory / f"{prefix}_2.npy"
        assert left.read_bytes() == right.read_bytes()
        assert np.isfinite(np.load(left, allow_pickle=False)).all()
    assert max(abs(value.pressure_mean) for value in resumed.diagnostics) < 1.0e-12
    assert max(value.velocity_mean_norm for value in resumed.diagnostics) < 1.0e-12


def test_periodic_compiler_rejects_unqualified_runtime_before_allocation(tmp_path):
    with pytest.raises(ValueError, match="periodic_spectral"):
        compile_simulation(
            _simulation(tmp_path, runtime_path="compiled_v2")
        )


def test_periodic_restart_rejects_tensor_tamper_before_advancing(tmp_path):
    first = run_simulation(
        _simulation(
            tmp_path,
            output_name="tamper_source",
            steps=1,
            checkpoint_interval=1,
        )
    )
    checkpoint = first.output_directory / "checkpoint_1"
    tensor = checkpoint / "spatial__Qxx.npy"
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
