"""R5-7 shared execution-device normalization and side-effect boundaries."""

from __future__ import annotations

import json
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
    compile_simulation,
    run_simulation,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.applications.channel_active_nematics import (
    run_channel_active_nematics,
)
from pssolver.applications.plane_beris_edwards import run_plane_beris_edwards
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.execution_device import (
    execution_device_normalization,
    resolve_allocated_execution_device,
)
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.functional import (
    FunctionalTensorSpec,
    FunctionalValueError,
    periodic_activity_functional_request,
)
from pssolver.geometries import PeriodicBox, RectangularChannel
from pssolver.models.active_nematics import CompleteStressBerisEdwards


CANONICAL_DEVICES = ("auto", "cpu", "cuda", "cuda:0", "cuda:2")
INVALID_DEVICES = ("", " cpu", "cpu ", "cpu:0", "cuda:01", "CUDA", "mps")


def _simulation(
    tmp_path: Path,
    *,
    family: str,
    device: str,
    output_name: str,
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
    periodic = family == "periodic"
    geometry = (
        PeriodicBox(shape=(8, 8, 8), lengths=(8.0, 8.0, 8.0))
        if periodic
        else RectangularChannel(shape=(8, 8, 8), lengths=(8.0, 8.0, 8.0))
    )
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": free_slip_velocity() if periodic else no_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    snapshot = tmp_path / f"{output_name}_snapshot"
    snapshot.mkdir()
    values = np.zeros((8, 8, 8, 5), dtype=np.float64)
    values[..., 0] = 0.2
    values[..., 3] = -0.1
    np.save(snapshot / "Q_0.npy", values, allow_pickle=False)
    options = {
        "tf32": "off",
        "molecular_field_linear_space": "spectral",
        "stress_divergence_sum_space": "spectral" if periodic else "physical",
        "pointwise_execution": "eager",
        "disable_q_gradient_reuse": False,
    }
    discretization = {}
    if not periodic:
        discretization["pressure_solver"] = {
            "algorithm": "preconditioned_conjugate_gradient",
            "relative_tolerance": 1.0e-10,
            "max_iterations": 40,
            "fixed_iterations": 12,
            "warm_start": True,
        }
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
            projected_transform_execution="truncated",
            spectral_storage="hermitian_half" if periodic else "full_complex",
            hermitian_axis=1 if periodic else None,
        ),
        time=TimeStepping(dt=0.001),
        initial_condition=SnapshotInitialCondition(snapshot, step=0),
        execution=TorchSpectralExecution(
            runtime_path=(
                "periodic_spectral" if periodic else "channel_complete_stress"
            ),
            device=device,
            options=options,
        ),
        output=Output(
            directory=tmp_path / output_name,
            steps=1,
            save_start_step=0,
            save_interval=10,
            diagnostic_interval=1,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
        discretization=discretization,
    )


@pytest.mark.parametrize("device", CANONICAL_DEVICES)
def test_legacy_declarations_accept_the_shared_canonical_device_grammar(
    tmp_path,
    device,
):
    plane = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=tmp_path / "plane",
        device=device,
    )
    channel = ChannelActiveNematicRunSpec(device=device)
    assert plane.device == device
    assert channel.device == device


@pytest.mark.parametrize("device", INVALID_DEVICES)
def test_legacy_declarations_reject_every_noncanonical_device_token(
    tmp_path,
    device,
):
    with pytest.raises(ValueError, match="invalid execution device"):
        create_plane_beris_edwards_run_spec(
            activity_number=18.0,
            output_dir=tmp_path / "plane",
            device=device,
        )
    with pytest.raises(ValueError, match="invalid execution device"):
        ChannelActiveNematicRunSpec(device=device)


def test_auto_normalization_and_functional_allocation_agree_on_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert execution_device_normalization("auto") == {
        "requested_execution_device": "auto",
        "resolved_execution_device": "cpu",
    }
    assert resolve_allocated_execution_device("auto") == torch.device("cpu")


@pytest.mark.parametrize("device", ("auto", "cpu:0", "cuda"))
def test_functional_tensor_identity_rejects_nonconcrete_or_noncanonical_device(
    device,
):
    with pytest.raises(FunctionalValueError):
        FunctionalTensorSpec(
            name="state",
            shape=(1,),
            dtype="float64",
            device=device,
            batch_axis=0,
            layout="batch",
            meaning="test state",
        )


def test_periodic_functional_request_accepts_auto_and_records_cpu(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    simulation = _simulation(
        tmp_path,
        family="periodic",
        device="auto",
        output_name="functional_auto",
    )
    request = periodic_activity_functional_request(simulation.specification)
    assert request.control_fields[0].tensor.device == "cpu"
    assert request.observations[0].tensor.device == "cpu"


@pytest.mark.parametrize("family", ("periodic", "channel"))
def test_cuda_unavailable_runtime_construction_leaves_no_output_directory(
    monkeypatch,
    tmp_path,
    family,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output_name = f"{family}_cuda"
    simulation = _simulation(
        tmp_path,
        family=family,
        device="cuda",
        output_name=output_name,
    )
    module = (
        "pssolver.applications.periodic_beris_edwards"
        if family == "periodic"
        else "pssolver.applications.channel_beris_edwards"
    )
    monkeypatch.setattr(
        f"{module}.build_package_simulation_runtime",
        lambda construction: (_ for _ in ()).throw(
            RuntimeError("simulated unavailable CUDA runtime")
        ),
    )
    compiled = compile_simulation(simulation)
    with pytest.raises(RuntimeError, match="unavailable CUDA"):
        run_simulation(compiled)
    assert not (tmp_path / output_name).exists()


@pytest.mark.parametrize("family", ("periodic", "channel"))
def test_output_metadata_persists_requested_and_resolved_device(
    monkeypatch,
    tmp_path,
    family,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output_name = f"{family}_auto"
    result = run_simulation(
        _simulation(
            tmp_path,
            family=family,
            device="auto",
            output_name=output_name,
        )
    )
    metadata = json.loads(
        (result.output_directory / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["normalization"] == {
        "requested_execution_device": "auto",
        "resolved_execution_device": "cpu",
    }


def test_legacy_plane_output_metadata_persists_device_normalization(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "legacy_plane_auto"
    request = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=output,
        nx=8,
        ny=8,
        nz=8,
        steps=1,
        save_start_step=0,
        save_interval=1,
        diagnostic_interval=1,
        device="auto",
        dtype="float64",
        pointwise_execution="eager",
        disable_spectral_refresh=True,
    )
    run_plane_beris_edwards(request)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["normalization"] == {
        "requested_execution_device": "auto",
        "resolved_execution_device": "cpu",
    }


def test_legacy_channel_output_metadata_persists_device_normalization(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    output = tmp_path / "legacy_channel_auto"
    request = ChannelActiveNematicRunSpec(
        shape=(8, 6, 5),
        lengths=(8.0, 6.0, 5.0),
        dt=1.0e-3,
        steps=1,
        save_interval=1,
        diagnostic_interval=1,
        generated_output_directory=output,
        device="auto",
        activity=0.2,
        pressure_relative_tolerance=1.0e-8,
        pressure_max_iterations=100,
        pressure_fixed_iterations=8,
    )
    run_channel_active_nematics(request)
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["normalization"] == {
        "requested_execution_device": "auto",
        "resolved_execution_device": "cpu",
    }
