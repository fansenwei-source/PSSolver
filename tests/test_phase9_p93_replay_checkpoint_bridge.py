"""P9.3 deterministic replay and periodic durable-state bridge tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import stat

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
)
from pssolver.applications.periodic_beris_edwards import load_periodic_initial_q
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.configuration.public_simulation_runner import compile_public_simulation
from pssolver.functional import (
    FunctionalCheckpointBridgeProtocol,
    FunctionalRuntimeIdentity,
    PeriodicActivityCheckpointBridge,
    build_functional_runtime,
    periodic_activity_functional_request,
)
from pssolver.geometries import PeriodicBox
from pssolver.io.checkpoint import (
    seal_checkpoint_metadata,
    write_functional_checkpoint_provenance,
)
from pssolver.models.active_nematics import CompleteStressBerisEdwards, Q_COMPONENTS
from pssolver.runtime.periodic_beris_edwards import (
    PeriodicRuntimeBuildRequest,
    build_periodic_beris_edwards_runtime,
)
from pssolver.workflows.periodic_checkpoint import (
    capture_periodic_checkpoint,
    load_periodic_checkpoint,
    restore_periodic_checkpoint,
    write_periodic_checkpoint,
)


def _simulation(tmp_path: Path, *, name: str = "p93") -> Simulation:
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
        friction=0.0,
        tangential_zero_mode_policy="zero_mean",
    )
    shape = (6, 6, 4)
    geometry = PeriodicBox(shape=shape, lengths=(6.0, 6.0, 4.0))
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    rng = np.random.default_rng(20260928)
    q = rng.normal(scale=0.02, size=(*shape, 5)).astype(np.float64)
    q[..., 0] += 0.2
    q[..., 3] -= 0.1
    source = tmp_path / f"{name}_source"
    source.mkdir()
    np.save(source / "Q_0.npy", q, allow_pickle=False)
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
        initial_condition=SnapshotInitialCondition(source, step=0),
        execution=TorchSpectralExecution(
            runtime_path="periodic_spectral",
            device="cpu",
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": True,
            },
        ),
        output=Output(
            directory=tmp_path / name,
            steps=1,
            save_interval=10,
            diagnostic_interval=1,
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
    )


def _runtime(tmp_path: Path):
    simulation = _simulation(tmp_path)
    request = periodic_activity_functional_request(simulation.specification)
    return simulation, request, build_functional_runtime(request)


def _control(runtime, value: float) -> dict[str, torch.Tensor]:
    spec = runtime.control_specs[0].tensor
    result = torch.full(spec.shape, value, dtype=torch.float64, device=spec.device)
    result[:, :3, :, :] *= 1.25
    return {"activity": result}


def _assert_state_equal(left, right):
    assert len(left) == len(right)
    assert all(torch.equal(a, b) for a, b in zip(left, right, strict=True))


def _advance(runtime, state, controls):
    for index, control in enumerate(controls):
        state = runtime.step(state, control, index)
    return state


def _production_adapter(simulation):
    compiled = compile_public_simulation(simulation.specification)
    initial_values, _, _ = load_periodic_initial_q(compiled.run_spec)
    adapter = build_periodic_beris_edwards_runtime(
        PeriodicRuntimeBuildRequest(
            run_spec=compiled.run_spec,
            initial_values=initial_values,
            device="cpu",
        )
    )
    return compiled.run_spec, adapter


def _adapter_state(adapter):
    fields = adapter.fields
    return (
        torch.stack(tuple(fields[name] for name in Q_COMPONENTS), dim=0),
        torch.stack(tuple(fields[f"{name}.hat"] for name in Q_COMPONENTS), dim=0),
    )


def test_p93_declares_bitwise_replay_and_versioned_bridge(tmp_path):
    _, _, runtime = _runtime(tmp_path)

    assert runtime.capabilities.deterministic_replay == "bitwise"
    assert runtime.capabilities.durable_checkpoint_bridge is True
    assert isinstance(runtime.checkpoint_bridge, FunctionalCheckpointBridgeProtocol)
    assert isinstance(runtime.checkpoint_bridge, PeriodicActivityCheckpointBridge)
    assert runtime.checkpoint_bridge.format_version == 3


def test_p93_replay_is_bitwise_after_intervening_calls_and_fresh_construction(
    tmp_path,
):
    simulation, request, runtime = _runtime(tmp_path)
    initial = runtime.initial_state()
    controls = [_control(runtime, value) for value in (0.01, 0.013, 0.02)]

    reference_next, reference_observation = runtime.step_and_observe(
        initial,
        controls[0],
        0,
    )
    unrelated = tuple(value + 1.0e-5 for value in initial)
    runtime.step_and_observe(unrelated, controls[2], 19)
    replay_next, replay_observation = runtime.step_and_observe(
        initial,
        controls[0],
        0,
    )

    _assert_state_equal(reference_next, replay_next)
    assert all(
        torch.equal(reference_observation[name], replay_observation[name])
        for name in reference_observation
    )

    first = _advance(runtime, initial, controls)
    runtime.observe(unrelated, controls[1])
    second = _advance(runtime, initial, controls)
    fresh = build_functional_runtime(request)
    third = _advance(fresh, fresh.initial_state(), controls)
    _assert_state_equal(first, second)
    _assert_state_equal(first, third)
    assert runtime.identity().canonical_sha256() == fresh.identity().canonical_sha256()
    assert simulation.specification == request.simulation


def test_p93_functional_export_is_v3_and_requires_explicit_production_migration(
    tmp_path,
):
    simulation, _, runtime = _runtime(tmp_path)
    uniform = {
        "activity": torch.full(
            runtime.control_specs[0].tensor.shape,
            0.01,
            dtype=torch.float64,
        )
    }
    state = runtime.initial_state()
    for step in range(3):
        state = runtime.step(state, uniform, step)
    before = tuple(value.clone() for value in state)
    directory = tmp_path / "functional_checkpoint"
    written = runtime.checkpoint_bridge.export_checkpoint(
        directory,
        state,
        completed_steps=3,
    )

    assert written == directory.resolve()
    assert stat.S_IMODE(directory.stat().st_mode) == 0o755
    _assert_state_equal(state, before)
    metadata = json.loads((directory / "checkpoint.json").read_text())
    assert metadata["format_version"] == 2
    assert metadata["functional_bridge"]["format_version"] == 3
    assert "checkpoint_compatibility_identity" in metadata["functional_bridge"]
    assert metadata["completed_steps"] == 3

    imported = runtime.checkpoint_bridge.import_checkpoint(directory)
    assert imported.completed_steps == 3
    assert imported.source_format == "periodic_functional_bridge_v3"
    _assert_state_equal(imported.state, state)

    run_spec, production = _production_adapter(simulation)
    checkpoint = load_periodic_checkpoint(directory)
    with pytest.raises(ValueError, match="explicit identity upgrade"):
        restore_periodic_checkpoint(
            production,
            checkpoint,
            run_spec=run_spec,
        )
    assert production.completed_steps == 0


def test_p93_imports_an_existing_production_checkpoint_without_repacking_loss(
    tmp_path,
):
    simulation, _, runtime = _runtime(tmp_path)
    run_spec, production = _production_adapter(simulation)
    production.advance(2)
    expected = tuple(value.detach().clone() for value in _adapter_state(production))
    directory = tmp_path / "production_checkpoint"
    write_periodic_checkpoint(
        directory,
        capture_periodic_checkpoint(
            production,
            run_spec=run_spec,
        ),
    )

    imported = runtime.checkpoint_bridge.import_checkpoint(directory)

    assert imported.completed_steps == 2
    assert imported.source_format == "periodic_production_v3"
    _assert_state_equal(imported.state, expected)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        (
            "checkpoint_compatibility_sha256",
            "0" * 64,
            "metadata checksum mismatch",
        ),
        ("api_version", "9.9", "metadata checksum mismatch"),
    ),
)
def test_p93_bridge_identity_fails_before_any_tensor_payload_is_loaded(
    tmp_path,
    monkeypatch,
    field,
    replacement,
    message,
):
    _, _, runtime = _runtime(tmp_path)
    directory = tmp_path / f"bad_{field}"
    runtime.checkpoint_bridge.export_checkpoint(
        directory,
        runtime.initial_state(),
        completed_steps=0,
    )
    metadata_path = directory / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["functional_bridge"][field] = replacement
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")

    def forbidden(*_args, **_kwargs):
        raise AssertionError("np.load must not run before identity preflight")

    monkeypatch.setattr(
        "pssolver.workflows.periodic_checkpoint.np.load",
        forbidden,
    )
    with pytest.raises(ValueError, match=message):
        runtime.checkpoint_bridge.import_checkpoint(directory)


def test_p93_bridge_rejects_checksum_nonfinite_and_existing_destination(tmp_path):
    _, _, runtime = _runtime(tmp_path)
    state = runtime.initial_state()
    checksum = tmp_path / "checksum"
    runtime.checkpoint_bridge.export_checkpoint(checksum, state, completed_steps=0)
    tensor_path = checksum / "spatial__Qxx.npy"
    payload = bytearray(tensor_path.read_bytes())
    payload[-1] ^= 1
    tensor_path.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum mismatch"):
        runtime.checkpoint_bridge.import_checkpoint(checksum)

    nonfinite = tmp_path / "nonfinite"
    runtime.checkpoint_bridge.export_checkpoint(nonfinite, state, completed_steps=0)
    tensor_path = nonfinite / "spatial__Qxx.npy"
    values = np.load(tensor_path, allow_pickle=False)
    values.flat[0] = np.nan
    np.save(tensor_path, values, allow_pickle=False)
    metadata_path = nonfinite / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["tensor_files"]["spatial"]["Qxx"]["sha256"] = hashlib.sha256(
        tensor_path.read_bytes()
    ).hexdigest()
    metadata.pop("metadata_sha256")
    metadata = seal_checkpoint_metadata(metadata)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    write_functional_checkpoint_provenance(
        nonfinite,
        metadata,
        metadata["functional_bridge"],
    )
    with pytest.raises(ValueError, match="NaN or Inf"):
        runtime.checkpoint_bridge.import_checkpoint(nonfinite)

    with pytest.raises(FileExistsError, match="already exists"):
        runtime.checkpoint_bridge.export_checkpoint(
            nonfinite,
            state,
            completed_steps=0,
        )


@pytest.mark.parametrize("mutation", ("completed_steps", "functional_bridge"))
def test_p93_functional_metadata_tamper_cannot_fall_back_to_production(
    tmp_path,
    monkeypatch,
    mutation,
):
    _, _, runtime = _runtime(tmp_path)
    directory = tmp_path / f"metadata_{mutation}"
    runtime.checkpoint_bridge.export_checkpoint(
        directory,
        runtime.initial_state(),
        completed_steps=0,
    )
    metadata_path = directory / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if mutation == "completed_steps":
        metadata["completed_steps"] = 999
    else:
        del metadata["functional_bridge"]
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("np.load must not run before metadata preflight")

    monkeypatch.setattr(
        "pssolver.workflows.periodic_checkpoint.np.load",
        forbidden,
    )
    with pytest.raises(ValueError, match="metadata checksum mismatch"):
        runtime.checkpoint_bridge.import_checkpoint(directory)


def test_p93_cross_identity_checkpoint_is_rejected(tmp_path, monkeypatch):
    _, _, runtime = _runtime(tmp_path)
    directory = tmp_path / "cross_identity"
    runtime.checkpoint_bridge.export_checkpoint(
        directory,
        runtime.initial_state(),
        completed_steps=0,
    )

    identity_metadata = runtime.identity().to_metadata()
    incompatible_scientific = dict(identity_metadata["scientific"])
    incompatible_scientific["incompatible_test_model"] = True
    incompatible_identity = FunctionalRuntimeIdentity(
        scientific=incompatible_scientific,
        discretization=identity_metadata["discretization"],
        execution=identity_metadata["execution"],
        state_layout=identity_metadata["state_layout"],
    )
    other = PeriodicActivityCheckpointBridge(
        state_spec=runtime.state_spec,
        functional_identity=incompatible_identity,
        production_runtime_identity_sha256="0" * 64,
        backend_restart={"kind": "periodic_stokes_stateless", "state_keys": []},
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("np.load must not run before runtime identity preflight")

    monkeypatch.setattr(
        "pssolver.workflows.periodic_checkpoint.np.load",
        forbidden,
    )
    with pytest.raises(ValueError, match="forward compatibility differs"):
        other.import_checkpoint(directory)
