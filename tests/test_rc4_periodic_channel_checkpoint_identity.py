"""RC4.2.3 Periodic/Channel production identity and legacy gates."""

from __future__ import annotations

from dataclasses import replace
import hashlib
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
)
from pssolver.applications.channel_beris_edwards import load_channel_initial_q
from pssolver.applications.periodic_beris_edwards import load_periodic_initial_q
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.configuration.simulation import InvocationSpec
from pssolver.geometries import PeriodicBox, RectangularChannel
from pssolver.io.checkpoint import seal_checkpoint_metadata
from pssolver.io.checkpoint_identity import SourceReleaseGeneration
from pssolver.models.active_nematics import (
    CompleteStressBerisEdwards,
    Q_COMPONENTS,
    VELOCITY_COMPONENTS,
)
from pssolver.runtime.channel_beris_edwards import (
    ChannelBerisEdwardsRuntimeBuildRequest,
    build_channel_beris_edwards_runtime,
)
from pssolver.runtime.periodic_beris_edwards import (
    PeriodicRuntimeBuildRequest,
    build_periodic_beris_edwards_runtime,
)
from pssolver.workflows import periodic_checkpoint as periodic_module
from pssolver.workflows.channel_beris_edwards import (
    CHECKPOINT_VERSION as CHANNEL_CHECKPOINT_VERSION,
    _load_checkpoint as load_channel_checkpoint,
    read_channel_beris_edwards_checkpoint_header,
    upgrade_channel_beris_edwards_checkpoint_directory,
    write_channel_beris_edwards_checkpoint,
)
from pssolver.workflows.periodic_checkpoint import (
    PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_V1,
    PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    capture_periodic_checkpoint,
    load_periodic_checkpoint,
    restore_periodic_checkpoint,
    upgrade_periodic_checkpoint_directory,
    write_periodic_checkpoint,
)


def _model():
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


def _snapshot(tmp_path: Path, name: str, shape: tuple[int, int, int]) -> Path:
    directory = tmp_path / name
    directory.mkdir()
    values = np.zeros((*shape, 5), dtype=np.float64)
    values[..., 0] = 0.2
    values[..., 3] = -0.1
    values += np.random.default_rng(4213).normal(
        scale=1.0e-3,
        size=values.shape,
    )
    np.save(directory / "Q_0.npy", values, allow_pickle=False)
    return directory


def _periodic_simulation(tmp_path: Path, storage: str, *, device="cpu"):
    shape = (6, 6, 4)
    model = _model()
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
    numerics = {
        "dtype": "float64",
        "dealias_rule": "cubic_half",
        "spectral_storage": storage,
    }
    if storage == "hermitian_half":
        numerics["hermitian_axis"] = 1
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=SpectralNumerics(**numerics),
        time=TimeStepping(dt=0.001, refresh={"mode": "disabled"}),
        initial_condition=SnapshotInitialCondition(
            _snapshot(tmp_path, f"periodic_{storage}_{device}", shape),
            step=0,
        ),
        execution=TorchSpectralExecution(
            runtime_path="periodic_spectral",
            device=device,
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "spectral",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": True,
            },
        ),
        output=Output(
            directory=tmp_path / f"periodic_output_{storage}_{device}",
            steps=1,
            save_interval=10,
            diagnostic_interval=1,
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
        ),
    )


def _periodic_runtime(simulation):
    compiled = compile_simulation(simulation)
    run_spec = compiled.application_request
    initial, _, _ = load_periodic_initial_q(run_spec)
    adapter = build_periodic_beris_edwards_runtime(
        PeriodicRuntimeBuildRequest(
            run_spec=run_spec,
            initial_values=initial,
            device="cpu",
        )
    )
    return run_spec, adapter


def _channel_simulation(tmp_path: Path, nx: int, *, device="cpu"):
    shape = (nx, 6, 6)
    model = _model()
    geometry = RectangularChannel(shape=shape, lengths=(float(nx), 6.0, 6.0))
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
        time=TimeStepping(dt=0.001),
        initial_condition=SnapshotInitialCondition(
            _snapshot(tmp_path, f"channel_{nx}_{device}", shape),
            step=0,
        ),
        execution=TorchSpectralExecution(
            runtime_path="channel_complete_stress",
            device=device,
            options={
                "tf32": "off",
                "molecular_field_linear_space": "spectral",
                "stress_divergence_sum_space": "physical",
                "pointwise_execution": "eager",
                "disable_q_gradient_reuse": False,
            },
        ),
        output=Output(
            directory=tmp_path / f"channel_output_{nx}_{device}",
            steps=1,
            save_interval=10,
            diagnostic_interval=1,
            save_start_step=0,
            diagnostics=True,
            save_hydrodynamics=True,
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
        invocation=InvocationSpec({}),
    )


def _channel_runtime(simulation):
    compiled = compile_simulation(simulation)
    run_spec = compiled.application_request
    initial, _, _ = load_channel_initial_q(run_spec)
    adapter = build_channel_beris_edwards_runtime(
        ChannelBerisEdwardsRuntimeBuildRequest(
            run_spec=run_spec,
            initial_values=initial,
            device="cpu",
        )
    )
    return run_spec, adapter


def _tree_sha256(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _periodic_legacy(checkpoint, generation):
    version = (
        PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_V1
        if generation is SourceReleaseGeneration.RC1
        else PERIODIC_LEGACY_WORKFLOW_CHECKPOINT_FORMAT_VERSION
    )
    return replace(
        checkpoint,
        format_version=version,
        compatibility_identity=None,
        migration_provenance=None,
    )


def _downgrade_channel(directory: Path, run_spec, generation) -> None:
    metadata_path = directory / "checkpoint.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata.pop("metadata_sha256", None)
    metadata.pop("compatibility_identity")
    metadata.pop("compatibility_identity_sha256")
    metadata.pop("run_provenance")
    metadata["runtime_identity_sha256"] = run_spec.runtime_identity_sha256()
    metadata["format_version"] = (
        1 if generation is SourceReleaseGeneration.RC1 else 2
    )
    if metadata["format_version"] == 2:
        metadata = seal_checkpoint_metadata(metadata)
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _assert_periodic_state_equal(left, right):
    assert all(
        torch.equal(left.fields[f"{name}{suffix}"], right.fields[f"{name}{suffix}"])
        for name in Q_COMPONENTS
        for suffix in ("", ".hat")
    )


def _assert_channel_state_equal(left, right):
    names = (*Q_COMPONENTS, *VELOCITY_COMPONENTS, "p")
    assert all(
        torch.equal(left.fields[f"{name}{suffix}"], right.fields[f"{name}{suffix}"])
        for name in names
        for suffix in ("", ".hat")
    )
    assert torch.equal(
        left.capture_pressure_guess(),
        right.capture_pressure_guess(),
    )


def test_m07_rc1_full_complex_rejects_before_periodic_tensor_load(
    tmp_path,
    monkeypatch,
):
    run_spec, source = _periodic_runtime(
        _periodic_simulation(tmp_path, "full_complex")
    )
    legacy = _periodic_legacy(
        capture_periodic_checkpoint(source, run_spec=run_spec),
        SourceReleaseGeneration.RC1,
    )
    directory = write_periodic_checkpoint(tmp_path / "periodic_rc1", legacy)
    monkeypatch.setattr(
        periodic_module,
        "_load_tensor",
        lambda *_args, **_kwargs: pytest.fail("tensor load started"),
    )

    with pytest.raises(ValueError, match="pre-repair Hermitian dynamics"):
        upgrade_periodic_checkpoint_directory(
            directory,
            tmp_path / "rejected",
            source_run_spec=run_spec,
            source_release_generation=SourceReleaseGeneration.RC1,
        )


def test_m08_rc1_hermitian_half_upgrades_and_restores(tmp_path):
    run_spec, source = _periodic_runtime(
        _periodic_simulation(tmp_path, "hermitian_half")
    )
    target_spec, target = _periodic_runtime(
        _periodic_simulation(tmp_path, "hermitian_half", device="auto")
    )
    source.advance(2)
    legacy = _periodic_legacy(
        capture_periodic_checkpoint(source, run_spec=run_spec),
        SourceReleaseGeneration.RC1,
    )
    directory = write_periodic_checkpoint(tmp_path / "periodic_rc1", legacy)
    before = _tree_sha256(directory)
    upgraded_path = upgrade_periodic_checkpoint_directory(
        directory,
        tmp_path / "periodic_v3",
        source_run_spec=run_spec,
        source_release_generation=SourceReleaseGeneration.RC1,
    )

    assert _tree_sha256(directory) == before
    upgraded = load_periodic_checkpoint(upgraded_path)
    assert upgraded.format_version == PERIODIC_WORKFLOW_CHECKPOINT_FORMAT_VERSION
    assert restore_periodic_checkpoint(target, upgraded, run_spec=target_spec) == 2
    _assert_periodic_state_equal(source, target)


@pytest.mark.parametrize(
    "generation",
    (SourceReleaseGeneration.RC2, SourceReleaseGeneration.RC3),
)
def test_m09_post_repair_periodic_checkpoint_upgrades(generation, tmp_path):
    run_spec, source = _periodic_runtime(
        _periodic_simulation(tmp_path, "full_complex")
    )
    legacy = _periodic_legacy(
        capture_periodic_checkpoint(source, run_spec=run_spec),
        generation,
    )
    directory = write_periodic_checkpoint(tmp_path / "legacy", legacy)
    upgraded = upgrade_periodic_checkpoint_directory(
        directory,
        tmp_path / "upgraded",
        source_run_spec=run_spec,
        source_release_generation=generation,
    )
    assert load_periodic_checkpoint(upgraded).migration_provenance[
        "hermitian_repair_applicability"
    ] == "source_is_post_repair"


def test_m10_rc1_even_channel_rejects_before_tensor_load(tmp_path, monkeypatch):
    run_spec, source = _channel_runtime(_channel_simulation(tmp_path, 6))
    directory = tmp_path / "channel_rc1"
    write_channel_beris_edwards_checkpoint(directory, source, run_spec=run_spec)
    _downgrade_channel(directory, run_spec, SourceReleaseGeneration.RC1)
    monkeypatch.setattr(
        "pssolver.workflows.channel_beris_edwards.np.load",
        lambda *_args, **_kwargs: pytest.fail("tensor load started"),
    )

    with pytest.raises(ValueError, match="pre-RC4.1 forward Nyquist dynamics"):
        upgrade_channel_beris_edwards_checkpoint_directory(
            directory,
            tmp_path / "rejected",
            source_run_spec=run_spec,
            source_release_generation=SourceReleaseGeneration.RC1,
        )


def test_m11_rc1_odd_channel_upgrades_and_restores(tmp_path):
    run_spec, source = _channel_runtime(_channel_simulation(tmp_path, 7))
    target_spec, target = _channel_runtime(
        _channel_simulation(tmp_path, 7, device="auto")
    )
    source.advance(1)
    directory = tmp_path / "channel_rc1"
    write_channel_beris_edwards_checkpoint(directory, source, run_spec=run_spec)
    _downgrade_channel(directory, run_spec, SourceReleaseGeneration.RC1)
    before = _tree_sha256(directory)
    upgraded = upgrade_channel_beris_edwards_checkpoint_directory(
        directory,
        tmp_path / "channel_v3",
        source_run_spec=run_spec,
        source_release_generation=SourceReleaseGeneration.RC1,
    )

    assert _tree_sha256(directory) == before
    header = read_channel_beris_edwards_checkpoint_header(upgraded)
    assert header.format_version == CHANNEL_CHECKPOINT_VERSION
    assert load_channel_checkpoint(upgraded, target, run_spec=target_spec) == 1
    _assert_channel_state_equal(source, target)


@pytest.mark.parametrize(
    "generation",
    (SourceReleaseGeneration.RC2, SourceReleaseGeneration.RC3),
)
def test_m12_post_nyquist_channel_checkpoint_upgrades(generation, tmp_path):
    run_spec, source = _channel_runtime(_channel_simulation(tmp_path, 6))
    directory = tmp_path / "legacy"
    write_channel_beris_edwards_checkpoint(directory, source, run_spec=run_spec)
    _downgrade_channel(directory, run_spec, generation)
    upgraded = upgrade_channel_beris_edwards_checkpoint_directory(
        directory,
        tmp_path / "upgraded",
        source_run_spec=run_spec,
        source_release_generation=generation,
    )
    metadata = json.loads((upgraded / "checkpoint.json").read_text())
    assert metadata["migration_provenance"][
        "forward_nyquist_applicability"
    ] == "source_is_post_nyquist_repair"


def test_unknown_periodic_and_channel_versions_reject_before_tensor_load(
    tmp_path,
    monkeypatch,
):
    periodic_spec, periodic = _periodic_runtime(
        _periodic_simulation(tmp_path, "hermitian_half")
    )
    periodic_dir = write_periodic_checkpoint(
        tmp_path / "periodic",
        capture_periodic_checkpoint(periodic, run_spec=periodic_spec),
    )
    channel_spec, channel = _channel_runtime(_channel_simulation(tmp_path, 7))
    channel_dir = tmp_path / "channel"
    write_channel_beris_edwards_checkpoint(
        channel_dir,
        channel,
        run_spec=channel_spec,
    )
    for directory in (periodic_dir, channel_dir):
        path = directory / "checkpoint.json"
        metadata = json.loads(path.read_text())
        metadata["format_version"] = 99
        path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(periodic_module, "_load_tensor", lambda *_a: pytest.fail())
    monkeypatch.setattr(
        "pssolver.workflows.channel_beris_edwards.np.load",
        lambda *_a: pytest.fail(),
    )

    with pytest.raises(ValueError, match="unsupported periodic"):
        load_periodic_checkpoint(periodic_dir)
    with pytest.raises(ValueError, match="unsupported complete-stress Channel"):
        read_channel_beris_edwards_checkpoint_header(channel_dir)


def test_current_identity_digest_tamper_rejects_before_tensor_load(
    tmp_path,
    monkeypatch,
):
    periodic_spec, periodic = _periodic_runtime(
        _periodic_simulation(tmp_path, "hermitian_half")
    )
    periodic_dir = write_periodic_checkpoint(
        tmp_path / "periodic",
        capture_periodic_checkpoint(periodic, run_spec=periodic_spec),
    )
    channel_spec, channel = _channel_runtime(_channel_simulation(tmp_path, 7))
    channel_dir = tmp_path / "channel"
    write_channel_beris_edwards_checkpoint(
        channel_dir,
        channel,
        run_spec=channel_spec,
    )
    for directory in (periodic_dir, channel_dir):
        path = directory / "checkpoint.json"
        metadata = json.loads(path.read_text())
        metadata.pop("metadata_sha256")
        metadata["compatibility_identity_sha256"] = "0" * 64
        metadata = seal_checkpoint_metadata(metadata)
        path.write_text(json.dumps(metadata), encoding="utf-8")
    monkeypatch.setattr(periodic_module, "_load_tensor", lambda *_a: pytest.fail())
    monkeypatch.setattr(
        "pssolver.workflows.channel_beris_edwards.np.load",
        lambda *_a: pytest.fail(),
    )

    with pytest.raises(ValueError, match="identity checksum mismatch"):
        load_periodic_checkpoint(periodic_dir)
    with pytest.raises(ValueError, match="identity checksum mismatch"):
        read_channel_beris_edwards_checkpoint_header(channel_dir)
