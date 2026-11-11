"""S-03 compatibility gates for public input and restart exceptions."""

from __future__ import annotations

from pathlib import Path

import pytest

import pssolver
from pssolver import compile_simulation, run_simulation
from pssolver.configuration import legacy_channel as channel_application
from pssolver.configuration import legacy_periodic as periodic_application
from pssolver.functional import (
    FunctionalCheckpointError,
    FunctionalCheckpointIOError,
    PeriodicActivityCheckpointBridge,
)
from pssolver.io.checkpoint import (
    CheckpointError,
    CheckpointIntegrityError,
    CheckpointIOError,
    CheckpointNotFoundError,
)
from pssolver.runtime import legacy_entry as channel_workflow
from pssolver.runtime import legacy_entry as periodic_workflow
from pssolver.runtime import legacy_entry as plane_application
from tests.test_phase8_p82_periodic_application import (
    _simulation as periodic_simulation,
)
from tests.test_phase8_p83_channel_complete_stress_application import (
    _simulation as channel_simulation,
)
from tests.test_rc5_r513_optional_hygiene import _plane_spec


@pytest.mark.parametrize(
    ("module", "simulation_factory", "loader"),
    (
        (
            periodic_application,
            periodic_simulation,
            periodic_application.load_periodic_initial_q,
        ),
        (
            channel_application,
            channel_simulation,
            channel_application.load_channel_initial_q,
        ),
    ),
)
def test_missing_snapshot_is_both_public_input_and_file_not_found(
    tmp_path,
    module,
    simulation_factory,
    loader,
):
    simulation = simulation_factory(tmp_path, output_name=f"missing_{module.__name__}")
    compiled = compile_simulation(simulation)
    snapshot = Path(
        compiled.source.initial_condition.parameters["directory"]
    ) / "Q_0.npy"
    snapshot.unlink()

    with pytest.raises(pssolver.SimulationInputFileNotFoundError) as caught:
        loader(compiled.application_request)

    assert isinstance(caught.value, pssolver.SimulationInputError)
    assert isinstance(caught.value, OSError)
    assert isinstance(caught.value, FileNotFoundError)
    assert caught.value.details["legacy_exception_type"] == "FileNotFoundError"


@pytest.mark.parametrize(
    ("module", "simulation_factory", "loader"),
    (
        (
            periodic_application,
            periodic_simulation,
            periodic_application.load_periodic_initial_q,
        ),
        (
            channel_application,
            channel_simulation,
            channel_application.load_channel_initial_q,
        ),
    ),
)
def test_unreadable_snapshot_is_both_public_input_and_os_error(
    tmp_path,
    monkeypatch,
    module,
    simulation_factory,
    loader,
):
    compiled = compile_simulation(
        simulation_factory(tmp_path, output_name=f"unreadable_{module.__name__}")
    )

    def reject(*_args, **_kwargs):
        raise ValueError("invalid npy header")

    monkeypatch.setattr(module.np, "load", reject)
    with pytest.raises(pssolver.SimulationInputIOError) as caught:
        loader(compiled.application_request)

    assert isinstance(caught.value, pssolver.SimulationInputError)
    assert isinstance(caught.value, OSError)
    assert isinstance(caught.value, ValueError)
    assert caught.value.details["legacy_exception_type"] == "ValueError"


@pytest.mark.parametrize(
    ("module", "simulation_factory", "loader"),
    (
        (
            periodic_application,
            periodic_simulation,
            periodic_application.load_periodic_initial_q,
        ),
        (
            channel_application,
            channel_simulation,
            channel_application.load_channel_initial_q,
        ),
    ),
)
def test_snapshot_hash_io_failure_retains_os_error_compatibility(
    tmp_path,
    monkeypatch,
    module,
    simulation_factory,
    loader,
):
    compiled = compile_simulation(
        simulation_factory(tmp_path, output_name=f"hash_{module.__name__}")
    )

    def reject(_path):
        raise PermissionError("snapshot became unreadable")

    hash_name = "_snapshot_sha256" if module is channel_application else "_sha256"
    monkeypatch.setattr(module, hash_name, reject)
    with pytest.raises(pssolver.SimulationInputIOError) as caught:
        loader(compiled.application_request)

    assert isinstance(caught.value, OSError)
    assert caught.value.details["legacy_exception_type"] == "PermissionError"


@pytest.mark.parametrize(
    ("simulation_factory", "output_name"),
    (
        (periodic_simulation, "periodic_missing_restart"),
        (channel_simulation, "channel_missing_restart"),
    ),
)
def test_application_restart_preflight_preserves_checkpoint_not_found(
    tmp_path,
    simulation_factory,
    output_name,
):
    output = tmp_path / output_name
    simulation = simulation_factory(
        tmp_path,
        output_name=output_name,
        steps=1,
        restart_from=tmp_path / "missing_checkpoint",
    )

    with pytest.raises(CheckpointNotFoundError) as caught:
        run_simulation(simulation)

    assert isinstance(caught.value, CheckpointError)
    assert isinstance(caught.value, OSError)
    assert isinstance(caught.value, FileNotFoundError)
    assert not output.exists()


def test_plane_application_preflight_preserves_checkpoint_integrity_error(
    tmp_path,
    monkeypatch,
):
    sentinel = CheckpointIntegrityError("corrupt Plane header")
    spec = _plane_spec(
        tmp_path,
        restart_from=tmp_path / "checkpoint",
        dry_run=True,
    )

    def reject(_path):
        raise sentinel

    monkeypatch.setattr(plane_application, "read_plane_checkpoint_header", reject)
    monkeypatch.setattr("pssolver.io.checkpoint.read_checkpoint_json", lambda *_args, **_kwargs: {})
    with pytest.raises(CheckpointIntegrityError) as caught:
        plane_application.run_v04_plane_translation(spec)

    assert caught.value is sentinel
    assert isinstance(caught.value, CheckpointError)
    assert isinstance(caught.value, ValueError)


@pytest.mark.parametrize(
    ("module", "workflow_type"),
    (
        (periodic_workflow, periodic_workflow.restore_periodic_state),
        (channel_workflow, channel_workflow.restore_channel_state),
    ),
)
def test_complete_stress_workflow_restore_preserves_checkpoint_error(
    tmp_path,
    monkeypatch,
    module,
    workflow_type,
):
    sentinel = CheckpointIntegrityError("corrupt complete-stress checkpoint")
    if workflow_type is periodic_workflow.restore_periodic_state:
        from tests.v0_5_s25_periodic_v04_checkpoint import production_handle

        spec, handle = production_handle(periodic_simulation(tmp_path))
        def invoke():
            return workflow_type(handle.engine, handle.state, None, run_spec=spec)

        patch_name = "validate_periodic_restore"
    else:
        import torch

        from pssolver.engine.engine import Engine

        compiled = compile_simulation(channel_simulation(tmp_path))
        spec = compiled.application_request
        engine = Engine(channel_application.translate_channel(spec), device="cpu", dtype=torch.float64)
        initial, _, _ = channel_application.load_channel_initial_q(spec)
        state = engine.initial_state(initial)

        def invoke():
            return workflow_type(engine, state, tmp_path / "checkpoint", run_spec=spec)

        patch_name = "load_v04_channel_checkpoint"

    def reject(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(module, patch_name, reject)
    with pytest.raises(CheckpointIntegrityError) as caught:
        invoke()

    assert caught.value is sentinel
    assert isinstance(caught.value, CheckpointError)
    assert isinstance(caught.value, ValueError)


def test_periodic_functional_export_uses_functional_translation(monkeypatch):
    sentinel = CheckpointIOError("filesystem rejected checkpoint export")
    bridge = object.__new__(PeriodicActivityCheckpointBridge)

    def reject(*_args, **_kwargs):
        raise sentinel

    monkeypatch.setattr(bridge, "_write_state_checkpoint", reject)
    with pytest.raises(FunctionalCheckpointIOError) as caught:
        bridge.export_checkpoint("unused", (), completed_steps=0)

    assert isinstance(caught.value, FunctionalCheckpointError)
    assert isinstance(caught.value, CheckpointIOError)
    assert isinstance(caught.value, OSError)
    assert caught.value.__cause__ is sentinel
