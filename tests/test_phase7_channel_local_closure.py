"""P7.5 package application, output, and same-runtime restart closure."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pssolver.applications.channel_active_nematics import run_channel_active_nematics
from pssolver.configuration.channel_active_nematics import ChannelActiveNematicRunSpec
from pssolver.workflows import load_channel_checkpoint


SHAPE = (8, 6, 5)
LENGTHS = (8.0, 6.0, 5.0)


def _spec(output: Path, runtime: str, **overrides) -> ChannelActiveNematicRunSpec:
    values = {
        "shape": SHAPE,
        "lengths": LENGTHS,
        "dt": 1.0e-3,
        "steps": 1,
        "save_interval": 100,
        "diagnostic_interval": 10,
        "generated_output_directory": output,
        "device": "cpu",
        "activity": 0.2,
        "pressure_relative_tolerance": 1.0e-8,
        "pressure_max_iterations": 100,
        "pressure_fixed_iterations": 8,
        "runtime_path": runtime,
    }
    values.update(overrides)
    return ChannelActiveNematicRunSpec(**values)


def _load_outputs(directory: Path, step: int):
    return tuple(np.load(directory / f"{name}_{step}.npy", allow_pickle=False) for name in ("Q", "u", "p"))


@pytest.mark.parametrize("steps", (1, 100))
def test_package_application_paths_are_byte_identical(tmp_path, steps):
    legacy_dir = tmp_path / f"legacy_{steps}"
    compiled_dir = tmp_path / f"compiled_{steps}"
    legacy = run_channel_active_nematics(_spec(legacy_dir, "legacy_channel", steps=steps))
    compiled = run_channel_active_nematics(_spec(compiled_dir, "compiled_channel_v2", steps=steps))

    assert legacy.final_step == compiled.final_step == steps
    for left, right in zip(_load_outputs(legacy_dir, steps), _load_outputs(compiled_dir, steps), strict=True):
        assert left.dtype == right.dtype == np.float32
        assert np.array_equal(left, right)
    for directory, runtime in ((legacy_dir, "legacy_channel"), (compiled_dir, "compiled_channel_v2")):
        metadata = json.loads((directory / "metadata.json").read_text())
        assert (directory / "COMPLETE").read_text() == "complete\n"
        assert metadata["status"] == "complete"
        assert metadata["completed_steps"] == steps
        assert metadata["runtime_selection"]["requested"] == runtime
        assert metadata["runtime_selection"]["effective"] == runtime
        assert metadata["runtime_selection"]["fallback_used"] is False
        assert np.isfinite(np.load(directory / "diagnostics.npy", allow_pickle=False).view(np.recarray).div_rel).all()


@pytest.mark.parametrize("runtime", ("legacy_channel", "compiled_channel_v2"))
def test_same_runtime_workflow_restart_matches_continuous(tmp_path, runtime):
    continuous_dir = tmp_path / f"continuous_{runtime}"
    segment_dir = tmp_path / f"segment_{runtime}"
    resumed_dir = tmp_path / f"resumed_{runtime}"
    run_channel_active_nematics(_spec(continuous_dir, runtime, steps=20, save_interval=20))
    segment = run_channel_active_nematics(_spec(segment_dir, runtime, steps=7, save_interval=7, checkpoint_interval=7))
    checkpoint_dir = segment_dir / "checkpoint_7"
    checkpoint = load_channel_checkpoint(checkpoint_dir)
    captured_pressure = checkpoint.pressure_guess.clone()
    resumed = run_channel_active_nematics(_spec(resumed_dir, runtime, steps=13, save_interval=20, restart_from=checkpoint_dir))

    assert segment.final_step == checkpoint.completed_steps == 7
    assert resumed.start_step == 7
    assert resumed.final_step == 20
    assert np.isfinite(captured_pressure.numpy()).all()
    for left, right in zip(_load_outputs(continuous_dir, 20), _load_outputs(resumed_dir, 20), strict=True):
        assert np.array_equal(left, right)
    metadata = json.loads((resumed_dir / "metadata.json").read_text())
    assert metadata["restart"]["source_runtime_path"] == runtime
    assert metadata["restart"]["source_completed_steps"] == 7
    assert metadata["restart"]["cross_runtime_adapter_used"] is False


def test_cross_runtime_workflow_restart_fails_without_complete(tmp_path):
    source_dir = tmp_path / "source"
    target_dir = tmp_path / "target"
    run_channel_active_nematics(_spec(source_dir, "legacy_channel", checkpoint_interval=1))

    with pytest.raises(ValueError, match="cross-runtime"):
        run_channel_active_nematics(_spec(target_dir, "compiled_channel_v2", restart_from=source_dir / "checkpoint_1"))
    assert not (target_dir / "COMPLETE").exists()


def test_application_refuses_nonempty_output_and_legacy_snapshot_mode(tmp_path):
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    (occupied / "owned.txt").write_text("user data\n")
    with pytest.raises(FileExistsError, match="nonempty"):
        run_channel_active_nematics(_spec(occupied, "legacy_channel"))
    assert (occupied / "owned.txt").read_text() == "user data\n"

    with pytest.raises(ValueError, match="legacy snapshot"):
        run_channel_active_nematics(_spec(tmp_path / "snapshot", "legacy_channel", initialization_mode="snapshot"))


def test_progress_contract_fails_closed_before_complete(tmp_path):
    output = tmp_path / "progress"
    with pytest.raises(ValueError, match="consecutive"):
        run_channel_active_nematics(_spec(output, "legacy_channel", steps=2), progress=(0, 2))
    assert not (output / "COMPLETE").exists()


def test_channel_entry_point_and_oracle_module_remain_unchanged():
    root = Path(__file__).resolve().parents[1]
    import hashlib
    assert hashlib.sha256((root / "Channel.py").read_bytes()).hexdigest() == "ea7087c4a22796bcefb77407505d2971cee948a9d2b643ced03d6dcb0a5d34b2"
    assert hashlib.sha256((root / "pssolver/channel.py").read_bytes()).hexdigest() == "3e85bd8e387eaeee48674998e08a3f97585f9d73ae3c7c4bd620d9ea1bd9605b"


def test_p75_record_binds_package_workflow_sources():
    import hashlib

    root = Path(__file__).resolve().parents[1]
    record = json.loads(
        (root / "notes/architecture_v0_2/phase_7_p75_local_closure.json").read_text()
    )

    def digest(relative: str) -> str:
        return hashlib.sha256((root / relative).read_bytes()).hexdigest()

    implementation = record["implementation"]
    assert record["classification"] == "PASS_P7_5_CHANNEL_LOCAL_CLOSURE"
    assert implementation["application_sha256"] == (
        "ee1889699388bed7e08ce419f20985e7afef1be6dbac26b913329e60ecd915f0"
    )
    assert digest(implementation["application"]) != implementation[
        "application_sha256"
    ]
    for name in ("compiled_runtime_bridge", "workflow", "observation"):
        assert digest(implementation[name]) == implementation[f"{name}_sha256"]
    assert record["authorization"]["p7_5_complete"] is True
    assert record["authorization"]["p7_6_h100_authorized"] is False
