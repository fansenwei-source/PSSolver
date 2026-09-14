"""Qualification tests for the shared Stage O.3 Plane workflow."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import pssolver
from pssolver.configuration import create_plane_beris_edwards_run_spec
from pssolver.workflows import (
    PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PLANE_WORKFLOW_SCHEMA_VERSION,
    load_plane_checkpoint,
    read_plane_checkpoint_header,
)


PROJECT_ROOT = Path(__file__).parents[1]
PRODUCTION_SCRIPT = PROJECT_ROOT / "Plane_beris_edwards_stokes.py"


def _base_arguments(output, runtime_path, steps):
    return [
        "--activity-number",
        "18",
        "--output-dir",
        str(output),
        "--runtime-path",
        runtime_path,
        "--device",
        "cpu",
        "--dtype",
        "float64",
        "--pointwise-execution",
        "eager",
        "--nx",
        "8",
        "--ny",
        "8",
        "--nz",
        "8",
        "--steps",
        str(steps),
        "--save-start-step",
        "0",
        "--save-interval",
        "2",
        "--diagnostic-interval",
        "1",
        "--spectral-refresh-steps",
        "2",
        "--save-hydrodynamics",
    ]


def _run(output, runtime_path, steps, *extra, check=True):
    result = subprocess.run(
        [
            sys.executable,
            str(PRODUCTION_SCRIPT),
            *_base_arguments(output, runtime_path, steps),
            *extra,
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if check:
        assert result.returncode == 0, result.stderr
    return result


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def test_runtime_identity_excludes_workflow_but_binds_numerics(tmp_path):
    common = {
        "activity_number": 18.0,
        "dtype": "float64",
        "pointwise_execution": "eager",
        "nx": 8,
        "ny": 8,
        "nz": 8,
        "save_start_step": 0,
    }
    left = create_plane_beris_edwards_run_spec(
        output_dir=tmp_path / "left",
        steps=8,
        **common,
    )
    right = create_plane_beris_edwards_run_spec(
        output_dir=tmp_path / "right",
        steps=3,
        checkpoint_interval=3,
        restart_from=tmp_path / "checkpoint",
        **common,
    )
    changed = create_plane_beris_edwards_run_spec(
        output_dir=tmp_path / "changed",
        steps=8,
        dt=0.005,
        **common,
    )
    canary = create_plane_beris_edwards_run_spec(
        output_dir=tmp_path / "canary",
        steps=8,
        runtime_path="separated_canary",
        **common,
    )
    changed_workflow = create_plane_beris_edwards_run_spec(
        output_dir=tmp_path / "changed_workflow",
        steps=3,
        diagnostics=True,
        initial_s=0.2,
        checkpoint_interval=3,
        restart_from=tmp_path / "checkpoint",
        **common,
    )

    assert left.runtime_identity_sha256() == right.runtime_identity_sha256()
    assert (
        left.runtime_identity_sha256()
        == changed_workflow.runtime_identity_sha256()
    )
    assert left.runtime_identity_sha256() != changed.runtime_identity_sha256()
    assert left.runtime_identity_sha256() != canary.runtime_identity_sha256()


def test_shared_workflow_preserves_output_contract_and_complete_is_last(tmp_path):
    output = tmp_path / "legacy"
    _run(output, "legacy_production", 3)

    expected = {
        *(f"{prefix}_{step}.npy" for prefix in ("Q", "u", "p") for step in (0, 2, 3)),
        "Q2D_initial.npy",
        "Q2D_defects.csv",
        "metadata.json",
        "COMPLETE",
    }
    assert {path.name for path in output.iterdir()} == expected
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["completed_steps"] == 3
    assert metadata["workflow"] == {
        "schema_version": PLANE_WORKFLOW_SCHEMA_VERSION,
        "runtime_path": "legacy_production",
        "checkpoint_format_version": (
            PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION
        ),
        "cross_backend_restart_supported": False,
        "start_step": 0,
        "requested_additional_steps": 3,
        "completion_marker_order": "metadata_then_COMPLETE",
        "saved_steps": [0, 2, 3],
        "checkpoint_steps": [],
        "final_step": 3,
    }
    assert (output / "COMPLETE").stat().st_mtime_ns >= (
        output / "metadata.json"
    ).stat().st_mtime_ns
    assert not tuple(output.glob(".*.tmp"))
    assert np.load(output / "Q_3.npy").shape == (8, 8, 8, 5)
    assert np.load(output / "u_3.npy").shape == (8, 8, 8, 3)
    assert np.load(output / "p_3.npy").shape == (8, 8, 8)


@pytest.mark.parametrize(
    "runtime_path",
    ("legacy_production", "separated_canary"),
)
def test_same_backend_split_restart_is_byte_exact(tmp_path, runtime_path):
    continuous = tmp_path / f"{runtime_path}_continuous"
    segment = tmp_path / f"{runtime_path}_segment"
    resumed = tmp_path / f"{runtime_path}_resumed"
    _run(continuous, runtime_path, 6)
    _run(segment, runtime_path, 3, "--checkpoint-interval", "3")
    checkpoint_path = segment / "checkpoint_3"
    checkpoint = load_plane_checkpoint(checkpoint_path)
    assert checkpoint.completed_steps == 3
    assert checkpoint.integrator_refresh_count == 1
    assert checkpoint.integrator_step_count == 1
    _run(
        resumed,
        runtime_path,
        3,
        "--restart-from",
        str(checkpoint_path),
    )

    for prefix in ("Q", "u", "p"):
        assert _sha256(continuous / f"{prefix}_6.npy") == _sha256(
            resumed / f"{prefix}_6.npy"
        )
    metadata = json.loads((resumed / "metadata.json").read_text())
    assert metadata["restart"]["source_completed_steps"] == 3
    assert metadata["restart"]["source_runtime_path"] == runtime_path
    assert metadata["workflow"]["start_step"] == 3
    assert metadata["workflow"]["final_step"] == 6


def test_cross_backend_restart_fails_before_output_or_solver(tmp_path):
    source = tmp_path / "legacy_source"
    target = tmp_path / "canary_target"
    _run(source, "legacy_production", 2, "--checkpoint-interval", "2")
    result = _run(
        target,
        "separated_canary",
        1,
        "--restart-from",
        str(source / "checkpoint_2"),
        check=False,
    )
    assert result.returncode != 0
    assert "cross-runtime Plane checkpoint restart is unsupported" in result.stderr
    assert not target.exists()


def test_checkpoint_checksum_tamper_is_fatal(tmp_path):
    source = tmp_path / "source"
    _run(source, "legacy_production", 2, "--checkpoint-interval", "2")
    checkpoint = source / "checkpoint_2"
    tensor = checkpoint / "evolved_spatial__Qxx.npy"
    contents = bytearray(tensor.read_bytes())
    contents[-1] ^= 1
    tensor.write_bytes(contents)

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_plane_checkpoint(checkpoint)


@pytest.mark.parametrize(
    "runtime_path",
    ("legacy_production", "separated_canary"),
)
def test_shared_diagnostics_schema_is_finite_for_both_paths(
    tmp_path,
    runtime_path,
):
    output = tmp_path / runtime_path
    _run(output, runtime_path, 2, "--diagnostics")
    values = np.load(output / "diagnostics.npy", allow_pickle=False)
    assert values.dtype.names == (
        "step",
        "div_max",
        "div_rms",
        "div_rel",
        "schur_iterations",
        "schur_abs_residual",
        "schur_rel_residual",
        "wall_normal_momentum_max",
        "wall_normal_momentum_rms",
    )
    assert values["step"].tolist() == [0, 1, 2]
    for name in values.dtype.names[1:]:
        assert np.isfinite(values[name]).all()
    header = (output / "diagnostics.csv").read_text().splitlines()[0]
    assert header.startswith("step,div_max,div_rms,div_rel")


def test_checkpoint_header_is_small_and_backend_explicit(tmp_path):
    output = tmp_path / "canary"
    _run(output, "separated_canary", 2, "--checkpoint-interval", "2")
    header = read_plane_checkpoint_header(output / "checkpoint_2")
    assert header.runtime_path.value == "separated_canary"
    assert header.completed_steps == 2
    assert len(header.runtime_identity_sha256) == 64


def test_o3_workflow_is_not_promoted_through_generic_or_channel_api():
    assert not hasattr(pssolver, "PlaneBerisEdwardsWorkflow")
    for relative in ("pssolver/solver.py", "pssolver/channel.py"):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "pssolver.workflows" not in source
        assert "PlaneBerisEdwardsWorkflow" not in source
