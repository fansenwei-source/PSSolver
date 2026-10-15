"""R5-13 low-risk metadata, restart, packaging, and device hygiene."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from pssolver.applications import plane_beris_edwards as plane_application
from pssolver.configuration import (
    PlaneRuntimePath,
    create_plane_beris_edwards_run_spec,
)
from pssolver.errors import SimulationExecutionError, SimulationRestartError


ROOT = Path(__file__).resolve().parents[1]


def _plane_spec(tmp_path: Path, **overrides: object):
    values = {
        "activity_number": 18.0,
        "output_dir": tmp_path / "output",
        "device": "cpu",
        "dtype": "float64",
        "pointwise_execution": "eager",
        "nx": 8,
        "ny": 8,
        "nz": 8,
        "steps": 1,
        "save_start_step": 0,
        "save_interval": 1,
        "diagnostic_interval": 1,
        "disable_spectral_refresh": True,
    }
    values.update(overrides)
    return create_plane_beris_edwards_run_spec(**values)


def _matching_checkpoint_header(spec, *, completed_steps: int):
    run_spec_identity = plane_application.plane_run_spec_dynamics_identity(
        spec
    ).to_metadata()
    forward = SimpleNamespace(
        to_metadata=lambda: {
            "kind": "forward_dynamics",
            "schema_version": 1,
            "version": "plane.forward.v2",
            "payload": {
                "run_spec_dynamics": run_spec_identity,
                "lifting_dynamics": None,
            },
        }
    )
    return SimpleNamespace(
        format_version=2,
        runtime_path=PlaneRuntimePath.LEGACY_PRODUCTION,
        completed_steps=completed_steps,
        compatibility_identity=SimpleNamespace(forward_dynamics=forward),
    )


def test_plane_dry_run_validates_absolute_restart_save_window(
    tmp_path,
    monkeypatch,
):
    spec = _plane_spec(
        tmp_path,
        restart_from=tmp_path / "checkpoint",
        dry_run=True,
        save_start_step=3,
    )
    monkeypatch.setattr(
        plane_application,
        "read_plane_checkpoint_header",
        lambda _path: _matching_checkpoint_header(spec, completed_steps=1),
    )

    with pytest.raises(SimulationRestartError) as captured:
        plane_application.run_plane_beris_edwards(spec)
    error = captured.value
    assert isinstance(error, SimulationExecutionError)
    assert isinstance(error, ValueError)
    assert error.to_metadata() == {
        "code": "simulation_restart_error",
        "message": "save_start_step must not exceed the absolute final step",
        "application_family": "plane_complete_stress_beris_edwards",
        "stage": "restart_validation",
        "step": 1,
        "details": {
            "absolute_final_step": 2,
            "reason": "save_start_step_after_final_step",
            "save_start_step": 3,
        },
    }
    assert not spec.output_dir.exists()


def test_provisional_writers_use_the_shared_checkpoint_transaction():
    for relative in (
        "pssolver/experimental/checkpointing.py",
        "pssolver/experimental/modal_block_checkpoint.py",
        "pssolver/integrators/reference_checkpoint.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert "checkpoint_directory_transaction(" in source
        assert "tempfile.mkdtemp(" not in source
        assert "os.replace(" not in source


def test_pdf_files_are_explicitly_binary():
    attributes = (ROOT / ".gitattributes").read_text(encoding="utf-8")
    assert attributes.splitlines() == ["*.pdf binary"]
