"""P5.5 explicit application connection for the compiled-v2 Plane path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from pssolver.configuration import (
    PlaneRuntimePath,
    create_plane_beris_edwards_run_spec,
    parse_plane_beris_edwards_run_spec,
)
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime import legacy_entry as application


ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / (
    "notes/architecture_v0_2/phase_5_p55_application_connection.json"
)


def _spec(tmp_path: Path, name: str, **overrides):
    values = {
        "activity_number": 18.0,
        "output_dir": tmp_path / name,
        "device": "cpu",
        "dtype": "float64",
        "pointwise_execution": "eager",
        "nx": 8,
        "ny": 8,
        "nz": 8,
        "steps": 4,
        "save_start_step": 0,
        "save_interval": 2,
        "diagnostic_interval": 1,
        "spectral_refresh_steps": 2,
        "save_hydrodynamics": True,
    }
    values.update(overrides)
    return create_plane_beris_edwards_run_spec(**values)


def _initial_values():
    coordinate = torch.arange(8**3, dtype=torch.float64).reshape(8, 8, 8)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    return {
        name: coordinate.mul((index + 1) * 1.0e-3)
        for index, name in enumerate(Q_COMPONENTS)
    }




def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_runtime_selector_default_is_unchanged_and_compiled_is_explicit(tmp_path):
    default = _spec(tmp_path, "default")
    compiled = _spec(tmp_path, "compiled", runtime_path="compiled_v2")
    parsed = parse_plane_beris_edwards_run_spec(
        [
            "--activity-number",
            "18",
            "--output-dir",
            str(tmp_path / "parsed"),
            "--runtime-path",
            "compiled_v2",
        ]
    )

    assert default.runtime_path is PlaneRuntimePath.LEGACY_PRODUCTION
    assert compiled.runtime_path is PlaneRuntimePath.COMPILED_V2
    assert parsed.runtime_path is PlaneRuntimePath.COMPILED_V2
    assert {member.value for member in PlaneRuntimePath} == {
        "legacy_production",
        "separated_canary",
        "compiled_v2",
    }






def test_compiled_dry_run_is_tensor_free_and_creates_no_output(
    tmp_path,
    monkeypatch,
    capsys,
):
    output = tmp_path / "dry_run"
    spec = _spec(
        tmp_path,
        "dry_run",
        runtime_path="compiled_v2",
        dry_run=True,
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("compiled dry-run must not construct tensors")

    monkeypatch.setattr(
        application,
        "Engine",
        forbidden,
    )
    monkeypatch.setattr(application, "create_initial_condition", forbidden)

    assert application.run_v04_plane_translation(spec, emit_metadata=True) is None
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["runtime_selection"]["requested"] == "compiled_v2"
    assert metadata["runtime_selection"]["effective"] == "compiled_v2"
    assert metadata["runtime_selection"]["fallback_allowed"] is False
    assert not output.exists()


def test_compiled_application_is_byte_identical_to_legacy_for_short_run(
    tmp_path,
):
    legacy = _spec(tmp_path, "legacy", runtime_path="legacy_production")
    compiled = _spec(tmp_path, "compiled", runtime_path="compiled_v2")

    legacy_result = application.run_v04_plane_translation(legacy)
    compiled_result = application.run_v04_plane_translation(compiled)

    assert legacy_result is not None
    assert compiled_result is not None
    assert legacy_result.final_step == compiled_result.final_step == 4
    for prefix in ("Q", "u", "p"):
        assert _sha256(legacy.output_dir / f"{prefix}_4.npy") == _sha256(
            compiled.output_dir / f"{prefix}_4.npy"
        )
    metadata = json.loads(
        (compiled.output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["runtime_selection"]["requested"] == "compiled_v2"
    assert metadata["runtime_selection"]["effective"] == "compiled_v2"
    assert metadata["runtime_selection"]["fallback_used"] is False
    assert metadata["workflow"]["runtime_path"] == "compiled_v2"
    assert (compiled.output_dir / "COMPLETE").is_file()


def test_compiled_application_runs_in_float32(tmp_path):
    compiled = _spec(
        tmp_path,
        "compiled_float32",
        runtime_path="compiled_v2",
        dtype="float32",
        steps=2,
    )

    result = application.run_v04_plane_translation(compiled)

    assert result is not None
    assert result.final_step == 2
    assert (compiled.output_dir / "COMPLETE").is_file()


def test_compiled_construction_failure_leaves_no_output_directory(
    tmp_path,
    monkeypatch,
):
    compiled = _spec(
        tmp_path,
        "compiled_construction_failure",
        runtime_path="compiled_v2",
        steps=2,
    )

    def fail(*args, **kwargs):
        raise RuntimeError("sentinel package-runtime construction failure")

    monkeypatch.setattr(
        application,
        "Engine",
        fail,
    )

    with pytest.raises(
        RuntimeError,
        match="sentinel package-runtime construction failure",
    ):
        application.run_v04_plane_translation(compiled)

    assert not compiled.output_dir.exists()


def test_compiled_application_split_restart_is_byte_identical(tmp_path):
    continuous = _spec(
        tmp_path,
        "continuous",
        runtime_path="compiled_v2",
        steps=4,
    )
    segment = _spec(
        tmp_path,
        "segment",
        runtime_path="compiled_v2",
        steps=2,
        checkpoint_interval=2,
    )
    application.run_v04_plane_translation(continuous)
    application.run_v04_plane_translation(segment)
    checkpoint = segment.output_dir / "checkpoint_2"
    resumed = _spec(
        tmp_path,
        "resumed",
        runtime_path="compiled_v2",
        steps=2,
        save_start_step=4,
        restart_from=checkpoint,
    )

    resumed_result = application.run_v04_plane_translation(resumed)

    assert resumed_result.start_step == 2
    assert resumed_result.final_step == 4
    assert resumed_result.saved_steps == (4,)
    assert not (resumed.output_dir / "Q2D_initial.npy").exists()
    assert not (resumed.output_dir / "Q2D_defects.csv").exists()
    metadata = json.loads(
        (resumed.output_dir / "metadata.json").read_text(encoding="utf-8")
    )
    assert metadata["initial_condition"]["runtime_construction_only"] is True
    assert metadata["initial_condition"]["artifacts_written"] is False
    assert metadata["initial_condition"]["trajectory_state_source"] == (
        "authenticated_checkpoint"
    )
    assert metadata["initial_condition"]["checkpoint_path"] == str(
        checkpoint.resolve()
    )
    for prefix in ("Q", "u", "p"):
        assert _sha256(continuous.output_dir / f"{prefix}_4.npy") == _sha256(
            resumed.output_dir / f"{prefix}_4.npy"
        )


def test_p55_qualification_record_freezes_opt_in_connection():
    record = json.loads(QUALIFICATION.read_text(encoding="utf-8"))

    assert record["classification"] == (
        "PASS_P5_5_OPT_IN_APPLICATION_CONNECTION"
    )
    assert record["runtime_paths"] == {
        "default": "legacy_production",
        "diagnostic_opt_in": "separated_canary",
        "compiled_opt_in": "compiled_v2",
    }
    assert record["connection"]["fallback_allowed"] is False
    assert record["validation"]["short_run_byte_identity"] == "PASS"
    assert record["validation"]["split_restart_byte_identity"] == "PASS"
    assert record["eligibility"]["p5_6_locally_eligible"] is True
    assert record["eligibility"]["phase_6_eligible"] is False
