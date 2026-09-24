"""P5.5 explicit application connection for the compiled-v2 Plane path."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

import pssolver.runtime as runtime_api
from pssolver.applications import plane_beris_edwards as application
from pssolver.configuration import (
    PlaneRuntimePath,
    create_plane_beris_edwards_run_spec,
    parse_plane_beris_edwards_run_spec,
)
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime import (
    PlaneRuntimeBuildRequest,
    build_plane_beris_edwards_runtime,
)
from pssolver.workflows import read_plane_checkpoint_header
from pssolver.workflows.plane_compiled_v2 import (
    build_plane_compiled_v2_runtime,
)


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


def _request(spec):
    return PlaneRuntimeBuildRequest(
        run_spec=spec,
        production_metadata={
            "configuration": spec.identity_metadata(),
            "runtime_selection": spec.runtime_selection_metadata(),
        },
        initial_values=_initial_values(),
        device="cpu",
    )


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


def test_compiled_factory_builds_private_adapter_without_fallback(tmp_path):
    spec = _spec(tmp_path, "factory", runtime_path="compiled_v2")
    adapter = build_plane_beris_edwards_runtime(
        _request(spec),
        compiled_builder=lambda: build_plane_compiled_v2_runtime(
            spec,
            device="cpu",
            initial_values=_initial_values(),
        ),
    )
    callbacks = []

    adapter.advance(
        2,
        pre_update_callback=lambda solver, step: callbacks.append(
            (solver is adapter.solver, step)
        ),
    )

    assert adapter.runtime_path is PlaneRuntimePath.COMPILED_V2
    assert adapter.completed_steps == 2
    assert callbacks == [(True, 0), (True, 1)]
    assert adapter.to_metadata()["requested"] == "compiled_v2"
    assert adapter.to_metadata()["effective"] == "compiled_v2"
    assert adapter.to_metadata()["fallback_used"] is False
    assert adapter.backend_restart_metadata()["kind"] == (
        "compiled_v2_plane_stateless"
    )
    assert not hasattr(runtime_api, "CompiledV2PlaneRuntimeAdapter")


def test_compiled_factory_propagates_construction_failure_without_fallback(
    tmp_path,
):
    spec = _spec(tmp_path, "failure", runtime_path="compiled_v2")

    with pytest.raises(RuntimeError, match="fallback is forbidden"):
        build_plane_beris_edwards_runtime(_request(spec))

    def fail():
        raise RuntimeError("sentinel compiled construction failure")

    with pytest.raises(
        RuntimeError,
        match="sentinel compiled construction failure",
    ):
        build_plane_beris_edwards_runtime(
            _request(spec),
            compiled_builder=fail,
        )


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
        "build_package_simulation_runtime",
        forbidden,
    )
    monkeypatch.setattr(application, "create_initial_condition", forbidden)

    assert application.run_plane_beris_edwards(spec, emit_metadata=True) is None
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

    legacy_result = application.run_plane_beris_edwards(legacy)
    compiled_result = application.run_plane_beris_edwards(compiled)

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
    application.run_plane_beris_edwards(continuous)
    application.run_plane_beris_edwards(segment)
    checkpoint = segment.output_dir / "checkpoint_2"
    header = read_plane_checkpoint_header(checkpoint)
    resumed = _spec(
        tmp_path,
        "resumed",
        runtime_path="compiled_v2",
        steps=2,
        restart_from=checkpoint,
    )

    application.run_plane_beris_edwards(resumed)

    assert header.runtime_path is PlaneRuntimePath.COMPILED_V2
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
