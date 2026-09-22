"""P5.4 disconnected observation, diagnostics, and checkpoint tests."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import pssolver.workflows as workflow_api
from pssolver.configuration import (
    PlaneRuntimePath,
    create_plane_beris_edwards_run_spec,
)
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.plane_beris_edwards import LegacyPlaneRuntimeAdapter
from pssolver.runtime.plane_compiled_v2_binding import (
    bind_plane_compiled_v2,
)
from pssolver.runtime.plane_compiled_v2_step import (
    build_plane_compiled_v2_step_program,
)
from pssolver.runtime.plane_legacy import build_legacy_plane_runtime
from pssolver.workflows.plane_checkpoint import (
    PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    capture_plane_checkpoint,
    load_plane_checkpoint,
    restore_plane_checkpoint,
    write_plane_checkpoint,
)
from pssolver.workflows.plane_compiled_v2 import (
    PlaneCompiledV2WorkflowAdapter,
)
from pssolver.workflows.plane_observation import (
    capture_plane_diagnostic,
    capture_plane_observation,
    write_plane_observation,
)


ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / (
    "notes/architecture_v0_2/phase_5_p54_observation_checkpoint.json"
)


def _spec(tmp_path: Path, name: str, **overrides):
    values = {
        "activity_number": 18.0,
        "output_dir": tmp_path / name,
        "device": "cpu",
        "dtype": "float64",
        "pointwise_execution": "eager",
        "nx": 6,
        "ny": 6,
        "nz": 6,
        "lx": 8.0,
        "ly": 9.0,
        "height": 20.0,
        "steps": 8,
        "save_start_step": 0,
        "save_interval": 1,
        "diagnostic_interval": 1,
        "diagnostics": True,
        "spectral_refresh_steps": 3,
    }
    values.update(overrides)
    return create_plane_beris_edwards_run_spec(**values)


def _initial_values():
    coordinate = torch.arange(6**3, dtype=torch.float64).reshape(6, 6, 6)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    return {
        name: coordinate.mul((index + 1) * 1.0e-3)
        for index, name in enumerate(Q_COMPONENTS)
    }


def _solver(spec):
    return build_legacy_plane_runtime(
        spec,
        device="cpu",
        initial_values=_initial_values(),
    )


def _compiled(tmp_path: Path, name: str, **overrides):
    spec = _spec(tmp_path, name, **overrides)
    solver, projector = _solver(spec)
    binding = bind_plane_compiled_v2(
        spec,
        solver=solver,
        projector=projector,
    )
    program = build_plane_compiled_v2_step_program(binding)
    adapter = PlaneCompiledV2WorkflowAdapter(binding, program)
    return spec, solver, projector, binding, program, adapter


def _legacy(tmp_path: Path, name: str):
    spec = _spec(tmp_path, name)
    solver, projector = _solver(spec)
    return spec, LegacyPlaneRuntimeAdapter(solver, projector)


def _advance(program, steps: int):
    for _ in range(steps):
        program.step()


def _assert_runtime_equal(left, right):
    assert torch.equal(left.fields.spatial, right.fields.spatial)
    assert torch.equal(left.fields.spectral, right.fields.spectral)
    assert left.integrator.runtime_state.to_metadata() == (
        right.integrator.runtime_state.to_metadata()
    )
    assert left.integrator._static_fields_are_current == (
        right.integrator._static_fields_are_current
    )


def test_output_views_are_zero_copy_and_observation_matches_legacy(tmp_path):
    _, legacy = _legacy(tmp_path, "legacy_observation")
    _, solver, _, _, program, adapter = _compiled(
        tmp_path,
        "compiled_observation",
    )
    legacy.advance(2)
    _advance(program, 2)
    legacy.synchronize_for_observation()
    adapter.synchronize_for_observation()

    expected = capture_plane_observation(legacy)
    actual = adapter.capture_observation()
    views = adapter.output_views

    assert actual.step == expected.step == 2
    assert np.array_equal(actual.q, expected.q)
    assert np.array_equal(actual.velocity, expected.velocity)
    assert np.array_equal(actual.pressure, expected.pressure)
    owner_pointer = solver.fields.spatial.untyped_storage().data_ptr()
    assert views.q.untyped_storage().data_ptr() == owner_pointer
    assert views.velocity.untyped_storage().data_ptr() == owner_pointer
    assert views.pressure.untyped_storage().data_ptr() == owner_pointer
    assert views.to_metadata()["zero_copy"] is True


def test_observation_files_are_byte_identical_to_legacy(tmp_path):
    _, legacy = _legacy(tmp_path, "legacy_files")
    _, _, _, _, program, adapter = _compiled(tmp_path, "compiled_files")
    legacy.advance(2)
    _advance(program, 2)
    legacy.synchronize_for_observation()
    adapter.synchronize_for_observation()
    expected = capture_plane_observation(legacy)
    actual = adapter.capture_observation()
    legacy_directory = tmp_path / "legacy_output"
    compiled_directory = tmp_path / "compiled_output"
    legacy_directory.mkdir()
    compiled_directory.mkdir()

    expected_paths = write_plane_observation(
        legacy_directory,
        expected,
        save_hydrodynamics=True,
    )
    actual_paths = write_plane_observation(
        compiled_directory,
        actual,
        save_hydrodynamics=True,
    )

    assert [path.name for path in expected_paths] == [
        path.name for path in actual_paths
    ]
    for expected_path, actual_path in zip(
        expected_paths,
        actual_paths,
        strict=True,
    ):
        assert expected_path.read_bytes() == actual_path.read_bytes()


def test_projected_force_and_pressure_diagnostics_match_legacy(tmp_path):
    spec, legacy = _legacy(tmp_path, "legacy_diagnostic")
    _, _, _, _, program, adapter = _compiled(
        tmp_path,
        "compiled_diagnostic",
    )
    legacy.advance(2)
    _advance(program, 2)
    legacy.synchronize_for_observation()
    adapter.synchronize_for_observation()

    expected = capture_plane_diagnostic(
        legacy,
        step=2,
        viscosity=spec.eta,
        friction=0.0,
    )
    actual = adapter.capture_diagnostic(
        step=2,
        viscosity=spec.eta,
        friction=0.0,
    )

    assert actual == expected
    assert torch.equal(
        adapter.projected_normal_force(),
        legacy.projected_normal_force(),
    )
    assert adapter.flow_diagnostics() == legacy.flow_diagnostics()


def test_disabled_diagnostics_do_not_block_observation_or_checkpoint(tmp_path):
    _, _, _, _, program, adapter = _compiled(
        tmp_path,
        "diagnostics_disabled",
        diagnostics=False,
    )
    program.step()

    observation = adapter.capture_observation(synchronize=True)
    checkpoint = adapter.capture_checkpoint()

    assert observation.step == checkpoint.completed_steps == 1
    with pytest.raises(RuntimeError, match="normal-force diagnostics are disabled"):
        adapter.projected_normal_force()
    with pytest.raises(RuntimeError, match="pressure diagnostics are disabled"):
        adapter.flow_diagnostics()


def test_checkpoint_v1_round_trip_preserves_existing_file_schema(tmp_path):
    _, _, _, _, program, adapter = _compiled(tmp_path, "checkpoint")
    _advance(program, 3)
    checkpoint = adapter.capture_checkpoint()
    directory = tmp_path / "checkpoint_v1"

    write_plane_checkpoint(directory, checkpoint)
    loaded = load_plane_checkpoint(directory)
    metadata = json.loads(
        (directory / "checkpoint.json").read_text(encoding="utf-8")
    )

    assert loaded.format_version == PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION == 1
    assert loaded.runtime_path is PlaneRuntimePath.LEGACY_PRODUCTION
    assert loaded.backend_restart["kind"] == "compiled_v2_plane_stateless"
    assert set(metadata) == {
        "backend_restart",
        "completed_steps",
        "evolved_components",
        "format_version",
        "integrator",
        "runtime_identity_sha256",
        "runtime_path",
        "tensor_files",
    }
    assert set(metadata["tensor_files"]) == {
        "evolved_spatial",
        "evolved_spectral",
    }
    assert sorted(path.name for path in directory.glob("*.npy")) == sorted(
        f"{kind}__{name}.npy"
        for kind in ("evolved_spatial", "evolved_spectral")
        for name in Q_COMPONENTS
    )


def test_continuous_and_split_checkpoint_restart_are_byte_identical(tmp_path):
    _, continuous, _, _, continuous_program, continuous_adapter = _compiled(
        tmp_path,
        "continuous",
    )
    _, _, _, _, segment_program, segment_adapter = _compiled(
        tmp_path,
        "segment",
    )
    _, resumed, _, _, resumed_program, resumed_adapter = _compiled(
        tmp_path,
        "resumed",
    )
    _advance(continuous_program, 7)
    _advance(segment_program, 3)
    checkpoint = segment_adapter.capture_checkpoint()

    assert resumed_adapter.restore_checkpoint(checkpoint) == 3
    _advance(resumed_program, 4)
    continuous_adapter.synchronize_for_observation()
    resumed_adapter.synchronize_for_observation()

    _assert_runtime_equal(continuous, resumed)
    assert continuous_program.state.progress.completed_steps == 7
    assert resumed_program.state.progress.completed_steps == 7
    continuous_observation = continuous_adapter.capture_observation()
    resumed_observation = resumed_adapter.capture_observation()
    assert continuous_observation.step == resumed_observation.step == 7
    assert np.array_equal(continuous_observation.q, resumed_observation.q)
    assert np.array_equal(
        continuous_observation.velocity,
        resumed_observation.velocity,
    )
    assert np.array_equal(
        continuous_observation.pressure,
        resumed_observation.pressure,
    )


def test_cross_runtime_checkpoints_are_rejected_before_mutation(tmp_path):
    spec, legacy = _legacy(tmp_path, "legacy_cross")
    _, solver, _, _, program, adapter = _compiled(tmp_path, "compiled_cross")
    legacy.advance(2)
    _advance(program, 2)
    legacy_checkpoint = capture_plane_checkpoint(
        legacy,
        runtime_identity_sha256=spec.runtime_identity_sha256(),
    )
    compiled_checkpoint = adapter.capture_checkpoint()
    legacy_before = legacy.solver.fields.spatial.clone()
    compiled_before = solver.fields.spatial.clone()

    with pytest.raises(ValueError, match="backend restart contract"):
        restore_plane_checkpoint(
            legacy,
            compiled_checkpoint,
            runtime_identity_sha256=spec.runtime_identity_sha256(),
        )
    with pytest.raises(ValueError, match="backend restart contract"):
        adapter.restore_checkpoint(legacy_checkpoint)

    assert torch.equal(legacy.solver.fields.spatial, legacy_before)
    assert torch.equal(solver.fields.spatial, compiled_before)
    assert legacy.completed_steps == adapter.completed_steps == 2


def _tampered_checkpoint(checkpoint, case: str):
    if case == "identity":
        return dataclasses.replace(
            checkpoint,
            runtime_identity_sha256="0" * 64,
        )
    if case == "backend":
        return dataclasses.replace(
            checkpoint,
            backend_restart={"kind": "wrong", "state_keys": []},
        )
    if case == "runtime_path":
        return dataclasses.replace(
            checkpoint,
            runtime_path=PlaneRuntimePath.SEPARATED_CANARY,
        )
    if case == "shape":
        values = dict(checkpoint.evolved_spectral)
        values["Qyz"] = values["Qyz"][..., :-1]
        return dataclasses.replace(checkpoint, evolved_spectral=values)
    if case == "dtype":
        values = dict(checkpoint.evolved_spatial)
        values["Qyz"] = values["Qyz"].to(torch.float32)
        return dataclasses.replace(checkpoint, evolved_spatial=values)
    if case == "nonfinite":
        checkpoint.evolved_spatial["Qyz"].reshape(-1)[0] = torch.nan
        return checkpoint
    raise AssertionError(case)


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("identity", "runtime identity"),
        ("backend", "backend restart contract"),
        ("runtime_path", "cross-runtime"),
        ("shape", "shape"),
        ("dtype", "dtype"),
        ("nonfinite", "non-finite"),
    ),
)
def test_checkpoint_tamper_is_rejected_before_state_or_clock_change(
    tmp_path,
    case,
    message,
):
    _, source, _, _, source_program, source_adapter = _compiled(
        tmp_path,
        f"source_{case}",
    )
    _, target, _, _, _, target_adapter = _compiled(
        tmp_path,
        f"target_{case}",
    )
    _advance(source_program, 2)
    checkpoint = _tampered_checkpoint(source_adapter.capture_checkpoint(), case)
    spatial_before = target.fields.spatial.clone()
    spectral_before = target.fields.spectral.clone()

    with pytest.raises((ValueError, RuntimeError), match=message):
        target_adapter.restore_checkpoint(checkpoint)

    assert torch.equal(target.fields.spatial, spatial_before)
    assert torch.equal(target.fields.spectral, spectral_before)
    assert target_adapter.completed_steps == 0


def test_adapter_metadata_records_transitional_v1_identity_contract(tmp_path):
    _, _, _, _, _, adapter = _compiled(tmp_path, "metadata")
    metadata = adapter.to_metadata()

    assert metadata["identity"] == (
        "compiled_v2_observation_checkpoint_adapter"
    )
    assert metadata["connected_runtime"] is None
    assert metadata["output_views"]["zero_copy"] is True
    assert metadata["checkpoint"] == {
        "format_version": 1,
        "runtime_path_carrier": "legacy_production",
        "backend_identity_enforced": True,
        "preflight_before_mutation": True,
        "cross_runtime_restore": False,
    }
    assert metadata["runtime_selector_added"] is False
    assert metadata["application_import_added"] is False
    assert metadata["implicit_fallback"] is False


def test_compiled_workflow_adapter_remains_private_and_selector_free():
    import pssolver

    name = "PlaneCompiledV2WorkflowAdapter"
    assert name not in pssolver.__all__
    assert name not in workflow_api.__all__
    assert not hasattr(workflow_api, name)
    assert {member.value for member in PlaneRuntimePath} == {
        "legacy_production",
        "separated_canary",
    }


def test_p54_qualification_record_matches_disconnected_scope():
    record = json.loads(QUALIFICATION.read_text(encoding="utf-8"))

    assert record["classification"] == (
        "PASS_P5_4_OBSERVATION_DIAGNOSTICS_CHECKPOINT_ADAPTER"
    )
    assert record["authorization"]["p5_4_implementation_authorized"] is True
    assert record["authorization"]["p5_5_implementation_authorized"] is False
    assert record["checkpoint"]["format_version"] == 1
    assert record["checkpoint"]["schema_changed"] is False
    assert record["connection"]["runtime_selector_added"] is False
    assert record["connection"]["fallback_allowed"] is False
    assert record["eligibility"]["p5_4_complete"] is True
    assert record["eligibility"]["p5_5_locally_eligible"] is True
    assert record["eligibility"]["phase_6_eligible"] is False
