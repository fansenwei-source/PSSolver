"""P5.6 local closure gates for the opt-in compiled Plane runtime."""

from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path

import pytest
import torch

from pssolver.configuration import create_plane_beris_edwards_run_spec
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.plane_beris_edwards import LegacyPlaneRuntimeAdapter
from pssolver.runtime.plane_compiled_v2_step import PlaneCompiledEulerStepProgram
from pssolver.runtime.plane_legacy import build_legacy_plane_runtime
from pssolver.workflows.plane_checkpoint import (
    capture_plane_checkpoint,
    restore_plane_checkpoint,
)
from pssolver.workflows.plane_compiled_v2 import (
    build_plane_compiled_v2_runtime,
)


ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / "notes/architecture_v0_2/phase_5_p56_local_closure.json"
PHASE6_PLAN = ROOT / (
    "notes/architecture_v0_2/phase_6_plane_compiled_v2_qualification_plan.json"
)


def _spec(tmp_path: Path, name: str, runtime_path: str, **overrides):
    values = {
        "activity_number": 18.0,
        "output_dir": tmp_path / name,
        "runtime_path": runtime_path,
        "device": "cpu",
        "dtype": "float64",
        "pointwise_execution": "eager",
        "nx": 6,
        "ny": 6,
        "nz": 6,
        "lx": 8.0,
        "ly": 9.0,
        "height": 20.0,
        "steps": 100,
        "save_start_step": 0,
        "save_interval": 100,
        "diagnostic_interval": 10,
        "spectral_refresh_steps": 7,
        "save_hydrodynamics": True,
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


def _legacy(tmp_path: Path, name: str):
    spec = _spec(tmp_path, name, "legacy_production")
    solver, projector = build_legacy_plane_runtime(
        spec,
        device="cpu",
        initial_values=_initial_values(),
    )
    return spec, LegacyPlaneRuntimeAdapter(solver, projector)


def _compiled(tmp_path: Path, name: str):
    spec = _spec(tmp_path, name, "compiled_v2")
    return spec, build_plane_compiled_v2_runtime(
        spec,
        device="cpu",
        initial_values=_initial_values(),
    )


def _assert_equal(left, right):
    assert torch.equal(left.fields.spatial, right.fields.spatial)
    assert torch.equal(left.fields.spectral, right.fields.spectral)
    assert torch.equal(left.fields.L_hat, right.fields.L_hat)
    assert left.completed_steps == right.completed_steps
    assert left.solver.integrator.runtime_state.to_metadata() == (
        right.solver.integrator.runtime_state.to_metadata()
    )


def _persistent_storage_identities(runtime):
    program = runtime._workflow._program
    state = program.state
    return {
        "fields_spatial": runtime.fields.spatial.untyped_storage().data_ptr(),
        "fields_spectral": runtime.fields.spectral.untyped_storage().data_ptr(),
        "linear_operator": runtime.fields.L_hat.untyped_storage().data_ptr(),
        "state_physical": state.physical.untyped_storage().data_ptr(),
        "state_spectral": state.spectral.untyped_storage().data_ptr(),
        "denominator": program._denominator.untyped_storage().data_ptr(),
    }


@pytest.mark.parametrize("steps", (1, 2, 100))
def test_compiled_runtime_is_byte_identical_at_closure_step_counts(
    tmp_path,
    steps,
):
    _, legacy = _legacy(tmp_path, f"legacy_{steps}")
    _, compiled = _compiled(tmp_path, f"compiled_{steps}")
    before = _persistent_storage_identities(compiled)

    legacy.advance(steps)
    compiled.advance(steps)

    _assert_equal(legacy, compiled)
    assert _persistent_storage_identities(compiled) == before
    assert compiled.to_metadata()["fallback_used"] is False
    workspace = compiled._workflow._program.workspace
    assert workspace.plan.required_bytes == 0
    assert workspace.active is False
    assert workspace.generation == steps


def test_compiled_checkpoint_restart_matches_continuous_and_rejects_cross_path(
    tmp_path,
):
    continuous_spec, continuous = _compiled(tmp_path, "continuous")
    _, source = _compiled(tmp_path, "source")
    _, resumed = _compiled(tmp_path, "resumed")
    _, legacy = _legacy(tmp_path, "legacy_target")

    continuous.advance(100)
    source.advance(37)
    checkpoint = capture_plane_checkpoint(
        source,
        runtime_identity_sha256=continuous_spec.runtime_identity_sha256(),
    )
    restore_plane_checkpoint(
        resumed,
        checkpoint,
        runtime_identity_sha256=continuous_spec.runtime_identity_sha256(),
    )
    resumed.advance(63)

    _assert_equal(continuous, resumed)
    legacy_before = legacy.fields.spatial.clone()
    with pytest.raises(ValueError, match="cross-runtime"):
        restore_plane_checkpoint(
            legacy,
            checkpoint,
            runtime_identity_sha256=continuous_spec.runtime_identity_sha256(),
        )
    assert torch.equal(legacy.fields.spatial, legacy_before)
    assert legacy.completed_steps == 0


def test_compiled_hot_loop_has_only_prebound_bounded_control():
    source = inspect.getsource(PlaneCompiledEulerStepProgram.step)
    tree = ast.parse(inspect.cleandoc(source))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }
    call_attributes = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }

    assert not any(isinstance(node, (ast.For, ast.While)) for node in ast.walk(tree))
    assert names.isdisjoint({"json", "getattr", "setattr", "globals", "locals"})
    assert attributes.isdisjoint(
        {"runtime_path", "registry", "capabilities", "to_metadata"}
    )
    assert call_attributes.isdisjoint(
        {"empty", "empty_like", "zeros", "zeros_like", "ones", "full", "clone"}
    )


def test_p56_record_closes_phase5_without_authorizing_phase6_execution():
    record = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    plan = json.loads(PHASE6_PLAN.read_text(encoding="utf-8"))

    assert record["classification"] == "PASS_P5_6_LOCAL_CLOSURE"
    assert record["validation"]["step_counts"] == [1, 2, 100]
    assert record["validation"]["fallback_forbidden"] == "PASS"
    assert record["eligibility"]["phase_5_complete"] is True
    assert record["eligibility"]["phase_6_planning_eligible"] is True
    assert record["authorization"]["phase_6_execution_authorized"] is False
    assert record["authorization"]["production_default_changed"] is False

    assert plan["baseline_runtime"] == "legacy_production"
    assert plan["candidate_runtime"] == "compiled_v2"
    assert plan["grids"] == [[128, 128, 32], [320, 320, 80]]
    assert plan["formal_h100_submission_count"] == 1
    assert plan["automatic_retry"] is False
    assert plan["authorization"]["execution_authorized"] is False
    assert plan["authorization"]["default_promotion_authorized"] is False
