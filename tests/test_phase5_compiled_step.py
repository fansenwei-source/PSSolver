"""P5.3 disconnected compiled projected-Euler step qualification."""

from __future__ import annotations

import ast
import dataclasses
import inspect
import json
from pathlib import Path

import pytest
import torch

import pssolver.runtime as runtime_api
from pssolver.configuration import (
    PlaneRuntimePath,
    create_plane_beris_edwards_run_spec,
)
from pssolver.diagnostics import (
    build_tensor_inventory,
    compare_tensor_inventories,
)
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.plane_compiled_v2_binding import (
    bind_plane_compiled_v2,
)
from pssolver.runtime.plane_compiled_v2_step import (
    PlaneCompiledEulerStepProgram,
    build_plane_compiled_v2_step_program,
)
from pssolver.runtime.plane_legacy import build_legacy_plane_runtime


ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION = ROOT / (
    "notes/architecture_v0_2/phase_5_p53_compiled_euler_step.json"
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


def _build(tmp_path: Path, name: str, **overrides):
    spec = _spec(tmp_path, name, **overrides)
    solver, projector = build_legacy_plane_runtime(
        spec,
        device="cpu",
        initial_values=_initial_values(),
    )
    return spec, solver, projector


def _candidate(tmp_path: Path, name: str, **overrides):
    spec, solver, projector = _build(tmp_path, name, **overrides)
    binding = bind_plane_compiled_v2(
        spec,
        solver=solver,
        projector=projector,
    )
    return solver, binding, build_plane_compiled_v2_step_program(binding)


def _assert_solver_state_equal(left, right):
    assert torch.equal(left.fields.spatial, right.fields.spatial)
    assert torch.equal(left.fields.spectral, right.fields.spectral)
    assert torch.equal(left.fields.L_hat, right.fields.L_hat)
    assert left.integrator.runtime_state.to_metadata() == (
        right.integrator.runtime_state.to_metadata()
    )
    assert left.integrator._static_fields_are_current == (
        right.integrator._static_fields_are_current
    )


@pytest.mark.parametrize("steps", (1, 2, 7))
def test_compiled_step_is_byte_identical_to_legacy_continuous_cpu(
    tmp_path,
    steps,
):
    _, legacy, _ = _build(tmp_path, f"legacy_{steps}")
    candidate, _, program = _candidate(tmp_path, f"candidate_{steps}")

    for _ in range(steps):
        legacy.integrator.step()
        program.step()

    _assert_solver_state_equal(legacy, candidate)
    assert candidate.integrator.runtime_state.progress.completed_steps == steps
    assert program.workspace.active is False
    assert program.workspace.generation == steps


def test_callback_position_and_refresh_order_are_frozen(tmp_path):
    spec, solver, projector = _build(
        tmp_path,
        "trace",
        spectral_refresh_steps=1,
    )
    binding = bind_plane_compiled_v2(
        spec,
        solver=solver,
        projector=projector,
    )
    trace: list[str] = []
    original = binding.operators

    def record(name, operation):
        def wrapped(state, workspace, generation):
            trace.append(name)
            return operation(state, workspace, generation)

        return wrapped

    traced_operators = dataclasses.replace(
        original,
        prepare_algebraic=record("prepare", original.prepare_algebraic),
        explicit_rhs=record("rhs", original.explicit_rhs),
        project_dynamic_spectra=record(
            "project",
            original.project_dynamic_spectra,
        ),
        inverse_dynamic_spectra=record(
            "inverse",
            original.inverse_dynamic_spectra,
        ),
        refresh_dynamic_spectra=record(
            "refresh",
            original.refresh_dynamic_spectra,
        ),
    )
    program = build_plane_compiled_v2_step_program(
        dataclasses.replace(binding, operators=traced_operators)
    )

    program.step(pre_update_callback=lambda: trace.append("callback"))

    assert trace == [
        "prepare",
        "callback",
        "rhs",
        "project",
        "inverse",
        "refresh",
    ]
    assert program.state.progress.completed_steps == 1
    assert program.state.progress.refresh_step_count == 0
    assert program.state.progress.refresh_count == 1


def test_restored_static_fields_are_consumed_once_then_invalidated(tmp_path):
    spec, solver, projector = _build(tmp_path, "static")
    solver.model.update_static_fields()
    solver.integrator._static_fields_are_current = True
    calls = 0
    original_update = solver.model.update_static_fields

    def counted_update():
        nonlocal calls
        calls += 1
        return original_update()

    solver.model.update_static_fields = counted_update
    binding = bind_plane_compiled_v2(
        spec,
        solver=solver,
        projector=projector,
    )
    program = build_plane_compiled_v2_step_program(binding)

    program.step()
    assert calls == 0
    assert solver.integrator._static_fields_are_current is False
    program.step()
    assert calls == 1


def test_compiled_program_retains_no_additional_tensor_storage(tmp_path):
    spec, solver, projector = _build(tmp_path, "inventory")
    binding = bind_plane_compiled_v2(
        spec,
        solver=solver,
        projector=projector,
    )
    before = build_tensor_inventory(
        {"binding": binding, "projector": projector, "solver": solver},
        device="cpu",
    )

    program = build_plane_compiled_v2_step_program(binding)
    after = build_tensor_inventory(
        {
            "binding": binding,
            "program": program,
            "projector": projector,
            "solver": solver,
        },
        device="cpu",
    )
    comparison = compare_tensor_inventories(before, after)

    assert comparison["unique_storage_identity_equal"] is True
    assert comparison["added_storage_identity_count"] == 0
    assert program.workspace.plan.required_bytes == 0
    assert program.workspace.plan.slots == ()


def test_project_failure_matches_legacy_clock_and_tensor_semantics(tmp_path):
    _, legacy, _ = _build(tmp_path, "legacy_failure")
    candidate, binding, _ = _candidate(tmp_path, "candidate_failure")

    def fail_project(state, workspace, generation):
        raise RuntimeError("synthetic projection failure")

    object.__setattr__(
        legacy.integrator._step_program,
        "project_dynamic_spectra",
        fail_project,
    )
    failed_binding = dataclasses.replace(
        binding,
        operators=dataclasses.replace(
            binding.operators,
            project_dynamic_spectra=fail_project,
        ),
    )
    program = build_plane_compiled_v2_step_program(failed_binding)

    with pytest.raises(RuntimeError, match="synthetic projection failure"):
        legacy.integrator.step()
    with pytest.raises(RuntimeError, match="synthetic projection failure"):
        program.step()

    _assert_solver_state_equal(legacy, candidate)
    assert program.state.progress.completed_steps == 0
    assert program.state.representations.current.value == "spectral"
    assert program.workspace.active is False
    assert legacy.integrator.runtime_workspace.active is False
    assert program.failure_semantics.tensor_rollback is False


def test_callback_failure_aborts_workspace_without_committing_progress(tmp_path):
    _, _, program = _candidate(tmp_path, "callback_failure")

    def fail():
        raise RuntimeError("synthetic callback failure")

    with pytest.raises(RuntimeError, match="synthetic callback failure"):
        program.step(pre_update_callback=fail)

    assert program.state.progress.completed_steps == 0
    assert program.state.representations.current.value == "synchronized"
    assert program.workspace.active is False
    assert program.workspace.generation == 1


def test_wrong_rhs_aborts_before_evolved_tensor_update(tmp_path):
    _, binding, _ = _candidate(tmp_path, "wrong_rhs")
    before = binding.state.spectral.clone()
    bad_binding = dataclasses.replace(
        binding,
        operators=dataclasses.replace(
            binding.operators,
            explicit_rhs=lambda state, workspace, generation: torch.zeros(1),
        ),
    )
    program = build_plane_compiled_v2_step_program(bad_binding)

    with pytest.raises(ValueError, match="explicit RHS"):
        program.step()

    assert torch.equal(program.state.spectral, before)
    assert program.state.progress.completed_steps == 0
    assert program.workspace.active is False


def test_metadata_freezes_disconnected_execution_and_failure_contract(tmp_path):
    _, _, program = _candidate(tmp_path, "metadata")
    metadata = program.to_metadata()

    assert metadata["identity"] == "compiled_v2_projected_euler"
    assert metadata["connected_runtime"] is None
    assert metadata["workspace"]["plan"]["required_bytes"] == 0
    assert metadata["operation_order"] == [
        "prepare_algebraic",
        "pre_update_callback",
        "explicit_rhs",
        "spectral_add_dt_rhs",
        "spectral_divide_by_denominator",
        "project_dynamic_spectra",
        "inverse_dynamic_spectra",
        "scheduled_spectral_refresh",
        "commit_progress",
    ]
    assert metadata["failure_semantics"] == {
        "progress_commit_last": True,
        "workspace_generation_aborted": True,
        "tensor_rollback": False,
        "algebraic_side_effect_rollback": False,
        "callback_side_effect_rollback": False,
    }
    assert metadata["implicit_fallback"] is False


def test_step_hot_path_has_no_configuration_or_field_name_discovery():
    source = inspect.getsource(PlaneCompiledEulerStepProgram.step)
    tree = ast.parse(inspect.cleandoc(source))
    names = {
        node.id for node in ast.walk(tree) if isinstance(node, ast.Name)
    }
    attributes = {
        node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)
    }

    assert "json" not in names
    assert "getattr" not in names
    assert "name_to_idx" not in attributes
    assert "runtime_path" not in attributes
    assert "to_metadata" not in attributes


def test_compiled_step_remains_private_and_selector_free():
    import pssolver

    assert "PlaneCompiledEulerStepProgram" not in pssolver.__all__
    assert "PlaneCompiledEulerStepProgram" not in runtime_api.__all__
    assert not hasattr(runtime_api, "PlaneCompiledEulerStepProgram")
    assert {member.value for member in PlaneRuntimePath} == {
        "legacy_production",
        "separated_canary",
    }


def test_p53_qualification_record_matches_disconnected_scope():
    record = json.loads(QUALIFICATION.read_text(encoding="utf-8"))

    assert record["classification"] == (
        "PASS_P5_3_COMPILED_EULER_STEP_PROGRAM"
    )
    assert record["authorization"]["p5_3_implementation_authorized"] is True
    assert record["authorization"]["p5_4_implementation_authorized"] is False
    assert record["connection"]["runtime_selector_added"] is False
    assert record["connection"]["production_import_added"] is False
    assert record["connection"]["fallback_allowed"] is False
    assert record["eligibility"]["p5_3_complete"] is True
    assert record["eligibility"]["p5_4_locally_eligible"] is True
    assert record["eligibility"]["phase_6_eligible"] is False
