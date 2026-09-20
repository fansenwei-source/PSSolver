"""P3.3 fixed step-program characterization."""

from __future__ import annotations

import pytest
import torch

import pssolver
from pssolver.execution.state import IntegratorProgress, RuntimeState
from pssolver.execution.workspace import (
    WorkspacePlan,
    WorkspaceSlotSpec,
)
from pssolver.integrators.step_program import (
    ProjectedSemiImplicitEulerStepProgram,
)


def _state(*, completed_steps: int = 0) -> RuntimeState:
    physical = torch.tensor(
        [[[1.0, -2.0]], [[0.5, 3.0]]],
        dtype=torch.float64,
    )
    spectral = torch.complex(physical, torch.zeros_like(physical))
    refresh_count, refresh_step = divmod(completed_steps, 2)
    return RuntimeState(
        component_names=("q", "r"),
        physical=physical.clone(),
        spectral=spectral.clone(),
        progress=IntegratorProgress(
            dt=0.125,
            completed_steps=completed_steps,
            refresh_interval=2,
            refresh_step_count=refresh_step,
            refresh_count=refresh_count,
        ),
    )


def _workspace():
    return WorkspacePlan(
        slots=(
            WorkspaceSlotSpec(
                "rhs",
                (2, 1, 2),
                torch.complex128,
                "explicit RHS",
            ),
        ),
        device="cpu",
    ).allocate()


def _program(trace: list[str]) -> ProjectedSemiImplicitEulerStepProgram:
    def prepare(state, workspace, generation):
        trace.append("prepare")

    def rhs(state, workspace, generation):
        trace.append("rhs")
        value = workspace.buffer("rhs", token=generation)
        value.copy_(torch.full_like(state.spectral, 0.25 + 0.5j))
        workspace.publish("rhs", token=generation)
        return workspace.require("rhs", token=generation)

    def project(state, workspace, generation):
        trace.append("project")
        state.spectral[..., -1].zero_()

    def inverse(state, workspace, generation):
        trace.append("inverse")
        state.physical.copy_(state.spectral.real)

    def refresh(state, workspace, generation):
        trace.append("refresh")
        state.spectral.copy_(torch.complex(state.physical, torch.zeros_like(state.physical)))

    denominator = torch.full((2, 1, 2), 1.25 + 0.0j, dtype=torch.complex128)
    return ProjectedSemiImplicitEulerStepProgram(
        denominator=denominator,
        prepare_algebraic=prepare,
        explicit_rhs=rhs,
        project_dynamic_spectra=project,
        inverse_dynamic_spectra=inverse,
        refresh_dynamic_spectra=refresh,
    )


def test_step_program_is_provisional_and_not_a_root_export():
    assert "ProjectedSemiImplicitEulerStepProgram" not in pssolver.__all__
    assert _program([]).to_metadata()["connected_runtime"] is None


def test_step_program_preserves_frozen_order_and_update_arithmetic():
    trace: list[str] = []
    state = _state()
    workspace = _workspace()
    initial = state.spectral.clone()
    program = _program(trace)
    expected = initial.clone()
    rhs = torch.full_like(expected, 0.25 + 0.5j)
    expected.add_(state.progress.dt * rhs)
    expected.div_(program.denominator)
    expected[..., -1].zero_()

    program.step(
        state,
        workspace,
        pre_update_callback=lambda: trace.append("callback"),
    )

    assert trace == ["prepare", "callback", "rhs", "project", "inverse"]
    assert torch.equal(state.spectral, expected)
    assert torch.equal(state.physical, expected.real)
    assert state.progress.completed_steps == 1
    assert state.progress.refresh_step_count == 1
    assert state.progress.refresh_count == 0
    assert state.representations.generation == 1
    assert state.representations.current.value == "synchronized"
    assert workspace.active is False


def test_scheduled_refresh_occurs_after_inverse_and_before_progress_commit():
    trace: list[str] = []
    state = _state(completed_steps=1)
    workspace = _workspace()
    program = _program(trace)

    program.step(state, workspace)

    assert trace == ["prepare", "rhs", "project", "inverse", "refresh"]
    assert state.progress.completed_steps == 2
    assert state.progress.refresh_step_count == 0
    assert state.progress.refresh_count == 1
    assert state.representations.generation == 1
    assert state.representations.current.value == "synchronized"


def test_failed_operation_aborts_workspace_and_does_not_commit_clock():
    trace: list[str] = []
    state = _state()
    workspace = _workspace()

    def fail(state, workspace, generation):
        trace.append("fail")
        raise RuntimeError("synthetic failure")

    program = _program(trace)
    object.__setattr__(program, "project_dynamic_spectra", fail)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        program.step(state, workspace)
    assert trace == ["prepare", "rhs", "fail"]
    assert state.progress.completed_steps == 0
    assert workspace.active is False


def test_step_program_rejects_mismatched_rhs_and_denominator():
    trace: list[str] = []
    state = _state()
    workspace = _workspace()
    program = _program(trace)
    object.__setattr__(
        program,
        "denominator",
        torch.ones((1,), dtype=torch.complex128),
    )
    with pytest.raises(ValueError, match="denominator shape"):
        program.step(state, workspace)

    program = _program([])
    object.__setattr__(
        program,
        "explicit_rhs",
        lambda state, workspace, generation: torch.zeros((1,)),
    )
    with pytest.raises(ValueError, match="explicit RHS"):
        program.step(state, workspace)
    assert state.progress.completed_steps == 0
    assert workspace.active is False
