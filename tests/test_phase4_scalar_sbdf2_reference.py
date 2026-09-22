"""Phase 4.2 disconnected scalar SBDF2 reference-path gates."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch

import pssolver
import pssolver.integrators as integrators
from pssolver.core.integrators import IntegratorScheme, IntegratorSpec
from pssolver.integrators.scalar_reference import (
    ScalarPeriodicReactionDiffusion,
    ScalarPeriodicReferenceStepper,
    ScalarReferenceState,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_PATH = (
    PROJECT_ROOT / "pssolver" / "integrators" / "scalar_reference.py"
)
P42_RECORD_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_4_p42_scalar_sbdf2_reference.json"
)
RUNTIME_PATHS = tuple((PROJECT_ROOT / "pssolver" / "runtime").glob("*.py"))

STARTUP_OPERATIONS = (
    "require_current_state_and_history",
    "prepare_algebraic",
    "pre_update_callback",
    "evaluate_current_explicit_rhs",
    "assemble_euler_implicit_rhs",
    "solve_implicit_operator",
    "project_dynamic_spectra",
    "inverse_dynamic_spectra",
    "scheduled_spectral_refresh",
    "commit_progress",
    "commit_integrator_history",
)

SBDF2_OPERATIONS = (
    "require_current_state_and_history",
    "prepare_algebraic",
    "pre_update_callback",
    "evaluate_current_explicit_rhs",
    "assemble_sbdf2_implicit_rhs",
    "solve_implicit_operator",
    "project_dynamic_spectra",
    "inverse_dynamic_spectra",
    "scheduled_spectral_refresh",
    "commit_progress",
    "commit_integrator_history",
)


def _stepper(
    *,
    scheme: IntegratorScheme = IntegratorScheme.SBDF2,
    dt: float = 0.01,
) -> ScalarPeriodicReferenceStepper:
    model = ScalarPeriodicReactionDiffusion()
    spec = (
        IntegratorSpec.sbdf2(dt=dt)
        if scheme is IntegratorScheme.SBDF2
        else IntegratorSpec.projected_semi_implicit_euler(dt=dt)
    )
    return ScalarPeriodicReferenceStepper(model=model, spec=spec)


def _assert_state_unchanged(
    state: ScalarReferenceState,
    physical_before: torch.Tensor,
    spectral_before: torch.Tensor,
    history_before: object,
) -> None:
    assert torch.equal(state.physical, physical_before)
    assert torch.equal(state.native_spectrum, spectral_before)
    assert state.history is history_before


def test_scalar_reference_module_does_not_import_production_runtime():
    tree = ast.parse(REFERENCE_PATH.read_text(encoding="utf-8"))
    imported_modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
    assert all(not name.startswith("pssolver.runtime") for name in imported_modules)
    assert all(not name.startswith("pssolver.execution") for name in imported_modules)
    assert "pssolver.integrators.step_program" not in imported_modules


def test_scalar_model_freezes_periodic_split_and_exact_solution():
    model = ScalarPeriodicReactionDiffusion(
        point_count=32,
        length=2.0 * math.pi,
        diffusivity=0.2,
        reaction_rate=0.15,
        initial_mode=3,
        initial_amplitude=0.75,
    )
    assert model.analytic_growth_rate == pytest.approx(-1.65)
    initial = model.initial_physical(dtype=torch.float64, device=torch.device("cpu"))
    exact = model.exact_physical(
        0.4,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    assert torch.allclose(exact, initial * math.exp(-1.65 * 0.4))
    metadata = model.to_metadata()
    assert metadata["split"] == {
        "implicit": "diffusion",
        "explicit": "linear_reaction",
    }


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("point_count", 7, "point_count"),
        ("point_count", 9, "point_count"),
        ("length", 0.0, "length"),
        ("diffusivity", -0.1, "diffusivity"),
        ("reaction_rate", math.inf, "reaction_rate"),
        ("initial_mode", 16, "initial_mode"),
        ("initial_amplitude", math.nan, "initial_amplitude"),
    ],
)
def test_scalar_model_rejects_invalid_parameters(field, value, message):
    arguments = {field: value}
    with pytest.raises(ValueError, match=message):
        ScalarPeriodicReactionDiffusion(**arguments)


def test_sbdf2_startup_uses_projected_semi_implicit_euler_and_commits_history():
    stepper = _stepper(dt=0.01)
    initial = stepper.initial_state()
    result = stepper.step(initial)
    linear = stepper.model.linear_eigenvalues(
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    rhs = stepper.model.reaction_rate * initial.native_spectrum
    expected = (initial.native_spectrum + 0.01 * rhs) / (1.0 - 0.01 * linear)
    assert result.scheme_used is IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
    assert result.startup_step is True
    assert result.operation_order == STARTUP_OPERATIONS
    assert torch.equal(result.state.native_spectrum, expected)
    assert result.state.completed_steps == 1
    assert result.state.history is not None
    assert (
        result.state.history.previous_evolved_native_spectrum
        is initial.native_spectrum
    )
    assert torch.equal(
        result.state.history.previous_explicit_native_spectral_rhs,
        rhs,
    )
    assert result.state.history.source_completed_steps == 0


def test_second_step_uses_frozen_sbdf2_formula_and_replaces_history_last():
    stepper = _stepper(dt=0.01)
    initial = stepper.initial_state()
    startup = stepper.step(initial).state
    old_history = startup.history
    assert old_history is not None
    result = stepper.step(startup)
    current_rhs = stepper.model.reaction_rate * startup.native_spectrum
    linear = stepper.model.linear_eigenvalues(
        dtype=torch.float64,
        device=torch.device("cpu"),
    )
    assembled = (
        2.0 / 0.01 * startup.native_spectrum
        - 0.5 / 0.01 * old_history.previous_evolved_native_spectrum
        + 2.0 * current_rhs
        - old_history.previous_explicit_native_spectral_rhs
    )
    expected = assembled / (3.0 / (2.0 * 0.01) - linear)
    assert result.scheme_used is IntegratorScheme.SBDF2
    assert result.startup_step is False
    assert result.operation_order == SBDF2_OPERATIONS
    assert torch.equal(result.state.native_spectrum, expected)
    assert result.state.history is not old_history
    assert result.state.history is not None
    assert (
        result.state.history.previous_evolved_native_spectrum
        is startup.native_spectrum
    )
    assert result.state.history.source_completed_steps == 1


def test_callback_occurs_after_prepare_and_before_explicit_rhs():
    stepper = _stepper()
    events = []

    def callback() -> None:
        events.append("callback_body")

    result = stepper.step(
        stepper.initial_state(),
        pre_update_callback=callback,
        stage_observer=events.append,
    )
    assert result.operation_order == STARTUP_OPERATIONS
    assert events.index("prepare_algebraic") < events.index("callback_body")
    assert events.index("callback_body") < events.index("pre_update_callback")
    assert events.index("pre_update_callback") < events.index(
        "evaluate_current_explicit_rhs"
    )


def test_refresh_clock_and_refresh_operation_are_deterministic():
    stepper = _stepper()
    state0 = stepper.initial_state(refresh_interval=2)
    first = stepper.step(state0)
    second = stepper.step(first.state)
    assert first.refreshed is False
    assert first.state.refresh_step_count == 1
    assert first.state.refresh_count == 0
    assert second.refreshed is True
    assert second.state.refresh_step_count == 0
    assert second.state.refresh_count == 1
    refreshed = stepper._project(torch.fft.rfft(second.state.physical))
    assert torch.equal(second.state.native_spectrum, refreshed)


@pytest.mark.parametrize("failed_stage", STARTUP_OPERATIONS)
def test_startup_failure_is_atomic_at_every_observable_stage(failed_stage):
    stepper = _stepper()
    state = stepper.initial_state(refresh_interval=3)
    physical_before = state.physical.clone()
    spectral_before = state.native_spectrum.clone()
    history_before = state.history

    def observer(stage: str) -> None:
        if stage == failed_stage:
            raise RuntimeError(f"injected failure at {stage}")

    with pytest.raises(RuntimeError, match="injected failure"):
        stepper.step(state, stage_observer=observer)
    _assert_state_unchanged(
        state,
        physical_before,
        spectral_before,
        history_before,
    )
    assert state.completed_steps == 0
    assert state.refresh_step_count == 0
    assert state.refresh_count == 0


@pytest.mark.parametrize("failed_stage", SBDF2_OPERATIONS)
def test_multistep_failure_is_atomic_at_every_observable_stage(failed_stage):
    stepper = _stepper()
    state = stepper.step(stepper.initial_state(refresh_interval=3)).state
    physical_before = state.physical.clone()
    spectral_before = state.native_spectrum.clone()
    history_before = state.history

    def observer(stage: str) -> None:
        if stage == failed_stage:
            raise RuntimeError(f"injected failure at {stage}")

    with pytest.raises(RuntimeError, match="injected failure"):
        stepper.step(state, stage_observer=observer)
    _assert_state_unchanged(
        state,
        physical_before,
        spectral_before,
        history_before,
    )
    assert state.completed_steps == 1
    assert state.refresh_step_count == 1
    assert state.refresh_count == 0


def test_callback_failure_is_atomic():
    stepper = _stepper()
    state = stepper.initial_state()
    physical_before = state.physical.clone()
    spectral_before = state.native_spectrum.clone()

    def fail() -> None:
        raise RuntimeError("callback failed")

    with pytest.raises(RuntimeError, match="callback failed"):
        stepper.step(state, pre_update_callback=fail)
    _assert_state_unchanged(state, physical_before, spectral_before, None)


def test_euler_reference_has_no_multistep_history():
    stepper = _stepper(scheme=IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER)
    state = stepper.run(steps=3, refresh_interval=None)
    assert state.completed_steps == 3
    assert state.history is None
    assert state.refresh_step_count == 3
    result = stepper.step(state)
    assert result.scheme_used is IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
    assert result.startup_step is False
    assert result.operation_order == STARTUP_OPERATIONS[:-1]


def test_sbdf2_rejects_missing_or_incompatible_history_before_advance():
    stepper = _stepper()
    advanced = stepper.step(stepper.initial_state()).state
    missing = replace(advanced, history=None)
    with pytest.raises(ValueError, match="requires complete history"):
        stepper.step(missing)
    assert advanced.history is not None
    incompatible = replace(
        advanced,
        history=replace(advanced.history, dt=0.02),
    )
    with pytest.raises(ValueError, match="changing dt"):
        stepper.step(incompatible)


def test_reference_metadata_freezes_scope_and_operation_order():
    stepper = _stepper(dt=0.0125)
    metadata = stepper.to_metadata()
    assert metadata["reference_only"] is True
    assert metadata["cpu_only"] is True
    assert metadata["functional_state_transition"] is True
    assert metadata["failure_atomic"] is True
    assert metadata["callback_timing"] == "after_prepare_before_explicit_rhs"
    assert metadata["history_commit_position"] == "last"
    assert metadata["startup_operation_order"] == list(STARTUP_OPERATIONS)
    assert metadata["sbdf2_operation_order"] == list(SBDF2_OPERATIONS)
    assert metadata["connection"] == {
        "runtime_state": False,
        "step_program": False,
        "plane": False,
        "production_checkpoint": False,
    }
    state = stepper.step(stepper.initial_state(refresh_interval=4)).state
    state_metadata = state.to_metadata(dt=stepper.spec.dt)
    assert state_metadata["completed_steps"] == 1
    assert state_metadata["time"] == 0.0125
    assert state_metadata["history"]["scheme"] == "sbdf2"


def test_scalar_reference_types_are_direct_import_only_and_disconnected():
    names = {
        "ScalarPeriodicReactionDiffusion",
        "ScalarPeriodicReferenceStepper",
        "ScalarReferenceState",
        "ScalarReferenceStepResult",
    }
    assert names.isdisjoint(pssolver.__all__)
    assert names.isdisjoint(integrators.__all__)
    for path in RUNTIME_PATHS:
        source = path.read_text(encoding="utf-8")
        assert "pssolver.integrators.scalar_reference" not in source


def test_phase4_p42_machine_record_matches_reference_only_contract():
    record = json.loads(P42_RECORD_PATH.read_text(encoding="utf-8"))
    assert record["status"] == "P4_2_LOCAL_COMPLETE_REFERENCE_ONLY"
    assert record["baseline_commit"] == (
        "9273c7b5fb73d3f04be7dc1bf74befb14656b3d3"
    )
    assert record["model"]["identity"] == "scalar_periodic_reaction_diffusion"
    assert record["model"]["implicit_term"] == "diffusion"
    assert record["model"]["explicit_term"] == "linear_reaction"
    assert record["semantics"]["startup"] == "projected_semi_implicit_euler"
    assert record["semantics"]["multistep"] == "constant_step_sbdf2"
    assert record["semantics"]["failure_atomic"] is True
    assert record["connection"] == {
        "runtime_state_connected": False,
        "step_program_connected": False,
        "plane_connected": False,
        "checkpoint_v1_changed": False,
        "package_root_exported": False,
        "integrators_package_exported": False,
    }
    assert record["validation"] == {
        "focused_tests_passed": 70,
        "complete_tests_passed": 1726,
        "subtests_passed": 8,
        "failures": 0,
        "archive_sources_verified": 54,
        "git_diff_check_passed": True,
    }
    assert record["eligibility"] == {
        "p4_2_complete": True,
        "p4_3_authorized": True,
        "phase_4_complete": False,
        "phase_5_authorized": False,
        "production_default_changed": False,
        "separated_canary_promoted": False,
    }
