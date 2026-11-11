"""R4 ratchets for exact-order stepping and spectral-refresh progress."""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pssolver.engine.progress import (
    IntegratorProgress,
    RuntimeState,
    _completed_steps_from_refresh_counters,
    _refresh_counters_are_consistent,
    _refresh_counters_from_completed_steps,
    _refresh_due_after_next_step,
)
from pssolver.integrator import SemiImplicitEulerIntegrator
from pssolver.integrators.state_backed import (
    _integrator_completed_steps,
    _restore_integrator_progress,
)
from pssolver.integrators.step_program import (
    ProjectedSemiImplicitEulerStepProgram,
)
from qualification.s25_source_paths import source_path
from tests.v0_3_s03_successor import resolve_s03_successor


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("interval", [None, 1, 2, 7])
@pytest.mark.parametrize("completed_steps", [0, 1, 2, 6, 7, 20])
def test_shared_refresh_clock_round_trips_exactly(interval, completed_steps):
    step_count, refresh_count = _refresh_counters_from_completed_steps(
        completed_steps,
        interval,
    )
    assert _completed_steps_from_refresh_counters(
        refresh_interval=interval,
        refresh_step_count=step_count,
        refresh_count=refresh_count,
    ) == completed_steps
    assert _refresh_counters_are_consistent(
        completed_steps=completed_steps,
        refresh_interval=interval,
        refresh_step_count=step_count,
        refresh_count=refresh_count,
    )
    assert _refresh_due_after_next_step(
        refresh_interval=interval,
        refresh_step_count=step_count,
    ) is (
        interval is not None and step_count + 1 >= interval
    )


def test_progress_rejects_inconsistent_clocks_without_normalizing_them():
    with pytest.raises(ValueError, match="spectral-refresh counters"):
        IntegratorProgress(
            dt=0.01,
            completed_steps=4,
            refresh_interval=3,
            refresh_step_count=0,
            refresh_count=1,
        )
    with pytest.raises(ValueError, match="spectral-refresh counters"):
        IntegratorProgress(
            dt=0.01,
            completed_steps=4,
            refresh_interval=None,
            refresh_step_count=4,
            refresh_count=1,
        )


def test_legacy_refresh_failure_keeps_the_original_counter_boundary():
    integrator = object.__new__(SemiImplicitEulerIntegrator)
    integrator._spectral_refresh_interval = 2
    integrator.step_count = 1
    integrator.refresh_count = 3

    def fail_refresh():
        raise RuntimeError("refresh failed")

    integrator._refresh_dynamic_spectra = fail_refresh
    with pytest.raises(RuntimeError, match="refresh failed"):
        integrator._advance_spectral_refresh_clock()
    assert integrator.step_count == 2
    assert integrator.refresh_count == 3


def test_runtime_adapters_share_completed_and_restore_progress_primitives():
    progress = IntegratorProgress(
        dt=0.01,
        completed_steps=5,
        refresh_interval=3,
        refresh_step_count=2,
        refresh_count=1,
    )
    state_backed = SimpleNamespace(
        runtime_state=RuntimeState(
            component_names=("q",),
            physical=torch.zeros(1),
            spectral=torch.zeros(1),
            progress=progress,
        )
    )
    assert _integrator_completed_steps(state_backed) == 5

    legacy = SimpleNamespace(
        spectral_refresh_interval=3,
        step_count=2,
        refresh_count=1,
    )
    assert _integrator_completed_steps(legacy) == 5

    calls = []

    def set_interval(value):
        calls.append(("interval", value))

    def restore(completed, *, static_fields_are_current):
        calls.append(("restore", completed, static_fields_are_current))

    legacy.set_spectral_refresh_interval = set_interval
    legacy.restore_progress = restore
    _restore_integrator_progress(
        legacy,
        completed_steps=5,
        spectral_refresh_interval=3,
        integrator_step_count=2,
        integrator_refresh_count=1,
        mismatch_message="exact adapter message",
    )
    assert calls == [("interval", 3), ("restore", 5, True)]

    with pytest.raises(RuntimeError, match="exact adapter message"):
        _restore_integrator_progress(
            legacy,
            completed_steps=5,
            spectral_refresh_interval=3,
            integrator_step_count=0,
            integrator_refresh_count=1,
            mismatch_message="exact adapter message",
        )


def test_projected_programs_delegate_to_the_same_exact_order_executor():
    generic = inspect.getsource(ProjectedSemiImplicitEulerStepProgram.step)
    assert "_execute_projected_euler_step(" in generic
    for source in (generic,):
        assert ".add_(" not in source
        assert ".div_(" not in source
        assert "commit_step(" not in source


def test_projected_runtimes_inherit_one_inverse_and_refresh_implementation():
    from pssolver.engine import program

    source = inspect.getsource(program.advance)
    assert "_evolved_transform(" in source
    assert "_refresh_due_after_next_step(" in source


def test_control_facing_functional_paths_remain_byte_identical_to_parent():
    expected = {
        "pssolver/functional/channel_activity_runtime.py": (
            "a5cb43e59467172271b4db393a022da9aec07593aca6d8d6f357dbce692b9365"
        ),
        "pssolver/functional/periodic_activity.py": (
            "ada4c53ffb38e24843ad1a2f74c2f5dec99e59196d0b57d916faaff62b14b0cd"
        ),
        "pssolver/models/active_nematics/beris_edwards.py": (
            "2b2e8d9a0555e1337cc314a5776b9fb5073eab1bafa6087f70b468eabe62906d"
        ),
        "pssolver/models/active_nematics/q_tensor.py": (
            "adae50e10adff6d0c6daeaaba215afd9664ebbcf9bfe0471fe6880d8f511601b"
        ),
    }
    for relative, digest in expected.items():
        assert hashlib.sha256(source_path(ROOT, relative).read_bytes()).hexdigest() == resolve_s03_successor(relative, digest)
