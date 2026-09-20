"""Phase 3.0/3.1 ownership and disconnected-state gates."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import pssolver
import pssolver.execution as execution
from pssolver.execution.state import (
    CurrentRepresentation,
    IntegratorProgress,
    RepresentationLedger,
    RuntimeState,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INVENTORY_PATH = (
    PROJECT_ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_3_state_inventory.json"
)
RUNTIME_MODULES = (
    PROJECT_ROOT / "pssolver" / "runtime" / "plane_legacy.py",
    PROJECT_ROOT / "pssolver" / "runtime" / "plane_beris_edwards.py",
)
STATE_BACKED_CORE = (
    PROJECT_ROOT / "pssolver" / "integrators" / "state_backed.py"
)


def test_phase3_inventory_freezes_ownership_and_connection_order():
    inventory = json.loads(INVENTORY_PATH.read_text(encoding="utf-8"))
    assert inventory["schema_version"] == 1
    assert inventory["phase"] == 3
    assert inventory["status"] == "P3_5_PRODUCTION_FACADE_CONNECTED_LOCAL"
    assert inventory["checkpoint"] == {
        "format_version": 1,
        "must_remain_readable": True,
        "persistent_payload": [
            "evolved_spatial_Q",
            "evolved_spectral_Q",
            "completed_steps",
            "spectral_refresh_interval",
            "integrator_step_count",
            "integrator_refresh_count",
            "declared_backend_restart_state",
        ],
    }
    assert inventory["connection_policy"] == {
        "legacy_production_touched_before_p3_5": False,
        "p3_4_h100_qualified": True,
        "production_default_changed": False,
        "p3_1_runtime_imports_state": False,
        "p3_2_runtime_imports_workspace": False,
        "p3_3_step_program_connected": False,
        "p3_4_separated_canary_connected": True,
        "p3_5_legacy_production_connected": True,
        "separated_canary_touched_before_p3_4": False,
    }
    assert inventory["timestep_oracle"] == [
        "synchronize_algebraic_if_needed",
        "pre_update_callback",
        "explicit_rhs",
        "spectral_add_dt_rhs",
        "spectral_divide_by_denominator",
        "project_dynamic_spectra",
        "inverse_dynamic_spectra",
        "scheduled_spectral_refresh",
        "commit_progress",
    ]


def test_state_is_not_promoted_and_shared_core_owns_runtime_imports():
    provisional = {
        "CurrentRepresentation",
        "IntegratorProgress",
        "RepresentationLedger",
        "RuntimeState",
    }
    assert provisional.isdisjoint(execution.__all__)
    assert provisional.isdisjoint(pssolver.__all__)
    legacy_source = RUNTIME_MODULES[0].read_text(encoding="utf-8")
    selector_source = RUNTIME_MODULES[1].read_text(encoding="utf-8")
    core_source = STATE_BACKED_CORE.read_text(encoding="utf-8")
    assert "pssolver.execution.state" not in legacy_source
    assert "RuntimeState" not in legacy_source
    assert "from pssolver.execution.state import RuntimeState" in selector_source
    assert "from pssolver.execution.state import (" in core_source
    assert "StateBackedProjectedIntegratorMixin" in legacy_source


def test_representation_ledger_models_updates_and_synchronization():
    ledger = RepresentationLedger()
    assert ledger.current is CurrentRepresentation.SYNCHRONIZED
    assert ledger.to_metadata() == {
        "generation": 0,
        "physical_generation": 0,
        "spectral_generation": 0,
        "current": "synchronized",
    }

    assert ledger.mark_spectral_updated() == 1
    assert ledger.current is CurrentRepresentation.SPECTRAL
    with pytest.raises(RuntimeError, match="physical representation is stale"):
        ledger.require_physical_current()
    ledger.mark_physical_synchronized()
    assert ledger.current is CurrentRepresentation.SYNCHRONIZED

    assert ledger.mark_physical_updated() == 2
    assert ledger.current is CurrentRepresentation.PHYSICAL
    with pytest.raises(RuntimeError, match="spectral representation is stale"):
        ledger.require_spectral_current()
    ledger.mark_spectral_synchronized()
    assert ledger.current is CurrentRepresentation.SYNCHRONIZED


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (
            {
                "generation": 2,
                "physical_generation": 1,
                "spectral_generation": 1,
            },
            "at least one representation",
        ),
        (
            {
                "generation": 1,
                "physical_generation": 2,
                "spectral_generation": 1,
            },
            "cannot lead state",
        ),
    ],
)
def test_representation_ledger_rejects_invalid_generation_states(kwargs, match):
    with pytest.raises(ValueError, match=match):
        RepresentationLedger(**kwargs)


def test_integrator_progress_matches_checkpoint_v1_counter_rules():
    progress = IntegratorProgress(
        dt=0.005,
        completed_steps=7,
        refresh_interval=3,
        refresh_step_count=1,
        refresh_count=2,
    )
    assert progress.time == pytest.approx(0.035)
    assert progress.refresh_due_after_next_step is False
    progress.commit_step(refreshed=False)
    assert progress.completed_steps == 8
    assert progress.refresh_due_after_next_step is True
    with pytest.raises(RuntimeError, match="refresh outcome"):
        progress.commit_step(refreshed=False)
    progress.commit_step(refreshed=True)
    assert progress.to_metadata()["spectral_refresh"] == {
        "interval": 3,
        "step_count": 0,
        "refresh_count": 3,
    }

    disabled = IntegratorProgress(
        dt=0.01,
        completed_steps=4,
        refresh_interval=None,
        refresh_step_count=4,
        refresh_count=0,
    )
    disabled.commit_step(refreshed=False)
    assert disabled.completed_steps == 5
    assert disabled.refresh_step_count == 5


@pytest.mark.parametrize(
    "kwargs",
    [
        {
            "dt": 0.01,
            "completed_steps": 4,
            "refresh_interval": 3,
            "refresh_step_count": 0,
            "refresh_count": 1,
        },
        {
            "dt": 0.01,
            "completed_steps": 4,
            "refresh_interval": None,
            "refresh_step_count": 4,
            "refresh_count": 1,
        },
    ],
)
def test_integrator_progress_rejects_inconsistent_counters(kwargs):
    with pytest.raises(ValueError, match="counters are inconsistent"):
        IntegratorProgress(**kwargs)


def test_runtime_state_adopts_storage_without_copy_or_conversion():
    physical = torch.arange(30, dtype=torch.float64).reshape(3, 1, 2, 5)
    spectral = torch.complex(physical, torch.zeros_like(physical))
    warm_start = torch.ones((1, 2, 5), dtype=torch.float64)
    progress = IntegratorProgress(dt=0.01, refresh_interval=20)
    ledger = RepresentationLedger()
    state = RuntimeState(
        component_names=("Qxx", "Qxy", "Qxz"),
        physical=physical,
        spectral=spectral,
        progress=progress,
        representations=ledger,
        persistent_algebraic={"pressure_guess": warm_start},
    )

    assert state.physical is physical
    assert state.spectral is spectral
    assert state.progress is progress
    assert state.representations is ledger
    assert state.physical_component("Qxy").untyped_storage().data_ptr() == (
        physical.untyped_storage().data_ptr()
    )
    assert state.spectral_component("Qxz").untyped_storage().data_ptr() == (
        spectral.untyped_storage().data_ptr()
    )
    assert state.persistent_algebraic["pressure_guess"] is warm_start

    physical[1, 0, 0, 0] = -7.0
    assert state.physical_component("Qxy")[0, 0, 0].item() == -7.0
    metadata = state.to_metadata()
    assert metadata["component_names"] == ["Qxx", "Qxy", "Qxz"]
    assert metadata["persistent_algebraic_names"] == ["pressure_guess"]


def test_runtime_state_validates_layout_device_and_names():
    physical = torch.zeros((2, 1, 4), dtype=torch.float64)
    spectral = torch.zeros((2, 1, 3), dtype=torch.complex128)
    progress = IntegratorProgress(dt=0.01)

    with pytest.raises(ValueError, match="unique"):
        RuntimeState(
            component_names=("q", "q"),
            physical=physical,
            spectral=spectral,
            progress=progress,
        )
    with pytest.raises(ValueError, match="leading dimension"):
        RuntimeState(
            component_names=("q",),
            physical=physical,
            spectral=spectral,
            progress=progress,
        )
    with pytest.raises(ValueError, match="batch dimensions"):
        RuntimeState(
            component_names=("q", "r"),
            physical=physical,
            spectral=torch.zeros((2, 2, 3), dtype=torch.complex128),
            progress=progress,
        )
