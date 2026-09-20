"""P3.4 local gates for the separated-canary state connection."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from pssolver.execution import AlgebraicExecutionPolicy
from pssolver.experimental.integrators import (
    ProjectedSemiImplicitEulerIntegrator,
    StateBackedProjectedSemiImplicitEulerIntegrator,
)
from pssolver.experimental.plane_shadow_driver import (
    _build_plane_shadow_runtime_from_production_metadata,
)
from pssolver.models.active_nematics import Q_COMPONENTS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PRODUCTION_SCRIPT = PROJECT_ROOT / "Plane_beris_edwards_stokes.py"


def _production_reference(tmp_path: Path) -> tuple[dict[str, object], dict]:
    output = tmp_path / "legacy_reference"
    result = subprocess.run(
        [
            sys.executable,
            str(PRODUCTION_SCRIPT),
            "--activity-number",
            "18",
            "--output-dir",
            str(output),
            "--device",
            "cpu",
            "--dtype",
            "float64",
            "--pointwise-execution",
            "eager",
            "--nx",
            "8",
            "--ny",
            "8",
            "--nz",
            "8",
            "--steps",
            "1",
            "--save-start-step",
            "0",
            "--save-interval",
            "1",
            "--spectral-refresh-steps",
            "2",
            "--save-hydrodynamics",
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    q = np.load(output / "Q_0.npy", allow_pickle=False)
    initial = {
        name: torch.from_numpy(np.array(q[..., index], copy=True))
        for index, name in enumerate(Q_COMPONENTS)
    }
    return metadata, initial


def _runtime(metadata, *, phase3: bool):
    runtime, comparison = _build_plane_shadow_runtime_from_production_metadata(
        metadata,
        device="cpu",
        algebraic_execution_policy=AlgebraicExecutionPolicy.batched(),
        connect_phase3_runtime_state=phase3,
    )
    assert comparison.compatible is True
    return runtime


def _assert_all_field_storage_equal(left, right):
    assert torch.equal(left.solver.fields.spatial, right.solver.fields.spatial)
    assert torch.equal(left.solver.fields.spectral, right.solver.fields.spectral)


def test_phase3_canary_matches_preconnection_runtime_step_by_step(tmp_path):
    metadata, initial = _production_reference(tmp_path)
    reference = _runtime(metadata, phase3=False)
    candidate = _runtime(metadata, phase3=True)
    reference.reset(initial)
    candidate.reset(initial)

    assert type(reference.solver.integrator) is (
        ProjectedSemiImplicitEulerIntegrator
    )
    assert type(candidate.solver.integrator) is (
        StateBackedProjectedSemiImplicitEulerIntegrator
    )
    _assert_all_field_storage_equal(reference, candidate)

    for completed_steps in range(1, 7):
        reference.solver.run(1)
        candidate.solver.run(1)
        _assert_all_field_storage_equal(reference, candidate)
        state = candidate.solver.integrator.runtime_state
        assert state.progress.completed_steps == completed_steps
        assert state.representations.generation == completed_steps
        assert state.representations.current.value == "synchronized"
        assert candidate.solver.integrator.step_count == (
            reference.solver.integrator.step_count
        )
        assert candidate.solver.integrator.refresh_count == (
            reference.solver.integrator.refresh_count
        )


def test_phase3_canary_adopts_reallocated_reset_storage_without_copy(tmp_path):
    metadata, initial = _production_reference(tmp_path)
    runtime = _runtime(metadata, phase3=True)
    runtime.reset(initial)
    integrator = runtime.solver.integrator
    state = integrator.runtime_state

    assert state.physical.untyped_storage().data_ptr() == (
        runtime.solver.fields.spatial.untyped_storage().data_ptr()
    )
    assert state.spectral.untyped_storage().data_ptr() == (
        runtime.solver.fields.spectral.untyped_storage().data_ptr()
    )
    old_physical = state.physical
    runtime.reset(initial)
    rebound = integrator.runtime_state
    assert rebound.physical is not old_physical
    assert rebound.physical.untyped_storage().data_ptr() == (
        runtime.solver.fields.spatial.untyped_storage().data_ptr()
    )
    assert rebound.progress.completed_steps == 0


def test_phase3_connection_metadata_is_explicit_and_workspace_is_bounded(tmp_path):
    metadata, _ = _production_reference(tmp_path)
    reference = _runtime(metadata, phase3=False)
    candidate = _runtime(metadata, phase3=True)

    assert "phase3_execution" not in reference.to_metadata()["time_integration"]
    value = candidate.to_metadata()["time_integration"]["phase3_execution"]
    assert value["connection_stage"] == "P3.4_separated_canary"
    assert value["legacy_storage_adopted_without_copy"] is True
    assert value["production_default_changed"] is False
    assert value["workspace"]["plan"]["required_bytes"] == 0
    assert value["step_program"]["connected_runtime"] is None
