"""P2.5 pre-migration oracles for the legacy Plane runtime consumer."""

from __future__ import annotations

from pathlib import Path

import torch

from pssolver.configuration import create_plane_beris_edwards_run_spec
from pssolver.configuration.legacy_plane import translate_plane
from pssolver.engine.engine import Engine
from pssolver.models.active_nematics import Q_COMPONENTS
from tests.v0_5_s27_plane_v04_checkpoint import (
    PlaneTestState,
    capture_plane_checkpoint,
    restore_plane_checkpoint,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LEGACY_RUNTIME_PATH = PROJECT_ROOT / "pssolver" / "runtime" / "plane_legacy.py"

FLAT_RUNTIME_READS = {
    "beta",
    "dealias_rule",
    "diagnostics",
    "disable_q_gradient_reuse",
    "dt",
    "dtype",
    "eta",
    "flow_alignment",
    "friction_mode_fric",
    "height",
    "ldg_a",
    "ldg_b",
    "ldg_c",
    "lx",
    "ly",
    "molecular_field_linear_space",
    "nx",
    "ny",
    "nz",
    "pointwise_execution",
    "projected_transform_execution",
    "shendruk_preset",
    "spectral_refresh_interval_steps",
    "spectral_storage",
    "stress_divergence_sum_space",
    "transform_execution_order",
    "zero_mode_policy",
}
COMPONENT_OWNERS = {
    "geometry": {"height", "lx", "ly", "nx", "ny", "nz"},
    "numerics": {
        "dealias_rule",
        "dtype",
        "projected_transform_execution",
        "spectral_storage",
        "transform_execution_order",
    },
    "physics": {
        "beta",
        "eta",
        "flow_alignment",
        "friction_mode_fric",
        "ldg_a",
        "ldg_b",
        "ldg_c",
        "shendruk_preset",
        "zero_mode_policy",
    },
    "time_stepping": {"dt", "spectral_refresh_interval_steps"},
    "execution": {
        "disable_q_gradient_reuse",
        "molecular_field_linear_space",
        "pointwise_execution",
        "stress_divergence_sum_space",
    },
    "workflow": {"diagnostics"},
}


def _spec(tmp_path: Path, **overrides: object):
    values = {
        "activity_number": 18.0,
        "output_dir": tmp_path / "unused",
        "device": "cpu",
        "dtype": "float64",
        "pointwise_execution": "eager",
        "nx": 8,
        "ny": 8,
        "nz": 8,
        "lx": 8.0,
        "ly": 9.0,
        "height": 20.0,
        "steps": 4,
        "save_start_step": 0,
        "save_interval": 1,
        "diagnostic_interval": 1,
        "spectral_refresh_steps": 3,
    }
    values.update(overrides)
    return create_plane_beris_edwards_run_spec(**values)


def _initial_values() -> dict[str, torch.Tensor]:
    coordinate = torch.arange(8**3, dtype=torch.float64).reshape(8, 8, 8)
    coordinate = (coordinate - coordinate.mean()) / coordinate.numel()
    return {
        name: coordinate.mul((index + 1) * 1.0e-3)
        for index, name in enumerate(Q_COMPONENTS)
    }












def _build(spec):
    engine = Engine(translate_plane(spec), device="cpu", dtype=torch.float64)
    return PlaneTestState(engine, engine.initial_state(_initial_values()))


def test_short_cpu_trajectory_and_checkpoint_restart_are_byte_exact(tmp_path):
    spec = _spec(tmp_path)
    continuous = _build(spec)
    segment = _build(spec)
    resumed = _build(spec)

    for _ in range(4):
        continuous.engine.step_(continuous.state)
    for _ in range(2):
        segment.engine.step_(segment.state)
    checkpoint = capture_plane_checkpoint(
        segment,
        run_spec=spec,
    )
    assert restore_plane_checkpoint(
        resumed,
        checkpoint,
        run_spec=spec,
    ) == 2
    for _ in range(2):
        resumed.engine.step_(resumed.state)
    continuous.engine.prepare_constraints_(continuous.state)
    resumed.engine.prepare_constraints_(resumed.state)

    assert continuous.completed_steps == resumed.completed_steps == 4
