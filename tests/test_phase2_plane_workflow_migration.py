"""P2.5 oracles for the Plane workflow consumer migration."""

from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from pssolver.configuration import (
    PlaneRuntimePath,
    create_plane_beris_edwards_run_spec,
)
from pssolver.workflows import plane_beris_edwards as workflow_module
from pssolver.workflows.plane_beris_edwards import PlaneBerisEdwardsWorkflow


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = (
    PROJECT_ROOT / "pssolver" / "workflows" / "plane_beris_edwards.py"
)
FLAT_WORKFLOW_READS = {
    "checkpoint_interval",
    "diagnostic_interval",
    "diagnostics",
    "eta",
    "friction_mode_fric",
    "restart_from",
    "runtime_path",
    "save_hydrodynamics",
    "save_interval",
    "save_start_step",
    "steps",
    "zero_mode_policy",
}
FACADE_METHOD_READS = {"runtime_identity_sha256"}


class _Adapter:
    runtime_path = PlaneRuntimePath.LEGACY_PRODUCTION

    def __init__(self):
        self.solver = SimpleNamespace(
            integrator=SimpleNamespace(refresh_count=0)
        )
        self.projector = SimpleNamespace(retained_axis_counts=(1, 1, 1))
        self.fields = {}
        self.completed_steps = 0

    def advance(self, steps, *, pre_update_callback=None):
        self.completed_steps += steps

    def synchronize_for_observation(self):
        return None

    def projected_normal_force(self):
        return None

    def flow_diagnostics(self):
        return {}

    def backend_restart_metadata(self):
        return {}

    def restore_progress(self, **_kwargs):
        return None

    def to_metadata(self):
        return {}


def _spec(tmp_path: Path, **overrides: object):
    values = {
        "activity_number": 18.0,
        "output_dir": tmp_path,
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
        "checkpoint_interval": 2,
        "diagnostics": True,
        "save_hydrodynamics": True,
        "spectral_refresh_time": None,
        "disable_spectral_refresh": True,
    }
    values.update(overrides)
    return create_plane_beris_edwards_run_spec(**values)


def _run_spec_reads() -> set[str]:
    tree = ast.parse(
        WORKFLOW_PATH.read_text(encoding="utf-8"),
        filename=str(WORKFLOW_PATH),
    )
    values = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        owner = node.value
        if (
            isinstance(owner, ast.Attribute)
            and isinstance(owner.value, ast.Name)
            and owner.value.id == "self"
            and owner.attr == "run_spec"
        ) or isinstance(owner, ast.Name) and owner.id == "run_spec":
            values.add(node.attr)
    return values


def test_workflow_uses_one_component_graph_and_only_facade_identity_method(
    tmp_path,
    monkeypatch,
):
    assert _run_spec_reads() == FACADE_METHOD_READS
    assert FLAT_WORKFLOW_READS == {
        "checkpoint_interval",
        "diagnostic_interval",
        "diagnostics",
        "eta",
        "friction_mode_fric",
        "restart_from",
        "runtime_path",
        "save_hydrodynamics",
        "save_interval",
        "save_start_step",
        "steps",
        "zero_mode_policy",
    }

    calls = []
    original = workflow_module.decompose_plane_beris_edwards_run_spec

    def spy(value):
        result = original(value)
        calls.append((value, result))
        return result

    monkeypatch.setattr(
        workflow_module,
        "decompose_plane_beris_edwards_run_spec",
        spy,
    )
    spec = _spec(tmp_path)
    workflow = PlaneBerisEdwardsWorkflow(
        _Adapter(),
        spec,
        tmp_path,
        {"initial_condition": {}, "numerics": {}},
    )
    assert len(calls) == 1
    assert calls[0][0] is spec
    assert workflow._components is calls[0][1]


@pytest.mark.parametrize(
    ("zero_mode_policy", "friction_mode_fric", "expected_friction"),
    (
        ("zero_mean", 0.25, 0.0),
        ("friction", 0.25, 0.25),
    ),
)
def test_diagnostics_output_and_checkpoint_values_are_frozen(
    tmp_path,
    monkeypatch,
    zero_mode_policy,
    friction_mode_fric,
    expected_friction,
):
    spec = _spec(
        tmp_path,
        zero_mode_policy=zero_mode_policy,
        friction_mode_fric=friction_mode_fric,
    )
    calls = []
    diagnostic = object()
    observation = object()
    checkpoint = SimpleNamespace(completed_steps=3)

    def capture_diagnostic(adapter, *, step, viscosity, friction):
        calls.append(("diagnostic", adapter, step, viscosity, friction))
        return diagnostic

    def capture_observation(adapter, *, step):
        calls.append(("observation", adapter, step))
        return observation

    def write_observation(directory, value, *, save_hydrodynamics):
        calls.append(
            ("write_observation", directory, value, save_hydrodynamics)
        )

    def capture_checkpoint(adapter, *, runtime_identity_sha256):
        calls.append(
            ("checkpoint", adapter, runtime_identity_sha256)
        )
        return checkpoint

    def write_checkpoint(directory, value):
        calls.append(("write_checkpoint", directory, value))

    monkeypatch.setattr(
        workflow_module,
        "capture_plane_diagnostic",
        capture_diagnostic,
    )
    monkeypatch.setattr(
        workflow_module,
        "capture_plane_observation",
        capture_observation,
    )
    monkeypatch.setattr(
        workflow_module,
        "write_plane_observation",
        write_observation,
    )
    monkeypatch.setattr(
        workflow_module,
        "capture_plane_checkpoint",
        capture_checkpoint,
    )
    monkeypatch.setattr(
        workflow_module,
        "write_plane_checkpoint",
        write_checkpoint,
    )

    adapter = _Adapter()
    workflow = PlaneBerisEdwardsWorkflow(
        adapter,
        spec,
        tmp_path,
        {"initial_condition": {}, "numerics": {}},
    )
    assert workflow._capture_diagnostic(2) is diagnostic
    assert workflow._save_observation(2) is observation
    workflow._save_checkpoint()

    assert calls == [
        ("diagnostic", adapter, 2, spec.eta, expected_friction),
        ("observation", adapter, 2),
        ("write_observation", tmp_path.resolve(), observation, True),
        ("checkpoint", adapter, spec.runtime_identity_sha256()),
        (
            "write_checkpoint",
            tmp_path.resolve() / "checkpoint_3",
            checkpoint,
        ),
    ]
    assert workflow._diagnostics == [diagnostic]
    assert workflow._saved_steps == [2]
    assert workflow._checkpoint_steps == [3]
