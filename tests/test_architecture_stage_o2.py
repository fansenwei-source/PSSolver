"""Characterization tests for the Stage O.2 Plane runtime adapters."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import pssolver
from pssolver.configuration import (
    PlaneRuntimePath,
    create_plane_beris_edwards_run_spec,
)
from pssolver.runtime import (
    LegacyPlaneRuntimeAdapter,
    PlaneRuntimeAdapterProtocol,
    PlaneRuntimeBuildRequest,
    build_plane_beris_edwards_runtime,
)


PROJECT_ROOT = Path(__file__).parents[1]
PRODUCTION_SCRIPT = PROJECT_ROOT / "Plane_beris_edwards_stokes.py"


class _DummyProjector:
    def retained_axis_counts(self, _boundaries):
        return (1, 1, 1)


class _DummyIntegrator:
    spectral_refresh_interval = None
    step_count = 0
    refresh_count = 0


class _DummySolver:
    def __init__(self):
        self.fields = {}
        self.integrator = _DummyIntegrator()
        self.advance_calls = []
        self.synchronizations = 0

    def run(self, steps, callback=None, pre_update_callback=None):
        self.advance_calls.append((steps, callback, pre_update_callback))

    def refresh_static_fields(self):
        self.synchronizations += 1


def _spec(tmp_path, **overrides):
    values = {
        "activity_number": 18.0,
        "output_dir": tmp_path / "run",
    }
    values.update(overrides)
    return create_plane_beris_edwards_run_spec(**values)


def _request(spec, *, metadata=None):
    if metadata is None:
        metadata = {
            "configuration": spec.identity_metadata(),
            "runtime_selection": spec.runtime_selection_metadata(),
        }
    return PlaneRuntimeBuildRequest(
        run_spec=spec,
        production_metadata=metadata,
        initial_values={},
        device="cpu",
    )


def _run_tiny(tmp_path, runtime_path):
    output = tmp_path / runtime_path
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
            "--disable-spectral-refresh",
            "--save-hydrodynamics",
            "--runtime-path",
            runtime_path,
        ],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    return output


def test_omitted_runtime_path_remains_legacy_and_canary_is_explicit(tmp_path):
    default = _spec(tmp_path)

    assert default.runtime_path is PlaneRuntimePath.LEGACY_PRODUCTION
    assert default.runtime_selection_metadata() == {
        "authority": ("pssolver.configuration.PlaneBerisEdwardsRunSpec.runtime_path"),
        "requested": "legacy_production",
        "effective": "legacy_production",
        "default": "legacy_production",
        "fallback_allowed": False,
    }


def test_legacy_adapter_implements_narrow_protocol_and_delegates(tmp_path):
    spec = _spec(tmp_path)
    solver = _DummySolver()
    projector = _DummyProjector()
    builder_calls = 0

    def builder():
        nonlocal builder_calls
        builder_calls += 1
        return solver, projector

    adapter = build_plane_beris_edwards_runtime(
        _request(spec),
        legacy_builder=builder,
    )
    assert isinstance(adapter, LegacyPlaneRuntimeAdapter)
    assert isinstance(adapter, PlaneRuntimeAdapterProtocol)
    assert adapter.runtime_path is PlaneRuntimePath.LEGACY_PRODUCTION
    assert adapter.completed_steps == 0
    assert builder_calls == 1

    def callback(_solver, _step):
        return None

    adapter.advance(2, pre_update_callback=callback)
    adapter.synchronize_for_observation()
    assert solver.advance_calls == [(2, None, callback)]
    assert solver.synchronizations == 1
    assert adapter.to_metadata()["separated_architecture"] is None


def test_mixed_configuration_authorities_fail_before_builder(tmp_path):
    spec = _spec(tmp_path)
    metadata = {
        "configuration": {
            **spec.identity_metadata(),
            "runtime_path": "separated_canary",
        },
        "runtime_selection": spec.runtime_selection_metadata(),
    }
    with pytest.raises(ValueError, match="mixed Plane runtime"):
        _request(spec, metadata=metadata)


def test_default_runtime_module_does_not_import_experimental_architecture():
    code = """
import json
import sys
from pathlib import Path
from pssolver.configuration import create_plane_beris_edwards_run_spec
from pssolver.runtime import PlaneRuntimeBuildRequest, build_plane_beris_edwards_runtime

class Solver:
    fields = {}
    integrator = object()
    def run(self, *args, **kwargs): pass
    def refresh_static_fields(self): pass
class Projector:
    def retained_axis_counts(self, boundaries): return (1, 1, 1)

spec = create_plane_beris_edwards_run_spec(
    activity_number=18.0,
    output_dir=Path('/tmp/stage-o2-unused'),
)
metadata = {
    'configuration': spec.identity_metadata(),
    'runtime_selection': spec.runtime_selection_metadata(),
}
request = PlaneRuntimeBuildRequest(spec, metadata, {}, 'cpu')
build_plane_beris_edwards_runtime(
    request,
    legacy_builder=lambda: (Solver(), Projector()),
)
print(json.dumps(sorted(
    name for name in sys.modules if name.startswith('pssolver.experimental')
)))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_o2_does_not_promote_or_touch_other_geometry_entry_points():
    assert not hasattr(pssolver, "PlaneRuntimePath")
    for relative in ("pssolver/solver.py",):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "PlaneRuntimePath" not in source
        assert "pssolver.runtime" not in source
