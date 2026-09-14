"""Characterization tests for the Stage O.2 Plane runtime adapters."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

import pssolver
from pssolver.configuration import (
    PlaneRuntimePath,
    create_plane_beris_edwards_run_spec,
    parse_plane_beris_edwards_run_spec,
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
    canary = _spec(tmp_path, runtime_path="separated_canary")

    assert default.runtime_path is PlaneRuntimePath.LEGACY_PRODUCTION
    assert canary.runtime_path is PlaneRuntimePath.SEPARATED_CANARY
    assert default.runtime_selection_metadata() == {
        "authority": (
            "pssolver.configuration.PlaneBerisEdwardsRunSpec.runtime_path"
        ),
        "requested": "legacy_production",
        "effective": "legacy_production",
        "default": "legacy_production",
        "fallback_allowed": False,
    }


def test_cli_runtime_selection_has_the_same_single_authority(tmp_path):
    spec = parse_plane_beris_edwards_run_spec(
        [
            "--activity-number",
            "18",
            "--output-dir",
            str(tmp_path / "run"),
            "--runtime-path",
            "separated_canary",
        ]
    )
    assert spec.runtime_path is PlaneRuntimePath.SEPARATED_CANARY
    assert spec.identity_metadata()["runtime_path"] == "separated_canary"
    assert spec.to_metadata()["runtime_path"] == "separated_canary"


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

    callback = lambda _solver, _step: None
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


def test_canary_rejects_legacy_q_gradient_cache_flag(tmp_path):
    with pytest.raises(
        ValueError,
        match="does not accept legacy Q-gradient cache flags",
    ):
        _spec(
            tmp_path,
            runtime_path="separated_canary",
            disable_q_gradient_reuse=True,
        )


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


def test_tiny_canary_uses_identical_initial_q_and_roundoff_step_parity(
    tmp_path,
):
    legacy = _run_tiny(tmp_path, "legacy_production")
    canary = _run_tiny(tmp_path, "separated_canary")
    legacy_metadata = json.loads((legacy / "metadata.json").read_text())
    canary_metadata = json.loads((canary / "metadata.json").read_text())

    assert legacy_metadata["runtime_selection"]["effective"] == (
        "legacy_production"
    )
    assert canary_metadata["runtime_selection"]["effective"] == (
        "separated_canary"
    )
    assert canary_metadata["runtime_selection"][
        "separated_architecture"
    ] is not None
    for key in ("raw_q_sha256", "projected_q_sha256"):
        assert (
            legacy_metadata["initial_condition"][key]
            == canary_metadata["initial_condition"][key]
        )

    for step in (0, 1):
        for prefix in ("Q", "u", "p"):
            reference = np.load(legacy / f"{prefix}_{step}.npy")
            candidate = np.load(canary / f"{prefix}_{step}.npy")
            assert reference.shape == candidate.shape
            assert reference.dtype == candidate.dtype == np.float64
            assert np.isfinite(candidate).all()
            difference = candidate - reference
            relative_l2 = np.linalg.norm(difference.ravel()) / max(
                np.linalg.norm(reference.ravel()),
                np.finfo(np.float64).tiny,
            )
            assert relative_l2 <= 1.0e-12


def test_o2_does_not_promote_or_touch_other_geometry_entry_points():
    assert not hasattr(pssolver, "PlaneRuntimePath")
    for relative in ("pssolver/solver.py", "pssolver/channel.py"):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "PlaneRuntimePath" not in source
        assert "pssolver.runtime" not in source
