"""Stage R contracts for package-owned Plane production assembly."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import torch

from pssolver.configuration import create_plane_beris_edwards_run_spec
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime import (
    DealiasedSemiImplicitEulerIntegrator,
    LegacyPlaneRuntimeAdapter,
    PlaneRuntimeBuildRequest,
    build_plane_beris_edwards_runtime,
)


PROJECT_ROOT = Path(__file__).parents[1]
PRODUCTION_SCRIPT = PROJECT_ROOT / "Plane_beris_edwards_stokes.py"
PRODUCTION_APPLICATION = (
    PROJECT_ROOT / "pssolver/applications/plane_beris_edwards.py"
)


def _request(tmp_path: Path) -> PlaneRuntimeBuildRequest:
    spec = create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=tmp_path / "unused",
        nx=8,
        ny=8,
        nz=8,
        steps=1,
        save_start_step=0,
        save_interval=1,
        diagnostic_interval=1,
        dtype="float64",
        pointwise_execution="eager",
        spectral_refresh_time=None,
        disable_spectral_refresh=True,
    )
    initial_values = {
        name: torch.zeros((8, 8, 8), dtype=torch.float64)
        for name in Q_COMPONENTS
    }
    metadata = {
        "configuration": spec.identity_metadata(),
        "runtime_selection": spec.runtime_selection_metadata(),
    }
    return PlaneRuntimeBuildRequest(
        run_spec=spec,
        production_metadata=metadata,
        initial_values=initial_values,
        device="cpu",
    )


def test_package_factory_owns_default_plane_production_assembly(tmp_path):
    adapter = build_plane_beris_edwards_runtime(_request(tmp_path))

    assert isinstance(adapter, LegacyPlaneRuntimeAdapter)
    assert isinstance(
        adapter.solver.integrator,
        DealiasedSemiImplicitEulerIntegrator,
    )
    assert adapter.solver.shape == (8, 8, 8)
    assert adapter.projector.rule == "cubic_half"
    assert adapter.to_metadata()["effective"] == "legacy_production"


def test_default_factory_remains_outside_experimental_architecture(tmp_path):
    code = f"""
import json
import sys
from pathlib import Path
import torch
from pssolver.configuration import create_plane_beris_edwards_run_spec
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime import PlaneRuntimeBuildRequest, build_plane_beris_edwards_runtime

spec = create_plane_beris_edwards_run_spec(
    activity_number=18.0,
    output_dir=Path({str(tmp_path / 'unused')!r}),
    nx=8,
    ny=8,
    nz=8,
    steps=1,
    save_start_step=0,
    save_interval=1,
    diagnostic_interval=1,
    dtype='float64',
    pointwise_execution='eager',
    disable_spectral_refresh=True,
)
metadata = {{
    'configuration': spec.identity_metadata(),
    'runtime_selection': spec.runtime_selection_metadata(),
}}
values = {{name: torch.zeros((8, 8, 8), dtype=torch.float64) for name in Q_COMPONENTS}}
request = PlaneRuntimeBuildRequest(spec, metadata, values, 'cpu')
build_plane_beris_edwards_runtime(request)
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


def test_top_level_driver_no_longer_owns_plane_numerical_assembly():
    source = PRODUCTION_SCRIPT.read_text(encoding="utf-8")
    application = PRODUCTION_APPLICATION.read_text(encoding="utf-8")

    assert "class DealiasedSemiImplicitEulerIntegrator" not in source
    assert "def build_legacy_plane_runtime" not in source
    assert "SpectralSolver(" not in source
    assert "legacy_builder=" not in source
    assert "build_plane_beris_edwards_runtime(" not in source
    assert "build_plane_beris_edwards_runtime(" in application


def test_stage_r_does_not_change_other_geometry_or_public_root_api():
    assert not hasattr(__import__("pssolver"), "build_legacy_plane_runtime")
    for relative in ("pssolver/solver.py", "pssolver/channel.py"):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "plane_legacy" not in source
