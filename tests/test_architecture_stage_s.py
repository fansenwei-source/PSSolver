"""Stage S contracts for the bounded PSSolver v0.1 Plane application."""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pssolver.applications import run_plane_beris_edwards
from pssolver.applications.plane_beris_edwards import main
from pssolver.configuration import create_plane_beris_edwards_run_spec
from pssolver.workflows import PlaneWorkflowResult


PROJECT_ROOT = Path(__file__).parents[1]
COMPATIBILITY_CLI = PROJECT_ROOT / "Plane_beris_edwards_stokes.py"
APPLICATION = PROJECT_ROOT / "pssolver/applications/plane_beris_edwards.py"
V01_SCOPE = PROJECT_ROOT / "notes/pssolver_v0_1_scope.md"
SETUP = PROJECT_ROOT / "setup.py"


def _arguments(output: Path) -> list[str]:
    return [
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
    ]


def _spec(output: Path, *, dry_run: bool = False):
    return create_plane_beris_edwards_run_spec(
        activity_number=18.0,
        output_dir=output,
        device="cpu",
        dtype="float64",
        pointwise_execution="eager",
        nx=8,
        ny=8,
        nz=8,
        steps=1,
        save_start_step=0,
        save_interval=1,
        diagnostic_interval=1,
        disable_spectral_refresh=True,
        save_hydrodynamics=True,
        dry_run=dry_run,
    )


def test_compatibility_cli_is_import_safe_and_thin():
    code = "import Plane_beris_edwards_stokes; print('import-safe')"
    result = subprocess.run(
        [sys.executable, "-c", code, "--unknown-host-argument"],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "import-safe"

    source = COMPATIBILITY_CLI.read_text(encoding="utf-8")
    assert len(source.splitlines()) <= 20
    assert "pssolver.applications.plane_beris_edwards import main" in source
    assert "parse_plane_beris_edwards_run_spec" not in source
    assert "build_plane_beris_edwards_runtime" not in source
    assert "torch" not in source
    assert "numpy" not in source


def test_application_module_owns_orchestration_and_exports_callable_api():
    source = APPLICATION.read_text(encoding="utf-8")
    assert "def run_plane_beris_edwards(" in source
    assert "def main(" in source
    assert "build_plane_beris_edwards_runtime(" in source
    assert "PlaneBerisEdwardsWorkflow(" in source

    with pytest.raises(TypeError, match="PlaneBerisEdwardsRunSpec"):
        run_plane_beris_edwards(object())


def test_programmatic_dry_run_does_not_create_output(tmp_path):
    output = tmp_path / "dry"
    assert run_plane_beris_edwards(_spec(output, dry_run=True)) is None
    assert not output.exists()


def test_main_accepts_explicit_argv_and_returns_status(tmp_path, capsys):
    output = tmp_path / "dry_cli"
    status = main([*_arguments(output), "--dry-run"])
    captured = capsys.readouterr()

    assert status == 0
    metadata = json.loads(captured.out)
    assert metadata["script"] == "Plane_beris_edwards_stokes.py"
    assert metadata["configuration"]["authority"].endswith(
        "PlaneBerisEdwardsRunSpec"
    )
    assert not output.exists()


def test_callable_api_and_compatibility_cli_preserve_one_step_trajectory(tmp_path):
    api_output = tmp_path / "api"
    cli_output = tmp_path / "cli"

    result = run_plane_beris_edwards(_spec(api_output))
    assert isinstance(result, PlaneWorkflowResult)
    assert result.start_step == 0
    assert result.final_step == 1

    cli = subprocess.run(
        [sys.executable, str(COMPATIBILITY_CLI), *_arguments(cli_output)],
        cwd=PROJECT_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert cli.returncode == 0, cli.stderr

    for name in ("Q_1.npy", "u_1.npy", "p_1.npy"):
        api_value = np.load(api_output / name)
        cli_value = np.load(cli_output / name)
        assert np.array_equal(api_value, cli_value)

    api_metadata = json.loads((api_output / "metadata.json").read_text())
    cli_metadata = json.loads((cli_output / "metadata.json").read_text())
    assert api_metadata["script"] == cli_metadata["script"]
    assert (
        api_metadata["runtime_selection"]
        == cli_metadata["runtime_selection"]
    )
    assert (
        api_metadata["implementation_provenance"]
        == cli_metadata["implementation_provenance"]
    )
    assert (
        _spec(api_output).runtime_identity_sha256()
        == _spec(cli_output).runtime_identity_sha256()
    )


def test_v01_scope_is_explicitly_bounded_before_stage_t():
    text = V01_SCOPE.read_text(encoding="utf-8")
    assert "legacy_production" in text
    assert "arbitrary PDE" in text
    assert "production Channel parity" in text
    assert "strict Shendruk" in text
    assert "Stage T starts only after" in text


def test_package_declares_the_supported_plane_console_entry_point():
    tree = ast.parse(SETUP.read_text(encoding="utf-8"))
    text = ast.unparse(tree)
    assert "pssolver-plane-beris-edwards=" in text
    assert "pssolver.applications.plane_beris_edwards:main" in text
