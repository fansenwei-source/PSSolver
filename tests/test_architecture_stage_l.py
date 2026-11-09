"""Qualification tests for the Stage L production/shadow trajectory gate."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import pssolver


PROJECT_ROOT = Path(__file__).parents[1]


def _production_command(
    output: Path,
    *,
    spectral_storage: str,
    projected_transform_execution: str,
) -> tuple[str, ...]:
    return (
        sys.executable,
        str(PROJECT_ROOT / "Plane_beris_edwards_stokes.py"),
        "--activity-number",
        "10.5",
        "--output-dir",
        str(output),
        "--height",
        "3.5",
        "--parameterization",
        "fixed-k",
        "--frank-k",
        "0.02",
        "--lx",
        "5",
        "--ly",
        "4",
        "--nx",
        "10",
        "--ny",
        "8",
        "--nz",
        "7",
        "--dt",
        "0.005",
        "--steps",
        "6",
        "--save-start-step",
        "0",
        "--save-interval",
        "1",
        "--diagnostic-interval",
        "1",
        "--seed",
        "24",
        "--num-defect-pairs",
        "1",
        "--defect-min-separation",
        "1",
        "--defect-core-radius",
        "0.3",
        "--twist-amplitude",
        "0.01",
        "--twist-modes",
        "1",
        "2",
        "--ldg-a",
        "0",
        "--ldg-b",
        "-0.3",
        "--ldg-c",
        "0.3",
        "--gamma",
        "2.94",
        "--flow-alignment",
        "0.31",
        "--eta",
        str(2.0 / 3.0),
        "--beta",
        "-1",
        "--initial-s",
        str(1.0 / 3.0),
        "--device",
        "cpu",
        "--dtype",
        "float64",
        "--zero-mode-policy",
        "zero_mean",
        "--dealias-rule",
        "cubic_half",
        "--projected-transform-execution",
        projected_transform_execution,
        "--molecular-field-linear-space",
        "spectral",
        "--stress-divergence-sum-space",
        "spectral",
        "--pointwise-execution",
        "eager",
        "--transform-execution-order",
        "real_first",
        "--spectral-storage",
        spectral_storage,
        "--tf32",
        "off",
        "--spectral-refresh-steps",
        "2",
        "--save-hydrodynamics",
    )


@pytest.fixture(
    scope="module",
    params=(
        ("hermitian_half", "truncated"),
        ("full_complex", "full"),
    ),
    ids=("qualified_default", "full_complex_rollback"),
)
def production_reference(tmp_path_factory, request):
    spectral_storage, projected_execution = request.param
    directory = tmp_path_factory.mktemp(f"stage_l_reference_{spectral_storage}") / "production"
    subprocess.run(
        _production_command(
            directory,
            spectral_storage=spectral_storage,
            projected_transform_execution=projected_execution,
        ),
        cwd=PROJECT_ROOT,
        env=dict(os.environ),
        check=True,
        capture_output=True,
        text=True,
    )
    return directory


def _run_shadow_cli(production: Path, shadow: Path) -> dict[str, object]:
    result = subprocess.run(
        (
            sys.executable,
            str(PROJECT_ROOT / "Plane_beris_edwards_shadow.py"),
            "--production-reference-dir",
            str(production),
            "--output-dir",
            str(shadow),
            "--confirm-steps",
            "6",
        ),
        cwd=PROJECT_ROOT,
        env=dict(os.environ),
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def test_stage_l_remains_opt_in_and_outside_production_paths():
    assert not hasattr(pssolver, "ProductionPlaneReference")
    assert not hasattr(pssolver, "PlaneShadowTrajectoryComparison")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/plane.py",
        "pssolver/solver.py",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "plane_shadow_driver" not in text
        assert "shadow_comparison" not in text
