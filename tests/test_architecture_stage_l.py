"""Qualification tests for the Stage L production/shadow trajectory gate."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

import pssolver
from pssolver.experimental import (
    build_plane_shadow_runtime_from_production_metadata,
    compare_plane_shadow_trajectories,
    load_production_plane_reference,
    run_plane_shadow_from_production_reference,
    write_plane_shadow_comparison,
)
from pssolver.experimental._shadow_support import file_sha256


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
    directory = (
        tmp_path_factory.mktemp(f"stage_l_reference_{spectral_storage}")
        / "production"
    )
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


def _input_identities(directory: Path) -> dict[str, str]:
    return {
        path.name: file_sha256(path)
        for path in sorted(directory.iterdir())
        if path.is_file()
    }


def test_real_production_and_shadow_clis_pass_short_cpu_trajectory_gate(
    production_reference,
    tmp_path,
):
    before = _input_identities(production_reference)
    shadow = tmp_path / "shadow"
    summary = _run_shadow_cli(production_reference, shadow)
    assert summary["classification"] == "shadow_run_complete"
    assert summary["configuration_compatible"] is True
    assert summary["saved_steps"] == [0, 1, 2, 3, 4, 5, 6]

    comparison_output = tmp_path / "comparison"
    result = subprocess.run(
        (
            sys.executable,
            str(
                PROJECT_ROOT
                / "scripts_plane/compare_plane_beris_edwards_shadow.py"
            ),
            "--production-dir",
            str(production_reference),
            "--shadow-dir",
            str(shadow),
            "--output-dir",
            str(comparison_output),
            "--relative-l2-tolerance",
            "1e-10",
        ),
        cwd=PROJECT_ROOT,
        env=dict(os.environ),
        check=True,
        capture_output=True,
        text=True,
    )
    report = json.loads(result.stdout)
    assert report["classification"] == "PASS"
    assert report["eligible_for_bounded_h100_gate"] is True
    assert report["configuration"]["compatible"] is True
    assert report["saved_steps"] == [0, 1, 2, 3, 4, 5, 6]
    assert report["array_count"] == 21
    assert report["maximum_gate_relative_l2"] < 1.0e-12
    assert all(value["passed"] for value in report["arrays"])
    assert (comparison_output / "COMPLETE").read_text() == "pass\n"
    assert _input_identities(production_reference) == before


def test_stage_l_reference_and_configuration_gates(production_reference):
    reference = load_production_plane_reference(production_reference)
    runtime, comparison = build_plane_shadow_runtime_from_production_metadata(
        reference.metadata
    )
    assert runtime.context.real_dtype.is_floating_point
    comparison.require_compatible()

    with pytest.raises(ValueError, match="exactly equal"):
        run_plane_shadow_from_production_reference(
            production_reference,
            production_reference.parent / "wrong_steps",
            confirmed_steps=2,
        )
    assert not (production_reference.parent / "wrong_steps").exists()

    unsupported = copy.deepcopy(reference.metadata)
    unsupported["numerics"]["q_gradient_reuse"]["enabled"] = False
    with pytest.raises(ValueError, match="Q-gradient reuse"):
        build_plane_shadow_runtime_from_production_metadata(unsupported)
    with pytest.raises(ValueError, match="CPU-only"):
        build_plane_shadow_runtime_from_production_metadata(
            reference.metadata,
            device="cuda",
        )


def test_comparison_reports_a_numerical_perturbation_as_failure(
    production_reference,
    tmp_path,
):
    shadow = tmp_path / "shadow"
    _run_shadow_cli(production_reference, shadow)
    perturbed = tmp_path / "perturbed"
    shutil.copytree(shadow, perturbed)
    q_path = perturbed / "Q_6.npy"
    q = np.load(q_path, allow_pickle=False)
    q[0, 0, 0, 0] += 1.0e-5
    np.save(q_path, q, allow_pickle=False)

    comparison = compare_plane_shadow_trajectories(
        production_reference,
        perturbed,
        relative_l2_tolerance=1.0e-10,
    )
    assert not comparison.passed
    failed = [value for value in comparison.arrays if not value.passed]
    assert [(value.step, value.field) for value in failed] == [(6, "Q")]
    report = write_plane_shadow_comparison(
        tmp_path / "failed_comparison",
        comparison,
    )
    assert json.loads(report.read_text())["classification"] == "FAIL"
    assert (report.parent / "FAILED").read_text() == "fail\n"
    assert not (report.parent / "COMPLETE").exists()


def test_stage_l_remains_opt_in_and_outside_production_paths():
    assert not hasattr(pssolver, "ProductionPlaneReference")
    assert not hasattr(pssolver, "PlaneShadowTrajectoryComparison")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
        "pssolver/solver.py",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "plane_shadow_driver" not in text
        assert "shadow_comparison" not in text
