"""CPU qualification for the bounded Stage O.4 evidence pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pssolver.experimental import (
    analyze_stage_o4_qualification,
    build_stage_o4_h100_plan,
    compare_plane_stage_o4_workflows,
)


PROJECT_ROOT = Path(__file__).parents[1]
PRODUCTION_SCRIPT = PROJECT_ROOT / "Plane_beris_edwards_stokes.py"


def _run_cpu(output: Path, runtime_path: str) -> None:
    result = subprocess.run(
        [
            sys.executable,
            str(PRODUCTION_SCRIPT),
            "--activity-number",
            "18",
            "--output-dir",
            str(output),
            "--runtime-path",
            runtime_path,
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
            "6",
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


def test_o4_real_cpu_dual_runtime_six_step_gate(tmp_path):
    legacy = tmp_path / "legacy"
    canary = tmp_path / "canary"
    _run_cpu(legacy, "legacy_production")
    _run_cpu(canary, "separated_canary")

    report = compare_plane_stage_o4_workflows(
        legacy,
        canary,
        left_runtime_path="legacy_production",
        right_runtime_path="separated_canary",
        expected_steps=tuple(range(7)),
        comparison_role="cross_runtime_six_step",
        require_initial_q_identity=True,
    )

    assert report["classification"] == "PASS"
    assert report["initial_q_identity_gate"] is True
    assert report["array_count"] == 21
    assert report["maximum_gate_relative_l2"] <= 1.0e-10
    assert report["production_default_changed"] is False


def test_o4_comparison_rejects_wrong_frame_set(tmp_path):
    legacy = tmp_path / "legacy"
    _run_cpu(legacy, "legacy_production")
    with pytest.raises(ValueError, match="frame set"):
        compare_plane_stage_o4_workflows(
            legacy,
            legacy,
            left_runtime_path="legacy_production",
            right_runtime_path="legacy_production",
            expected_steps=(6,),
            comparison_role="legacy_same_backend_restart",
            require_byte_identity=True,
        )


def _write_json(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _comparison(role: str, steps: tuple[int, ...], *, exact: bool) -> dict[str, object]:
    arrays = [
        {
            "field": field,
            "step": step,
            "left_sha256": "b" * 64 if field == "Q" and step == 0 else "c" * 64,
            "right_sha256": "b" * 64 if field == "Q" and step == 0 else "d" * 64,
            "byte_identical": exact or (field == "Q" and step == 0),
            "gate_relative_l2": 2.0e-15,
        }
        for step in steps
        for field in ("Q", "u", "p")
    ]
    return {
        "qualification_stage": "O.4",
        "comparison_role": role,
        "classification": "PASS",
        "scientific_signature": {"fixed": True},
        "expected_steps": list(steps),
        "array_count": len(arrays),
        "arrays": arrays,
        "relative_l2_tolerance": 1.0e-10,
        "maximum_gate_relative_l2": 2.0e-15,
        "numerical_gate": True,
        "require_byte_identity": exact,
        "byte_identity_gate": True,
        "require_initial_q_identity": role.startswith("cross_runtime"),
        "initial_q_identity_gate": True,
        "production_default_changed": False,
    }


def _profile_config() -> dict[str, object]:
    return {
        "shape": [128, 128, 32],
        "lengths": [100.0, 100.0, 20.0],
        "device": "cuda",
        "dtype": "float64",
        "dt": 0.005,
        "dealias_rule": "cubic_half",
        "projected_transform_execution": "truncated",
        "spectral_storage": "hermitian_half",
        "spectral_refresh_interval": 2,
        "warmup_steps": 10,
        "profile_steps": 20,
        "timing_scope": "whole_timestep",
    }


def _materialization() -> dict[str, int]:
    return {
        "physical_materializations": 29,
        "on_demand_physical_materializations": 0,
        "physical_materialization_batches": 4,
        "batched_physical_components": 29,
        "singleton_materialization_batches": 0,
        "maximum_materialization_batch_size": 15,
        "physical_island_prefetches": 7,
        "physical_island_requested_components": 35,
        "unmaterialized_published_components": 25,
    }


def _legacy_profile(mean: float, allocated: int = 100) -> dict[str, object]:
    return {
        "config": _profile_config(),
        "environment": {
            "cuda_available": True,
            "device_name": "NVIDIA H100 PCIe",
            "cuda_matmul_allow_tf32": False,
        },
        "throughput": {"mean_timestep_seconds": mean},
        "memory": {
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": allocated * 2,
        },
        "timings": {
            "transform_forward": {"calls": 140},
            "transform_inverse": {"calls": 640},
        },
        "profile_input": {"initial_q_sha256": "b" * 64},
    }


def _canary_profile(mean: float, allocated: int = 101) -> dict[str, object]:
    return {
        "qualification_stage": "N.4",
        "classification": "PROFILE_COMPLETE",
        "mode": "candidate",
        "finite": True,
        "configuration": _profile_config(),
        "configuration_authority": "unified_execution_policy",
        "algebraic_execution_policy": {"mode": "batched_physical_islands"},
        "environment": {
            "cuda_available": True,
            "device_name": "NVIDIA H100 PCIe",
            "cuda_matmul_allow_tf32": False,
        },
        "throughput": {"mean_timestep_seconds": mean},
        "memory": {
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": allocated * 2,
        },
        "production_initial_q_sha256": "b" * 64,
        "transform_call_audit": {
            "forward_calls_per_step": 7.0,
            "inverse_calls_per_step": 32.0,
            "execution_policy": {"mode": "batched_physical_islands"},
            "representation_reuse": {"retained_pairs_after_generation": 0},
            "physical_materialization": _materialization(),
        },
    }


def _analysis_inputs(tmp_path: Path):
    comparisons = {
        "six": _write_json(
            tmp_path / "six.json",
            _comparison("cross_runtime_six_step", tuple(range(7)), exact=False),
        ),
        "hundred": _write_json(
            tmp_path / "hundred.json",
            _comparison("cross_runtime_hundred_step", (0, 100), exact=False),
        ),
        "legacy_restart": _write_json(
            tmp_path / "legacy_restart.json",
            _comparison("legacy_same_backend_restart", (6,), exact=True),
        ),
        "canary_restart": _write_json(
            tmp_path / "canary_restart.json",
            _comparison("canary_same_backend_restart", (6,), exact=True),
        ),
    }
    legacy = [
        _write_json(
            tmp_path / f"legacy_{index}.json",
            _legacy_profile(value),
        )
        for index, value in enumerate((0.0100, 0.0101, 0.0099), start=1)
    ]
    canary = [
        _write_json(
            tmp_path / f"canary_{index}.json",
            _canary_profile(value),
        )
        for index, value in enumerate((0.0101, 0.0100, 0.0100), start=1)
    ]
    return comparisons, legacy, canary


def test_o4_analysis_accepts_complete_non_regressing_evidence(tmp_path):
    comparisons, legacy, canary = _analysis_inputs(tmp_path)
    report = analyze_stage_o4_qualification(
        comparisons["six"],
        comparisons["hundred"],
        comparisons["legacy_restart"],
        comparisons["canary_restart"],
        legacy,
        canary,
    )

    assert report["classification"] == "A_recommended"
    assert report["eligible_for_stage_o5_decision"] is True
    assert report["eligible_for_production_promotion"] is False
    assert report["gate_failures"] == []
    assert all(report["gates"].values())


def test_o4_analysis_reports_performance_failure_without_promotion(tmp_path):
    comparisons, legacy, canary = _analysis_inputs(tmp_path)
    for path in canary:
        value = json.loads(path.read_text())
        value["throughput"]["mean_timestep_seconds"] = 0.011
        path.write_text(json.dumps(value), encoding="utf-8")

    report = analyze_stage_o4_qualification(
        comparisons["six"],
        comparisons["hundred"],
        comparisons["legacy_restart"],
        comparisons["canary_restart"],
        legacy,
        canary,
    )
    assert report["classification"] == "B_neutral"
    assert report["eligible_for_stage_o5_decision"] is False
    assert report["gates"]["performance_non_regression"] is False
    assert report["eligible_for_production_promotion"] is False


def test_o4_plan_is_read_only_bounded_balanced_and_non_promoting(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    control = tmp_path / "control"
    scratch = tmp_path / "scratch"

    plan = build_stage_o4_h100_plan(
        project_root=project,
        control_root=control,
        scratch_root=scratch,
        python=python,
        expected_commit="a" * 40,
    )

    assert plan["qualification_stage"] == "O.4"
    assert plan["planning_only"] is True
    assert plan["runtime_default_before"] == "legacy_production"
    assert plan["runtime_default_may_change"] is False
    assert plan["fixed_gates"]["same_backend_restart_byte_exact"] is True
    assert len(plan["commands"]["trajectories"]) == 8
    assert len(plan["commands"]["comparisons"]) == 4
    assert [
        (item["trial"], item["runtime"])
        for item in plan["commands"]["profiles_balanced"]
    ] == [
        (1, "legacy"),
        (1, "canary"),
        (2, "canary"),
        (2, "legacy"),
        (3, "legacy"),
        (3, "canary"),
    ]
    assert not control.exists()
    assert not scratch.exists()


def test_o4_cli_wrappers_are_import_safe():
    for relative in (
        "scripts_plane/compare_plane_stage_o4_workflows.py",
        "scripts_plane/analyze_plane_stage_o4_qualification.py",
        "scripts_plane/plan_plane_stage_o4_qualification.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "if __name__ == \"__main__\"" in source
        assert "Plane_beris_edwards_stokes" not in source
