"""Qualification contracts for Stage Q.4 H100 execution."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

from pssolver.experimental._shadow_support import file_sha256
from pssolver.experimental.stage_q4_plan import build_stage_q4_h100_plan
from pssolver.experimental.stage_q4_qualification import (
    STAGE_Q4_TRAJECTORY_STEPS,
    _validate_compiled_rhs,
    analyze_stage_q4_qualification,
)
from pssolver.models.active_nematics.beris_edwards import (
    BerisEdwardsPointwiseKernels,
)


def _compiled_rhs() -> dict[str, object]:
    return {
        "owner": "geometry_executor",
        "implementation_name": "plane_beris_edwards_pointwise_explicit_rhs",
        "observability": {
            "fallback_to_model": False,
            "pointwise_kernels": {
                "requested": "compile",
                "effective": "compile",
                "fallback_allowed": False,
                "fallback_reason": None,
                "compile": {
                    "backend": "inductor",
                    "fullgraph": True,
                    "dynamic": False,
                },
            },
        },
    }


def test_compiled_rhs_gate_uses_the_runtime_metadata_schema():
    actual = _compiled_rhs()
    assert _validate_compiled_rhs(actual)

    runtime_metadata = {
        "owner": "geometry_executor",
        "implementation_name": "plane_beris_edwards_pointwise_explicit_rhs",
        "observability": {
            "fallback_to_model": False,
            "pointwise_kernels": BerisEdwardsPointwiseKernels(
                "compile"
            ).metadata(),
        },
    }
    assert _validate_compiled_rhs(runtime_metadata)

    old_synthetic = json.loads(json.dumps(actual))
    pointwise = old_synthetic["observability"]["pointwise_kernels"]
    pointwise["compile"]["fallback_allowed"] = pointwise.pop(
        "fallback_allowed"
    )
    pointwise["compile"]["fallback_reason"] = pointwise.pop(
        "fallback_reason"
    )
    assert not _validate_compiled_rhs(old_synthetic)


def _q3_report(path: Path) -> str:
    report = {
        "qualification_stage": "Q.3",
        "classification": "ALGEBRAIC_TARGET_REFINEMENT_COMPLETE",
        "candidate_design": {
            "selected_candidate": "preplanned_algebraic_batch_workspace",
            "scope": {
                "geometry": "PlaneSlab",
                "model": "BerisEdwardsPlaneCoupledModel",
                "retain_tensor_values_across_timesteps": False,
            },
        },
        "eligible_for_stage_q4_candidate_implementation": True,
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
    }
    path.write_text(json.dumps(report), encoding="utf-8")
    return file_sha256(path)


def _assembly(role: str) -> dict[str, object]:
    candidate = role == "candidate"
    return {
        "policy": {
            "mode": (
                "preallocated_workspace" if candidate else "copy_cat"
            )
        },
        "copy_cat_batches": 2 if candidate else 8,
        "preallocated_workspace_batches": 6 if candidate else 0,
        "workspace_count": 4 if candidate else 0,
        "workspace_allocated_bytes": 4096 if candidate else 0,
        "workspace_active_count": 0,
        "workspace_retains_timestep_inputs": False,
        "retained_tensor_references": 0,
        "source_attribution": (
            {
                "algebraic.nematic_stress.outputs": {
                    "copy_cat_batches": 0,
                    "preallocated_workspace_batches": 6,
                },
                "explicit_rhs.outputs": {
                    "copy_cat_batches": 2,
                    "preallocated_workspace_batches": 0,
                },
            }
            if candidate
            else {
                "algebraic.nematic_stress.outputs": {
                    "copy_cat_batches": 6,
                    "preallocated_workspace_batches": 0,
                },
                "explicit_rhs.outputs": {
                    "copy_cat_batches": 2,
                    "preallocated_workspace_batches": 0,
                },
            }
        ),
    }


def _trajectory(root: Path, role: str, *, perturb: bool = False) -> Path:
    directory = root / role
    directory.mkdir()
    (directory / "COMPLETE").write_text("complete\n", encoding="utf-8")
    for step in range(STAGE_Q4_TRAJECTORY_STEPS + 1):
        for field, components in (("Q", 5), ("u", 3), ("p", 1)):
            value = np.full(
                (components, 2, 2, 2),
                float(step + components),
                dtype=np.float64,
            )
            if perturb and role == "candidate" and field == "Q" and step == 6:
                value[0, 0, 0, 0] += 1.0
            np.save(directory / f"{field}_{step}.npy", value)
    metrics = {
        "qualification_stage": "Q.4",
        "classification": "TRAJECTORY_COMPLETE",
        "measurement_role": role,
        "completed_steps": STAGE_Q4_TRAJECTORY_STEPS,
        "saved_steps": list(range(STAGE_Q4_TRAJECTORY_STEPS + 1)),
        "projected_batch_assembly_policy": {
            "mode": (
                "preallocated_workspace"
                if role == "candidate"
                else "copy_cat"
            )
        },
        "production_default_changed": False,
        "production_metadata_sha256": "1" * 64,
        "production_initial_q_sha256": "2" * 64,
        "explicit_rhs_execution": _compiled_rhs(),
    }
    (directory / "stage_q4_metrics.json").write_text(
        json.dumps(metrics),
        encoding="utf-8",
    )
    return directory


def _profiles(root: Path, role: str) -> list[Path]:
    paths = []
    for trial in range(1, 4):
        path = root / f"{role}_{trial}.json"
        report = {
            "qualification_stage": "Q.4",
            "classification": "PROFILE_COMPLETE",
            "measurement_role": role,
            "configuration": {
                "shape": [320, 320, 80],
                "warmup_steps": 10,
                "profile_steps": 20,
            },
            "projected_batch_assembly_policy": {
                "mode": (
                    "preallocated_workspace"
                    if role == "candidate"
                    else "copy_cat"
                )
            },
            "finite": True,
            "production_default_changed": False,
            "environment": {
                "device_name": "NVIDIA H100 PCIe",
                "cuda_matmul_allow_tf32": False,
            },
            "production_metadata_sha256": "1" * 64,
            "production_initial_q_sha256": "2" * 64,
            "explicit_rhs_execution": _compiled_rhs(),
            "throughput": {
                "mean_timestep_seconds": (
                    0.079 + trial * 1.0e-4
                    if role == "candidate"
                    else 0.100 + trial * 1.0e-4
                )
            },
            "memory": {
                "peak_allocated_bytes": (
                    1020 if role == "candidate" else 1000
                )
            },
            "batch_assembly_diagnostics": _assembly(role),
        }
        path.write_text(json.dumps(report), encoding="utf-8")
        paths.append(path)
    return paths


def test_stage_q4_accepts_fast_equivalent_workspace_candidate(tmp_path):
    q3 = tmp_path / "q3.json"
    q3_sha = _q3_report(q3)
    baseline = _trajectory(tmp_path, "baseline")
    candidate = _trajectory(tmp_path, "candidate")
    baseline_profiles = _profiles(tmp_path, "baseline")
    candidate_profiles = _profiles(tmp_path, "candidate")

    report = analyze_stage_q4_qualification(
        baseline,
        candidate,
        baseline_profiles,
        candidate_profiles,
        stage_q3_report=q3,
        expected_stage_q3_sha256=q3_sha,
    )

    assert report["classification"] == "A_recommended"
    assert report["trajectory"]["maximum_relative_l2"] == 0.0
    assert report["gates"] == {
        "numerical_equivalence": True,
        "workspace_lifecycle": True,
        "r320_performance_improvement": True,
        "r320_memory_non_regression": True,
        "candidate_safety_non_regression": True,
    }
    assert report["eligible_for_stage_q5_decision"] is True
    assert report["eligible_for_production_promotion"] is False


def test_stage_q4_rejects_a_numerically_different_candidate(tmp_path):
    q3 = tmp_path / "q3.json"
    q3_sha = _q3_report(q3)
    baseline = _trajectory(tmp_path, "baseline")
    candidate = _trajectory(tmp_path, "candidate", perturb=True)
    report = analyze_stage_q4_qualification(
        baseline,
        candidate,
        _profiles(tmp_path, "baseline"),
        _profiles(tmp_path, "candidate"),
        stage_q3_report=q3,
        expected_stage_q3_sha256=q3_sha,
    )
    assert report["classification"] == "C_rejected"
    assert report["gates"]["numerical_equivalence"] is False
    assert report["eligible_for_stage_q5_decision"] is False


def test_stage_q4_plan_is_balanced_and_does_not_change_defaults(tmp_path):
    project = tmp_path / "project"
    reference = tmp_path / "reference"
    project.mkdir()
    reference.mkdir()
    q3 = tmp_path / "q3.json"
    q3_sha = _q3_report(q3)
    plan = build_stage_q4_h100_plan(
        project_root=project,
        production_reference=reference,
        control_root=tmp_path / "control",
        scratch_root=tmp_path / "scratch",
        python=sys.executable,
        expected_commit="a" * 40,
        stage_q3_report=q3,
        expected_stage_q3_sha256=q3_sha,
    )
    commands = plan["commands"]["profiles_balanced"]
    assert [command["role"] for command in commands] == [
        "baseline",
        "candidate",
        "candidate",
        "baseline",
        "baseline",
        "candidate",
    ]
    assert len(plan["commands"]["trajectories"]) == 2
    assert plan["fixed_contract"]["trajectory_saved_steps"] == list(range(7))
    assert plan["fixed_contract"]["changes_production_default"] is False
