"""Read-only command planning for Stage O.4.1 H100 qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .stage_n1_plan import _existing_directory, _new_directory
from .stage_o4_qualification import _sha256
from .stage_o41_qualification import (
    STAGE_O41_DECISION_SHAPE,
    STAGE_O41_DIAGNOSTIC_SHAPE,
    STAGE_O41_MAXIMUM_MEAN_TIMESTEP_RATIO,
    STAGE_O41_MAXIMUM_MEMORY_RATIO,
    STAGE_O41_MAXIMUM_PAIRED_TIMESTEP_RATIO,
    STAGE_O41_PROFILE_TRIALS,
    STAGE_O41_RETAINED_GROWTH_FLOOR_BYTES,
    STAGE_O41_RETAINED_GROWTH_FRACTION,
)


def _require_commit(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("expected_commit must be a full lowercase Git SHA")
    return value


def build_stage_o41_h100_plan(
    *,
    project_root: str | Path,
    control_root: str | Path,
    scratch_root: str | Path,
    python: str | Path,
    expected_commit: str,
    stage_o4_report: str | Path,
    expected_stage_o4_sha256: str,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Return the bounded two-scale profile and analysis command plan."""

    project = _existing_directory(project_root, "project root")
    control = _new_directory(control_root, "control root")
    scratch = _new_directory(scratch_root, "scratch root")
    if control == scratch or control in scratch.parents or scratch in control.parents:
        raise ValueError("control and scratch roots must be independent")
    python_path = Path(python).expanduser().resolve()
    if not python_path.is_file():
        raise FileNotFoundError(f"Python executable is missing: {python_path}")
    commit = _require_commit(expected_commit)
    report = Path(stage_o4_report).expanduser().resolve()
    if not report.is_file():
        raise FileNotFoundError(f"Stage O.4 report is missing: {report}")
    if _sha256(report) != expected_stage_o4_sha256:
        raise ValueError("Stage O.4 report SHA-256 differs from expectation")
    if not isinstance(expected_gpu_name, str) or not expected_gpu_name.strip():
        raise ValueError("expected_gpu_name must not be empty")

    python_text = str(python_path)
    driver = str(project / "Plane_beris_edwards_stokes.py")
    legacy_profiler = str(project / "benchmarks/profile_beris_edwards_timestep.py")
    canary_profiler = str(project / "benchmarks/profile_plane_shadow_stage_n4.py")
    analyzer = str(project / "scripts_plane/analyze_plane_stage_o41_qualification.py")
    references = scratch / "references"
    profiles = control / "profiles"

    shapes = {
        "r128": STAGE_O41_DIAGNOSTIC_SHAPE,
        "r320": STAGE_O41_DECISION_SHAPE,
    }

    def reference_command(label: str, shape: tuple[int, int, int]) -> list[str]:
        return [
            python_text,
            driver,
            "--activity-number",
            "18",
            "--output-dir",
            str(references / label),
            "--runtime-path",
            "legacy_production",
            "--height",
            "20",
            "--parameterization",
            "fixed-k",
            "--frank-k",
            "0.012345679012345678",
            "--lx",
            "100",
            "--ly",
            "100",
            "--nx",
            str(shape[0]),
            "--ny",
            str(shape[1]),
            "--nz",
            str(shape[2]),
            "--dt",
            "0.005",
            "--steps",
            "1",
            "--save-start-step",
            "0",
            "--save-interval",
            "1",
            "--seed",
            "24",
            "--dtype",
            "float64",
            "--device",
            "cuda",
            "--tf32",
            "off",
            "--dealias-rule",
            "cubic_half",
            "--projected-transform-execution",
            "truncated",
            "--molecular-field-linear-space",
            "spectral",
            "--stress-divergence-sum-space",
            "spectral",
            "--pointwise-execution",
            "compile",
            "--transform-execution-order",
            "real_first",
            "--spectral-storage",
            "hermitian_half",
            "--spectral-refresh-steps",
            "2",
            "--save-hydrodynamics",
        ]

    reference_commands = [
        {
            "role": f"{label}_legacy_reference",
            "argv": reference_command(label, shape),
        }
        for label, shape in shapes.items()
    ]

    def legacy_profile_command(
        label: str,
        shape: tuple[int, int, int],
        trial: int,
    ) -> list[str]:
        return [
            python_text,
            legacy_profiler,
            "--shape",
            ",".join(str(value) for value in shape),
            "--lengths",
            "100,100,20",
            "--device",
            "cuda",
            "--dtype",
            "float64",
            "--dt",
            "0.005",
            "--dealias-rule",
            "cubic_half",
            "--projected-transform-execution",
            "truncated",
            "--warmup-steps",
            "10",
            "--profile-steps",
            "20",
            "--spectral-refresh-interval",
            "2",
            "--molecular-field-linear-space",
            "spectral",
            "--stress-divergence-sum-space",
            "spectral",
            "--pointwise-execution",
            "compile",
            "--transform-execution-order",
            "real_first",
            "--spectral-storage",
            "hermitian_half",
            "--initial-q-path",
            str(references / label / "Q_0.npy"),
            "--timing-scope",
            "whole_timestep",
            "--output",
            str(profiles / f"{label}_legacy_{trial}.json"),
        ]

    def canary_profile_command(label: str, trial: int) -> list[str]:
        return [
            python_text,
            canary_profiler,
            "--production-reference-dir",
            str(references / label),
            "--mode",
            "candidate",
            "--warmup-steps",
            "10",
            "--profile-steps",
            "20",
            "--transform-audit-steps",
            "2",
            "--expected-gpu-name",
            expected_gpu_name,
            "--output",
            str(profiles / f"{label}_canary_{trial}.json"),
        ]

    profile_commands = []
    for label, shape in shapes.items():
        for trial in range(1, STAGE_O41_PROFILE_TRIALS + 1):
            commands = {
                "legacy": legacy_profile_command(label, shape, trial),
                "canary": canary_profile_command(label, trial),
            }
            order = ("legacy", "canary") if trial % 2 == 1 else (
                "canary",
                "legacy",
            )
            profile_commands.extend(
                {
                    "shape_role": label,
                    "trial": trial,
                    "runtime": runtime,
                    "argv": commands[runtime],
                }
                for runtime in order
            )

    analysis = [
        python_text,
        analyzer,
        "--stage-o4-report",
        str(report),
        "--expected-stage-o4-sha256",
        expected_stage_o4_sha256,
    ]
    for label in shapes:
        for runtime in ("legacy", "canary"):
            option = f"--{runtime}-{label}-profile"
            for trial in range(1, STAGE_O41_PROFILE_TRIALS + 1):
                analysis.extend(
                    (option, str(profiles / f"{label}_{runtime}_{trial}.json"))
                )
    analysis.extend(
        (
            "--expected-gpu-name",
            expected_gpu_name,
            "--output",
            str(control / "stage_o41_qualification.json"),
        )
    )

    return {
        "schema_version": 1,
        "qualification_stage": "O.4.1",
        "planning_only": True,
        "expected_commit": commit,
        "expected_gpu_name": expected_gpu_name,
        "stage_o4_evidence": {
            "path": str(report),
            "sha256": expected_stage_o4_sha256,
            "classification_must_remain": "B_neutral",
        },
        "paths": {
            "project_root": str(project),
            "control_root": str(control),
            "scratch_root": str(scratch),
        },
        "runtime_default_before": "legacy_production",
        "runtime_default_may_change": False,
        "eligible_result": "stage_o5_decision_only",
        "fixed_gates": {
            "diagnostic_shape": list(STAGE_O41_DIAGNOSTIC_SHAPE),
            "decision_shape": list(STAGE_O41_DECISION_SHAPE),
            "profile_trials_per_runtime_and_shape": STAGE_O41_PROFILE_TRIALS,
            "warmup_steps": 10,
            "profile_steps": 20,
            "maximum_mean_timestep_ratio": (
                STAGE_O41_MAXIMUM_MEAN_TIMESTEP_RATIO
            ),
            "maximum_paired_timestep_ratio": (
                STAGE_O41_MAXIMUM_PAIRED_TIMESTEP_RATIO
            ),
            "minimum_paired_pass_count": 2,
            "maximum_phase_memory_ratio": STAGE_O41_MAXIMUM_MEMORY_RATIO,
            "retained_growth_floor_bytes": (
                STAGE_O41_RETAINED_GROWTH_FLOOR_BYTES
            ),
            "retained_growth_fraction": STAGE_O41_RETAINED_GROWTH_FRACTION,
            "cross_runtime_transform_identity_required": False,
            "per_runtime_transform_plan_stability_required": True,
            "canary_transform_nonincrease_required": True,
            "strict_transform_reduction_required": True,
        },
        "commands": {
            "references": reference_commands,
            "profiles_balanced": profile_commands,
            "analysis": analysis,
        },
        "stop_conditions": [
            "Git HEAD, clean-worktree, or O.4 evidence identity gate fails",
            "CUDA unavailable or GPU name mismatch",
            "any command exits nonzero",
            "profile configuration or initial-Q identity differs",
            "non-finite timing, transform, lifecycle, or memory evidence",
            "R320 performance or phase-aware memory gate fails",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the Stage O.4.1 plan")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--stage-o4-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o4-sha256", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    args = parser.parse_args(argv)
    plan = build_stage_o41_h100_plan(
        project_root=args.project_root,
        control_root=args.control_root,
        scratch_root=args.scratch_root,
        python=args.python,
        expected_commit=args.expected_commit,
        stage_o4_report=args.stage_o4_report,
        expected_stage_o4_sha256=args.expected_stage_o4_sha256,
        expected_gpu_name=args.expected_gpu_name,
    )
    print(json.dumps(plan, allow_nan=False, indent=2, sort_keys=True))
    return 0


__all__ = ["build_stage_o41_h100_plan", "main"]
