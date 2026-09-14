"""Read-only command planning for Stage O.4 H100 qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .stage_n1_plan import _existing_directory, _new_directory


def _require_commit(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError("expected_commit must be a full lowercase Git SHA")
    return value


def build_stage_o4_h100_plan(
    *,
    project_root: str | Path,
    control_root: str | Path,
    scratch_root: str | Path,
    python: str | Path,
    expected_commit: str,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Return a bounded single-job trajectory/restart/profile command plan."""

    project = _existing_directory(project_root, "project root")
    control = _new_directory(control_root, "control root")
    scratch = _new_directory(scratch_root, "scratch root")
    if control == scratch or control in scratch.parents or scratch in control.parents:
        raise ValueError("control and scratch roots must be independent")
    python_path = Path(python).expanduser().resolve()
    if not python_path.is_file():
        raise FileNotFoundError(f"Python executable is missing: {python_path}")
    commit = _require_commit(expected_commit)
    if not isinstance(expected_gpu_name, str) or not expected_gpu_name.strip():
        raise ValueError("expected_gpu_name must not be empty")

    python_text = str(python_path)
    driver = str(project / "Plane_beris_edwards_stokes.py")
    compare = str(project / "scripts_plane/compare_plane_stage_o4_workflows.py")
    runs = scratch / "runs"
    comparisons = control / "comparisons"
    profiles = control / "profiles"

    def run_command(
        name: str,
        runtime_path: str,
        steps: int,
        save_interval: int,
        *extra: str,
    ) -> list[str]:
        return [
            python_text,
            driver,
            "--activity-number",
            "18",
            "--output-dir",
            str(runs / name),
            "--runtime-path",
            runtime_path,
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
            "128",
            "--ny",
            "128",
            "--nz",
            "32",
            "--dt",
            "0.005",
            "--steps",
            str(steps),
            "--save-start-step",
            "0",
            "--save-interval",
            str(save_interval),
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
            *extra,
        ]

    trajectory_commands = [
        {
            "role": "legacy_six_step",
            "argv": run_command(
                "legacy_six",
                "legacy_production",
                6,
                1,
                "--diagnostics",
                "--diagnostic-interval",
                "1",
            ),
        },
        {
            "role": "canary_six_step",
            "argv": run_command(
                "canary_six",
                "separated_canary",
                6,
                1,
                "--diagnostics",
                "--diagnostic-interval",
                "1",
            ),
        },
        {
            "role": "legacy_hundred_step",
            "argv": run_command("legacy_hundred", "legacy_production", 100, 100),
        },
        {
            "role": "canary_hundred_step",
            "argv": run_command("canary_hundred", "separated_canary", 100, 100),
        },
    ]
    for runtime_path, prefix in (
        ("legacy_production", "legacy"),
        ("separated_canary", "canary"),
    ):
        trajectory_commands.extend(
            (
                {
                    "role": f"{prefix}_restart_segment",
                    "argv": run_command(
                        f"{prefix}_segment",
                        runtime_path,
                        3,
                        3,
                        "--checkpoint-interval",
                        "3",
                    ),
                },
                {
                    "role": f"{prefix}_restart_resume",
                    "argv": run_command(
                        f"{prefix}_resumed",
                        runtime_path,
                        3,
                        3,
                        "--restart-from",
                        str(runs / f"{prefix}_segment" / "checkpoint_3"),
                    ),
                },
            )
        )

    def comparison_command(
        *,
        left: str,
        right: str,
        left_runtime: str,
        right_runtime: str,
        steps: Sequence[int],
        role: str,
        exact: bool = False,
        q0: bool = False,
    ) -> list[str]:
        argv = [
            python_text,
            compare,
            "--left-dir",
            str(runs / left),
            "--right-dir",
            str(runs / right),
            "--left-runtime-path",
            left_runtime,
            "--right-runtime-path",
            right_runtime,
            "--comparison-role",
            role,
            "--relative-l2-tolerance",
            "1e-10",
        ]
        for step in steps:
            argv.extend(("--expected-step", str(step)))
        if exact:
            argv.extend(("--require-byte-identity", "--allow-extra-steps"))
        if q0:
            argv.append("--require-initial-q-identity")
        argv.extend(("--output", str(comparisons / f"{role}.json")))
        return argv

    comparison_commands = [
        comparison_command(
            left="legacy_six",
            right="canary_six",
            left_runtime="legacy_production",
            right_runtime="separated_canary",
            steps=tuple(range(7)),
            role="cross_runtime_six_step",
            q0=True,
        ),
        comparison_command(
            left="legacy_hundred",
            right="canary_hundred",
            left_runtime="legacy_production",
            right_runtime="separated_canary",
            steps=(0, 100),
            role="cross_runtime_hundred_step",
            q0=True,
        ),
        comparison_command(
            left="legacy_six",
            right="legacy_resumed",
            left_runtime="legacy_production",
            right_runtime="legacy_production",
            steps=(6,),
            role="legacy_same_backend_restart",
            exact=True,
        ),
        comparison_command(
            left="canary_six",
            right="canary_resumed",
            left_runtime="separated_canary",
            right_runtime="separated_canary",
            steps=(6,),
            role="canary_same_backend_restart",
            exact=True,
        ),
    ]

    legacy_profiles = [profiles / f"legacy_{trial}.json" for trial in range(1, 4)]
    canary_profiles = [profiles / f"canary_{trial}.json" for trial in range(1, 4)]
    profile_commands = []
    for trial in range(1, 4):
        commands = {
            "legacy": [
                python_text,
                str(project / "benchmarks/profile_beris_edwards_timestep.py"),
                "--shape",
                "128,128,32",
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
                str(runs / "legacy_six" / "Q_0.npy"),
                "--timing-scope",
                "whole_timestep",
                "--output",
                str(legacy_profiles[trial - 1]),
            ],
            "canary": [
                python_text,
                str(project / "benchmarks/profile_plane_shadow_stage_n4.py"),
                "--production-reference-dir",
                str(runs / "legacy_six"),
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
                str(canary_profiles[trial - 1]),
            ],
        }
        order = ("legacy", "canary") if trial % 2 == 1 else ("canary", "legacy")
        profile_commands.extend(
            {"trial": trial, "runtime": runtime, "argv": commands[runtime]}
            for runtime in order
        )

    analysis = [
        python_text,
        str(project / "scripts_plane/analyze_plane_stage_o4_qualification.py"),
        "--six-step-comparison",
        str(comparisons / "cross_runtime_six_step.json"),
        "--hundred-step-comparison",
        str(comparisons / "cross_runtime_hundred_step.json"),
        "--legacy-restart-comparison",
        str(comparisons / "legacy_same_backend_restart.json"),
        "--canary-restart-comparison",
        str(comparisons / "canary_same_backend_restart.json"),
    ]
    for path in legacy_profiles:
        analysis.extend(("--legacy-profile", str(path)))
    for path in canary_profiles:
        analysis.extend(("--canary-profile", str(path)))
    analysis.extend(
        (
            "--expected-gpu-name",
            expected_gpu_name,
            "--output",
            str(control / "stage_o4_qualification.json"),
        )
    )

    return {
        "schema_version": 1,
        "qualification_stage": "O.4",
        "planning_only": True,
        "expected_commit": commit,
        "expected_gpu_name": expected_gpu_name,
        "runtime_default_before": "legacy_production",
        "runtime_default_may_change": False,
        "eligible_result": "stage_o5_decision_only",
        "paths": {
            "project_root": str(project),
            "control_root": str(control),
            "scratch_root": str(scratch),
        },
        "fixed_gates": {
            "six_step_saved_steps": list(range(7)),
            "hundred_step_saved_steps": [0, 100],
            "relative_l2_tolerance": 1.0e-10,
            "same_backend_restart_byte_exact": True,
            "initial_q_sha256_identical": True,
            "profile_shape": [128, 128, 32],
            "profile_trials": 3,
            "warmup_steps": 10,
            "profile_steps": 20,
            "maximum_mean_timestep_ratio": 1.03,
            "maximum_paired_timestep_ratio": 1.05,
            "minimum_paired_non_regression_fraction": "2/3",
            "maximum_memory_ratio": 1.03,
            "transform_call_counts_must_match": True,
            "canary_lifecycle_must_be_consistent": True,
        },
        "commands": {
            "trajectories": trajectory_commands,
            "comparisons": comparison_commands,
            "profiles_balanced": profile_commands,
            "analysis": analysis,
        },
        "stop_conditions": [
            "Git HEAD or clean-worktree gate fails",
            "CUDA unavailable or GPU name mismatch",
            "any command exits nonzero",
            "missing or incorrectly ordered COMPLETE marker",
            "scientific signature or initial-Q identity mismatch",
            "shape, float64 dtype, or finite-value gate fails",
            "same-backend restart is not byte exact",
            "trajectory relative-L2 error exceeds 1e-10",
            "profile, transform-count, lifecycle, memory, or timing gate fails",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the Stage O.4 H100 plan")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    args = parser.parse_args(argv)
    plan = build_stage_o4_h100_plan(
        project_root=args.project_root,
        control_root=args.control_root,
        scratch_root=args.scratch_root,
        python=args.python,
        expected_commit=args.expected_commit,
        expected_gpu_name=args.expected_gpu_name,
    )
    print(json.dumps(plan, allow_nan=False, indent=2, sort_keys=True))
    return 0


__all__ = ["build_stage_o4_h100_plan", "main"]
