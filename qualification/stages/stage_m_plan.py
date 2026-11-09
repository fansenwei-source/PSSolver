"""Read-only command planning for the bounded Stage M H100 qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def _absolute_new_root(value: str | Path, description: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"{description} already exists: {path}")
    return path


def build_stage_m_h100_plan(
    *,
    project_root: str | Path,
    control_root: str | Path,
    scratch_root: str | Path,
    python: str | Path,
    expected_commit: str,
    expected_gpu_name: str = "H100",
    profile_trials: int = 3,
) -> dict[str, object]:
    """Return exact argv vectors without creating files or using CUDA."""

    project = Path(project_root).expanduser().resolve()
    python_path = Path(python).expanduser().resolve()
    if not project.is_dir():
        raise FileNotFoundError(f"project root is missing: {project}")
    if not python_path.is_file():
        raise FileNotFoundError(f"Python executable is missing: {python_path}")
    if (
        not isinstance(expected_commit, str)
        or len(expected_commit) != 40
        or any(character not in "0123456789abcdef" for character in expected_commit)
    ):
        raise ValueError("expected_commit must be a full lowercase Git SHA")
    if not isinstance(expected_gpu_name, str) or not expected_gpu_name.strip():
        raise ValueError("expected_gpu_name must not be empty")
    if (
        not isinstance(profile_trials, int)
        or isinstance(profile_trials, bool)
        or profile_trials <= 0
    ):
        raise ValueError("profile_trials must be a positive integer")
    control = _absolute_new_root(control_root, "control root")
    scratch = _absolute_new_root(scratch_root, "scratch root")
    if control == scratch or control in scratch.parents or scratch in control.parents:
        raise ValueError("control and scratch roots must be independent")

    production = scratch / "trajectory_production"
    shadow = scratch / "trajectory_shadow"
    comparison = control / "trajectory_comparison"
    profile_directory = control / "profiles"
    python_text = str(python_path)
    production_command = [
        python_text,
        str(project / "Plane_beris_edwards_stokes.py"),
        "--activity-number", "18",
        "--output-dir", str(production),
        "--height", "20",
        "--parameterization", "fixed-k",
        "--frank-k", "0.012345679012345678",
        "--lx", "100",
        "--ly", "100",
        "--nx", "128",
        "--ny", "128",
        "--nz", "32",
        "--dt", "0.005",
        "--steps", "6",
        "--save-start-step", "0",
        "--save-interval", "1",
        "--diagnostic-interval", "1",
        "--seed", "24",
        "--device", "cuda",
        "--dtype", "float64",
        "--zero-mode-policy", "zero_mean",
        "--dealias-rule", "cubic_half",
        "--projected-transform-execution", "truncated",
        "--molecular-field-linear-space", "spectral",
        "--stress-divergence-sum-space", "spectral",
        "--pointwise-execution", "compile",
        "--transform-execution-order", "real_first",
        "--spectral-storage", "hermitian_half",
        "--tf32", "off",
        "--spectral-refresh-steps", "2",
        "--save-hydrodynamics",
    ]
    shadow_command = [
        python_text,
        str(project / "Plane_beris_edwards_shadow_h100.py"),
        "--production-reference-dir", str(production),
        "--output-dir", str(shadow),
        "--confirm-steps", "6",
        "--expected-gpu-name", expected_gpu_name,
    ]
    comparison_command = [
        python_text,
        str(project / "scripts_plane/compare_plane_beris_edwards_shadow.py"),
        "--production-dir", str(production),
        "--shadow-dir", str(shadow),
        "--output-dir", str(comparison),
        "--relative-l2-tolerance", "1e-10",
    ]

    common_profile = [
        "--shape", "128,128,32",
        "--lengths", "100,100,20",
        "--device", "cuda",
        "--dtype", "float64",
        "--dt", "0.005",
        "--dealias-rule", "cubic_half",
        "--projected-transform-execution", "truncated",
        "--warmup-steps", "3",
        "--profile-steps", "10",
        "--spectral-refresh-interval", "2",
        "--molecular-field-linear-space", "spectral",
        "--stress-divergence-sum-space", "spectral",
        "--pointwise-execution", "compile",
        "--transform-execution-order", "real_first",
        "--spectral-storage", "hermitian_half",
        "--seed", "24",
        "--initial-q-path", str(production / "Q_0.npy"),
    ]
    profile_commands = []
    production_profiles = []
    shadow_profiles = []
    for trial in range(1, profile_trials + 1):
        production_output = profile_directory / f"production_{trial}.json"
        shadow_output = profile_directory / f"shadow_{trial}.json"
        production_profiles.append(production_output)
        shadow_profiles.append(shadow_output)
        commands = {
            "production": [
                python_text,
                str(project / "benchmarks/profile_beris_edwards_timestep.py"),
                *common_profile,
                "--output", str(production_output),
            ],
            "shadow": [
                python_text,
                str(project / "benchmarks/profile_plane_shadow_timestep.py"),
                "--production-reference-dir", str(production),
                "--warmup-steps", "3",
                "--profile-steps", "10",
                "--expected-gpu-name", expected_gpu_name,
                "--output", str(shadow_output),
            ],
        }
        order = (
            ("production", "shadow")
            if trial % 2 == 1
            else ("shadow", "production")
        )
        profile_commands.extend(
            {
                "trial": trial,
                "implementation": name,
                "argv": commands[name],
            }
            for name in order
        )

    analysis_command = [
        python_text,
        str(project / "scripts_plane/analyze_plane_shadow_h100_qualification.py"),
        "--trajectory-comparison", str(comparison / "comparison.json"),
    ]
    for path in production_profiles:
        analysis_command.extend(("--production-profile", str(path)))
    for path in shadow_profiles:
        analysis_command.extend(("--shadow-profile", str(path)))
    analysis_command.extend(
        (
            "--expected-gpu-name", expected_gpu_name,
            "--output-dir", str(control / "final_analysis"),
        )
    )
    return {
        "schema_version": 1,
        "qualification_stage": "M",
        "planning_only": True,
        "expected_commit": expected_commit,
        "expected_gpu_name": expected_gpu_name,
        "production_path_changed": False,
        "long_simulation_authorized": False,
        "default_promotion_authorized": False,
        "profile_performance_is_observational": True,
        "paths": {
            "project_root": str(project),
            "control_root": str(control),
            "scratch_root": str(scratch),
            "production_trajectory": str(production),
            "shadow_trajectory": str(shadow),
        },
        "fixed_numerical_gate": {
            "shape": [128, 128, 32],
            "steps": 6,
            "saved_steps": list(range(7)),
            "relative_l2_tolerance": 1.0e-10,
            "dtype": "float64",
            "tf32": False,
            "spectral_refresh_interval_steps": 2,
        },
        "commands": {
            "trajectory_production": production_command,
            "trajectory_shadow": shadow_command,
            "trajectory_comparison": comparison_command,
            "profiles_balanced": profile_commands,
            "analysis": analysis_command,
        },
        "stop_conditions": [
            "Git HEAD or worktree cleanliness mismatch",
            "CUDA unavailable or GPU name mismatch",
            "any command exits nonzero",
            "missing COMPLETE marker or expected JSON",
            "configuration mismatch",
            "NaN or Inf",
            "trajectory relative-L2 error exceeds 1e-10",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a read-only Stage M H100 qualification plan."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--profile-trials", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_stage_m_h100_plan(
        project_root=args.project_root,
        control_root=args.control_root,
        scratch_root=args.scratch_root,
        python=args.python,
        expected_commit=args.expected_commit,
        expected_gpu_name=args.expected_gpu_name,
        profile_trials=args.profile_trials,
    )
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


__all__ = ["build_stage_m_h100_plan", "main"]
