"""Read-only planning for Stage N.1 H100 qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


def _existing_directory(value: str | Path, description: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{description} is missing: {path}")
    return path


def _new_directory(value: str | Path, description: str) -> Path:
    path = Path(value).expanduser().resolve()
    if path.exists():
        raise FileExistsError(f"{description} already exists: {path}")
    return path


def build_stage_n1_h100_plan(
    *,
    project_root: str | Path,
    production_reference: str | Path,
    control_root: str | Path,
    scratch_root: str | Path,
    python: str | Path,
    expected_commit: str,
    expected_gpu_name: str = "H100",
    profile_trials: int = 3,
) -> dict[str, object]:
    """Return exact candidate-trajectory and balanced-profile commands."""

    project = _existing_directory(project_root, "project root")
    production = _existing_directory(
        production_reference,
        "production reference",
    )
    control = _new_directory(control_root, "control root")
    scratch = _new_directory(scratch_root, "scratch root")
    if control == scratch or control in scratch.parents or scratch in control.parents:
        raise ValueError("control and scratch roots must be independent")
    python_path = Path(python).expanduser().resolve()
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

    python_text = str(python_path)
    candidate = scratch / "trajectory_candidate"
    comparison = control / "trajectory_comparison"
    profiles = control / "profiles"
    trajectory = [
        python_text,
        str(project / "Plane_beris_edwards_shadow_stage_n1.py"),
        "--production-reference-dir",
        str(production),
        "--output-dir",
        str(candidate),
        "--confirm-steps",
        "6",
        "--expected-gpu-name",
        expected_gpu_name,
    ]
    compare = [
        python_text,
        str(project / "scripts_plane/compare_plane_beris_edwards_shadow.py"),
        "--production-dir",
        str(production),
        "--shadow-dir",
        str(candidate),
        "--output-dir",
        str(comparison),
        "--relative-l2-tolerance",
        "1e-10",
    ]
    profile_commands = []
    control_profiles = []
    candidate_profiles = []
    for trial in range(1, profile_trials + 1):
        paths = {
            "control": profiles / f"control_{trial}.json",
            "candidate": profiles / f"candidate_{trial}.json",
        }
        control_profiles.append(paths["control"])
        candidate_profiles.append(paths["candidate"])
        commands = {
            mode: [
                python_text,
                str(project / "benchmarks/profile_plane_shadow_stage_n1.py"),
                "--production-reference-dir",
                str(production),
                "--mode",
                mode,
                "--warmup-steps",
                "10",
                "--profile-steps",
                "20",
                "--transform-audit-steps",
                "2",
                "--expected-gpu-name",
                expected_gpu_name,
                "--output",
                str(paths[mode]),
            ]
            for mode in ("control", "candidate")
        }
        order = (
            ("control", "candidate")
            if trial % 2 == 1
            else ("candidate", "control")
        )
        profile_commands.extend(
            {
                "trial": trial,
                "mode": mode,
                "argv": commands[mode],
            }
            for mode in order
        )
    analysis = [
        python_text,
        str(project / "scripts_plane/analyze_plane_shadow_stage_n1.py"),
        "--trajectory-comparison",
        str(comparison / "comparison.json"),
    ]
    for path in control_profiles:
        analysis.extend(("--control-profile", str(path)))
    for path in candidate_profiles:
        analysis.extend(("--candidate-profile", str(path)))
    analysis.extend(
        (
            "--expected-gpu-name",
            expected_gpu_name,
            "--output",
            str(control / "stage_n1_qualification.json"),
        )
    )
    return {
        "schema_version": 1,
        "qualification_stage": "N.1",
        "planning_only": True,
        "expected_commit": expected_commit,
        "expected_gpu_name": expected_gpu_name,
        "production_path_changed": False,
        "production_promotion_authorized": False,
        "long_simulation_authorized": False,
        "paths": {
            "project_root": str(project),
            "production_reference": str(production),
            "control_root": str(control),
            "scratch_root": str(scratch),
            "candidate_trajectory": str(candidate),
        },
        "fixed_gates": {
            "trajectory_steps": 6,
            "saved_steps": list(range(7)),
            "relative_l2_tolerance": 1.0e-10,
            "profile_trials": profile_trials,
            "warmup_steps": 10,
            "profile_steps": 20,
            "transform_audit_steps": 2,
            "minimum_mean_speedup": 1.02,
            "minimum_faster_trials_fraction": "2/3",
            "maximum_memory_ratio": 1.05,
            "candidate_forward_calls_must_decrease": True,
            "candidate_inverse_calls_must_not_increase": True,
            "generation_must_retain_no_tensor_pairs": True,
        },
        "commands": {
            "candidate_trajectory": trajectory,
            "trajectory_comparison": compare,
            "profiles_balanced": profile_commands,
            "analysis": analysis,
        },
        "stop_conditions": [
            "Git HEAD or worktree cleanliness mismatch",
            "production-reference identity mismatch",
            "CUDA unavailable or GPU name mismatch",
            "any command exits nonzero",
            "missing COMPLETE marker or expected JSON",
            "configuration mismatch",
            "NaN or Inf",
            "trajectory relative-L2 error exceeds 1e-10",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print the Stage N.1 H100 plan")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--production-reference", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--profile-trials", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_stage_n1_h100_plan(
        project_root=args.project_root,
        production_reference=args.production_reference,
        control_root=args.control_root,
        scratch_root=args.scratch_root,
        python=args.python,
        expected_commit=args.expected_commit,
        expected_gpu_name=args.expected_gpu_name,
        profile_trials=args.profile_trials,
    )
    print(json.dumps(plan, allow_nan=False, indent=2, sort_keys=True))
    return 0


__all__ = ["build_stage_n1_h100_plan", "main"]
