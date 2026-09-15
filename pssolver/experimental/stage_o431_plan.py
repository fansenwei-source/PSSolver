"""Planning-only contract for the Stage O.4.3.1 H100 qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .stage_n1_plan import _existing_directory, _new_directory
from .stage_o4_qualification import _sha256
from .stage_o41_plan import _require_commit
from .stage_o43_qualification import (
    STAGE_O43_DECISION_SHAPE,
    STAGE_O43_DIAGNOSTIC_SHAPE,
    STAGE_O43_MAXIMUM_MEAN_TIMESTEP_RATIO,
    STAGE_O43_MAXIMUM_MEMORY_RATIO,
    STAGE_O43_MAXIMUM_PAIRED_TIMESTEP_RATIO,
    STAGE_O43_PROFILE_TRIALS,
    STAGE_O43_RELATIVE_L2_TOLERANCE,
    STAGE_O43_TRAJECTORY_STEPS,
)


def build_stage_o431_h100_plan(
    *,
    project_root: str | Path,
    control_root: str | Path,
    scratch_root: str | Path,
    python: str | Path,
    r128_production_reference_dir: str | Path,
    r320_production_reference_dir: str | Path,
    expected_commit: str,
    stage_o42_report: str | Path,
    expected_stage_o42_sha256: str,
    stage_o43_report: str | Path,
    expected_stage_o43_sha256: str,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Return bounded natural-storage-view qualification commands."""

    project = _existing_directory(project_root, "project root")
    control = _new_directory(control_root, "control root")
    scratch = _new_directory(scratch_root, "scratch root")
    if control == scratch or control in scratch.parents or scratch in control.parents:
        raise ValueError("control and scratch roots must be independent")
    python_path = Path(python).expanduser().resolve()
    if not python_path.is_file():
        raise FileNotFoundError(f"Python executable is missing: {python_path}")
    references = {
        "r128": _existing_directory(
            r128_production_reference_dir,
            "R128 production reference",
        ),
        "r320": _existing_directory(
            r320_production_reference_dir,
            "R320 production reference",
        ),
    }
    commit = _require_commit(expected_commit)
    evidence = {
        "o42": (
            Path(stage_o42_report).expanduser().resolve(),
            expected_stage_o42_sha256,
        ),
        "o43": (
            Path(stage_o43_report).expanduser().resolve(),
            expected_stage_o43_sha256,
        ),
    }
    for stage, (path, expected_sha256) in evidence.items():
        if not path.is_file():
            raise FileNotFoundError(f"Stage {stage} report is missing: {path}")
        if _sha256(path) != expected_sha256:
            raise ValueError(f"Stage {stage} report SHA-256 differs")
    if not isinstance(expected_gpu_name, str) or not expected_gpu_name.strip():
        raise ValueError("expected_gpu_name must not be empty")

    python_text = str(python_path)
    trajectory_driver = str(
        project / "benchmarks/run_plane_stage_o431_trajectory.py"
    )
    profiler = str(project / "benchmarks/profile_plane_stage_o431.py")
    analyzer = str(
        project / "scripts_plane/analyze_plane_stage_o431_qualification.py"
    )
    roles = ("baseline", "candidate")
    trajectory_paths = {
        role: scratch / "trajectories" / role for role in roles
    }
    trajectory_commands = [
        {
            "role": role,
            "argv": [
                python_text,
                trajectory_driver,
                "--production-reference-dir",
                str(references["r128"]),
                "--output-dir",
                str(trajectory_paths[role]),
                "--role",
                role,
                "--confirm-steps",
                str(STAGE_O43_TRAJECTORY_STEPS),
                "--expected-gpu-name",
                expected_gpu_name,
            ],
        }
        for role in roles
    ]

    profiles = control / "profiles"
    shapes = {
        "r128": STAGE_O43_DIAGNOSTIC_SHAPE,
        "r320": STAGE_O43_DECISION_SHAPE,
    }
    profile_paths = {
        (scale, role, trial): profiles / f"{scale}_{role}_{trial}.json"
        for scale in shapes
        for role in roles
        for trial in range(1, STAGE_O43_PROFILE_TRIALS + 1)
    }
    profile_commands = []
    for scale in shapes:
        for trial in range(1, STAGE_O43_PROFILE_TRIALS + 1):
            order = roles if trial % 2 == 1 else tuple(reversed(roles))
            for role in order:
                profile_commands.append(
                    {
                        "scale": scale,
                        "trial": trial,
                        "role": role,
                        "argv": [
                            python_text,
                            profiler,
                            "--production-reference-dir",
                            str(references[scale]),
                            "--role",
                            role,
                            "--warmup-steps",
                            "10",
                            "--profile-steps",
                            "20",
                            "--expected-gpu-name",
                            expected_gpu_name,
                            "--output",
                            str(profile_paths[(scale, role, trial)]),
                        ],
                    }
                )

    analysis = [
        python_text,
        analyzer,
        "--baseline-trajectory",
        str(trajectory_paths["baseline"]),
        "--candidate-trajectory",
        str(trajectory_paths["candidate"]),
    ]
    for scale in shapes:
        for role in roles:
            for trial in range(1, STAGE_O43_PROFILE_TRIALS + 1):
                analysis.extend(
                    (
                        f"--{role}-{scale}-profile",
                        str(profile_paths[(scale, role, trial)]),
                    )
                )
    analysis.extend(
        (
            "--stage-o42-report",
            str(evidence["o42"][0]),
            "--expected-stage-o42-sha256",
            expected_stage_o42_sha256,
            "--stage-o43-report",
            str(evidence["o43"][0]),
            "--expected-stage-o43-sha256",
            expected_stage_o43_sha256,
            "--expected-gpu-name",
            expected_gpu_name,
            "--output",
            str(control / "stage_o431_qualification.json"),
        )
    )
    return {
        "schema_version": 1,
        "qualification_stage": "O.4.3.1",
        "planning_only": True,
        "expected_commit": commit,
        "expected_gpu_name": expected_gpu_name,
        "paths": {
            "project_root": str(project),
            "control_root": str(control),
            "scratch_root": str(scratch),
            "r128_production_reference_dir": str(references["r128"]),
            "r320_production_reference_dir": str(references["r320"]),
        },
        "prior_evidence": {
            stage: {"path": str(path), "sha256": sha256}
            for stage, (path, sha256) in evidence.items()
        },
        "candidate_contract": {
            "baseline_output_publication": "deferred_stack",
            "baseline_batch_assembly": "copy_cat",
            "candidate_output_publication": "deferred_stack",
            "candidate_batch_assembly": "contiguous_storage_view",
            "candidate_republishes_algebraic_outputs": False,
            "candidate_adds_persistent_tensor_storage": False,
            "fallback": "copy_cat",
            "changes_equations": False,
            "changes_transform_order": False,
            "changes_field_lifetime_boundary": False,
            "production_default_may_change": False,
        },
        "qualification_contract": {
            "trajectory_steps": STAGE_O43_TRAJECTORY_STEPS,
            "relative_l2_tolerance": STAGE_O43_RELATIVE_L2_TOLERANCE,
            "profile_trials_per_role_and_scale": STAGE_O43_PROFILE_TRIALS,
            "profile_shapes": {
                name: list(shape) for name, shape in shapes.items()
            },
            "warmup_steps": 10,
            "profile_steps": 20,
            "maximum_mean_timestep_ratio": (
                STAGE_O43_MAXIMUM_MEAN_TIMESTEP_RATIO
            ),
            "maximum_paired_timestep_ratio": (
                STAGE_O43_MAXIMUM_PAIRED_TIMESTEP_RATIO
            ),
            "maximum_memory_ratio": STAGE_O43_MAXIMUM_MEMORY_RATIO,
            "eligible_result": "stage_o44_decision_only",
        },
        "commands": {
            "trajectories": trajectory_commands,
            "profiles_balanced": profile_commands,
            "analysis": analysis,
        },
        "stop_conditions": [
            "Git HEAD, clean-worktree, or prior-evidence identity gate fails",
            "production reference validation or fixed-shape identity fails",
            "CUDA unavailable or GPU name mismatch",
            "any trajectory or profile command exits nonzero",
            "any output contains NaN or Inf",
            "analysis output is missing or malformed",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the Stage O.4.3.1 plan")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--r128-production-reference-dir", type=Path, required=True
    )
    parser.add_argument(
        "--r320-production-reference-dir", type=Path, required=True
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--stage-o42-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o42-sha256", required=True)
    parser.add_argument("--stage-o43-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o43-sha256", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    args = parser.parse_args(argv)
    plan = build_stage_o431_h100_plan(
        project_root=args.project_root,
        control_root=args.control_root,
        scratch_root=args.scratch_root,
        python=args.python,
        r128_production_reference_dir=args.r128_production_reference_dir,
        r320_production_reference_dir=args.r320_production_reference_dir,
        expected_commit=args.expected_commit,
        stage_o42_report=args.stage_o42_report,
        expected_stage_o42_sha256=args.expected_stage_o42_sha256,
        stage_o43_report=args.stage_o43_report,
        expected_stage_o43_sha256=args.expected_stage_o43_sha256,
        expected_gpu_name=args.expected_gpu_name,
    )
    print(json.dumps(plan, allow_nan=False, indent=2, sort_keys=True))
    return 0


__all__ = ["build_stage_o431_h100_plan", "main"]
