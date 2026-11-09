"""Planning-only contract for Stage Q.4 H100 qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .stage_n1_plan import _existing_directory, _new_directory
from .stage_o41_plan import _require_commit
from .stage_q4_qualification import (
    STAGE_Q4_PROFILE_TRIALS,
    STAGE_Q4_TRAJECTORY_STEPS,
    _read_q3_evidence,
)


def build_stage_q4_h100_plan(
    *,
    project_root: str | Path,
    production_reference: str | Path,
    control_root: str | Path,
    scratch_root: str | Path,
    python: str | Path,
    expected_commit: str,
    stage_q3_report: str | Path,
    expected_stage_q3_sha256: str,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Return the six profiles, two trajectories, and final analysis."""

    project = _existing_directory(project_root, "project root")
    reference = _existing_directory(
        production_reference,
        "R320 production reference",
    )
    control = _new_directory(control_root, "control root")
    scratch = _new_directory(scratch_root, "scratch root")
    if control == scratch or control in scratch.parents or scratch in control.parents:
        raise ValueError("control and scratch roots must be independent")
    python_path = Path(python).expanduser().resolve()
    if not python_path.is_file():
        raise FileNotFoundError(f"Python executable is missing: {python_path}")
    commit = _require_commit(expected_commit)
    q3_path, _ = _read_q3_evidence(
        stage_q3_report,
        expected_stage_q3_sha256,
    )
    if not isinstance(expected_gpu_name, str) or not expected_gpu_name.strip():
        raise ValueError("expected_gpu_name must not be empty")

    python_text = str(python_path)
    profiler = str(project / "benchmarks/profile_plane_stage_q4.py")
    trajectory_runner = str(
        project / "scripts_plane/run_plane_stage_q4_trajectory.py"
    )
    analyzer = str(project / "scripts_plane/analyze_plane_stage_q4.py")
    profile_paths = {
        role: [
            control / "profiles" / f"{role}_trial_{trial}.json"
            for trial in range(1, STAGE_Q4_PROFILE_TRIALS + 1)
        ]
        for role in ("baseline", "candidate")
    }
    profile_commands = []
    for trial in range(1, STAGE_Q4_PROFILE_TRIALS + 1):
        order = (
            ("baseline", "candidate")
            if trial % 2 == 1
            else ("candidate", "baseline")
        )
        for role in order:
            profile_commands.append(
                {
                    "trial": trial,
                    "role": role,
                    "argv": [
                        python_text,
                        profiler,
                        "--production-reference-dir",
                        str(reference),
                        "--role",
                        role,
                        "--warmup-steps",
                        "10",
                        "--profile-steps",
                        "20",
                        "--expected-gpu-name",
                        expected_gpu_name,
                        "--output",
                        str(profile_paths[role][trial - 1]),
                    ],
                }
            )

    trajectories = {
        role: scratch / "trajectories" / role
        for role in ("baseline", "candidate")
    }
    trajectory_commands = [
        {
            "role": role,
            "argv": [
                python_text,
                trajectory_runner,
                "--production-reference-dir",
                str(reference),
                "--output-dir",
                str(trajectories[role]),
                "--role",
                role,
                "--confirm-steps",
                str(STAGE_Q4_TRAJECTORY_STEPS),
                "--expected-gpu-name",
                expected_gpu_name,
            ],
        }
        for role in ("baseline", "candidate")
    ]
    analysis = [
        python_text,
        analyzer,
        "--baseline-trajectory",
        str(trajectories["baseline"]),
        "--candidate-trajectory",
        str(trajectories["candidate"]),
    ]
    for path in profile_paths["baseline"]:
        analysis.extend(("--baseline-profile", str(path)))
    for path in profile_paths["candidate"]:
        analysis.extend(("--candidate-profile", str(path)))
    analysis.extend(
        (
            "--stage-q3-report",
            str(q3_path),
            "--expected-stage-q3-sha256",
            expected_stage_q3_sha256,
            "--expected-gpu-name",
            expected_gpu_name,
            "--output",
            str(control / "stage_q4_qualification.json"),
        )
    )
    return {
        "schema_version": 1,
        "qualification_stage": "Q.4",
        "planning_only": True,
        "expected_commit": commit,
        "expected_gpu_name": expected_gpu_name,
        "paths": {
            "project_root": str(project),
            "production_reference": str(reference),
            "control_root": str(control),
            "scratch_root": str(scratch),
        },
        "stage_q3_evidence": {
            "path": str(q3_path),
            "sha256": expected_stage_q3_sha256,
            "selected_candidate": "preplanned_algebraic_batch_workspace",
        },
        "fixed_contract": {
            "shape": list((320, 320, 80)),
            "profile_trials_per_role": STAGE_Q4_PROFILE_TRIALS,
            "warmup_steps": 10,
            "profile_steps": 20,
            "trajectory_steps": STAGE_Q4_TRAJECTORY_STEPS,
            "trajectory_saved_steps": list(
                range(STAGE_Q4_TRAJECTORY_STEPS + 1)
            ),
            "baseline_batch_assembly": "copy_cat",
            "candidate_batch_assembly": "preallocated_workspace",
            "algebraic_execution_policy": "batched_physical_islands",
            "algebraic_output_publication_policy": "deferred_stack",
            "explicit_rhs_execution": "compile",
            "changes_equations": False,
            "changes_transform_order_or_count": False,
            "changes_boundary_signatures": False,
            "retains_timestep_inputs": False,
            "changes_production_default": False,
        },
        "commands": {
            "profiles_balanced": profile_commands,
            "trajectories": trajectory_commands,
            "analysis": analysis,
        },
        "stop_conditions": [
            "Git HEAD, worktree cleanliness, or Stage Q.3 identity fails",
            "production-reference identity or R320 shape differs",
            "CUDA unavailable or GPU name mismatch",
            "any command exits nonzero or produces non-finite fields",
            "workspace aliases output or retains timestep input tensors",
            "missing COMPLETE marker, array, profile, or result JSON",
            "trajectory relative-L2 error exceeds 1e-10",
            "input identity changes during qualification",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the Stage Q.4 plan")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--production-reference", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--stage-q3-report", type=Path, required=True)
    parser.add_argument("--expected-stage-q3-sha256", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    args = parser.parse_args(argv)
    report = build_stage_q4_h100_plan(
        project_root=args.project_root,
        production_reference=args.production_reference,
        control_root=args.control_root,
        scratch_root=args.scratch_root,
        python=args.python,
        expected_commit=args.expected_commit,
        stage_q3_report=args.stage_q3_report,
        expected_stage_q3_sha256=args.expected_stage_q3_sha256,
        expected_gpu_name=args.expected_gpu_name,
    )
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0


__all__ = ["build_stage_q4_h100_plan", "main"]
