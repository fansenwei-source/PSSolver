"""Read-only planning for bounded Stage N H100 diagnostics."""

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


def build_stage_n_h100_plan(
    *,
    project_root: str | Path,
    production_reference: str | Path,
    control_root: str | Path,
    python: str | Path,
    expected_commit: str,
    expected_gpu_name: str = "H100",
    diagnostic_trials: int = 3,
) -> dict[str, object]:
    """Return exact diagnostic argv vectors without creating output."""

    project = _existing_directory(project_root, "project root")
    production = _existing_directory(
        production_reference,
        "production reference",
    )
    control = _new_directory(control_root, "control root")
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
        not isinstance(diagnostic_trials, int)
        or isinstance(diagnostic_trials, bool)
        or diagnostic_trials <= 0
    ):
        raise ValueError("diagnostic_trials must be a positive integer")

    profiles = control / "profiles"
    commands = []
    outputs = []
    for trial in range(1, diagnostic_trials + 1):
        output = profiles / f"shadow_stage_n_{trial}.json"
        outputs.append(output)
        commands.append(
            {
                "trial": trial,
                "argv": [
                    str(python_path),
                    str(project / "benchmarks/profile_plane_shadow_stage_n.py"),
                    "--production-reference-dir",
                    str(production),
                    "--warmup-steps",
                    "10",
                    "--profile-steps",
                    "20",
                    "--operator-audit-steps",
                    "2",
                    "--expected-gpu-name",
                    expected_gpu_name,
                    "--output",
                    str(output),
                ],
            }
        )
    analysis = [
        str(python_path),
        str(project / "scripts_plane/analyze_plane_shadow_stage_n.py"),
    ]
    for output in outputs:
        analysis.extend(("--profile", str(output)))
    analysis.extend(("--output", str(control / "stage_n_diagnostics.json")))
    return {
        "schema_version": 1,
        "qualification_stage": "N",
        "planning_only": True,
        "architecture_decision": "retain_shadow_without_promotion",
        "expected_commit": expected_commit,
        "expected_gpu_name": expected_gpu_name,
        "production_path_changed": False,
        "production_promotion_authorized": False,
        "optimization_authorized": False,
        "paths": {
            "project_root": str(project),
            "production_reference": str(production),
            "control_root": str(control),
        },
        "fixed_diagnostic_gate": {
            "diagnostic_trials": diagnostic_trials,
            "warmup_steps": 10,
            "profile_steps": 20,
            "operator_audit_steps": 2,
            "semantic_timing_and_operator_audit_are_separate": True,
        },
        "commands": {
            "diagnostic_profiles": commands,
            "analysis": analysis,
        },
        "stop_conditions": [
            "Git HEAD or worktree cleanliness mismatch",
            "production-reference identity mismatch",
            "CUDA unavailable or GPU name mismatch",
            "any command exits nonzero",
            "missing or non-finite diagnostic output",
        ],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Print a read-only Stage N H100 diagnostic plan."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--production-reference", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--diagnostic-trials", type=int, default=3)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    plan = build_stage_n_h100_plan(
        project_root=args.project_root,
        production_reference=args.production_reference,
        control_root=args.control_root,
        python=args.python,
        expected_commit=args.expected_commit,
        expected_gpu_name=args.expected_gpu_name,
        diagnostic_trials=args.diagnostic_trials,
    )
    print(json.dumps(plan, allow_nan=False, indent=2, sort_keys=True))
    return 0


__all__ = ["build_stage_n_h100_plan", "main"]
