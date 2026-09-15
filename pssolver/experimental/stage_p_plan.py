"""Planning-only contract for the Plane Stage P H100 diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .stage_n1_plan import _existing_directory, _new_directory
from .stage_o41_plan import _require_commit
from .stage_o_closure import (
    build_stage_o_closure_decision,
    stage_o_closure_identity_sha256,
)
from .stage_p_diagnostics import (
    STAGE_P_OPERATOR_AUDIT_STEPS,
    STAGE_P_PROFILE_TRIALS,
    STAGE_P_SEMANTIC_STEPS,
    STAGE_P_SHAPE,
    STAGE_P_THROUGHPUT_STEPS,
    STAGE_P_WARMUP_STEPS,
)


def build_stage_p_h100_plan(
    *,
    project_root: str | Path,
    control_root: str | Path,
    python: str | Path,
    r320_production_reference_dir: str | Path,
    expected_commit: str,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Return six balanced profiles and one diagnostic analysis command."""

    project = _existing_directory(project_root, "project root")
    control = _new_directory(control_root, "control root")
    reference = _existing_directory(
        r320_production_reference_dir,
        "R320 production reference directory",
    )
    python_path = Path(python).expanduser().resolve()
    if not python_path.is_file():
        raise FileNotFoundError(f"Python executable is missing: {python_path}")
    commit = _require_commit(expected_commit)
    if not isinstance(expected_gpu_name, str) or not expected_gpu_name.strip():
        raise ValueError("expected_gpu_name must not be empty")

    closure = build_stage_o_closure_decision(project)
    closure_sha256 = stage_o_closure_identity_sha256(closure)
    profiler = str(project / "benchmarks/profile_plane_stage_p.py")
    analyzer = str(project / "scripts_plane/analyze_plane_stage_p.py")
    profiles = control / "profiles"

    paths = {
        role: tuple(
            profiles / f"{role}_trial_{trial}.json"
            for trial in range(1, STAGE_P_PROFILE_TRIALS + 1)
        )
        for role in ("legacy_production", "separated_canary")
    }

    profile_commands = []
    orders = (
        ("legacy_production", "separated_canary"),
        ("separated_canary", "legacy_production"),
        ("legacy_production", "separated_canary"),
    )
    for trial, order in enumerate(orders, start=1):
        for role in order:
            profile_commands.append(
                {
                    "trial": trial,
                    "runtime_role": role,
                    "argv": [
                        str(python_path),
                        profiler,
                        "--production-reference-dir",
                        str(reference),
                        "--runtime-role",
                        role,
                        "--expected-gpu-name",
                        expected_gpu_name,
                        "--output",
                        str(paths[role][trial - 1]),
                    ],
                }
            )

    analysis = [str(python_path), analyzer]
    for role, option in (
        ("legacy_production", "--production-profile"),
        ("separated_canary", "--canary-profile"),
    ):
        for path in paths[role]:
            analysis.extend((option, str(path)))
    analysis.extend(
        (
            "--expected-stage-o-closure-sha256",
            closure_sha256,
            "--output",
            str(control / "stage_p_diagnostic.json"),
        )
    )

    return {
        "schema_version": 1,
        "qualification_stage": "P",
        "planning_only": True,
        "classification": "PLANNING_COMPLETE",
        "architecture_decision": (
            "measure_production_canary_operator_kernel_gap"
        ),
        "expected_commit": commit,
        "expected_gpu_name": expected_gpu_name,
        "stage_o_closure": {
            "sha256": closure_sha256,
            "classification": closure.to_metadata()["classification"],
            "materialization_layout_line_remains_closed": True,
        },
        "paths": {
            "project_root": str(project),
            "control_root": str(control),
            "r320_production_reference_dir": str(reference),
        },
        "diagnostic_contract": {
            "shape": list(STAGE_P_SHAPE),
            "profile_trials_per_runtime": STAGE_P_PROFILE_TRIALS,
            "balanced_order": [list(order) for order in orders],
            "warmup_steps_per_window": STAGE_P_WARMUP_STEPS,
            "noninstrumented_throughput_steps": STAGE_P_THROUGHPUT_STEPS,
            "operator_audit_steps": STAGE_P_OPERATOR_AUDIT_STEPS,
            "semantic_steps": STAGE_P_SEMANTIC_STEPS,
            "measurement_windows_are_separate": True,
            "raw_profiler_trace_retained": False,
            "large_simulation_arrays_written": False,
            "changes_equations": False,
            "changes_runtime_implementation": False,
            "changes_production_default": False,
            "optimization_candidate_authorized": False,
            "eligible_result": "stage_q_target_review_only",
        },
        "commands": {
            "profiles_balanced": profile_commands,
            "analysis": analysis,
        },
        "authorizations": {
            "single_h100_diagnostic_job": True,
            "stage_q_target_review": True,
            "stage_q_candidate_implementation": False,
            "production_promotion": False,
            "default_change": False,
            "benchmark_or_long_run": False,
        },
        "stop_conditions": [
            "Git HEAD or clean-worktree gate fails",
            "Stage O closure identity changes",
            "production reference validation or R320 identity fails",
            "CUDA unavailable or GPU name mismatch",
            "any profile exits nonzero or produces non-finite state",
            "raw trace retention or bounded-event contract is violated",
            "profile configuration or initial-Q identity differs",
            "analysis exits nonzero or any input identity changes",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the Stage P H100 plan")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--r320-production-reference-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    args = parser.parse_args(argv)
    plan = build_stage_p_h100_plan(
        project_root=args.project_root,
        control_root=args.control_root,
        python=args.python,
        r320_production_reference_dir=args.r320_production_reference_dir,
        expected_commit=args.expected_commit,
        expected_gpu_name=args.expected_gpu_name,
    )
    print(json.dumps(plan, allow_nan=False, indent=2, sort_keys=True))
    return 0


__all__ = ["build_stage_p_h100_plan", "main"]
