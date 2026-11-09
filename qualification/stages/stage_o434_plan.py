"""Planning-only contract for Stage O.4.3.4 attribution diagnostics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .stage_n1_plan import _existing_directory, _new_directory
from .stage_o41_plan import _require_commit
from .stage_o434_diagnostics import _require_neutral_o433_evidence


def build_stage_o434_h100_plan(
    *,
    project_root: str | Path,
    control_root: str | Path,
    python: str | Path,
    r320_production_reference_dir: str | Path,
    expected_commit: str,
    stage_o433_report: str | Path,
    expected_stage_o433_sha256: str,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Return three read-only R320 profiles and one attribution analysis."""

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
    evidence, _ = _require_neutral_o433_evidence(
        stage_o433_report,
        expected_stage_o433_sha256,
    )
    if not isinstance(expected_gpu_name, str) or not expected_gpu_name.strip():
        raise ValueError("expected_gpu_name must not be empty")

    profiler = str(project / "benchmarks/profile_plane_stage_o434.py")
    analyzer = str(
        project / "scripts_plane/analyze_plane_stage_o434_diagnostic.py"
    )
    profile_paths = tuple(
        control / "profiles" / f"r320_candidate_attribution_trial_{trial}.json"
        for trial in range(1, 4)
    )
    profiles = []
    for trial, output in enumerate(profile_paths, start=1):
        profiles.append(
            {
                "trial": trial,
                "argv": [
                    str(python_path),
                    profiler,
                    "--production-reference-dir",
                    str(reference),
                    "--warmup-steps",
                    "10",
                    "--profile-steps",
                    "20",
                    "--expected-gpu-name",
                    expected_gpu_name,
                    "--output",
                    str(output),
                ],
            }
        )
    analysis = [
        str(python_path),
        analyzer,
    ]
    for path in profile_paths:
        analysis.extend(("--profile", str(path)))
    analysis.extend(
        (
            "--stage-o433-report",
            str(evidence),
            "--expected-stage-o433-sha256",
            expected_stage_o433_sha256,
            "--output",
            str(control / "stage_o434_diagnostic.json"),
        )
    )
    return {
        "schema_version": 1,
        "qualification_stage": "O.4.3.4",
        "planning_only": True,
        "expected_commit": commit,
        "expected_gpu_name": expected_gpu_name,
        "paths": {
            "project_root": str(project),
            "control_root": str(control),
            "r320_production_reference_dir": str(reference),
        },
        "stage_o433_evidence": {
            "path": str(evidence),
            "sha256": expected_stage_o433_sha256,
            "classification_must_remain": "B_neutral",
        },
        "diagnostic_contract": {
            "shape": list((320, 320, 80)),
            "profile_trials": 3,
            "warmup_steps": 10,
            "profile_steps": 20,
            "runtime_variant": "o433_boundary_packed_candidate",
            "source_level_batch_counts": True,
            "source_level_logical_input_bytes": True,
            "source_level_copy_materialized_output_bytes": True,
            "source_level_deferred_cuda_timing": True,
            "unattributed_batches_forbidden": True,
            "retained_tensor_references": 0,
            "changes_equations": False,
            "changes_tensor_layout": False,
            "changes_transform_order": False,
            "changes_field_lifetime": False,
            "production_default_may_change": False,
            "eligible_result": "target_selection_review_only",
        },
        "commands": {
            "profiles": profiles,
            "analysis": analysis,
        },
        "stop_conditions": [
            "Git HEAD, clean-worktree, or O.4.3.3 identity gate fails",
            "production reference validation or R320 identity fails",
            "CUDA unavailable or GPU name mismatch",
            "any profile exits nonzero or contains non-finite values",
            "any projected batch remains unattributed",
            "source counter structure differs across trials",
            "analysis exits nonzero or input identity changes",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the Stage O.4.3.4 plan")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument(
        "--r320-production-reference-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--stage-o433-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o433-sha256", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    args = parser.parse_args(argv)
    plan = build_stage_o434_h100_plan(
        project_root=args.project_root,
        control_root=args.control_root,
        python=args.python,
        r320_production_reference_dir=args.r320_production_reference_dir,
        expected_commit=args.expected_commit,
        stage_o433_report=args.stage_o433_report,
        expected_stage_o433_sha256=args.expected_stage_o433_sha256,
        expected_gpu_name=args.expected_gpu_name,
    )
    print(json.dumps(plan, allow_nan=False, indent=2, sort_keys=True))
    return 0


__all__ = ["build_stage_o434_h100_plan", "main"]
