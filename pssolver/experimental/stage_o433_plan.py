"""Planning-only contract for Stage O.4.3.3 producer-owned packing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .stage_o431_plan import build_stage_o431_h100_plan
from .stage_o43_qualification import _require_neutral_stage_o431_evidence


def build_stage_o433_h100_plan(
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
    stage_o431_report: str | Path,
    expected_stage_o431_sha256: str,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Return one bounded H/stress producer-layout qualification plan."""

    o431_path, _ = _require_neutral_stage_o431_evidence(
        stage_o431_report,
        expected_stage_o431_sha256,
    )
    plan = build_stage_o431_h100_plan(
        project_root=project_root,
        control_root=control_root,
        scratch_root=scratch_root,
        python=python,
        r128_production_reference_dir=r128_production_reference_dir,
        r320_production_reference_dir=r320_production_reference_dir,
        expected_commit=expected_commit,
        stage_o42_report=stage_o42_report,
        expected_stage_o42_sha256=expected_stage_o42_sha256,
        stage_o43_report=stage_o43_report,
        expected_stage_o43_sha256=expected_stage_o43_sha256,
        expected_gpu_name=expected_gpu_name,
    )
    project = Path(plan["paths"]["project_root"])
    control = Path(plan["paths"]["control_root"])
    replacements = {
        str(project / "benchmarks/run_plane_stage_o431_trajectory.py"): str(
            project / "benchmarks/run_plane_stage_o433_trajectory.py"
        ),
        str(project / "benchmarks/profile_plane_stage_o431.py"): str(
            project / "benchmarks/profile_plane_stage_o433.py"
        ),
        str(
            project / "scripts_plane/analyze_plane_stage_o431_qualification.py"
        ): str(
            project / "scripts_plane/analyze_plane_stage_o433_qualification.py"
        ),
        str(control / "stage_o431_qualification.json"): str(
            control / "stage_o433_qualification.json"
        ),
    }
    for command_group in ("trajectories", "profiles_balanced"):
        for command in plan["commands"][command_group]:
            command["argv"] = [
                replacements.get(value, value) for value in command["argv"]
            ]
    analysis = [
        replacements.get(value, value)
        for value in plan["commands"]["analysis"]
    ]
    output_index = analysis.index("--output")
    analysis[output_index:output_index] = [
        "--stage-o431-report",
        str(o431_path),
        "--expected-stage-o431-sha256",
        expected_stage_o431_sha256,
    ]
    plan["commands"]["analysis"] = analysis
    plan["qualification_stage"] = "O.4.3.3"
    plan["prior_evidence"]["o431"] = {
        "path": str(o431_path),
        "sha256": expected_stage_o431_sha256,
    }
    plan["candidate_contract"] = {
        "baseline_output_publication": "deferred_stack",
        "baseline_batch_assembly": "contiguous_storage_view",
        "baseline_producer_output_layout": "component_mapping",
        "candidate_output_publication": "deferred_stack",
        "candidate_batch_assembly": "contiguous_storage_view",
        "candidate_producer_output_layout": "boundary_packed",
        "adapted_producers": ["molecular_field", "nematic_stress"],
        "packing_site": "inside_pointwise_kernel",
        "post_kernel_stack": False,
        "post_kernel_cat": False,
        "candidate_adds_persistent_tensor_storage": False,
        "cross_generation_reuse": False,
        "fallback": "component_mapping",
        "changes_equations": False,
        "changes_transform_order": False,
        "changes_field_lifetime_boundary": False,
        "changes_q_explicit_rhs": False,
        "changes_gradient_producers": False,
        "production_default_may_change": False,
    }
    plan["qualification_contract"]["eligible_result"] = (
        "stage_o44_decision_only"
    )
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the Stage O.4.3.3 plan")
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
    parser.add_argument("--stage-o431-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o431-sha256", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    args = parser.parse_args(argv)
    plan = build_stage_o433_h100_plan(
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
        stage_o431_report=args.stage_o431_report,
        expected_stage_o431_sha256=args.expected_stage_o431_sha256,
        expected_gpu_name=args.expected_gpu_name,
    )
    print(json.dumps(plan, allow_nan=False, indent=2, sort_keys=True))
    return 0


__all__ = ["build_stage_o433_h100_plan", "main"]
