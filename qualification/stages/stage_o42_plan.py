"""Planning-only contract for the Stage O.4.2 H100 diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .stage_n1_plan import _existing_directory, _new_directory
from .stage_o4_qualification import _sha256
from .stage_o41_plan import _require_commit


def build_stage_o42_h100_plan(
    *,
    project_root: str | Path,
    control_root: str | Path,
    python: str | Path,
    production_reference_dir: str | Path,
    expected_commit: str,
    stage_o41_report: str | Path,
    expected_stage_o41_sha256: str,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    """Return two read-only profile commands and one CPU analysis command."""

    project = _existing_directory(project_root, "project root")
    control = _new_directory(control_root, "control root")
    reference = _existing_directory(
        production_reference_dir,
        "production reference directory",
    )
    python_path = Path(python).expanduser().resolve()
    if not python_path.is_file():
        raise FileNotFoundError(f"Python executable is missing: {python_path}")
    commit = _require_commit(expected_commit)
    evidence = Path(stage_o41_report).expanduser().resolve()
    if not evidence.is_file():
        raise FileNotFoundError(f"Stage O.4.1 report is missing: {evidence}")
    if _sha256(evidence) != expected_stage_o41_sha256:
        raise ValueError("Stage O.4.1 report SHA-256 differs from expectation")
    if not isinstance(expected_gpu_name, str) or not expected_gpu_name.strip():
        raise ValueError("expected_gpu_name must not be empty")

    profiler = str(project / "benchmarks/diagnose_plane_stage_o42.py")
    analyzer = str(project / "scripts_plane/analyze_plane_stage_o42_diagnostics.py")
    profiles = control / "profiles"
    profile_paths = {
        role: profiles / f"r320_{role}_diagnostic.json" for role in ("legacy", "canary")
    }

    def profile_command(role: str) -> list[str]:
        return [
            str(python_path),
            profiler,
            "--production-reference-dir",
            str(reference),
            "--role",
            role,
            "--warmup-steps",
            "10",
            "--diagnostic-steps",
            "2",
            "--operator-audit-steps",
            "2",
            "--expected-gpu-name",
            expected_gpu_name,
            "--output",
            str(profile_paths[role]),
        ]

    analysis = [
        str(python_path),
        analyzer,
        "--legacy-profile",
        str(profile_paths["legacy"]),
        "--canary-profile",
        str(profile_paths["canary"]),
        "--stage-o41-report",
        str(evidence),
        "--expected-stage-o41-sha256",
        expected_stage_o41_sha256,
        "--output",
        str(control / "stage_o42_diagnostics.json"),
    ]
    return {
        "schema_version": 1,
        "qualification_stage": "O.4.2",
        "planning_only": True,
        "expected_commit": commit,
        "expected_gpu_name": expected_gpu_name,
        "paths": {
            "project_root": str(project),
            "control_root": str(control),
            "production_reference_dir": str(reference),
        },
        "stage_o41_evidence": {
            "path": str(evidence),
            "sha256": expected_stage_o41_sha256,
            "classification_must_remain": "B_neutral",
        },
        "diagnostic_contract": {
            "shape": [320, 320, 80],
            "runtime_roles": ["legacy", "canary"],
            "profile_count": 2,
            "warmup_steps": 10,
            "diagnostic_steps": 2,
            "operator_audit_steps": 2,
            "operator_audit_settling_steps": 2,
            "runtime_object_graph_inventory": True,
            "runtime_roots_only": True,
            "global_gc_traversal": False,
            "overlapping_storage_address_ranges_coalesced": True,
            "allocator_block_reconciliation": True,
            "allocator_counter_stability_checked": True,
            "storage_identity_priming_pass": True,
            "repeated_inventory_identity_checked": True,
            "priming_inventory_preserved": True,
            "canonical_inventory_hashes_preserved": True,
            "path_level_inventory_diff_preserved": True,
            "post_priming_inventory_verification": True,
            "custom_mapping_getitem_forbidden": True,
            "storage_accounting_consistency_is_diagnostic": True,
            "production_default_may_change": False,
            "eligible_result": "stage_o43_optimization_design_only",
        },
        "commands": {
            "profiles": [
                {"role": role, "argv": profile_command(role)}
                for role in ("legacy", "canary")
            ],
            "analysis": analysis,
        },
        "stop_conditions": [
            "Git HEAD, clean-worktree, or Stage O.4.1 identity gate fails",
            "production reference validation or R320 identity fails",
            "CUDA unavailable or GPU name mismatch",
            "any command exits nonzero",
            "tensor inventory is truncated or incomplete",
            "configuration, initial-Q identity, shape, dtype, or finite gate fails",
        ],
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Print the Stage O.4.2 plan")
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--control-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--production-reference-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--stage-o41-report", type=Path, required=True)
    parser.add_argument("--expected-stage-o41-sha256", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    args = parser.parse_args(argv)
    plan = build_stage_o42_h100_plan(
        project_root=args.project_root,
        control_root=args.control_root,
        python=args.python,
        production_reference_dir=args.production_reference_dir,
        expected_commit=args.expected_commit,
        stage_o41_report=args.stage_o41_report,
        expected_stage_o41_sha256=args.expected_stage_o41_sha256,
        expected_gpu_name=args.expected_gpu_name,
    )
    print(json.dumps(plan, allow_nan=False, indent=2, sort_keys=True))
    return 0


__all__ = ["build_stage_o42_h100_plan", "main"]
