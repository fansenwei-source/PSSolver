#!/usr/bin/env python3
"""Plan or execute the rc5 G-1 corrected 48-profile measurement matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.analyze_rc5_g1_measurement_window_h100 import analyze
from benchmarks.plan_rc5_g1_measurement_window_recovery import build_plan
from benchmarks.rc458_venv_identity import lexical_python_path
from benchmarks.run_rc5_g1_focused_profile_matrix import (
    _atomic_json,
    _preflight,
    build_command_plan as _build_command_plan,
    execute as _execute,
)


SCHEMA = "pssolver.rc5.g1.measurement_window_command_plan.v1"
PROGRESS_SCHEMA = "pssolver.rc5.g1.measurement_window_progress.v1"
COMPLETE_SCHEMA = "pssolver.rc5.g1.measurement_window_matrix_complete.v1"
FAILURE_SCHEMA = "pssolver.rc5.g1.measurement_window_analysis_failure.v1"
FAILURE_CLASSIFICATION = "FAIL_RC5_G1_CORRECTED_MEASUREMENT_WINDOW_GATE"


def build_command_plan(
    *,
    baseline_python: Path,
    candidate_python: Path,
    support_root: Path,
    output_root: Path,
) -> dict[str, object]:
    return _build_command_plan(
        baseline_python=baseline_python,
        candidate_python=candidate_python,
        support_root=support_root,
        output_root=output_root,
        plan=build_plan(),
        command_schema=SCHEMA,
    )


def execute(command_plan: dict[str, object]) -> dict[str, object]:
    return _execute(
        command_plan,
        analyzer=analyze,
        progress_schema=PROGRESS_SCHEMA,
        failure_schema=FAILURE_SCHEMA,
        failure_classification=FAILURE_CLASSIFICATION,
        complete_schema=COMPLETE_SCHEMA,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    plan = build_command_plan(
        baseline_python=lexical_python_path(args.baseline_python),
        candidate_python=lexical_python_path(args.candidate_python),
        support_root=args.support_root.resolve(),
        output_root=args.output_root.absolute(),
    )
    _atomic_json(args.command_plan, plan)
    if args.plan_only:
        _preflight(plan)
    else:
        execute(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
