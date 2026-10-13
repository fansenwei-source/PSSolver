#!/usr/bin/env python3
"""Run the RC4.5.8 matrix while preserving A/B venv entry points."""

from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.rc458_venv_identity import lexical_python_path
from benchmarks.run_rc458_profile_matrix import (
    _atomic_json,
    _validate_sha256,
    build_command_plan,
    execute,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--baseline-wheel-sha256", required=True)
    parser.add_argument("--candidate-wheel-sha256", required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    _validate_sha256(args.baseline_wheel_sha256, "baseline wheel identity")
    _validate_sha256(args.candidate_wheel_sha256, "candidate wheel identity")
    plan = build_command_plan(
        baseline_python=lexical_python_path(args.baseline_python),
        candidate_python=lexical_python_path(args.candidate_python),
        baseline_wheel_sha256=args.baseline_wheel_sha256,
        candidate_wheel_sha256=args.candidate_wheel_sha256,
        support_root=args.support_root.resolve(),
        output_root=args.output_root.resolve(),
    )
    _atomic_json(args.command_plan, plan)
    if not args.plan_only:
        execute(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
