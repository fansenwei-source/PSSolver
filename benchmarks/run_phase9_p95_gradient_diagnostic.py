#!/usr/bin/env python3
"""Run the two-grid P9.5 gradient diagnostic in fresh subprocesses."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmarks.analyze_phase9_p95_gradient_diagnostic import (
    _atomic_json,
    _load,
    analyze,
)
from benchmarks.diagnose_phase9_p95_gradient_epsilon import (
    ACTIVITY_EPSILONS,
    FROZEN_RELATIVE_TOLERANCE,
)
from benchmarks.run_phase9_p95_h100_qualification import validate_inputs


def validate_plan(plan: dict[str, object]) -> int:
    if plan.get("phase") != "P9.5-gradient-diagnostic":
        raise ValueError("plan is not the P9.5 gradient diagnostic plan")
    execution = plan.get("execution")
    if not isinstance(execution, dict):
        raise ValueError("diagnostic execution contract is absent")
    trials = execution.get("trials_per_grid")
    if trials != 3:
        raise ValueError("diagnostic plan must use three trials per grid")
    if execution.get("diagnostic_process_count") != 6:
        raise ValueError("diagnostic plan must require six fresh processes")
    if execution.get("analysis_only") is not True:
        raise ValueError("diagnostic plan must remain analysis-only")
    if plan.get("activity_epsilons") != list(ACTIVITY_EPSILONS):
        raise ValueError("diagnostic epsilon sweep differs")
    if plan.get("frozen_relative_tolerance") != FROZEN_RELATIVE_TOLERANCE:
        raise ValueError("diagnostic relative tolerance differs")
    if execution.get("dtype") != "float64" or execution.get("dt") != 0.001:
        raise ValueError("diagnostic precision or timestep differs")
    if execution.get("base_activity") != 0.013 or execution.get("tf32") is not False:
        raise ValueError("diagnostic activity or TF32 contract differs")
    if execution.get("fresh_process_per_trial") is not True:
        raise ValueError("each diagnostic trial must use a fresh process")
    if execution.get("submission_count_max") != 1:
        raise ValueError("diagnostic plan must allow exactly one H100 submission")
    if execution.get("automatic_retry") is not False:
        raise ValueError("diagnostic plan must prohibit automatic retry")
    grids = plan.get("grids")
    if not isinstance(grids, list) or any(
        not isinstance(grid, dict) for grid in grids
    ):
        raise ValueError("diagnostic grid contract is absent")
    if [grid.get("id") for grid in grids] != [
        "R128",
        "R320",
    ]:
        raise ValueError("diagnostic grid contract differs")
    expected_grids = {
        "R128": ([128, 128, 32], [100.0, 100.0, 20.0]),
        "R320": ([320, 320, 80], [100.0, 100.0, 20.0]),
    }
    for grid in grids:
        expected_shape, expected_lengths = expected_grids[grid["id"]]
        if grid.get("shape") != expected_shape:
            raise ValueError(f"{grid['id']} diagnostic shape differs")
        if grid.get("lengths") != expected_lengths:
            raise ValueError(f"{grid['id']} diagnostic lengths differ")
    return trials


def build_commands(
    plan: dict[str, object],
    *,
    initial_paths: dict[str, Path],
    output_directory: Path,
    python_executable: str,
) -> list[dict[str, object]]:
    trials = validate_plan(plan)
    commands = []
    for grid in plan["grids"]:
        for trial in range(1, trials + 1):
            output = output_directory / (
                f"{grid['id']}_gradient_diagnostic_trial{trial}.json"
            )
            argv = [
                python_executable,
                "-m",
                "benchmarks.diagnose_phase9_p95_gradient_epsilon",
                "--grid-id",
                grid["id"],
                "--trial",
                str(trial),
                "--shape",
                *(str(value) for value in grid["shape"]),
                "--lengths",
                *(str(value) for value in grid["lengths"]),
                "--initial-q-path",
                str(initial_paths[grid["id"]]),
                "--device",
                "cuda",
                "--output",
                str(output),
            ]
            commands.append(
                {
                    "grid": grid["id"],
                    "trial": trial,
                    "output": str(output),
                    "argv": argv,
                }
            )
    if len(commands) != 6 or len({item["output"] for item in commands}) != 6:
        raise ValueError("diagnostic command plan must contain six unique outputs")
    return commands


def execute(
    commands: list[dict[str, object]],
    *,
    output_directory: Path,
    expected_commit: str,
    result_path: Path,
) -> dict[str, object]:
    output_directory = output_directory.expanduser().resolve()
    if output_directory.exists() and any(output_directory.iterdir()):
        raise FileExistsError("refusing to reuse a non-empty diagnostic directory")
    output_directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for record in commands:
        subprocess.run(record["argv"], check=True, env=environment)
    reports = [_load(path) for path in sorted(output_directory.glob("*.json"))]
    result = analyze(reports, expected_commit=expected_commit)
    _atomic_json(result_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-analysis-only-h100", action="store_true")
    args = parser.parse_args(argv)
    plan = _load(args.plan)
    validate_plan(plan)
    paths = validate_inputs(plan, args.inputs)
    commands = build_commands(
        plan,
        initial_paths=paths,
        output_directory=args.diagnostics,
        python_executable=sys.executable,
    )
    command_plan = {
        "schema_version": 1,
        "phase": "P9.5-gradient-diagnostic",
        "kind": "p95_gradient_diagnostic_command_plan",
        "expected_commit": args.expected_commit,
        "commands": commands,
    }
    _atomic_json(args.command_plan, command_plan)
    if not args.execute:
        print(json.dumps(command_plan, indent=2, sort_keys=True))
        return 0
    if not args.confirm_analysis_only_h100:
        raise ValueError("execution requires --confirm-analysis-only-h100")
    execute(
        commands,
        output_directory=args.diagnostics,
        expected_commit=args.expected_commit,
        result_path=args.result,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
