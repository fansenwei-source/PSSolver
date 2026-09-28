#!/usr/bin/env python3
"""Build or execute the frozen P9.5 fresh-process H100 profile matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from benchmarks.analyze_phase9_p95_functional_qualification import (
    _atomic_json,
    _load,
    analyze,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_inputs(plan: dict[str, object], input_root: Path) -> dict[str, Path]:
    input_root = input_root.expanduser().resolve()
    manifest_path = input_root / "input_manifest.json"
    manifest = _load(manifest_path)
    if manifest.get("phase") != "P9.5":
        raise ValueError("input manifest is not a P9.5 manifest")
    records = manifest.get("grids")
    if not isinstance(records, list):
        raise ValueError("input manifest grids are absent")
    indexed = {record.get("id"): record for record in records}
    result = {}
    for grid in plan["grids"]:
        record = indexed.get(grid["id"])
        if not isinstance(record, dict):
            raise ValueError(f"input manifest lacks {grid['id']}")
        if record.get("shape") != grid["shape"]:
            raise ValueError(f"input shape differs for {grid['id']}")
        path = input_root / record["path"]
        if not path.is_file() or path.name != "Q_0.npy":
            raise FileNotFoundError(f"canonical input is absent for {grid['id']}")
        if _sha256(path) != record.get("sha256"):
            raise ValueError(f"input SHA-256 differs for {grid['id']}")
        result[grid["id"]] = path
    if set(indexed) != set(result):
        raise ValueError("input manifest grid set differs from the frozen plan")
    return result


def build_commands(
    plan: dict[str, object],
    *,
    initial_paths: dict[str, Path],
    profile_directory: Path,
    python_executable: str,
) -> list[dict[str, object]]:
    profile_directory = profile_directory.expanduser().resolve()
    roles = tuple(plan["roles"])
    commands = []
    for grid in plan["grids"]:
        for trial in range(1, plan["trials"] + 1):
            offset = (trial - 1) % len(roles)
            balanced_roles = roles[offset:] + roles[:offset]
            for role in balanced_roles:
                output = profile_directory / f"{grid['id']}_{role}_trial{trial}.json"
                command = [
                    python_executable,
                    "-m",
                    "benchmarks.profile_periodic_functional",
                    "--role",
                    role,
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
                    "--dtype",
                    "float64",
                    "--dt",
                    str(plan["dt"]),
                    "--base-activity",
                    str(plan["base_activity"]),
                    "--warmup-steps",
                    str(plan["warmup_steps"]),
                    "--profile-steps",
                    str(plan["profile_steps"]),
                    "--output",
                    str(output),
                ]
                commands.append(
                    {
                        "grid": grid["id"],
                        "role": role,
                        "trial": trial,
                        "output": str(output),
                        "argv": command,
                    }
                )
    if len(commands) != plan["profile_count"]:
        raise ValueError("generated command count differs from frozen plan")
    if len({item["output"] for item in commands}) != len(commands):
        raise ValueError("generated profile output paths are not unique")
    return commands


def execute_matrix(
    commands: list[dict[str, object]],
    *,
    profile_directory: Path,
    plan: dict[str, object],
    expected_commit: str,
    result_path: Path,
) -> dict[str, object]:
    profile_directory = profile_directory.expanduser().resolve()
    if profile_directory.exists() and any(profile_directory.iterdir()):
        raise FileExistsError("refusing to reuse a non-empty profile directory")
    profile_directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    for record in commands:
        subprocess.run(record["argv"], check=True, env=environment)
    reports = [
        _load(path) for path in sorted(profile_directory.glob("*.json"))
    ]
    result = analyze(plan, reports, expected_commit=expected_commit)
    _atomic_json(result_path, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-single-h100-job", action="store_true")
    args = parser.parse_args(argv)
    plan = _load(args.plan)
    paths = validate_inputs(plan, args.inputs)
    commands = build_commands(
        plan,
        initial_paths=paths,
        profile_directory=args.profiles,
        python_executable=sys.executable,
    )
    command_plan = {
        "schema_version": 1,
        "phase": "P9.5",
        "kind": "p95_fresh_process_command_plan",
        "expected_commit": args.expected_commit,
        "commands": commands,
    }
    _atomic_json(args.command_plan, command_plan)
    if not args.execute:
        print(json.dumps(command_plan, indent=2, sort_keys=True))
        return 0
    if not args.confirm_single_h100_job:
        raise ValueError("execution requires --confirm-single-h100-job")
    execute_matrix(
        commands,
        profile_directory=args.profiles,
        plan=plan,
        expected_commit=args.expected_commit,
        result_path=args.result,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
