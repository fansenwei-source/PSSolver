#!/usr/bin/env python3
"""Execute a fail-closed, shell-free rc5 G-1 H100 stage plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from benchmarks.analyze_rc5_g1_h100 import PASS_CLASSIFICATION
from benchmarks.plan_rc5_g1_h100 import (
    CONSUMER_SOURCE,
    PROVIDER_BASELINE,
    PROVIDER_CANDIDATE,
)


SCHEMA = "pssolver.rc5.g1.h100_stage_plan.v1"
COMPLETE_SCHEMA = "pssolver.rc5.g1.h100_stage_plan_complete.v1"
STAGES = (
    "environment",
    "cuda_only",
    "tf32_scope",
    "checkpoints",
    "numerical",
    "profiles",
    "consumer",
    "assemble",
    "analyze",
)
EXPECTED_SCHEMAS = {
    "environment": "pssolver.rc5.g1.environment.v1",
    "cuda_only": "pssolver.rc5.g1.cuda_only.v1",
    "tf32_scope": "pssolver.rc4_5_8.tf32_scope_h100.v1",
    "checkpoints": "pssolver.rc5.g1.checkpoint_h100.v1",
    "numerical": "pssolver.rc5.g1.numerical_h100.v1",
    "profiles": "pssolver.rc5.g1.profile_matrix_complete.v1",
    "consumer": "pssolver.rc5.g1.consumer_h100.v1",
    "assemble": "pssolver.rc5.g1.h100_auxiliary.v1",
    "analyze": "pssolver.rc5.g1.single_h100_analysis.v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    expected = json.loads(payload)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("stage JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _load_plan(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"regular stage plan required: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("stage plan must be a JSON object")
    return value


def validate_plan(plan: dict[str, object], output_root: Path) -> list[dict[str, object]]:
    output_root = output_root.expanduser().resolve()
    if plan.get("schema") != SCHEMA:
        raise ValueError("stage plan schema differs")
    if plan.get("provider_baseline_commit") != PROVIDER_BASELINE:
        raise ValueError("baseline commit differs")
    if plan.get("provider_candidate_commit") != PROVIDER_CANDIDATE:
        raise ValueError("candidate commit differs")
    if plan.get("consumer_source_commit") != CONSUMER_SOURCE:
        raise ValueError("Consumer source differs")
    stages = plan.get("stages")
    if not isinstance(stages, list) or [row.get("name") for row in stages if isinstance(row, dict)] != list(STAGES):
        raise ValueError("stage order differs")
    for row in stages:
        if not isinstance(row, dict):
            raise TypeError("stage must be an object")
        name = str(row["name"])
        command = row.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(value, str) and value for value in command
        ):
            raise TypeError(f"{name} command must be a nonempty argv list")
        if not Path(command[0]).is_absolute():
            raise ValueError(f"{name} executable must be absolute")
        if Path(command[0]).name in {"bash", "sh", "dash", "zsh"} and len(command) > 1 and command[1] == "-c":
            raise ValueError(f"{name} may not use a shell command string")
        artifact = Path(str(row.get("artifact", ""))).expanduser().resolve()
        try:
            artifact.relative_to(output_root)
        except ValueError as exc:
            raise ValueError(f"{name} artifact escapes output root") from exc
        if artifact.exists() or artifact.is_symlink():
            raise FileExistsError(f"{name} artifact already exists: {artifact}")
        if row.get("expected_schema") != EXPECTED_SCHEMAS[name]:
            raise ValueError(f"{name} expected schema differs")
    return stages


def execute(plan: dict[str, object], output_root: Path) -> dict[str, object]:
    output_root = output_root.absolute()
    stages = validate_plan(plan, output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    logs = output_root / "logs"
    logs.mkdir()
    completed: list[dict[str, object]] = []
    for sequence, row in enumerate(stages, start=1):
        name = str(row["name"])
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "NVIDIA_TF32_OVERRIDE": "0",
            }
        )
        overrides = row.get("environment", {})
        if not isinstance(overrides, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in overrides.items()
        ):
            raise TypeError(f"{name} environment must contain string pairs")
        if {"PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"} & set(overrides):
            raise ValueError(f"{name} may not override Python or loader paths")
        environment.update(overrides)
        result = subprocess.run(
            row["command"],
            cwd=output_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        (logs / f"{sequence:02d}_{name}.stdout").write_text(result.stdout, encoding="utf-8")
        (logs / f"{sequence:02d}_{name}.stderr").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"stage {name} failed with {result.returncode}")
        artifact = Path(str(row["artifact"])).absolute()
        if not artifact.is_file() or artifact.is_symlink():
            raise FileNotFoundError(f"stage {name} did not create its regular artifact")
        payload = json.loads(artifact.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != row["expected_schema"]:
            raise ValueError(f"stage {name} artifact schema differs")
        if name in {"environment", "cuda_only", "tf32_scope", "checkpoints", "numerical", "profiles", "consumer"} and payload.get("passed") is not True:
            raise ValueError(f"stage {name} did not pass")
        if name == "analyze" and (
            payload.get("classification") != PASS_CLASSIFICATION
            or payload.get("qualification_complete") is not True
        ):
            raise ValueError("final analyzer did not pass")
        completed.append(
            {"sequence": sequence, "name": name, "artifact": str(artifact), "sha256": _sha256(artifact)}
        )
        _atomic_json(
            output_root / "progress" / f"{sequence:02d}_{name}.json",
            {"schema": "pssolver.rc5.g1.h100_stage_progress.v1", "completed": completed},
        )
    result = {
        "schema": COMPLETE_SCHEMA,
        "passed": True,
        "stage_count": len(completed),
        "completed": completed,
        "classification": PASS_CLASSIFICATION,
        "qualification_complete": True,
    }
    _atomic_json(output_root / "STAGES_COMPLETE.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    plan = _load_plan(args.plan.absolute())
    if args.preflight_only:
        validate_plan(plan, args.output_root.absolute())
    else:
        execute(plan, args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
