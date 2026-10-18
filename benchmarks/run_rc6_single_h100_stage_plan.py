#!/usr/bin/env python3
"""Freeze or execute the shell-free RC6 single-H100 stage plan."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from benchmarks.analyze_rc6_single_h100 import PASS_CLASSIFICATION
from benchmarks.plan_rc6_single_h100_qualification import (
    CONSUMER_SOURCE,
    PROVIDER_BASELINE,
    PROVIDER_CANDIDATE,
)


DRAFT_SCHEMA = "pssolver.rc6.h100_stage_plan_draft.v1"
SCHEMA = "pssolver.rc6.h100_stage_plan.v1"
COMPLETE_SCHEMA = "pssolver.rc6.h100_stage_plan_complete.v1"
STAGES = (
    "environment",
    "cuda_only",
    "contract_smokes",
    "tf32_scope",
    "current_checkpoints",
    "released_checkpoints",
    "numerical",
    "profiles",
    "consumer",
    "assemble",
    "analyze",
)
EXPECTED_SCHEMAS = {
    "environment": "pssolver.rc6.environment.v1",
    "cuda_only": "pssolver.rc6.cuda_only.v1",
    "contract_smokes": "pssolver.rc6.contract_smokes.v1",
    "tf32_scope": "pssolver.rc4_5_8.tf32_scope_h100.v1",
    "current_checkpoints": "pssolver.rc6.current_checkpoint_h100.v1",
    "released_checkpoints": "pssolver.rc6.released_checkpoint_h100.v1",
    "numerical": "pssolver.rc6.numerical_h100.v1",
    "profiles": "pssolver.rc6.profile_matrix_complete.v1",
    "consumer": "pssolver.rc6.consumer_h100.v1",
    "assemble": "pssolver.rc6.h100_auxiliary.v1",
    "analyze": "pssolver.rc6.single_h100_analysis.v1",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().absolute()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    expected = json.loads(payload)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("stage-plan JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _load(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"regular stage-plan JSON required: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError("stage plan must be a JSON object")
    return value


def freeze_plan(draft: dict[str, object], output_root: Path) -> dict[str, object]:
    if draft.get("schema") != DRAFT_SCHEMA:
        raise ValueError("stage-plan draft schema differs")
    stages = draft.get("stages")
    if not isinstance(stages, list):
        raise TypeError("stage-plan draft stages must be a list")
    frozen: list[dict[str, object]] = []
    for row in stages:
        if not isinstance(row, dict):
            raise TypeError("stage-plan row must be an object")
        command = row.get("command")
        if not isinstance(command, list) or not command or not all(
            isinstance(value, str) and value for value in command
        ):
            raise TypeError("stage command must be a nonempty argv list")
        if "-c" in command[1:]:
            raise ValueError("inline Python command strings are forbidden")
        executable = Path(command[0])
        scripts = [Path(value) for value in command[1:] if Path(value).is_absolute() and Path(value).suffix == ".py"]
        if not executable.is_absolute() or len(scripts) != 1:
            raise ValueError("each stage must name one absolute executable and one absolute Python script")
        files = (("python_executable", executable), ("python_script", scripts[0]))
        bindings: list[dict[str, str]] = []
        for kind, path in files:
            if not path.is_file():
                raise FileNotFoundError(f"command file missing: {path}")
            if kind == "python_script" and path.is_symlink():
                raise FileNotFoundError(f"Python helper must be regular and non-symlink: {path}")
            bindings.append(
                {
                    "kind": kind,
                    "path": str(path),
                    "resolved_path": str(path.resolve()),
                    "sha256": _sha256(path),
                }
            )
        item = dict(row)
        item["command_file_bindings"] = bindings
        frozen.append(item)
    result = {**draft, "schema": SCHEMA, "stages": frozen}
    validate_plan(result, output_root)
    return result


def validate_plan(plan: dict[str, object], output_root: Path) -> list[dict[str, object]]:
    output_root = output_root.expanduser().resolve()
    if plan.get("schema") != SCHEMA:
        raise ValueError("stage plan schema differs")
    expected_identities = {
        "provider_baseline_commit": PROVIDER_BASELINE,
        "provider_candidate_commit": PROVIDER_CANDIDATE,
        "consumer_source_commit": CONSUMER_SOURCE,
    }
    for name, expected in expected_identities.items():
        if plan.get(name) != expected:
            raise ValueError(f"{name} differs")
    stages = plan.get("stages")
    if not isinstance(stages, list) or [row.get("name") for row in stages if isinstance(row, dict)] != list(STAGES):
        raise ValueError("stage order differs")
    for row in stages:
        if not isinstance(row, dict):
            raise TypeError("stage must be an object")
        name = str(row["name"])
        command = row.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(value, str) and value for value in command):
            raise TypeError(f"{name} command must be a nonempty argv list")
        if not Path(command[0]).is_absolute():
            raise ValueError(f"{name} executable must be absolute")
        if "-c" in command[1:]:
            raise ValueError(f"{name} may not use an inline command string")
        if Path(command[0]).name in {"bash", "sh", "dash", "zsh"}:
            raise ValueError(f"{name} may not use a shell executable")
        bindings = row.get("command_file_bindings")
        if not isinstance(bindings, list) or len(bindings) != 2:
            raise TypeError(f"{name} must have two command-file bindings")
        bound: dict[str, dict[str, object]] = {}
        for binding in bindings:
            if not isinstance(binding, dict):
                raise TypeError(f"{name} binding must be an object")
            path = Path(str(binding.get("path", "")))
            kind = binding.get("kind")
            if not path.is_absolute() or kind not in {"python_executable", "python_script"}:
                raise ValueError(f"{name} binding identity differs")
            if not path.is_file() or (kind == "python_script" and path.is_symlink()):
                raise FileNotFoundError(f"{name} bound command file missing: {path}")
            if binding.get("resolved_path") != str(path.resolve()) or binding.get("sha256") != _sha256(path):
                raise ValueError(f"{name} bound command file SHA-256 differs: {path}")
            bound[str(path)] = binding
        scripts = [value for value in command[1:] if Path(value).is_absolute() and Path(value).suffix == ".py"]
        if command[0] not in bound or bound[command[0]]["kind"] != "python_executable":
            raise ValueError(f"{name} executable is not correctly hash-bound")
        if len(scripts) != 1 or scripts[0] not in bound or bound[scripts[0]]["kind"] != "python_script":
            raise ValueError(f"{name} script is not correctly hash-bound")
        environment = row.get("environment", {})
        if not isinstance(environment, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in environment.items()):
            raise TypeError(f"{name} environment must contain string pairs")
        if {"PYTHONPATH", "PYTHONHOME", "LD_LIBRARY_PATH"} & set(environment):
            raise ValueError(f"{name} may not override Python or loader paths")
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
    output_root = output_root.expanduser().absolute()
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
        environment.update(row.get("environment", {}))
        result = subprocess.run(
            row["command"], cwd=output_root, env=environment, check=False,
            capture_output=True, text=True,
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
        if name != "analyze" and payload.get("passed") is not True:
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
            {"schema": "pssolver.rc6.h100_stage_progress.v1", "completed": completed},
        )
    result = {
        "schema": COMPLETE_SCHEMA,
        "provider_baseline_commit": PROVIDER_BASELINE,
        "provider_candidate_commit": PROVIDER_CANDIDATE,
        "consumer_source_commit": CONSUMER_SOURCE,
        "completed": completed,
        "passed": len(completed) == len(STAGES),
    }
    _atomic_json(output_root / "STAGE_PLAN_COMPLETE.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--freeze-draft", type=Path)
    modes.add_argument("--plan", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.freeze_draft is not None:
        if args.output is None:
            parser.error("--output is required with --freeze-draft")
        _atomic_json(args.output, freeze_plan(_load(args.freeze_draft), args.output_root))
        return 0
    execute(_load(args.plan), args.output_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
