#!/usr/bin/env python3
"""Plan or execute the frozen rc5 G-1 96-profile matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from benchmarks.plan_rc5_g1_h100 import build_plan
from benchmarks.rc458_venv_identity import lexical_python_path


SCHEMA = "pssolver.rc5.g1.profile_command_plan.v1"
COMPLETE_SCHEMA = "pssolver.rc5.g1.profile_matrix_complete.v1"


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
            raise RuntimeError("matrix JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def build_command_plan(
    *,
    baseline_python: Path,
    candidate_python: Path,
    support_root: Path,
    output_root: Path,
) -> dict[str, object]:
    plan = build_plan()
    script = support_root / "benchmarks" / "profile_rc5_g1_runtime_family.py"
    rows: list[dict[str, object]] = []
    for entry in plan["profile_matrix"]:
        role = str(entry["provider_role"])
        python = baseline_python if role == "baseline" else candidate_python
        grid = plan["grid_contracts"][entry["grid"]]
        output = output_root / "profiles" / f"{entry['key']}.json"
        cache = output_root / "compiler_cache" / str(entry["key"])
        command = [
            str(python),
            str(script),
            "--family",
            str(entry["family"]),
            "--provider-role",
            role,
            "--provider-commit",
            str(entry["provider_commit"]),
            "--provider-wheel-sha256",
            str(entry["provider_wheel_sha256"]),
            "--trial",
            str(entry["trial"]),
            "--shape",
            ",".join(str(value) for value in grid["shape"]),
            "--lengths",
            ",".join(str(value) for value in grid["lengths"]),
            "--device",
            "cuda",
            "--dtype",
            str(entry["dtype"]),
            "--warmup-steps",
            str(entry["warmup_steps"]),
            "--profile-steps",
            str(entry["timed_steps"]),
            "--output",
            str(output),
        ]
        rows.append(
            {
                "sequence": entry["sequence"],
                "key": entry["key"],
                "provider_role": role,
                "python": str(python),
                "provider_wheel_sha256": entry["provider_wheel_sha256"],
                "support_script": str(script),
                "output": str(output),
                "compiler_cache": str(cache),
                "command": command,
            }
        )
    return {
        "schema": SCHEMA,
        "profile_count": len(rows),
        "baseline_python": str(baseline_python),
        "candidate_python": str(candidate_python),
        "support_root": str(support_root),
        "output_root": str(output_root),
        "rows": rows,
    }


def _preflight(command_plan: dict[str, object]) -> None:
    rows = command_plan["rows"]
    if not isinstance(rows, list) or len(rows) != 96:
        raise RuntimeError("command plan must contain 96 rows")
    if [row["sequence"] for row in rows] != list(range(1, 97)):
        raise RuntimeError("command plan sequence differs")
    if len({row["key"] for row in rows}) != 96:
        raise RuntimeError("command plan keys are not unique")
    for name in ("baseline_python", "candidate_python"):
        if not Path(str(command_plan[name])).is_file():
            raise FileNotFoundError(f"Python executable required: {command_plan[name]}")
    support = Path(str(command_plan["support_root"])).resolve()
    if (support / "pssolver").exists() or (support / "pssolver").is_symlink():
        raise RuntimeError("support root must not contain a PSSolver source tree")
    script = support / "benchmarks" / "profile_rc5_g1_runtime_family.py"
    if not script.is_file() or script.is_symlink():
        raise FileNotFoundError(f"profile support script missing: {script}")
    for row in rows:
        _validate_sha256(str(row["provider_wheel_sha256"]), "wheel identity")
        for name in ("output", "compiler_cache"):
            path = Path(str(row[name]))
            if path.exists() or path.is_symlink():
                raise FileExistsError(f"{name} already exists: {path}")


def execute(command_plan: dict[str, object]) -> dict[str, object]:
    _preflight(command_plan)
    output_root = Path(str(command_plan["output_root"])).absolute()
    (output_root / "profiles").mkdir(parents=True, exist_ok=False)
    completed: list[dict[str, object]] = []
    for row in command_plan["rows"]:
        cache = Path(str(row["compiler_cache"])).absolute()
        triton = cache / "triton"
        inductor = cache / "torchinductor"
        triton.mkdir(parents=True)
        inductor.mkdir()
        env = dict(os.environ)
        env.update(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "NVIDIA_TF32_OVERRIDE": "0",
                "TRITON_CACHE_DIR": str(triton),
                "TORCHINDUCTOR_CACHE_DIR": str(inductor),
                "PYTHONPATH": str(Path(str(command_plan["support_root"])).resolve()),
            }
        )
        result = subprocess.run(
            row["command"],
            cwd=output_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        logs = output_root / "logs"
        logs.mkdir(exist_ok=True)
        (logs / f"{row['key']}.stdout").write_text(result.stdout, encoding="utf-8")
        (logs / f"{row['key']}.stderr").write_text(result.stderr, encoding="utf-8")
        if result.returncode != 0:
            raise RuntimeError(f"profile {row['key']} failed with {result.returncode}")
        path = Path(str(row["output"]))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("config", {}).get("family") not in str(row["key"]):
            raise RuntimeError(f"profile identity differs: {row['key']}")
        completed.append(
            {"sequence": row["sequence"], "key": row["key"], "sha256": _sha256(path)}
        )
        _atomic_json(
            output_root / "progress" / f"{int(row['sequence']):03d}.json",
            {"schema": "pssolver.rc5.g1.profile_progress.v1", "completed": completed},
        )
    result = {
        "schema": COMPLETE_SCHEMA,
        "profile_count": len(completed),
        "completed": completed,
        "passed": len(completed) == 96,
    }
    _atomic_json(output_root / "MATRIX_COMPLETE.json", result)
    return result


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
    if not args.plan_only:
        execute(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
