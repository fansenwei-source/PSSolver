#!/usr/bin/env python3
"""Plan or execute the frozen RC4.5.8 96-cell profile matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from benchmarks.plan_rc458_tf32_h100 import build_plan


SCHEMA = "pssolver.rc4_5_8.profile_command_plan.v1"
COMPLETE_SCHEMA = "pssolver.rc4_5_8.profile_matrix_complete.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
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


def build_command_plan(
    *,
    baseline_python: Path,
    candidate_python: Path,
    baseline_wheel_sha256: str,
    candidate_wheel_sha256: str,
    support_root: Path,
    output_root: Path,
) -> dict[str, object]:
    plan = build_plan()
    script = support_root / "benchmarks" / "profile_rc458_runtime_family.py"
    rows = []
    for entry in plan["profile_matrix"]:
        role = str(entry["provider_role"])
        python = baseline_python if role == "baseline" else candidate_python
        wheel = baseline_wheel_sha256 if role == "baseline" else candidate_wheel_sha256
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
            wheel,
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
                "provider_wheel_sha256": wheel,
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


def _validate_sha256(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _preflight(command_plan: dict[str, object]) -> None:
    rows = command_plan["rows"]
    if not isinstance(rows, list) or len(rows) != 96:
        raise RuntimeError("command plan must contain 96 rows")
    if [row["sequence"] for row in rows] != list(range(1, 97)):
        raise RuntimeError("command plan sequence differs")
    if len({row["key"] for row in rows}) != 96:
        raise RuntimeError("command plan keys are not unique")
    for path_name in ("baseline_python", "candidate_python"):
        path = Path(str(command_plan[path_name])).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Python executable required: {path}")
    support = Path(str(command_plan["support_root"])).resolve()
    if (support / "pssolver").exists() or (support / "pssolver").is_symlink():
        raise RuntimeError("support root must not contain a PSSolver source tree")
    script = support / "benchmarks" / "profile_rc458_runtime_family.py"
    if not script.is_file() or script.is_symlink():
        raise FileNotFoundError(f"profile support script missing: {script}")
    for row in rows:
        _validate_sha256(str(row["provider_wheel_sha256"]), "wheel identity")
        output = Path(str(row["output"]))
        if output.exists() or output.is_symlink():
            raise FileExistsError(f"profile output already exists: {output}")
        cache = Path(str(row["compiler_cache"]))
        if cache.exists() or cache.is_symlink():
            raise FileExistsError(f"compiler cache already exists: {cache}")


def execute(command_plan: dict[str, object]) -> dict[str, object]:
    _preflight(command_plan)
    output_root = Path(str(command_plan["output_root"])).resolve()
    profiles = output_root / "profiles"
    profiles.mkdir(parents=True, exist_ok=False)
    completed = []
    for row in command_plan["rows"]:
        cache = Path(str(row["compiler_cache"])).resolve()
        triton = cache / "triton"
        inductor = cache / "torchinductor"
        triton.mkdir(parents=True)
        inductor.mkdir(parents=True)
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
        completed_process = subprocess.run(
            row["command"],
            cwd=output_root,
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )
        logs = output_root / "logs"
        logs.mkdir(exist_ok=True)
        (logs / f"{row['key']}.stdout").write_text(
            completed_process.stdout, encoding="utf-8"
        )
        (logs / f"{row['key']}.stderr").write_text(
            completed_process.stderr, encoding="utf-8"
        )
        if completed_process.returncode != 0:
            raise RuntimeError(
                f"profile {row['key']} failed with {completed_process.returncode}"
            )
        path = Path(str(row["output"]))
        value = json.loads(path.read_text(encoding="utf-8"))
        if value.get("config", {}).get("family") not in str(row["key"]):
            raise RuntimeError(f"profile identity differs: {row['key']}")
        completed.append(
            {
                "sequence": row["sequence"],
                "key": row["key"],
                "sha256": _sha256(path),
            }
        )
        _atomic_json(
            output_root / "progress" / f"{int(row['sequence']):03d}.json",
            {
                "schema": "pssolver.rc4_5_8.profile_progress.v1",
                "completed": completed,
            },
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
        baseline_python=args.baseline_python.resolve(),
        candidate_python=args.candidate_python.resolve(),
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
