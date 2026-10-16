#!/usr/bin/env python3
"""Plan or execute the analysis-only rc5 G-1 timing instrumentation matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from benchmarks.analyze_rc5_g1_timing_instrumentation import analyze
from benchmarks.plan_rc5_g1_timing_instrumentation import build_plan
from benchmarks.rc458_venv_identity import lexical_python_path


SCHEMA = "pssolver.rc5.g1.timing_instrumentation_command_plan.v1"
PROGRESS_SCHEMA = "pssolver.rc5.g1.timing_instrumentation_progress.v1"
COMPLETE_SCHEMA = "pssolver.rc5.g1.timing_instrumentation_complete.v1"


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
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def build_command_plan(
    *, candidate_python: Path, support_root: Path, output_root: Path
) -> dict[str, object]:
    plan = build_plan()
    script = support_root / "benchmarks" / "diagnose_rc5_g1_timing_instrumentation.py"
    rows = []
    for entry in plan["profile_matrix"]:
        output = output_root / "profiles" / f"{entry['key']}.json"
        cache = output_root / "compiler_cache" / str(entry["key"])
        rows.append(
            {
                "sequence": entry["sequence"],
                "key": entry["key"],
                "python": str(candidate_python),
                "output": str(output),
                "compiler_cache": str(cache),
                "command": [
                    str(candidate_python),
                    str(script),
                    "--family",
                    str(entry["family"]),
                    "--provider-role",
                    "candidate",
                    "--provider-commit",
                    str(entry["provider_commit"]),
                    "--provider-wheel-sha256",
                    str(entry["provider_wheel_sha256"]),
                    "--trial",
                    str(entry["trial"]),
                    "--shape",
                    ",".join(str(value) for value in entry["shape"]),
                    "--lengths",
                    ",".join(str(value) for value in entry["lengths"]),
                    "--device",
                    "cuda",
                    "--dtype",
                    str(entry["dtype"]),
                    "--warmup-steps",
                    str(entry["warmup_steps"]),
                    "--post-flush-rewarm-steps",
                    str(entry["post_flush_rewarm_steps"]),
                    "--profile-steps",
                    str(entry["timed_steps"]),
                    "--output",
                    str(output),
                ],
            }
        )
    return {
        "schema": SCHEMA,
        "analysis_only": True,
        "profile_count": len(rows),
        "candidate_python": str(candidate_python),
        "support_root": str(support_root),
        "output_root": str(output_root),
        "rows": rows,
    }


def preflight(command_plan: dict[str, object]) -> None:
    rows = command_plan.get("rows")
    if not isinstance(rows, list) or len(rows) != 12:
        raise RuntimeError("instrumentation command plan must contain 12 rows")
    if [row["sequence"] for row in rows] != list(range(1, 13)):
        raise RuntimeError("instrumentation command sequence differs")
    if len({row["key"] for row in rows}) != 12:
        raise RuntimeError("instrumentation command keys are not unique")
    python = Path(str(command_plan["candidate_python"]))
    if not python.is_file():
        raise FileNotFoundError("candidate installed-wheel Python is missing")
    support = Path(str(command_plan["support_root"])).resolve()
    if (support / "pssolver").exists() or (support / "pssolver").is_symlink():
        raise RuntimeError("support root must not contain a PSSolver source package")
    script = support / "benchmarks" / "diagnose_rc5_g1_timing_instrumentation.py"
    if not script.is_file() or script.is_symlink():
        raise FileNotFoundError("instrumentation profile support is missing")
    output = Path(str(command_plan["output_root"]))
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"instrumentation output already exists: {output}")
    for row in rows:
        command = row.get("command")
        if not isinstance(command, list) or not command:
            raise TypeError("instrumentation command must be an argument vector")
        if (
            "--device" not in command
            or command[command.index("--device") + 1] != "cuda"
        ):
            raise ValueError("instrumentation command must request cuda")


def execute(command_plan: dict[str, object]) -> dict[str, object]:
    preflight(command_plan)
    output_root = Path(str(command_plan["output_root"])).absolute()
    profiles = output_root / "profiles"
    profiles.mkdir(parents=True, exist_ok=False)
    completed = []
    for row in command_plan["rows"]:
        cache = Path(str(row["compiler_cache"])).absolute()
        triton = cache / "triton"
        inductor = cache / "torchinductor"
        triton.mkdir(parents=True)
        inductor.mkdir()
        environment = dict(os.environ)
        environment.update(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "NVIDIA_TF32_OVERRIDE": "0",
                "TRITON_CACHE_DIR": str(triton),
                "TORCHINDUCTOR_CACHE_DIR": str(inductor),
                "PYTHONPATH": str(
                    Path(str(command_plan["support_root"])).resolve()
                ),
            }
        )
        result = subprocess.run(
            row["command"],
            cwd=output_root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        logs = output_root / "logs"
        logs.mkdir(exist_ok=True)
        (logs / f"{row['key']}.stdout").write_text(
            result.stdout, encoding="utf-8"
        )
        (logs / f"{row['key']}.stderr").write_text(
            result.stderr, encoding="utf-8"
        )
        if result.returncode != 0:
            raise RuntimeError(f"instrumentation profile {row['key']} failed")
        path = Path(str(row["output"]))
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("analysis_only") is not True:
            raise RuntimeError(
                f"instrumentation profile {row['key']} claims formal use"
            )
        completed.append(
            {
                "sequence": row["sequence"],
                "key": row["key"],
                "sha256": _sha256(path),
            }
        )
        _atomic_json(
            output_root / "progress" / f"{int(row['sequence']):03d}.json",
            {"schema": PROGRESS_SCHEMA, "completed": completed},
        )
    analysis = analyze(profiles)
    _atomic_json(output_root / "analysis.json", analysis)
    result = {
        "schema": COMPLETE_SCHEMA,
        "classification": analysis["classification"],
        "analysis_only": True,
        "profile_count": len(completed),
        "instrumentation_evidence_complete": (
            len(completed) == 12
            and analysis["instrumentation_evidence_complete"] is True
        ),
        "formal_requalification_performed": False,
        "completed": completed,
    }
    _atomic_json(output_root / "INSTRUMENTATION_COMPLETE.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--command-plan", type=Path, required=True)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args(argv)
    plan = build_command_plan(
        candidate_python=lexical_python_path(args.candidate_python),
        support_root=args.support_root.resolve(),
        output_root=args.output_root.absolute(),
    )
    _atomic_json(args.command_plan, plan)
    if args.plan_only:
        preflight(plan)
    else:
        execute(plan)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
