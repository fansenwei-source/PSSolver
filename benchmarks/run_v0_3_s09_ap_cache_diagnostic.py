#!/usr/bin/env python3
"""Execute the 72-process S-09 A-p cache diagnostic matrix."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Mapping


SCHEMA = "pssolver.v0_3.s09.ap_cache_diagnostic_matrix.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_replace_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(payload):
            raise RuntimeError("matrix progress JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _triplet(values: object) -> str:
    if not isinstance(values, list) or len(values) != 3:
        raise TypeError("shape and lengths must be three-item lists")
    return ",".join(str(value) for value in values)


def _isolated_import(python: Path) -> dict[str, object]:
    code = (
        "import json,pathlib,sys,pssolver;"
        "p=pathlib.Path(pssolver.__file__).resolve();"
        "pure=pathlib.Path(sys.prefix)/'lib'/"
        "('python%d.%d'%sys.version_info[:2])/'site-packages';"
        "print(json.dumps({'python':sys.executable,'prefix':sys.prefix,"
        "'pssolver':str(p),'version':pssolver.__version__,"
        "'inside_venv':p.is_relative_to(pure.resolve()),'sys_path':sys.path}))"
    )
    completed = subprocess.run(
        [str(python), "-I", "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    value = json.loads(completed.stdout)
    if value["inside_venv"] is not True:
        raise RuntimeError(f"installed-wheel identity failed for {python}: {value!r}")
    return value


def _command(
    row: Mapping[str, object],
    *,
    python: Path,
    child: Path,
    profile: Path,
    output: Path,
    tf32: Path,
) -> list[str]:
    return [
        str(python),
        str(child),
        "--tf32-bootstrap-output",
        str(tf32),
        "--profile-script",
        str(profile),
        "--family",
        str(row["family"]),
        "--diagnostic-role",
        str(row["diagnostic_role"]),
        "--provider-role",
        str(row["provider_role"]),
        "--provider-commit",
        str(row["provider_commit"]),
        "--provider-wheel-sha256",
        str(row["provider_wheel_sha256"]),
        "--trial",
        str(row["trial"]),
        "--shape",
        _triplet(row["shape"]),
        "--lengths",
        _triplet(row["lengths"]),
        "--execution-device",
        str(row["device"]),
        "--dtype",
        str(row["dtype"]),
        "--warmup-steps",
        str(row["warmup_steps"]),
        "--profile-steps",
        str(row["timed_steps"]),
        "--output",
        str(output),
    ]


def execute(
    *,
    plan_path: Path,
    output_root: Path,
    baseline_python: Path,
    candidate_python: Path,
    support_root: Path,
) -> dict[str, object]:
    plan_path = plan_path.resolve()
    output_root = output_root.absolute()
    support_root = support_root.resolve()
    if output_root.exists() or output_root.is_symlink():
        raise FileExistsError(f"diagnostic output root must not exist: {output_root}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    rows = plan.get("profiles")
    if not isinstance(rows, list) or len(rows) != 72:
        raise ValueError("diagnostic plan must contain exactly 72 profiles")
    child = support_root / "benchmarks/run_v0_3_s09_ap_cache_diagnostic_child.py"
    profile = support_root / "benchmarks/profile_v0_3_s09_ap_cache_diagnostic.py"
    revert_proxy = support_root / "benchmarks/v0_3_s09_revert_pressure_proxy.py"
    for python in (baseline_python, candidate_python):
        if not python.is_file():
            raise FileNotFoundError(f"working venv Python required: {python}")
    for path in (child, profile, revert_proxy):
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(f"regular diagnostic support file required: {path}")

    output_root.mkdir(parents=True)
    (output_root / "profiles").mkdir()
    (output_root / "logs").mkdir()
    (output_root / "tf32").mkdir()
    (output_root / "compile_caches").mkdir()
    environment = {
        "baseline": _isolated_import(baseline_python),
        "candidate": _isolated_import(candidate_python),
        "support_root": str(support_root),
        "support_files": {
            str(path.relative_to(support_root)): _sha256(path)
            for path in (child, profile, revert_proxy)
        },
    }
    _atomic_replace_json(output_root / "environment.json", environment)

    completed_rows: list[dict[str, object]] = []
    for expected_sequence, row in enumerate(rows, start=1):
        if int(row.get("sequence", -1)) != expected_sequence:
            raise ValueError("profile sequence is not contiguous")
        key = str(row["key"])
        role = str(row["diagnostic_role"])
        python = baseline_python if role == "baseline" else candidate_python
        output = output_root / "profiles" / f"{key}.json"
        tf32 = output_root / "tf32" / f"{key}.json"
        stdout = output_root / "logs" / f"{key}.stdout"
        stderr = output_root / "logs" / f"{key}.stderr"
        cache = output_root / "compile_caches" / key
        triton = cache / "triton"
        inductor = cache / "torchinductor"
        triton.mkdir(parents=True)
        inductor.mkdir(parents=True)
        if any(triton.iterdir()) or any(inductor.iterdir()):
            raise RuntimeError("per-profile compile caches were not initially empty")
        command = _command(
            row,
            python=python,
            child=child,
            profile=profile,
            output=output,
            tf32=tf32,
        )
        env = dict(os.environ)
        env.update(
            {
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": str(support_root),
                "TRITON_CACHE_DIR": str(triton),
                "TORCHINDUCTOR_CACHE_DIR": str(inductor),
            }
        )
        if plan.get("mode") == "cpu_rehearsal":
            env["TORCH_COMPILE_DISABLE"] = "1"
        with stdout.open("xb") as stdout_stream, stderr.open("xb") as stderr_stream:
            completed = subprocess.run(
                command,
                cwd=support_root,
                env=env,
                stdout=stdout_stream,
                stderr=stderr_stream,
                check=False,
            )
        if completed.returncode != 0:
            failure = {
                "schema": SCHEMA,
                "classification": "DIAGNOSTIC_PROFILE_CHILD_FAILED",
                "failed_key": key,
                "returncode": completed.returncode,
                "completed_profiles": len(completed_rows),
                "qualification_complete": False,
                "s09_complete": False,
            }
            _atomic_replace_json(output_root / "failure.json", failure)
            raise RuntimeError(f"diagnostic profile {key} failed with {completed.returncode}")
        result = json.loads(output.read_text(encoding="utf-8"))
        if result.get("diagnostic_role") != role or result.get("finite") is not True:
            raise RuntimeError(f"invalid diagnostic result for {key}")
        if (
            role == "candidate_revert_proxy"
            and not result["cache_control"]["proved_revert_proxy"]
        ):
            raise RuntimeError(f"revert-proxy proof missing for {key}")
        completed_rows.append(
            {
                "sequence": expected_sequence,
                "key": key,
                "diagnostic_role": role,
                "result": str(output),
                "result_sha256": _sha256(output),
                "tf32": str(tf32),
                "tf32_sha256": _sha256(tf32),
                "stdout": str(stdout),
                "stderr": str(stderr),
                "command": command,
                "returncode": completed.returncode,
            }
        )
        _atomic_replace_json(
            output_root / "progress.json",
            {
                "schema": SCHEMA,
                "planned_profiles": 72,
                "completed_profiles": len(completed_rows),
                "last_completed": key,
            },
        )

    manifest = {
        "schema": SCHEMA,
        "classification": "COMPLETE_S09_AP_CACHE_DIAGNOSTIC_MATRIX",
        "diagnostic_only": True,
        "qualification_complete": False,
        "s09_complete": False,
        "plan": str(plan_path),
        "plan_sha256": _sha256(plan_path),
        "planned_profiles": 72,
        "completed_profiles": 72,
        "environment": environment,
        "profiles": completed_rows,
    }
    _atomic_replace_json(output_root / "matrix_manifest.json", manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    args = parser.parse_args(argv)
    execute(
        plan_path=args.plan,
        output_root=args.output_root,
        baseline_python=args.baseline_python,
        candidate_python=args.candidate_python,
        support_root=args.support_root,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
