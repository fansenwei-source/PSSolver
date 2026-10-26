#!/usr/bin/env python3
"""Rehearse every non-formal S-09 job stage with installed wheels on CPU."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import sysconfig
import tempfile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.plan_v0_3_s09_h100 import (
    BASELINE_WHEEL_SHA256,
    CANDIDATE_WHEEL_SHA256,
    CONSUMER_WHEEL_SHA256,
    build_plan,
)


SCHEMA = "pssolver.v0_3.s09.cpu_rehearsal.v1"
STAGES = (
    "installed_wheel_environment",
    "cuda_device_identity_assertions",
    "installed_wheel_cuda_only_tests",
    "cpu_short_horizon_hermitian",
    "tf32_forward_backward_scope",
    "checkpoint_restart_and_released_fixture_matrix",
    "numerical_gradient_taylor_and_replay",
    "independent_control_consumer",
    "formal_profile_matrix",
    "assembler_and_analyzer_import",
)


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
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def _record_summary(distribution_name: str) -> dict[str, object]:
    distribution = importlib.metadata.distribution(distribution_name)
    record = Path(distribution._path) / "RECORD"  # type: ignore[attr-defined]
    if not record.is_file() or record.is_symlink():
        raise FileNotFoundError(f"ordinary wheel RECORD required: {record}")
    return {
        "path": str(record.resolve()),
        "sha256": _sha256(record),
        "entry_count": sum(1 for _ in record.open(encoding="utf-8")),
    }


def _internal_inspect(role: str) -> dict[str, object]:
    import pssolver

    purelib = Path(sysconfig.get_path("purelib")).resolve()
    provider_import = Path(pssolver.__file__).resolve()
    if not provider_import.is_relative_to(purelib):
        raise RuntimeError("Provider import is outside this environment's site-packages")
    value: dict[str, object] = {
        "role": role,
        "executable": sys.executable,
        "prefix": sys.prefix,
        "purelib": str(purelib),
        "sys_path": list(sys.path),
        "pssolver_version": importlib.metadata.version("pssolver"),
        "pssolver_import": str(provider_import),
        "pssolver_record": _record_summary("pssolver"),
    }
    if role == "consumer":
        import pssolver_control

        consumer_import = Path(pssolver_control.__file__).resolve()
        if not consumer_import.is_relative_to(purelib):
            raise RuntimeError("Consumer import is outside this environment's site-packages")
        value.update(
            {
                "pssolver_control_version": importlib.metadata.version(
                    "pssolver-control"
                ),
                "pssolver_control_import": str(consumer_import),
                "pssolver_control_record": _record_summary("pssolver-control"),
            }
        )
    return value


def _environment(support_root: Path) -> dict[str, str]:
    value = dict(os.environ)
    for name in ("PYTHONHOME", "LD_LIBRARY_PATH"):
        value.pop(name, None)
    value.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "NVIDIA_TF32_OVERRIDE": "0",
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPATH": str(support_root),
        }
    )
    return value


def _run(
    *, name: str, command: list[str], cwd: Path, environment: dict[str, str], logs: Path
) -> dict[str, object]:
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = logs / f"{name}.stdout"
    stderr = logs / f"{name}.stderr"
    stdout.write_text(completed.stdout, encoding="utf-8")
    stderr.write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"CPU rehearsal stage failed: {name}")
    return {
        "name": name,
        "command": command,
        "exit_code": completed.returncode,
        "stdout_sha256": _sha256(stdout),
        "stderr_sha256": _sha256(stderr),
        "passed": True,
    }


def _inspect(
    python: Path, role: str, script: Path, support: Path, logs: Path
) -> dict[str, object]:
    command = [str(python), "-I", str(script), "--internal-inspect", role]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=_environment(support),
    )
    (logs / f"inspect_{role}.stdout").write_text(completed.stdout, encoding="utf-8")
    (logs / f"inspect_{role}.stderr").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"installed-wheel inspection failed: {role}")
    return json.loads(completed.stdout)


def _pip_check(python: Path, role: str, support: Path, logs: Path) -> dict[str, object]:
    command = [str(python), "-I", "-m", "pip", "check"]
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=_environment(support),
    )
    (logs / f"pip_check_{role}.stdout").write_text(completed.stdout, encoding="utf-8")
    (logs / f"pip_check_{role}.stderr").write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"pip check failed: {role}")
    return {"exit_code": 0, "stdout": completed.stdout, "passed": True}


def execute(args: argparse.Namespace) -> dict[str, object]:
    output = args.output_root.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"CPU rehearsal output already exists: {output}")
    provider_support = args.provider_support.expanduser().resolve()
    consumer_support = args.consumer_support.expanduser().resolve()
    provider_reference = args.provider_reference.expanduser().resolve()
    for root, required in (
        (provider_support, "benchmarks"),
        (consumer_support, "tests"),
        (provider_reference, "pssolver"),
    ):
        if not (root / required).is_dir() or root.is_symlink():
            raise FileNotFoundError(f"ordinary rehearsal root required: {root}")
    if (provider_support / "pssolver").exists() or (
        consumer_support / "pssolver_control"
    ).exists():
        raise RuntimeError("CPU rehearsal support shadows an installed package")
    output.mkdir(parents=True)
    logs = output / "logs"
    results = output / "results"
    logs.mkdir()
    results.mkdir()
    script = Path(__file__).resolve()
    pythons = {
        "baseline": args.baseline_python.expanduser().absolute(),
        "candidate": args.candidate_python.expanduser().absolute(),
        "consumer": args.consumer_python.expanduser().absolute(),
    }
    inspections = {
        role: _inspect(python, role, script, provider_support, logs)
        for role, python in pythons.items()
    }
    expected_versions = {
        "baseline": "0.2.2",
        "candidate": "0.3.0.dev0",
        "consumer": "0.3.0.dev0",
    }
    if any(
        inspections[role]["pssolver_version"] != version
        for role, version in expected_versions.items()
    ):
        raise RuntimeError("CPU rehearsal Provider version differs")
    if inspections["consumer"].get("pssolver_control_version") != "0.0.1.dev0":
        raise RuntimeError("CPU rehearsal Consumer version differs")
    pip_checks = {
        role: _pip_check(python, role, provider_support, logs)
        for role, python in pythons.items()
    }
    stages: list[dict[str, object]] = [
        {"name": STAGES[0], "passed": True},
        {"name": STAGES[1], "skipped": True, "reason": "CPU rehearsal"},
        {
            "name": STAGES[2],
            "skipped": True,
            "reason": "CUDA-only node execution; node list and JUnit wiring validated statically",
        },
    ]
    environment = _environment(provider_support)
    node_root = Path(tempfile.mkdtemp(prefix="pssolver-s09-cpu-rehearsal-", dir="/tmp"))
    try:
        pytest_tmp = node_root / "pytest_tmp"
        pytest_tmp.mkdir()
        environment["TMPDIR"] = str(pytest_tmp)
        commands = [
            (
                STAGES[3],
                pythons["candidate"],
                [
                    "benchmarks/run_v0_3_s09_long_horizon.py",
                    "--python", str(pythons["candidate"]),
                    "--support-root", str(provider_support),
                    "--case-root", str(output / "long_horizon_cases"),
                    "--device", "cpu", "--horizon", "0.04",
                    "--output", str(results / "long_horizon.json"),
                ],
            ),
            (
                STAGES[4],
                pythons["candidate"],
                [
                    "benchmarks/run_v0_3_s09_tf32_backward.py",
                    "--support-root", str(provider_support), "--device", "cpu",
                    "--output", str(results / "tf32_backward.json"),
                ],
            ),
            (
                STAGES[5],
                pythons["candidate"],
                [
                    "benchmarks/run_v0_3_s09_checkpoints.py",
                    "--test-root", str(provider_support),
                    "--junit", str(logs / "checkpoints.junit.xml"),
                    "--device", "cpu", "--skip-cross-device",
                    "--output", str(results / "checkpoints.json"),
                ],
            ),
            (
                STAGES[6],
                pythons["candidate"],
                [
                    "benchmarks/run_v0_3_s09_numerical.py",
                    "--baseline-python", str(pythons["baseline"]),
                    "--candidate-python", str(pythons["candidate"]),
                    "--support-root", str(provider_support), "--device", "cpu",
                    "--output", str(results / "numerical.json"),
                ],
            ),
            (
                STAGES[7],
                pythons["consumer"],
                [
                    "benchmarks/run_v0_3_s09_consumer_h100.py",
                    "--support-root", str(consumer_support),
                    "--provider-reference", str(provider_reference),
                    "--junit", str(logs / "consumer.junit.xml"),
                    "--device", "cpu",
                    "--output", str(results / "consumer.json"),
                ],
            ),
        ]
        for name, python, arguments in commands:
            stages.append(
                _run(
                    name=name,
                    command=[str(python), str(provider_support / arguments[0]), *arguments[1:]],
                    cwd=output,
                    environment=environment,
                    logs=logs,
                )
            )
        stages.append(
            _run(
                name=STAGES[8],
                command=[
                    str(pythons["candidate"]),
                    str(provider_support / "benchmarks/run_v0_3_s09_profile_matrix.py"),
                    "--baseline-python", str(pythons["baseline"]),
                    "--candidate-python", str(pythons["candidate"]),
                    "--support-root", str(provider_support),
                    "--output-root", str(output / "formal_profiles_not_executed"),
                    "--command-plan", str(results / "profile_command_plan.json"),
                    "--plan-only",
                ],
                cwd=output,
                environment=environment,
                logs=logs,
            )
        )
        profile_plan = json.loads(
            (results / "profile_command_plan.json").read_text(encoding="utf-8")
        )
        if profile_plan.get("profile_count") != 192:
            raise RuntimeError("CPU rehearsal profile plan count differs")
        stages[-1]["formal_execution_skipped"] = True
        stages[-1]["profile_count"] = 192
        stages.append(
            _run(
                name=STAGES[9],
                command=[
                    str(pythons["candidate"]), "-c",
                    (
                        "from benchmarks.assemble_v0_3_s09_h100 import STAGES; "
                        "from benchmarks.analyze_v0_3_s09_h100 import PASS_CLASSIFICATION; "
                        "assert len(STAGES) == 7; assert PASS_CLASSIFICATION"
                    ),
                ],
                cwd=output,
                environment=environment,
                logs=logs,
            )
        )
    finally:
        shutil.rmtree(node_root)
    if [row["name"] for row in stages] != list(STAGES):
        raise RuntimeError("CPU rehearsal stage order differs")
    if not all(row.get("passed") is True or row.get("skipped") is True for row in stages):
        raise RuntimeError("CPU rehearsal contains an incomplete stage")
    result = {
        "schema": SCHEMA,
        "classification": "PASS_S09_INSTALLED_WHEEL_CPU_REHEARSAL",
        "installed_wheel_hashes": {
            "baseline": BASELINE_WHEEL_SHA256,
            "candidate": CANDIDATE_WHEEL_SHA256,
            "consumer": CONSUMER_WHEEL_SHA256,
        },
        "inspections": inspections,
        "pip_checks": pip_checks,
        "stage_order": list(STAGES),
        "stages": stages,
        "skipped_only": [
            "CUDA device identity assertions",
            "CUDA-only test execution",
            "cross-device checkpoint cells",
            "formal 192-profile execution",
        ],
        "formal_profile_count_preserved": len(build_plan()["profile_matrix"]),
        "passed": True,
    }
    _atomic_json(output / "cpu_rehearsal.json", result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--internal-inspect", choices=("baseline", "candidate", "consumer"))
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--consumer-python", type=Path)
    parser.add_argument("--provider-support", type=Path)
    parser.add_argument("--consumer-support", type=Path)
    parser.add_argument("--provider-reference", type=Path)
    parser.add_argument("--output-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.internal_inspect is not None:
        print(json.dumps(_internal_inspect(args.internal_inspect), allow_nan=False, sort_keys=True))
        return 0
    required = (
        "baseline_python", "candidate_python", "consumer_python",
        "provider_support", "consumer_support", "provider_reference", "output_root",
    )
    missing = [name for name in required if getattr(args, name) is None]
    if missing:
        raise ValueError(f"missing CPU rehearsal arguments: {missing}")
    execute(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
