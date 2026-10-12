#!/usr/bin/env python3
"""Build the non-executing RC4.4.7 command plan.

The generated commands run from an external qualification-support directory
that contains ``benchmarks/`` but no ``pssolver/`` package.  Consequently the
two fresh interpreters must import PSSolver from their installed wheels while
sharing the same content-addressed qualification code.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmarks.analyze_rc447_pressure_h100 import COMMITS, WHEELS


SCHEMA = "pssolver.rc4_4_7.channel_pressure_command_plan.v1"
SUPPORT_FILES = (
    "benchmarks/__init__.py",
    "benchmarks/profile_beris_edwards_timestep.py",
    "benchmarks/profile_channel_runtime_timestep.py",
    "benchmarks/profile_rc447_channel_pressure.py",
    "benchmarks/run_rc447_cuda_tests.py",
    "benchmarks/qualify_rc447_pressure_cuda.py",
    "benchmarks/analyze_rc447_pressure_h100.py",
)
GRIDS = (
    ("R128x20x20", "128,20,20"),
    ("R512x40x40", "512,40,40"),
)
ORDER = (("baseline", "candidate"), ("candidate", "baseline"), ("baseline", "candidate"))


def _ordinary_file(path: Path, name: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise ValueError(f"{name} must be an ordinary file: {path}")
    return path


def build_plan(
    *,
    baseline_python: Path,
    candidate_python: Path,
    support_root: Path,
    output_root: Path,
    consumer_root: Path,
    provider_test_root: Path,
) -> dict[str, object]:
    baseline_python = _ordinary_file(baseline_python, "baseline Python")
    candidate_python = _ordinary_file(candidate_python, "candidate Python")
    support_root = support_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    consumer_root = consumer_root.expanduser().resolve()
    provider_test_root = provider_test_root.expanduser().resolve()
    if (support_root / "pssolver").exists():
        raise ValueError("support root must not contain a pssolver source package")
    missing = [name for name in SUPPORT_FILES if not (support_root / name).is_file()]
    if missing:
        raise ValueError(f"qualification support is incomplete: {missing}")
    roles = {
        "baseline": {
            "python": str(baseline_python),
            "commit": COMMITS["baseline"],
            "wheel_sha256": WHEELS["baseline"],
        },
        "candidate": {
            "python": str(candidate_python),
            "commit": COMMITS["candidate"],
            "wheel_sha256": WHEELS["candidate"],
        },
    }
    commands: list[dict[str, object]] = []
    sequence = 0
    profile_paths: dict[str, list[str]] = {"baseline": [], "candidate": []}
    for grid, shape in GRIDS:
        for trial, pair in enumerate(ORDER, start=1):
            for role in pair:
                sequence += 1
                name = f"{sequence:02d}_{role}_{grid}_trial{trial}"
                profile = output_root / "profiles" / f"{name}.json"
                profile_paths[role].append(str(profile))
                cache = output_root / "compiler_caches" / name
                commands.append(
                    {
                        "sequence": sequence,
                        "kind": "profile",
                        "name": name,
                        "role": role,
                        "grid": grid,
                        "trial": trial,
                        "cwd": str(support_root),
                        "environment": {
                            "PYTHONNOUSERSITE": "1",
                            "PYTHONDONTWRITEBYTECODE": "1",
                            "NVIDIA_TF32_OVERRIDE": "0",
                            "TRITON_CACHE_DIR": str(cache / "triton"),
                            "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
                        },
                        "argv": [
                            roles[role]["python"],
                            "-m",
                            "benchmarks.profile_rc447_channel_pressure",
                            "--role",
                            role,
                            "--provider-commit",
                            roles[role]["commit"],
                            "--provider-wheel-sha256",
                            roles[role]["wheel_sha256"],
                            "--trial",
                            str(trial),
                            "--shape",
                            shape,
                            "--lengths",
                            "32,5,5",
                            "--output",
                            str(profile),
                        ],
                    }
                )
    correctness = output_root / "correctness"
    cuda_test_command = {
        "stage": 1,
        "kind": "cuda_only_tests",
        "name": "seven_cuda_only_tests",
        "cwd": str(support_root),
        "environment": {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NVIDIA_TF32_OVERRIDE": "0",
        },
        "argv": [
            str(candidate_python),
            "-m",
            "benchmarks.run_rc447_cuda_tests",
            "--provider-root",
            str(provider_test_root),
            "--output",
            str(output_root / "cuda_tests.json"),
        ],
    }
    correctness_command = {
        "stage": 2,
        "kind": "candidate_correctness",
        "name": "candidate_pressure_correctness",
        "cwd": str(support_root),
        "environment": {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "NVIDIA_TF32_OVERRIDE": "0",
        },
        "argv": [
            str(candidate_python),
            "-m",
            "benchmarks.qualify_rc447_pressure_cuda",
            "--output",
            str(correctness),
            "--consumer-root",
            str(consumer_root),
            "--provider-test-root",
            str(provider_test_root),
        ],
    }
    analyzer_argv = [
        str(candidate_python),
        "-m",
        "benchmarks.analyze_rc447_pressure_h100",
    ]
    for path in profile_paths["baseline"]:
        analyzer_argv.extend(("--baseline-profile", path))
    for path in profile_paths["candidate"]:
        analyzer_argv.extend(("--candidate-profile", path))
    analyzer_argv.extend(
        (
            "--correctness-report",
            str(correctness / "report.json"),
            "--cuda-test-report",
            str(output_root / "cuda_tests.json"),
            "--output",
            str(output_root / "analysis" / "qualification.json"),
        )
    )
    analyzer_command = {
        "stage": 4,
        "kind": "analyzer",
        "name": "cumulative_fail_closed_analyzer",
        "cwd": str(support_root),
        "environment": {
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
        "argv": analyzer_argv,
    }
    return {
        "schema": SCHEMA,
        "execution_started": False,
        "h100_submission_authorized": False,
        "support_root": str(support_root),
        "support_root_contains_pssolver": False,
        "support_files": list(SUPPORT_FILES),
        "roles": roles,
        "profile_count": len(commands),
        "cuda_test_command": cuda_test_command,
        "profile_commands": commands,
        "correctness_command": correctness_command,
        "analyzer_command": analyzer_command,
        "execution_order": [
            "cuda_test_command",
            "correctness_command",
            "profile_commands",
            "analyzer_command",
        ],
        "formal_h100_submission_limit": 1,
        "automatic_retry": False,
    }


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-python", type=Path, required=True)
    parser.add_argument("--candidate-python", type=Path, required=True)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--consumer-root", type=Path, required=True)
    parser.add_argument("--provider-test-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(
        args.output,
        build_plan(
            baseline_python=args.baseline_python,
            candidate_python=args.candidate_python,
            support_root=args.support_root,
            output_root=args.output_root,
            consumer_root=args.consumer_root,
            provider_test_root=args.provider_test_root,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
