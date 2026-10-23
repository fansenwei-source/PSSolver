#!/usr/bin/env python3
"""Assemble and validate the non-profile S-09 H100 stage reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmarks.analyze_v0_3_s09_h100 import AUXILIARY_SCHEMA, _validate_auxiliary
from benchmarks.plan_v0_3_s09_h100 import build_plan


STAGES = (
    "node_local_tmp_environment_and_host_compiler",
    "installed_wheel_cuda_only_tests",
    "cuda_long_horizon_hermitian",
    "tf32_forward_backward_scope",
    "checkpoint_restart_and_released_fixture_matrix",
    "numerical_gradient_taylor_and_replay",
    "independent_control_consumer",
)


def _load(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"regular stage JSON required: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"stage report must be an object: {path}")
    return value


def assemble(reports: list[dict[str, object]]) -> dict[str, object]:
    if len(reports) != len(STAGES):
        raise ValueError("S-09 auxiliary stage count differs")
    ordered: list[dict[str, object]] = []
    for expected, report in zip(STAGES, reports, strict=True):
        if report.get("name") != expected:
            raise ValueError(f"S-09 auxiliary stage order differs: {expected}")
        ordered.append(report)
    result = {
        "schema": AUXILIARY_SCHEMA,
        "passed": True,
        "formal_h100_submission_count": 1,
        "automatic_retry": False,
        "second_job_submitted": False,
        "stages": ordered,
    }
    _validate_auxiliary(result, build_plan())
    return result


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in STAGES:
        parser.add_argument(f"--{name.replace('_', '-')}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = [_load(getattr(args, name)) for name in STAGES]
    _atomic_json(args.output, assemble(reports))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
