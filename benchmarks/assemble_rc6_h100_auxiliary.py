#!/usr/bin/env python3
"""Validate and assemble the non-profile RC6 H100 subgate reports."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from benchmarks.analyze_rc6_single_h100 import AUXILIARY_SCHEMA, _validate_auxiliary
from benchmarks.plan_rc6_single_h100_qualification import build_plan


def _load(path: Path) -> dict[str, object]:
    path = path.expanduser().resolve()
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"regular JSON file required: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def assemble(
    *,
    environment: dict[str, object],
    cuda_only: dict[str, object],
    contract_smokes: dict[str, object],
    tf32_scope: dict[str, object],
    current_checkpoints: dict[str, object],
    released_checkpoints: dict[str, object],
    numerical: dict[str, object],
    consumer: dict[str, object],
) -> dict[str, object]:
    if current_checkpoints.get("schema") != "pssolver.rc6.current_checkpoint_h100.v1":
        raise ValueError("current checkpoint schema differs")
    if released_checkpoints.get("schema") != "pssolver.rc6.released_checkpoint_h100.v1":
        raise ValueError("released checkpoint schema differs")
    checkpoints = {
        "schema": "pssolver.rc6.checkpoint_h100.v1",
        "provider_commit": current_checkpoints.get("provider_commit"),
        "consumer_source_commit": current_checkpoints.get("consumer_source_commit"),
        "passed": (
            current_checkpoints.get("passed") is True
            and released_checkpoints.get("passed") is True
        ),
        "current_family_count": current_checkpoints.get("family_count"),
        "direction_count": current_checkpoints.get("direction_count"),
        "continuous_split_restart_passed": current_checkpoints.get(
            "continuous_split_restart_passed"
        ),
        "transactional_restore_passed": current_checkpoints.get(
            "transactional_restore_passed"
        ),
        "negative_gates_passed": current_checkpoints.get("negative_gates_passed"),
        "rejected_restart_output_remains_absent": current_checkpoints.get(
            "rejected_restart_output_remains_absent"
        ),
        "released_primary_case_count": released_checkpoints.get("primary_case_count"),
        "released_dispositions_match_cpu_record": released_checkpoints.get(
            "dispositions_match_cpu_record"
        ),
        "source_checkpoints_unchanged": (
            current_checkpoints.get("source_checkpoints_unchanged") is True
            and released_checkpoints.get("source_checkpoints_unchanged") is True
        ),
    }
    result = {
        "schema": AUXILIARY_SCHEMA,
        "passed": True,
        "formal_h100_submission_count": 1,
        "automatic_retry": False,
        "second_job_submitted": False,
        "environment": environment,
        "cuda_only": cuda_only,
        "contract_smokes": contract_smokes,
        "tf32_scope": tf32_scope,
        "checkpoints": checkpoints,
        "numerical": numerical,
        "consumer": consumer,
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
    expected = json.loads(payload)
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("auxiliary JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--environment", type=Path, required=True)
    parser.add_argument("--cuda-only", type=Path, required=True)
    parser.add_argument("--contract-smokes", type=Path, required=True)
    parser.add_argument("--tf32-scope", type=Path, required=True)
    parser.add_argument("--current-checkpoints", type=Path, required=True)
    parser.add_argument("--released-checkpoints", type=Path, required=True)
    parser.add_argument("--numerical", type=Path, required=True)
    parser.add_argument("--consumer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = assemble(
        environment=_load(args.environment),
        cuda_only=_load(args.cuda_only),
        contract_smokes=_load(args.contract_smokes),
        tf32_scope=_load(args.tf32_scope),
        current_checkpoints=_load(args.current_checkpoints),
        released_checkpoints=_load(args.released_checkpoints),
        numerical=_load(args.numerical),
        consumer=_load(args.consumer),
    )
    _atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
