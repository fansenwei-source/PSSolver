#!/usr/bin/env python3
"""Run the exact rc5 G-1 installed-wheel CUDA-only test gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.plan_rc5_g1_h100 import (
    PROVIDER_CANDIDATE,
    PROVIDER_CANDIDATE_WHEEL_SHA256,
    build_plan,
)
from benchmarks.run_rc426_cuda_tests import main as cuda_main


SCHEMA = "pssolver.rc5.g1.cuda_only.v1"


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
            raise RuntimeError("CUDA-only JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def summarize(detail: dict[str, object], policy: dict[str, object]) -> dict[str, object]:
    expected = list(build_plan()["cuda_only_tests"])
    outcomes = detail.get("outcomes")
    if not isinstance(outcomes, dict):
        raise TypeError("CUDA-only outcomes are missing")
    passed_nodes = outcomes.get("passed")
    if not isinstance(passed_nodes, list):
        raise TypeError("CUDA-only passed nodes are missing")
    failed = outcomes.get("failed")
    skipped = outcomes.get("skipped")
    xfailed = outcomes.get("xfailed")
    xpassed = outcomes.get("xpassed")
    collection_errors = outcomes.get("collection_errors")
    if detail.get("requested_nodeids") != expected:
        raise ValueError("CUDA-only requested node IDs differ")
    if sorted(passed_nodes) != sorted(expected):
        raise ValueError("CUDA-only passed node IDs differ")
    if any((failed, skipped, xfailed, xpassed, collection_errors)):
        raise ValueError("CUDA-only test outcomes contain a non-pass result")
    enforcement = policy.get("enforcement")
    if not isinstance(enforcement, dict) or enforcement.get("passed") is not True:
        raise ValueError("CUDA-only TF32 policy enforcement failed")
    if detail.get("passed") is not True or detail.get("pytest_exit_code") != 0:
        raise ValueError("CUDA-only detailed runner did not pass")
    return {
        "schema": SCHEMA,
        "provider_commit": PROVIDER_CANDIDATE,
        "provider_wheel_sha256": PROVIDER_CANDIDATE_WHEEL_SHA256,
        "passed": True,
        "passed_count": len(passed_nodes),
        "failed_count": 0,
        "skipped_count": 0,
        "xfailed_count": 0,
        "xpassed_count": 0,
        "deselected_count": 0,
        "collection_error_count": 0,
        "node_ids": expected,
        "tf32_policy": policy,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--detail-output", type=Path, required=True)
    parser.add_argument("--policy-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    root = args.test_root.expanduser().resolve()
    if not root.is_dir() or root.is_symlink():
        raise FileNotFoundError(f"ordinary test root required: {root}")
    if (root / "pssolver").exists() or (root / "pssolver").is_symlink():
        raise RuntimeError("CUDA-only support root shadows installed PSSolver")
    outputs = [
        path.expanduser().absolute()
        for path in (args.detail_output, args.policy_output, args.output)
    ]
    if len(set(outputs)) != 3:
        raise ValueError("CUDA-only outputs must be distinct")
    previous = Path.cwd()
    try:
        os.chdir(root)
        command: list[str] = []
        for node in build_plan()["cuda_only_tests"]:
            command.extend(("--node", str(node)))
        command.extend(
            (
                "--expected",
                "7",
                "--output",
                str(outputs[0]),
                "--policy-output",
                str(outputs[1]),
            )
        )
        cuda_main(command)
    finally:
        os.chdir(previous)
    detail = json.loads(outputs[0].read_text(encoding="utf-8"))
    policy = json.loads(outputs[1].read_text(encoding="utf-8"))
    _atomic_json(outputs[2], summarize(detail, policy))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
