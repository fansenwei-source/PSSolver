#!/usr/bin/env python3
"""Run the seven RC4.2.6 CUDA-only nodes with strict outcome accounting."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "pssolver.rc4_2_6.cuda_only_tests.v2"
TF32_POLICY_SCHEMA = "pssolver.rc4_2_6.cuda_test_tf32_policy.v1"


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError("CUDA-test JSON round-trip differs")
    finally:
        if temporary.exists():
            temporary.unlink()


def _exception_record(error: BaseException) -> dict[str, str]:
    return {"type": type(error).__name__, "message": str(error)}


def _read_flag(getter: Any) -> dict[str, object]:
    try:
        return {"value": bool(getter()), "exception": None}
    except Exception as error:  # pragma: no cover - exercised through fakes
        return {"value": None, "exception": _exception_record(error)}


def _set_flag(setter: Any) -> dict[str, object]:
    try:
        setter()
        return {"applied": True, "exception": None}
    except Exception as error:  # pragma: no cover - exercised through fakes
        return {"applied": False, "exception": _exception_record(error)}


def _enforce_tf32_policy(torch_module: object | None = None) -> dict[str, object]:
    """Disable TF32 in the same process that will enter ``pytest.main``."""

    if torch_module is None:
        import torch as torch_module

    def matmul_flag() -> object:
        return torch_module.backends.cuda.matmul.allow_tf32  # type: ignore[attr-defined]

    def cudnn_flag() -> object:
        return torch_module.backends.cudnn.allow_tf32  # type: ignore[attr-defined]

    before = {
        "cuda_matmul_allow_tf32": _read_flag(matmul_flag),
        "cudnn_allow_tf32": _read_flag(cudnn_flag),
    }

    def disable_matmul_tf32() -> None:
        torch_module.backends.cuda.matmul.allow_tf32 = False  # type: ignore[attr-defined]

    def disable_cudnn_tf32() -> None:
        torch_module.backends.cudnn.allow_tf32 = False  # type: ignore[attr-defined]

    assignments = {
        "torch.backends.cuda.matmul.allow_tf32 = False": _set_flag(
            disable_matmul_tf32
        ),
        "torch.backends.cudnn.allow_tf32 = False": _set_flag(
            disable_cudnn_tf32
        ),
    }
    after = {
        "cuda_matmul_allow_tf32": _read_flag(matmul_flag),
        "cudnn_allow_tf32": _read_flag(cudnn_flag),
    }
    failed_flags = [
        name
        for name, observation in after.items()
        if observation["exception"] is not None or observation["value"] is not False
    ]
    passed = not failed_flags and all(
        result["applied"] is True and result["exception"] is None
        for result in assignments.values()
    )
    return {
        "schema": TF32_POLICY_SCHEMA,
        "policy": "explicit_python_flags_before_pytest_main",
        "before": before,
        "assignments": assignments,
        "after": after,
        "enforcement": {
            "passed": passed,
            "failed_flags": failed_flags,
        },
    }


class Results:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[dict[str, str]] = []
        self.skipped: list[dict[str, str]] = []
        self.xfailed: list[str] = []
        self.xpassed: list[str] = []
        self.collection_errors: list[dict[str, str]] = []

    @staticmethod
    def _detail(report: Any) -> str:
        return str(getattr(report, "longrepr", ""))

    def pytest_collectreport(self, report: Any) -> None:
        if bool(getattr(report, "failed", False)):
            self.collection_errors.append(
                {
                    "nodeid": str(getattr(report, "nodeid", "collection")),
                    "detail": self._detail(report),
                }
            )

    def pytest_runtest_logreport(self, report: Any) -> None:
        nodeid = str(report.nodeid)
        wasxfail = getattr(report, "wasxfail", None)
        if report.when != "call":
            if report.failed or (report.when == "setup" and report.skipped):
                target = self.skipped if report.skipped else self.failed
                target.append({"nodeid": nodeid, "detail": self._detail(report)})
            return
        if wasxfail is not None:
            (self.xfailed if report.skipped else self.xpassed).append(nodeid)
        elif report.passed:
            self.passed.append(nodeid)
        elif report.failed:
            self.failed.append({"nodeid": nodeid, "detail": self._detail(report)})
        elif report.skipped:
            self.skipped.append({"nodeid": nodeid, "detail": self._detail(report)})

    def metadata(self) -> dict[str, object]:
        return {
            "passed": sorted(self.passed),
            "failed": self.failed,
            "skipped": self.skipped,
            "xfailed": sorted(self.xfailed),
            "xpassed": sorted(self.xpassed),
            "collection_errors": self.collection_errors,
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--node", action="append", required=True)
    parser.add_argument("--expected", type=int, default=7)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy-output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if len(arguments.node) != arguments.expected:
        raise ValueError("CUDA-only node list count differs")
    if len(set(arguments.node)) != len(arguments.node):
        raise ValueError("CUDA-only node list contains duplicates")
    output = Path(arguments.output).expanduser().resolve()
    policy_output = Path(arguments.policy_output).expanduser().resolve()
    if output == policy_output:
        raise ValueError("CUDA-test output and TF32-policy output must differ")

    tf32_policy = _enforce_tf32_policy()
    _atomic_json(policy_output, tf32_policy)
    tf32_policy_identity = {
        "path": str(policy_output),
        "sha256": hashlib.sha256(policy_output.read_bytes()).hexdigest(),
        "enforcement_passed": tf32_policy["enforcement"]["passed"],  # type: ignore[index]
    }
    if not tf32_policy_identity["enforcement_passed"]:
        _atomic_json(
            output,
            {
                "schema": SCHEMA,
                "expected": arguments.expected,
                "requested_nodeids": arguments.node,
                "pytest_started": False,
                "pytest_exit_code": None,
                "tf32_policy": tf32_policy_identity,
                "outcomes": Results().metadata(),
                "passed": False,
            },
        )
        raise RuntimeError("CUDA-test TF32 policy enforcement failed")

    import pytest

    collector = Results()
    exit_code = pytest.main(
        ["-q", "-p", "no:cacheprovider", *arguments.node],
        plugins=[collector],
    )
    outcomes = collector.metadata()
    passed = (
        int(exit_code) == 0
        and len(outcomes["passed"]) == arguments.expected
        and not outcomes["failed"]
        and not outcomes["skipped"]
        and not outcomes["xfailed"]
        and not outcomes["xpassed"]
        and not outcomes["collection_errors"]
    )
    report = {
        "schema": SCHEMA,
        "expected": arguments.expected,
        "requested_nodeids": arguments.node,
        "pytest_started": True,
        "pytest_exit_code": int(exit_code),
        "tf32_policy": tf32_policy_identity,
        "outcomes": outcomes,
        "passed": passed,
    }
    _atomic_json(output, report)
    if not passed:
        raise RuntimeError("CUDA-only test gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
