#!/usr/bin/env python3
"""Run the seven RC4.2.6 CUDA-only nodes with strict outcome accounting."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "pssolver.rc4_2_6.cuda_only_tests.v1"


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
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if len(arguments.node) != arguments.expected:
        raise ValueError("CUDA-only node list count differs")
    if len(set(arguments.node)) != len(arguments.node):
        raise ValueError("CUDA-only node list contains duplicates")
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
        "pytest_exit_code": int(exit_code),
        "outcomes": outcomes,
        "passed": passed,
    }
    _atomic_json(Path(arguments.output), report)
    if not passed:
        raise RuntimeError("CUDA-only test gate failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
