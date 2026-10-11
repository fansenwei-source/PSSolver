"""Run the RC4.3.6 lifting closure against one installed PSSolver wheel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
from typing import Any


CLASSIFICATION = "PASS_RC4_3_6_INSTALLED_WHEEL_LIFTING_CPU_CLOSURE"


def _is_relative_to(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _ordinary_file(value: str, description: str) -> Path:
    requested = Path(value).expanduser()
    path = requested.resolve()
    if requested.is_symlink() or not path.is_file():
        raise ValueError(f"{description} must be an ordinary file: {path}")
    return path


def _ordinary_directory(value: str, description: str) -> Path:
    requested = Path(value).expanduser()
    path = requested.resolve()
    if requested.is_symlink() or not path.is_dir():
        raise ValueError(f"{description} must be an ordinary directory: {path}")
    return path


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"output already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    if temporary.exists() or temporary.is_symlink():
        raise FileExistsError(f"temporary output already exists: {temporary}")
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError("atomic JSON round-trip differs")
    except BaseException:
        if temporary.exists():
            temporary.unlink()
        raise


class _PytestResults:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[dict[str, str]] = []
        self.skipped: list[dict[str, str]] = []
        self.xfailed: list[str] = []
        self.xpassed: list[str] = []
        self.collection_errors: list[dict[str, str]] = []

    @staticmethod
    def _longrepr(report: Any) -> str:
        return str(getattr(report, "longrepr", ""))

    def pytest_collectreport(self, report: Any) -> None:
        if bool(getattr(report, "failed", False)):
            self.collection_errors.append(
                {
                    "nodeid": str(getattr(report, "nodeid", "collection")),
                    "detail": self._longrepr(report),
                }
            )

    def pytest_runtest_logreport(self, report: Any) -> None:
        nodeid = str(report.nodeid)
        wasxfail = getattr(report, "wasxfail", None)
        if report.when == "setup":
            if report.skipped:
                self.skipped.append(
                    {"nodeid": nodeid, "detail": self._longrepr(report)}
                )
            elif report.failed:
                self.failed.append(
                    {"nodeid": nodeid, "detail": self._longrepr(report)}
                )
            return
        if report.when == "teardown":
            if report.failed:
                self.failed.append(
                    {"nodeid": nodeid, "detail": self._longrepr(report)}
                )
            return
        if wasxfail is not None:
            (self.xfailed if report.skipped else self.xpassed).append(nodeid)
        elif report.passed:
            self.passed.append(nodeid)
        elif report.failed:
            self.failed.append(
                {"nodeid": nodeid, "detail": self._longrepr(report)}
            )
        elif report.skipped:
            self.skipped.append(
                {"nodeid": nodeid, "detail": self._longrepr(report)}
            )

    def to_metadata(self) -> dict[str, object]:
        return {
            "collected": (
                len(self.passed)
                + len(self.failed)
                + len(self.skipped)
                + len(self.xfailed)
                + len(self.xpassed)
            ),
            "passed": len(self.passed),
            "failed": self.failed,
            "skipped": self.skipped,
            "xfailed": self.xfailed,
            "xpassed": self.xpassed,
            "collection_errors": self.collection_errors,
            "passed_nodeids": sorted(self.passed),
        }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--purelib", required=True)
    parser.add_argument("--forbidden-source-root", action="append", default=[])
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--expected-tests", required=True, type=int)
    parser.add_argument("--output", required=True)
    return parser


def _version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    purelib = _ordinary_directory(arguments.purelib, "purelib")
    test_file = _ordinary_file(arguments.test_file, "test file")
    forbidden = tuple(
        _ordinary_directory(value, "forbidden source root")
        for value in arguments.forbidden_source_root
    )

    import pssolver

    imported = Path(pssolver.__file__).resolve()
    if not _is_relative_to(imported, purelib):
        raise RuntimeError("pssolver was not imported from installed-wheel purelib")
    resolved_sys_path = [
        Path(entry or os.getcwd()).expanduser().resolve()
        for entry in sys.path
        if Path(entry or os.getcwd()).exists()
    ]
    shadow_entries = [
        str(entry)
        for entry in resolved_sys_path
        if any(entry == root or _is_relative_to(entry, root) for root in forbidden)
    ]
    if shadow_entries:
        raise RuntimeError(f"source shadow paths are present: {shadow_entries}")

    import numpy
    import pytest
    import scipy
    import torch

    collector = _PytestResults()
    exit_code = pytest.main(
        ["-q", "-p", "no:cacheprovider", str(test_file)],
        plugins=[collector],
    )
    results = collector.to_metadata()
    passed = (
        int(exit_code) == 0
        and results["collected"] == arguments.expected_tests
        and results["passed"] == arguments.expected_tests
        and not results["failed"]
        and not results["skipped"]
        and not results["xfailed"]
        and not results["xpassed"]
        and not results["collection_errors"]
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "classification": (
            CLASSIFICATION
            if passed
            else "FAIL_RC4_3_6_INSTALLED_WHEEL_LIFTING_CPU_CLOSURE"
        ),
        "passed": passed,
        "installed_identity": {
            "pssolver_import": str(imported),
            "purelib": str(purelib),
            "sysconfig_purelib": str(Path(sysconfig.get_paths()["purelib"]).resolve()),
            "source_shadow_import": False,
            "forbidden_source_roots": [str(path) for path in forbidden],
        },
        "environment": {
            "python": platform.python_version(),
            "pssolver": _version("pssolver"),
            "torch": torch.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "pytest": pytest.__version__,
            "cuda_available": torch.cuda.is_available(),
        },
        "tests": results,
        "qualification_boundary": {
            "public_lifted_plane_remains_fail_closed": True,
            "localized_policy_publicly_qualified": False,
            "h100_executed": False,
        },
    }
    _atomic_json(Path(arguments.output).expanduser().resolve(), report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
