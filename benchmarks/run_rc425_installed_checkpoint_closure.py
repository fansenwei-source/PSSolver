"""Run the RC4.2.5 checkpoint/consumer matrix from installed wheels only."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig
from typing import Any


CLASSIFICATION = "PASS_RC4_2_5_INSTALLED_WHEEL_CPU_CLOSURE"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    payload = json.dumps(
        value,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        if loaded != value:
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
    parser.add_argument("--provider-purelib", required=True)
    parser.add_argument("--consumer-purelib", required=True)
    parser.add_argument("--forbidden-source-root", action="append", default=[])
    parser.add_argument("--provider-test", action="append", default=[])
    parser.add_argument("--consumer-test", action="append", default=[])
    parser.add_argument("--expected-provider-tests", required=True, type=int)
    parser.add_argument("--expected-consumer-tests", required=True, type=int)
    parser.add_argument("--output", required=True)
    return parser


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    provider_purelib = _ordinary_directory(
        arguments.provider_purelib,
        "provider purelib",
    )
    consumer_purelib = _ordinary_directory(
        arguments.consumer_purelib,
        "consumer purelib",
    )
    forbidden = tuple(
        _ordinary_directory(value, "forbidden source root")
        for value in arguments.forbidden_source_root
    )
    provider_tests = tuple(
        _ordinary_file(value, "provider test")
        for value in arguments.provider_test
    )
    consumer_tests = tuple(
        _ordinary_file(value, "consumer test")
        for value in arguments.consumer_test
    )
    if not provider_tests or not consumer_tests:
        raise ValueError("both provider and consumer test sets are required")

    import pssolver
    import pssolver_control
    from pssolver.functional import (
        CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION,
        PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION,
    )
    from pssolver.functional.api import (
        FUNCTIONAL_API_VERSION,
        FUNCTIONAL_COMPATIBILITY_POLICY_VERSION,
    )
    from pssolver_control.adapters.pssolver import (
        SUPPORTED_FUNCTIONAL_API_VERSION,
    )

    provider_import = Path(pssolver.__file__).resolve()
    consumer_import = Path(pssolver_control.__file__).resolve()
    if not _is_relative_to(provider_import, provider_purelib):
        raise RuntimeError("pssolver was not imported from provider purelib")
    if not _is_relative_to(consumer_import, consumer_purelib):
        raise RuntimeError("pssolver_control was not imported from consumer purelib")
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
    if FUNCTIONAL_API_VERSION != "1.0":
        raise RuntimeError("functional API version differs")
    if SUPPORTED_FUNCTIONAL_API_VERSION != FUNCTIONAL_API_VERSION:
        raise RuntimeError("consumer functional API version differs")
    if FUNCTIONAL_COMPATIBILITY_POLICY_VERSION != 3:
        raise RuntimeError("functional compatibility policy version differs")
    if PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION != 3:
        raise RuntimeError("Periodic functional bridge version differs")
    if CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION != 3:
        raise RuntimeError("Channel functional bridge version differs")

    import pytest

    collector = _PytestResults()
    exit_code = pytest.main(
        [
            "-q",
            "-p",
            "no:cacheprovider",
            *(str(path) for path in provider_tests),
            *(str(path) for path in consumer_tests),
        ],
        plugins=[collector],
    )
    results = collector.to_metadata()
    expected = arguments.expected_provider_tests + arguments.expected_consumer_tests
    passed = (
        int(exit_code) == 0
        and results["collected"] == expected
        and results["passed"] == expected
        and not results["failed"]
        and not results["skipped"]
        and not results["xfailed"]
        and not results["xpassed"]
        and not results["collection_errors"]
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "classification": CLASSIFICATION if passed else "FAIL_RC4_2_5_INSTALLED_WHEEL_CPU_CLOSURE",
        "passed": passed,
        "installed_identity": {
            "provider_import": str(provider_import),
            "provider_purelib": str(provider_purelib),
            "consumer_import": str(consumer_import),
            "consumer_purelib": str(consumer_purelib),
            "source_shadow_import": False,
            "forbidden_source_roots": [str(path) for path in forbidden],
        },
        "protocol": {
            "functional_api_version": FUNCTIONAL_API_VERSION,
            "consumer_supported_functional_api_version": (
                SUPPORTED_FUNCTIONAL_API_VERSION
            ),
            "functional_compatibility_policy_version": (
                FUNCTIONAL_COMPATIBILITY_POLICY_VERSION
            ),
            "periodic_functional_bridge_format_version": (
                PERIODIC_FUNCTIONAL_BRIDGE_FORMAT_VERSION
            ),
            "channel_functional_bridge_format_version": (
                CHANNEL_FUNCTIONAL_BRIDGE_FORMAT_VERSION
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "executable": str(Path(sys.executable).resolve()),
            "purelib": sysconfig.get_paths()["purelib"],
            "torch": _distribution_version("torch"),
            "numpy": _distribution_version("numpy"),
            "scipy": _distribution_version("scipy"),
            "pytest": _distribution_version("pytest"),
            "pssolver": _distribution_version("pssolver"),
            "pssolver_control": _distribution_version("pssolver-control"),
        },
        "test_inputs": {
            "provider": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in provider_tests
            ],
            "consumer": [
                {"path": str(path), "sha256": _sha256(path)}
                for path in consumer_tests
            ],
            "expected_provider_tests": arguments.expected_provider_tests,
            "expected_consumer_tests": arguments.expected_consumer_tests,
        },
        "pytest": results,
    }
    _atomic_json(Path(arguments.output).expanduser().resolve(), report)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
