#!/usr/bin/env python3
"""Run the S-09 Control suite against installed Provider/Consumer wheels."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import sysconfig


SCHEMA = "pssolver.v0_3.s09.installed_consumer_cpu.v1"
ALLOWED_SKIPS = {
    "tests/test_checkpoint_memory.py::test_reverse_sweep_holds_one_segment[20]",
    "tests/test_checkpoint_memory.py::test_reverse_sweep_holds_one_segment[100]",
    "tests/test_gpu_optional.py::test_cuda_adjoint_matches_autograd_with_cpu_checkpoints",
    "tests/test_gpu_optional.py::test_cuda_and_cpu_toy_agree_to_rounding",
    "tests/test_gpu_optional.py::test_cpu_parameters_on_cuda_problem_fail_explicitly",
    "tests/test_warmup_experiments.py::test_r5_benchmark",
}


class Results:
    def __init__(self) -> None:
        self.passed: list[str] = []
        self.failed: list[dict[str, str]] = []
        self.skipped: list[dict[str, str]] = []
        self.xfailed: list[str] = []
        self.xpassed: list[str] = []
        self.collection_errors: list[str] = []

    def pytest_runtest_logreport(self, report) -> None:
        if report.when != "call":
            if report.failed:
                self.failed.append({"nodeid": report.nodeid, "longrepr": str(report.longrepr)})
            elif report.skipped:
                # Module/class/function-level skip conditions are reported in
                # the setup phase rather than the call phase.
                if hasattr(report, "wasxfail"):
                    self.xfailed.append(report.nodeid)
                else:
                    self.skipped.append(
                        {"nodeid": report.nodeid, "reason": str(report.longrepr)}
                    )
            return
        if report.passed:
            if hasattr(report, "wasxfail"):
                self.xpassed.append(report.nodeid)
            else:
                self.passed.append(report.nodeid)
        elif report.skipped:
            if hasattr(report, "wasxfail"):
                self.xfailed.append(report.nodeid)
            else:
                self.skipped.append({"nodeid": report.nodeid, "reason": str(report.longrepr)})
        elif report.failed:
            self.failed.append({"nodeid": report.nodeid, "longrepr": str(report.longrepr)})

    def pytest_collectreport(self, report) -> None:
        if report.failed:
            self.collection_errors.append(str(report.longrepr))

    def metadata(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "xfailed": self.xfailed,
            "xpassed": self.xpassed,
            "collection_errors": self.collection_errors,
        }


def _atomic_json(path: Path, value: object) -> None:
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


def execute(args: argparse.Namespace) -> dict[str, object]:
    import importlib.metadata
    import pssolver
    import pssolver_control
    import pytest

    support = args.support_root.expanduser().resolve()
    provider = args.provider_reference.expanduser().resolve()
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    provider_import = Path(pssolver.__file__).resolve()
    consumer_import = Path(pssolver_control.__file__).resolve()
    if not provider_import.is_relative_to(purelib) or not consumer_import.is_relative_to(purelib):
        raise RuntimeError("installed-wheel import gate failed")
    for relative in ("tests", "src/pssolver_control", "docs"):
        if not (support / relative).is_dir():
            raise RuntimeError(f"Consumer archive support is missing {relative}")
    support_source = (support / "src").resolve()
    if any(
        Path(value or os.curdir).expanduser().resolve() == support_source
        for value in sys.path
    ):
        raise RuntimeError("Consumer archive source directory is present on sys.path")
    if not (provider / "pssolver" / "__init__.py").is_file():
        raise FileNotFoundError("Provider reference source is missing")
    previous = os.environ.get("PSSOLVER_REFERENCE_ROOT")
    benchmark = os.environ.pop("PSSOLVER_CONTROL_BENCHMARK", None)
    os.environ["PSSOLVER_REFERENCE_ROOT"] = str(provider)
    collector = Results()
    junit = args.junit.expanduser().absolute()
    try:
        exit_code = int(
            pytest.main(
                [
                    "-q",
                    "-c", "/dev/null",
                    "--rootdir", str(support),
                    # The source suite intentionally imports conftest.py,
                    # periodic_runtime.py, and channel_runtime.py as top-level
                    # test helpers.  Prepend only the tests directory; the
                    # archive's src directory remains absent from sys.path and
                    # the installed package was imported above.
                    "--import-mode=prepend",
                    "-p", "no:cacheprovider",
                    f"--junitxml={junit}",
                    str(support / "tests"),
                ],
                plugins=[collector],
            )
        )
    finally:
        if previous is None:
            os.environ.pop("PSSOLVER_REFERENCE_ROOT", None)
        else:
            os.environ["PSSOLVER_REFERENCE_ROOT"] = previous
        if benchmark is not None:
            os.environ["PSSOLVER_CONTROL_BENCHMARK"] = benchmark
    outcomes = collector.metadata()
    skips = {str(value["nodeid"]) for value in outcomes["skipped"]}
    passed = len(outcomes["passed"])
    post_provider_import = Path(pssolver.__file__).resolve()
    post_consumer_import = Path(pssolver_control.__file__).resolve()
    failure_reasons: list[str] = []
    if exit_code != 0:
        failure_reasons.append(f"pytest exit code is {exit_code}")
    if passed != args.expected_passed:
        failure_reasons.append(
            f"passed count is {passed}, expected {args.expected_passed}"
        )
    if skips != ALLOWED_SKIPS:
        failure_reasons.append(f"skip identities differ: {sorted(skips)}")
    for key in ("failed", "xfailed", "xpassed", "collection_errors"):
        if outcomes[key]:
            failure_reasons.append(f"outcome contains {key}")
    if post_provider_import != provider_import or post_consumer_import != consumer_import:
        failure_reasons.append("installed-wheel import identity changed during pytest")
    result = {
        "schema": SCHEMA,
        "passed": not failure_reasons,
        "pytest_exit_code": exit_code,
        "failure_reasons": failure_reasons,
        "passed_count": passed,
        "skipped_count": len(skips),
        "skip_node_ids": sorted(skips),
        "failed_count": 0,
        "xfailed_count": 0,
        "xpassed_count": 0,
        "collection_error_count": 0,
        "pssolver_version": importlib.metadata.version("pssolver"),
        "pssolver_control_version": importlib.metadata.version("pssolver-control"),
        "pssolver_import": str(provider_import),
        "pssolver_control_import": str(consumer_import),
        "purelib": str(purelib),
        "source_shadow_import": False,
        "support_root": str(support),
        "support_source_on_sys_path": False,
        "sys_path": list(sys.path),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--provider-reference", type=Path, required=True)
    parser.add_argument("--expected-passed", type=int, default=351)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = execute(args)
    _atomic_json(args.output.expanduser().absolute(), result)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
