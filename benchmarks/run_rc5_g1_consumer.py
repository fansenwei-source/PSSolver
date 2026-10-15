#!/usr/bin/env python3
"""Run the frozen installed-wheel PSSolver-Control rc5 H100 gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.plan_rc5_g1_h100 import (
    CONSUMER_SOURCE,
    CONSUMER_WHEEL_SHA256,
    build_plan,
)
from benchmarks.rc458_venv_identity import validate_installed_venv_identity
from benchmarks.run_rc426_cuda_tests import Results, _enforce_tf32_policy


SCHEMA = "pssolver.rc5.g1.consumer_h100.v1"
RC4_PERIODIC_IDENTITY = (
    "f83d7b368bda23aacdf1c70e77de974df1c05a3e8b0c7886191dd82c14ade416"
)


class ConsumerResults(Results):
    def __init__(self) -> None:
        super().__init__()
        self.deselected: list[str] = []

    def pytest_deselected(self, items: list[object]) -> None:
        self.deselected.extend(str(getattr(item, "nodeid", "")) for item in items)

    def metadata(self) -> dict[str, object]:
        value = super().metadata()
        value["deselected"] = sorted(self.deselected)
        return value


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
            raise RuntimeError("Consumer JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _tree(directory: Path) -> dict[str, str]:
    return {
        path.relative_to(directory).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    }


def _capture_identities(consumer_root: Path, work_root: Path) -> dict[str, object]:
    import torch
    from pssolver_control.adapters.pssolver import build_periodic_activity_stepper
    from pssolver_control.adapters.pssolver_channel import (
        build_channel_activity_stepper,
    )
    from pssolver_control.exceptions import PSSolverIntegrationUnavailable

    tests = consumer_root / "tests"
    sys.path.insert(0, str(tests))
    try:
        import channel_runtime
        import periodic_runtime

        periodic = build_periodic_activity_stepper(
            periodic_runtime.simulation(work_root / "periodic").specification
        )
        fixed = build_channel_activity_stepper(
            channel_runtime.simulation(work_root / "channel_fixed").specification
        )
        tolerance = build_channel_activity_stepper(
            channel_runtime.simulation(
                work_root / "channel_tolerance",
                pressure_solver=channel_runtime.TOLERANCE_PRESSURE_SOLVER,
            ).specification
        )

        negative_root = work_root / "old_pin_negative"
        specification = periodic_runtime.simulation(negative_root).specification
        before = _tree(negative_root)
        try:
            build_periodic_activity_stepper(
                specification,
                expected_identity_sha256=RC4_PERIODIC_IDENTITY,
            )
        except PSSolverIntegrationUnavailable as error:
            old_pin = {
                "rejected": True,
                "exception_type": type(error).__name__,
                "message": str(error),
                "target_unchanged": _tree(negative_root) == before,
            }
        else:
            old_pin = {
                "rejected": False,
                "exception_type": None,
                "message": None,
                "target_unchanged": _tree(negative_root) == before,
            }
        identities = {
            "periodic": periodic.identity()["provider_identity_sha256"],
            "channel_fixed": fixed.identity()["provider_identity_sha256"],
            "channel_tolerance": tolerance.identity()[
                "provider_identity_sha256"
            ],
        }
        controls = torch.full(
            (1, *periodic_runtime.SHAPE),
            periodic_runtime.ACTIVITY,
            dtype=periodic_runtime.F64,
            device=periodic.initial_state()[0].device,
        )
        stepped = periodic.step(periodic.initial_state(), controls, 0)
        finite = all(bool(torch.isfinite(value).all().item()) for value in stepped)
        return {"identities": identities, "old_pin": old_pin, "finite_smoke": finite}
    finally:
        sys.path.remove(str(tests))


def summarize(
    *,
    exit_code: int,
    outcomes: dict[str, object],
    identity_report: dict[str, object],
    tf32_policy: dict[str, object],
) -> dict[str, object]:
    contract = build_plan()["consumer_h100_gate"]
    lists: dict[str, list[object]] = {}
    for name in (
        "passed",
        "failed",
        "skipped",
        "xfailed",
        "xpassed",
        "collection_errors",
        "deselected",
    ):
        value = outcomes.get(name)
        if not isinstance(value, list):
            raise TypeError(f"Consumer outcome {name} is missing")
        lists[name] = value
    skipped_node_ids = [
        str(value.get("nodeid"))
        for value in lists["skipped"]
        if isinstance(value, dict)
    ]
    identities = identity_report.get("identities")
    old_pin = identity_report.get("old_pin")
    if not isinstance(identities, dict) or not isinstance(old_pin, dict):
        raise TypeError("Consumer identity report is incomplete")
    expected_identities = {
        "periodic": contract["periodic_identity_sha256"],
        "channel_fixed": contract["channel_fixed_identity_sha256"],
        "channel_tolerance": contract["channel_tolerance_identity_sha256"],
    }
    if exit_code != 0:
        raise ValueError("Consumer pytest exit code differs")
    if len(lists["passed"]) != contract["expected_passed"]:
        raise ValueError("Consumer passed count differs")
    if skipped_node_ids != [contract["allowed_skip"]]:
        raise ValueError("Consumer skipped node differs")
    if any(
        lists[name]
        for name in (
            "failed",
            "xfailed",
            "xpassed",
            "collection_errors",
            "deselected",
        )
    ):
        raise ValueError("Consumer suite contains a forbidden outcome")
    if identities != expected_identities:
        raise ValueError("Consumer Provider identities differ")
    if old_pin.get("rejected") is not True or old_pin.get("target_unchanged") is not True:
        raise ValueError("old Periodic pin negative gate failed")
    if identity_report.get("finite_smoke") is not True:
        raise ValueError("Consumer finite smoke failed")
    enforcement = tf32_policy.get("enforcement")
    if not isinstance(enforcement, dict) or enforcement.get("passed") is not True:
        raise ValueError("Consumer TF32 policy enforcement failed")
    return {
        "schema": SCHEMA,
        "passed": True,
        "source_commit": CONSUMER_SOURCE,
        "wheel_sha256": CONSUMER_WHEEL_SHA256,
        "passed_count": len(lists["passed"]),
        "skipped_count": len(lists["skipped"]),
        "failed_count": 0,
        "xfailed_count": 0,
        "xpassed_count": 0,
        "deselected_count": 0,
        "collection_error_count": 0,
        "skip_node_id": skipped_node_ids[0],
        "identities": identities,
        "old_periodic_pin_rejected_before_mutation": True,
        "identity_report": identity_report,
        "tf32_policy": tf32_policy,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    import pssolver
    import pssolver_control
    import pytest

    consumer_root = args.consumer_support.expanduser().resolve()
    provider_root = args.provider_reference.expanduser().resolve()
    work_root = args.work_root.expanduser().absolute()
    if not consumer_root.is_dir() or consumer_root.is_symlink():
        raise FileNotFoundError(f"ordinary Consumer support root required: {consumer_root}")
    if not provider_root.is_dir() or provider_root.is_symlink():
        raise FileNotFoundError(f"ordinary Provider reference root required: {provider_root}")
    if work_root.exists() or work_root.is_symlink():
        raise FileExistsError(f"Consumer work root already exists: {work_root}")
    work_root.mkdir(parents=True)
    if not (consumer_root / "tests").is_dir() or not (consumer_root / "src").is_dir():
        raise FileNotFoundError("Consumer support layout is incomplete")
    if not (provider_root / "pssolver").is_dir():
        raise FileNotFoundError("Provider reference source tree is incomplete")

    identity = validate_installed_venv_identity(
        requested_python=Path(sys.executable),
        imported_module=Path(pssolver.__file__),
    )
    consumer_identity = validate_installed_venv_identity(
        requested_python=Path(sys.executable),
        imported_module=Path(pssolver_control.__file__),
    )
    tf32_policy = _enforce_tf32_policy()
    enforcement = tf32_policy["enforcement"]
    if not isinstance(enforcement, dict) or enforcement.get("passed") is not True:
        raise RuntimeError("cannot enforce Consumer TF32-off policy")

    previous_reference = os.environ.get("PSSOLVER_REFERENCE_ROOT")
    previous_benchmark = os.environ.pop("PSSOLVER_CONTROL_BENCHMARK", None)
    os.environ["PSSOLVER_REFERENCE_ROOT"] = str(provider_root)
    collector = ConsumerResults()
    try:
        exit_code = pytest.main(
            [
                "-q",
                "-c",
                "/dev/null",
                "--rootdir",
                str(consumer_root),
                "-p",
                "no:cacheprovider",
                str(consumer_root / "tests"),
            ],
            plugins=[collector],
        )
    finally:
        if previous_reference is None:
            os.environ.pop("PSSOLVER_REFERENCE_ROOT", None)
        else:
            os.environ["PSSOLVER_REFERENCE_ROOT"] = previous_reference
        if previous_benchmark is not None:
            os.environ["PSSOLVER_CONTROL_BENCHMARK"] = previous_benchmark
    identity_report = _capture_identities(
        consumer_root,
        Path(tempfile.mkdtemp(prefix="identity_", dir=work_root)),
    )
    result = summarize(
        exit_code=int(exit_code),
        outcomes=collector.metadata(),
        identity_report=identity_report,
        tf32_policy=tf32_policy,
    )
    result["installed_provider_identity"] = identity
    result["installed_consumer_identity"] = consumer_identity
    result["provider_reference_root"] = str(provider_root)
    result["consumer_support_root"] = str(consumer_root)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--consumer-support", type=Path, required=True)
    parser.add_argument("--provider-reference", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
