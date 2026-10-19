#!/usr/bin/env python3
"""Run the frozen RC6 PSSolver-Control installed-wheel H100 gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.plan_rc6_single_h100_qualification import (
    CONSUMER_SOURCE,
    CONSUMER_WHEEL_SHA256,
    build_plan,
)
from benchmarks.rc458_venv_identity import validate_installed_venv_identity
from benchmarks.run_rc426_cuda_tests import _enforce_tf32_policy
from benchmarks.run_rc5_g1_consumer import ConsumerResults, _capture_identities


SCHEMA = "pssolver.rc6.consumer_h100.v1"
GUARD_NODE = "tests/test_pssolver_rc6_guard_order_regression.py::test_rc6_guard_order_regression_is_closed_without_runtime_or_h100_changes"
REQUIRED_NODES = (
    "tests/test_pssolver_periodic_consumer.py::test_real_runtime_satisfies_the_stepper_contract",
    "tests/test_pssolver_channel_consumer.py::test_channel_runtime_satisfies_control_stepper_contract",
    "tests/test_pssolver_periodic_consumer.py::test_gradient_matches_full_history_autograd",
    "tests/test_pssolver_channel_consumer.py::test_channel_checkpointed_adjoint_matches_full_history",
)


def _execution_on_device(execution: object, device: str):
    """Rebuild a public Torch execution request for one concrete device.

    ``TorchSpectralExecution`` has a deliberately narrower public constructor
    than its ``ExecutionSpec`` dataclass base.  In particular, replaying the
    inherited dataclass fields with ``dataclasses.replace`` passes ``backend``
    to that constructor and fails before the Consumer runtime is built.
    """
    from pssolver import TorchSpectralExecution

    if not isinstance(execution, TorchSpectralExecution):
        raise TypeError("Consumer execution must be TorchSpectralExecution")
    if execution.backend != "torch_spectral":
        raise ValueError("Consumer execution backend differs")
    if not isinstance(device, str) or not device:
        raise ValueError("target device must be a non-empty string")

    source_metadata = execution.to_metadata()
    options = dict(execution.options)
    source_device = options.pop("device", None)
    fallback_allowed = options.pop("fallback_allowed", None)
    if not isinstance(source_device, str) or not source_device:
        raise ValueError("Consumer execution source device is missing")
    if fallback_allowed is not False:
        raise ValueError("Consumer execution must prohibit fallback")

    rebound = TorchSpectralExecution(
        runtime_path=execution.runtime_path,
        device=device,
        options=options,
    )
    expected_metadata = dict(source_metadata)
    expected_options = dict(source_metadata["options"])
    expected_options["device"] = device
    expected_metadata["options"] = expected_options
    if rebound.to_metadata() != expected_metadata:
        raise RuntimeError("Consumer execution device rebind changed metadata")
    if execution.to_metadata() != source_metadata:
        raise RuntimeError("Consumer execution source was mutated")
    return rebound


def _cuda_finite_smokes(consumer_root: Path, work_root: Path) -> dict[str, bool]:
    from dataclasses import replace
    import torch
    from pssolver_control.adapters.pssolver import build_periodic_activity_stepper
    from pssolver_control.adapters.pssolver_channel import build_channel_activity_stepper

    tests = consumer_root / "tests"
    sys.path.insert(0, str(tests))
    try:
        import channel_runtime
        import periodic_runtime

        results: dict[str, bool] = {}
        for name, module, builder in (
            ("periodic", periodic_runtime, build_periodic_activity_stepper),
            ("channel", channel_runtime, build_channel_activity_stepper),
        ):
            simulation = module.simulation(work_root / name)
            simulation = replace(
                simulation,
                execution=_execution_on_device(simulation.execution, "cuda"),
            )
            stepper = builder(simulation.specification)
            state = stepper.initial_state()
            controls = torch.full(
                (1, *module.SHAPE),
                module.ACTIVITY,
                dtype=module.F64,
                device=state[0].device,
            )
            evolved = stepper.step(state, controls, 0)
            observed = stepper.observe(evolved, controls)
            results[name] = (
                str(state[0].device) == "cuda:0"
                and all(bool(torch.isfinite(value).all().item()) for value in evolved)
                and all(bool(torch.isfinite(value).all().item()) for value in observed.values())
            )
        return results
    finally:
        sys.path.remove(str(tests))


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


def summarize(*, exit_code: int, outcomes: dict[str, object], identity_report: dict[str, object], tf32_policy: dict[str, object]) -> dict[str, object]:
    plan = build_plan()
    contract = plan["consumer_h100_gate"]
    identity = plan["consumer"]
    lists: dict[str, list[object]] = {}
    for name in ("passed", "failed", "skipped", "xfailed", "xpassed", "collection_errors", "deselected"):
        value = outcomes.get(name)
        if not isinstance(value, list):
            raise TypeError(f"Consumer outcome missing: {name}")
        lists[name] = value
    passed = {str(value) for value in lists["passed"]}
    skipped = [str(value.get("nodeid")) for value in lists["skipped"] if isinstance(value, dict)]
    if exit_code != 0 or len(passed) != contract["expected_passed"]:
        raise ValueError("Consumer pytest result differs")
    if skipped != [contract["allowed_skip"]]:
        raise ValueError("Consumer skip identity differs")
    if any(lists[name] for name in ("failed", "xfailed", "xpassed", "collection_errors", "deselected")):
        raise ValueError("Consumer suite contains a forbidden outcome")
    expected_identities = {
        "periodic": identity["periodic_identity_sha256"],
        "channel_fixed": identity["channel_fixed_identity_sha256"],
        "channel_tolerance": identity["channel_tolerance_identity_sha256"],
    }
    if identity_report.get("identities") != expected_identities:
        raise ValueError("Consumer Provider identities differ")
    old_pin = identity_report.get("old_pin")
    if not isinstance(old_pin, dict) or old_pin.get("rejected") is not True or old_pin.get("target_unchanged") is not True:
        raise ValueError("old Provider identity negative gate failed")
    if identity_report.get("finite_smoke") is not True:
        raise ValueError("Periodic CUDA smoke failed")
    cuda_smokes = identity_report.get("cuda_finite_smokes")
    if cuda_smokes != {"periodic": True, "channel": True}:
        raise ValueError("Periodic/Channel CUDA finite smokes failed")
    if GUARD_NODE not in passed or not set(REQUIRED_NODES).issubset(passed):
        raise ValueError("Consumer semantic gate nodes are missing")
    enforcement = tf32_policy.get("enforcement")
    if not isinstance(enforcement, dict) or enforcement.get("passed") is not True:
        raise ValueError("Consumer TF32 policy enforcement failed")
    return {
        "schema": SCHEMA,
        "passed": True,
        "source_commit": CONSUMER_SOURCE,
        "wheel_sha256": CONSUMER_WHEEL_SHA256,
        "passed_count": len(passed),
        "skipped_count": 1,
        "failed_count": 0,
        "xfailed_count": 0,
        "xpassed_count": 0,
        "deselected_count": 0,
        "collection_error_count": 0,
        "skip_node_id": skipped[0],
        "identities": expected_identities,
        "guard_order_regression_passed": True,
        "periodic_and_channel_cuda_execution_finite": True,
        "checkpoint_replay_and_gradient_gates_passed": True,
        "scientific_control_result": False,
        "cuda_finite_smokes": cuda_smokes,
        "old_periodic_pin_rejected_before_mutation": True,
        "tf32_policy": tf32_policy,
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    import pssolver
    import pssolver_control
    import pytest

    consumer_root = args.consumer_support.expanduser().resolve()
    provider_root = args.provider_reference.expanduser().resolve()
    work_root = args.work_root.expanduser().absolute()
    if not consumer_root.is_dir() or consumer_root.is_symlink() or not (consumer_root / "tests").is_dir():
        raise FileNotFoundError("ordinary Consumer support root required")
    if not provider_root.is_dir() or provider_root.is_symlink() or not (provider_root / "pssolver").is_dir():
        raise FileNotFoundError("ordinary Provider reference root required")
    if work_root.exists() or work_root.is_symlink():
        raise FileExistsError(f"Consumer work root already exists: {work_root}")
    work_root.mkdir(parents=True)
    provider_identity = validate_installed_venv_identity(requested_python=Path(sys.executable), imported_module=Path(pssolver.__file__))
    consumer_identity = validate_installed_venv_identity(requested_python=Path(sys.executable), imported_module=Path(pssolver_control.__file__))
    if provider_identity.get("passed") is not True or consumer_identity.get("passed") is not True:
        raise RuntimeError("installed-wheel identity failed")
    tf32_policy = _enforce_tf32_policy()
    previous_reference = os.environ.get("PSSOLVER_REFERENCE_ROOT")
    previous_benchmark = os.environ.pop("PSSOLVER_CONTROL_BENCHMARK", None)
    os.environ["PSSOLVER_REFERENCE_ROOT"] = str(provider_root)
    collector = ConsumerResults()
    try:
        exit_code = int(pytest.main(["-q", "-c", "/dev/null", "--rootdir", str(consumer_root), "-p", "no:cacheprovider", str(consumer_root / "tests")], plugins=[collector]))
    finally:
        if previous_reference is None:
            os.environ.pop("PSSOLVER_REFERENCE_ROOT", None)
        else:
            os.environ["PSSOLVER_REFERENCE_ROOT"] = previous_reference
        if previous_benchmark is not None:
            os.environ["PSSOLVER_CONTROL_BENCHMARK"] = previous_benchmark
    identity_report = _capture_identities(consumer_root, Path(tempfile.mkdtemp(prefix="identity_", dir=work_root)))
    identity_report["cuda_finite_smokes"] = _cuda_finite_smokes(
        consumer_root,
        Path(tempfile.mkdtemp(prefix="cuda_finite_", dir=work_root)),
    )
    result = summarize(exit_code=exit_code, outcomes=collector.metadata(), identity_report=identity_report, tf32_policy=tf32_policy)
    result["installed_provider_identity"] = provider_identity
    result["installed_consumer_identity"] = consumer_identity
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
