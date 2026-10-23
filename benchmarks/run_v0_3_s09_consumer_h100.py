#!/usr/bin/env python3
"""Run the independent installed-wheel Control gate before S-09 profiling."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import tempfile

from benchmarks.plan_v0_3_s09_h100 import (
    CANDIDATE_COMMIT,
    CANDIDATE_WHEEL_SHA256,
    CONSUMER_RUNTIME_COMMIT,
    CONSUMER_WHEEL_SHA256,
    build_plan,
)
from benchmarks.run_rc6_consumer import _execution_on_device
from benchmarks.run_v0_3_s09_installed_consumer_cpu import Results


SCHEMA = "pssolver.v0_3.s09.consumer_h100.v1"


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


def _finite_smokes(root: Path, work: Path) -> tuple[dict[str, bool], dict[str, str]]:
    import torch
    from pssolver_control.adapters.pssolver import build_periodic_activity_stepper
    from pssolver_control.adapters.pssolver_channel import build_channel_activity_stepper

    tests = root / "tests"
    sys.path.insert(0, str(tests))
    try:
        import channel_runtime
        import periodic_runtime

        finite: dict[str, bool] = {}
        identities: dict[str, str] = {}
        for name, module, builder in (
            ("periodic", periodic_runtime, build_periodic_activity_stepper),
            ("channel", channel_runtime, build_channel_activity_stepper),
        ):
            simulation = module.simulation(work / name)
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
            with stepper.execution_scope():
                evolved = stepper.step(state, controls, 0)
                cost = sum(value.square().mean() for value in evolved)
                gradient = torch.autograd.grad(cost, controls)[0]
            observed = stepper.observe(evolved, controls)
            finite[name] = (
                str(state[0].device) == "cuda:0"
                and all(bool(torch.isfinite(value).all().item()) for value in evolved)
                and all(bool(torch.isfinite(value).all().item()) for value in observed.values())
                and bool(torch.isfinite(gradient).all().item())
            )
            identities[name] = str(stepper.identity()["provider_identity_sha256"])
        return finite, identities
    finally:
        sys.path.remove(str(tests))


def run(args: argparse.Namespace) -> dict[str, object]:
    import pssolver
    import pssolver_control
    import pytest
    import torch

    plan = build_plan()
    contract = plan["consumer_h100_gate"]
    support = args.support_root.expanduser().resolve()
    provider = args.provider_reference.expanduser().resolve()
    if "site-packages" not in str(Path(pssolver.__file__).resolve()):
        raise RuntimeError("Consumer H100 Provider is not an installed wheel")
    if "site-packages" not in str(Path(pssolver_control.__file__).resolve()):
        raise RuntimeError("Consumer H100 package is not an installed wheel")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Consumer H100 gate requires one CUDA device")
    previous = os.environ.get("PSSOLVER_REFERENCE_ROOT")
    benchmark = os.environ.pop("PSSOLVER_CONTROL_BENCHMARK", None)
    os.environ["PSSOLVER_REFERENCE_ROOT"] = str(provider)
    collector = Results()
    try:
        exit_code = int(
            pytest.main(
                [
                    "-q", "-c", "/dev/null", "--rootdir", str(support),
                    "--import-mode=prepend", "-p", "no:cacheprovider",
                    f"--junitxml={args.junit.expanduser().absolute()}",
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
    skip_ids = [row["nodeid"] for row in outcomes["skipped"]]
    if exit_code != 0 or len(outcomes["passed"]) != contract["expected_passed"]:
        raise RuntimeError("Consumer H100 pytest outcome differs")
    if skip_ids != [contract["allowed_skip"]]:
        raise RuntimeError(f"Consumer H100 skip identity differs: {skip_ids}")
    for name in ("failed", "xfailed", "xpassed", "collection_errors"):
        if outcomes[name]:
            raise RuntimeError(f"Consumer H100 suite contains {name}")
    with tempfile.TemporaryDirectory(prefix="pssolver-s09-consumer-") as temporary:
        finite, identities = _finite_smokes(support, Path(temporary))
    expected_identities = {
        "periodic": contract["periodic_identity_sha256"],
        "channel": contract["channel_tolerance_identity_sha256"],
    }
    if finite != {"periodic": True, "channel": True} or identities != expected_identities:
        raise RuntimeError("Consumer CUDA smoke or Provider identity differs")
    return {
        "schema": SCHEMA,
        "name": "independent_control_consumer",
        "provider_candidate_commit": CANDIDATE_COMMIT,
        "candidate_wheel_sha256": CANDIDATE_WHEEL_SHA256,
        "consumer_source_commit": CONSUMER_RUNTIME_COMMIT,
        "consumer_wheel_sha256": CONSUMER_WHEEL_SHA256,
        "installed_wheels_only": True,
        "functional_api_1_1": True,
        "provider_identity_migrated": True,
        "periodic_gradient": True,
        "channel_gradient": True,
        "checkpointed_adjoint": True,
        "negative_controls": True,
        "finite_cuda_smokes": True,
        "passed_count": len(outcomes["passed"]),
        "skipped_count": len(skip_ids),
        "skip_node_ids": skip_ids,
        "identities": identities,
        "passed": True,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--support-root", type=Path, required=True)
    parser.add_argument("--provider-reference", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, run(args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
