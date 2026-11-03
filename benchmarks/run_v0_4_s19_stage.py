#!/usr/bin/env python3
"""Installed-wheel S-19 stage entry points shared by CPU rehearsal and H100.

Frozen S-09 fixture implementations remain byte-for-byte unchanged. Their
reports are rebound explicitly to S-19 identities; no numerical policy or
runtime function is replaced. This script never submits a scheduler job.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import sysconfig
import tempfile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks._artifact_io import atomic_write_json, file_sha256
from benchmarks._s19_compare import HardQualificationError
from benchmarks.plan_v0_4_s19_h100 import (
    CANDIDATE_COMMIT, CANDIDATE_WHEEL_SHA256, CONSUMER_RUNTIME_COMMIT,
    CONSUMER_WHEEL_SHA256, build_plan,
)
from benchmarks.run_v0_4_s19_self_compare import _installed_identity

CHECKPOINT_TESTS = (
    "tests/test_rc5_legacy_functional_checkpoints.py",
    "tests/test_rc5_r53_legacy_channel_checkpoint.py",
    "tests/test_v0_3_s08_historical_cpu_coverage.py",
    "tests/test_v0_3_s05_checkpoint_metadata_integrity.py",
    "tests/test_v0_2_0_checkpoint_fixtures.py",
)


def common(args):
    import torch
    from benchmarks.run_rc5_g1_numerical import _tf32_off
    identity = _installed_identity("candidate", args.provider_wheel)
    if args.device == "cpu":
        if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
            raise HardQualificationError("CPU rehearsal must hide CUDA")
    elif (not torch.cuda.is_available() or torch.cuda.device_count() != 1
          or "H100" not in torch.cuda.get_device_name(0)
          or str(torch.empty((), device="cuda").device) != "cuda:0"):
        raise HardQualificationError("expected one allocated cuda:0 H100")
    for path in (args.support, args.consumer_support):
        if path is not None and any((path / name).exists() for name in ("pssolver", "pssolver_control")):
            raise HardQualificationError("support shadows an installed package")
    _tf32_off()
    return identity


def run_pytest(root, tests, junit, basetemp, *, import_mode='importlib'):
    import pytest
    from benchmarks.run_rc426_cuda_tests import Results
    collector = Results()
    previous = Path.cwd()
    try:
        os.chdir(root)
        code = int(pytest.main(["-q", "-c", "/dev/null", "--rootdir", str(root),
                               "-p", "no:cacheprovider", f"--import-mode={import_mode}",
                               f"--junitxml={junit}", f"--basetemp={basetemp}", *tests], plugins=[collector]))
    finally:
        os.chdir(previous)
    report = collector.metadata()
    report.update(returncode=code, junit_sha256=file_sha256(junit), junit_written=True)
    for key in ("failed", "xfailed", "xpassed", "collection_errors"):
        if report[key]:
            raise HardQualificationError(f"pytest {key}: {report[key]}")
    if code != 0:
        raise HardQualificationError(f"pytest exit status: {code}")
    return report


def cuda_tests(args):
    from benchmarks.run_rc426_cuda_tests import _enforce_tf32_policy
    policy = _enforce_tf32_policy()
    if policy["enforcement"]["passed"] is not True:
        raise HardQualificationError("CUDA test TF32 policy")
    tests = build_plan()["cuda_only_tests"]
    report = run_pytest(args.support, tests, args.output.with_suffix(".junit.xml"), args.work / "pytest")
    if args.device == "cuda":
        if report["passed"] != sorted(tests) or report["skipped"]:
            raise HardQualificationError("CUDA node outcomes differ")
    elif report["passed"] or sorted(row["nodeid"] for row in report["skipped"]) != sorted(tests):
        raise HardQualificationError("CPU-only CUDA-stage rehearsal outcome differs")
    return {"name": "installed_wheel_cuda_only_tests", "node_ids": tests,
            "passed_count": len(report["passed"]), "nonpassing_count": len(report["skipped"]),
            "junit_written": True, "outcomes": report, "tf32_policy": policy,
            "cpu_rehearsal_exclusion": args.device == "cpu"}


def long_horizon(args):
    from benchmarks import run_v0_3_s09_long_horizon as frozen
    from benchmarks.check_periodic_hermitian_stability import main as run_case
    reports = []
    horizon = 200.0 if args.device == "cuda" else 0.02
    for case, storage in frozen.CASES:
        destination = args.work / f"{case}_{storage}.json"
        code = run_case(["--case", case, "--spectral-storage", storage, "--device", args.device,
                         "--horizon", str(horizon), "--tolerance-factor", "64.0", "--output", str(destination)])
        if code != 0:
            raise HardQualificationError("Hermitian child failed")
        reports.append(json.loads(destination.read_text()))
    result = frozen.summarize(reports, device=args.device, horizon=horizon)
    result["reports"] = reports
    return result


def tf32(args):
    from benchmarks.run_v0_3_s09_tf32_backward import run
    return run(args.support, device=args.device)


def checkpoints(args):
    from benchmarks.run_v0_3_s09_checkpoints import _cross_device
    fixture_root = args.support / "tests" / "fixtures"
    before = {str(p.relative_to(fixture_root)): file_sha256(p)
              for p in sorted(fixture_root.rglob("*")) if p.is_file()}
    report = run_pytest(args.support, CHECKPOINT_TESTS, args.output.with_suffix(".junit.xml"), args.work / "pytest")
    if report["skipped"] or len(report['passed']) != 97:
        raise HardQualificationError("checkpoint suite has unexpected skips")
    after = {str(p.relative_to(fixture_root)): file_sha256(p)
             for p in sorted(fixture_root.rglob("*")) if p.is_file()}
    if before != after:
        raise HardQualificationError("read-only released checkpoint fixtures changed")
    # No replacement for cross-device CPU/CUDA execution is claimed locally.
    cells = _cross_device(args.work / "cross_device") if args.device == "cuda" else []
    if args.device == "cuda" and len(cells) != 8:
        raise HardQualificationError("cross-device coverage differs")
    return {"name": "checkpoint_restart_and_released_fixture_matrix", "current_matrix": True,
            "released_fixtures": True, "sources_unchanged": True, "fixture_sha256": before,
            "restart_exact": all(row["continuous_split_restart_equal"] is True for row in cells),
            "negative_gates": all(row[name]["rejected"] is True and row[name]["target_unchanged"] is True
                                  for row in cells for name in ("payload_tamper", "identity_mismatch")),
            "cross_device_cell_count": len(cells), "cross_device_cells": cells,
            "pytest_outcomes": report, "cpu_cross_device_excluded": args.device == "cpu"}


def consumer(args):
    import pssolver_control
    from benchmarks.run_v0_3_s09_consumer_h100 import _pinned_identities, _finite_smokes
    from benchmarks.run_v0_3_s09_installed_consumer_cpu import ALLOWED_SKIPS
    purelib = Path(sysconfig.get_path("purelib")).resolve()
    if (not Path(pssolver_control.__file__).resolve().is_relative_to(purelib)
            or importlib.metadata.version("pssolver-control") != build_plan()["consumer"]["version"]
            or file_sha256(args.consumer_wheel) != CONSUMER_WHEEL_SHA256):
        raise HardQualificationError("installed Consumer identity mismatch")
    os.environ["PSSOLVER_REFERENCE_ROOT"] = str(args.provider_reference)
    os.environ.pop("PSSOLVER_CONTROL_BENCHMARK", None)
    # Consumer conftest supplies its own test helper imports. Provider/Consumer
    # packages are already loaded from their own venv and verified again below.
    sys.path.insert(0, str(args.consumer_support / "tests"))
    report = run_pytest(args.consumer_support, [str(args.consumer_support / "tests")],
                        args.output.with_suffix(".junit.xml"), args.work / "pytest", import_mode='prepend')
    contract = build_plan()["consumer_h100_gate"]
    expected = contract["expected_passed"] if args.device == "cuda" else contract["cpu_expected_passed"]
    skips = sorted(row["nodeid"] for row in report["skipped"])
    allowed = [contract["allowed_skip"]] if args.device == "cuda" else sorted(ALLOWED_SKIPS)
    if len(report["passed"]) != expected or skips != allowed:
        raise HardQualificationError("Consumer exact test coverage differs")
    identities = _pinned_identities(args.consumer_support, args.work / "identities")
    finite, execution_identities = _finite_smokes(args.consumer_support, args.work / "execution", device=args.device)
    expected_ids = {name: contract[f"{name}_identity_sha256"]
                    for name in ("periodic", "channel_fixed", "channel_tolerance")}
    if identities != expected_ids or finite != {name: True for name in expected_ids}:
        raise HardQualificationError("Consumer fixed identity/finite/gradient smoke")
    origins = {}
    for name, module in tuple(sys.modules.items()):
        if name == "pssolver" or name.startswith("pssolver.") or name == "pssolver_control" or name.startswith("pssolver_control."):
            source = getattr(module, "__file__", None)
            if source:
                path = Path(source).resolve()
                if not path.is_relative_to(purelib):
                    raise HardQualificationError(f"Consumer source-shadow module: {name}")
                origins[name] = str(path)
    return {"name": "independent_control_consumer", "consumer_source_commit": CONSUMER_RUNTIME_COMMIT,
            "consumer_wheel_sha256": CONSUMER_WHEEL_SHA256, "passed_count": len(report["passed"]),
            "skip_node_ids": skips, "outcomes": report, "identities": identities,
            "execution_identities": execution_identities, "finite_smokes": finite,
            "module_origins": origins, "installed_wheels_only": True, "functional_api_1_1": True,
            "provider_identity_migrated": True, "periodic_gradient": True, "channel_gradient": True,
            "checkpointed_adjoint": True, "negative_controls": True, "finite_cuda_smokes": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("cuda_tests", "long_horizon", "tf32", "checkpoints", "consumer"))
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    for name in ("support", "provider-wheel", "output", "work"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("consumer-support", "provider-reference", "consumer-wheel"):
        parser.add_argument("--" + name, type=Path)
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=False)
    identity = common(args)
    report = globals()[args.stage](args)
    report.update(schema="pssolver.v0_4.s19.stage.v1", classification="PASS", passed=True,
                  provider_candidate_commit=CANDIDATE_COMMIT, candidate_wheel_sha256=CANDIDATE_WHEEL_SHA256,
                  execution_device=args.device, finite=True, fallback_used=False,
                  formal_h100_contract=args.device == "cuda", installed_identity=identity)
    atomic_write_json(args.output, report, overwrite=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
