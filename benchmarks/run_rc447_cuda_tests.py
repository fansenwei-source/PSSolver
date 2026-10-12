#!/usr/bin/env python3
"""Run the seven frozen CUDA-only tests in an installed-wheel process."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import torch


SCHEMA = "pssolver.rc4_4_7.cuda_only_tests.v1"
PASS = "PASS_RC4_4_7_SEVEN_CUDA_ONLY_TESTS"
NODE_IDS = (
    "tests/test_bounded_axis_execution_plan.py::test_cuda_plan_key_uses_the_allocated_device_identity",
    "tests/test_benchmark_bounded_axis_applicability.py::test_cuda_smoke_records_actual_device_and_memory",
    "tests/test_phase4_combined_sbdf2_modal_block.py::test_cuda_canary_binds_concrete_device_and_remains_finite",
    "tests/test_phase8_p845_h100_qualification.py::test_unindexed_cuda_request_binds_lifting_to_allocated_device_identity",
    "tests/test_phase8_p856_finite_q_h100_qualification.py::test_cuda_runtime_matches_cpu_and_binds_allocated_device",
    "tests/test_phase8_p856_finite_q_h100_qualification.py::test_cuda_continuous_split_and_file_restart_are_exact",
    "tests/test_rc4_plane_nyquist_cuda.py::test_plane_nyquist_repair_binds_allocated_cuda_device_and_storage_equivalence",
)


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        if temporary.exists():
            temporary.unlink()


class OutcomeRecorder:
    def __init__(self):
        self.collected: list[str] = []
        self.outcomes: dict[str, str] = {}

    def pytest_collection_finish(self, session) -> None:
        self.collected = [item.nodeid for item in session.items]

    def pytest_runtest_logreport(self, report) -> None:
        if report.when == "setup" and report.skipped:
            self.outcomes[report.nodeid] = "skipped"
        elif report.when == "setup" and report.failed:
            self.outcomes[report.nodeid] = "error"
        elif report.when == "call":
            self.outcomes[report.nodeid] = report.outcome


def classify(returncode: int, collected: list[str], outcomes: dict[str, str]) -> tuple[bool, list[str]]:
    problems: list[str] = []
    if returncode != 0:
        problems.append(f"pytest return code is {returncode}")
    if len(collected) != len(NODE_IDS):
        problems.append(f"collected {len(collected)} tests, expected {len(NODE_IDS)}")
    if len(set(collected)) != len(collected):
        problems.append("duplicate collected node ID")
    missing_outcome = [node for node in collected if node not in outcomes]
    if missing_outcome:
        problems.append(f"missing outcomes: {missing_outcome}")
    wrong = {node: outcome for node, outcome in outcomes.items() if outcome != "passed"}
    if wrong:
        problems.append(f"non-pass outcomes: {wrong}")
    if len(outcomes) != len(NODE_IDS):
        problems.append(f"recorded {len(outcomes)} outcomes, expected {len(NODE_IDS)}")
    return not problems, problems


def run(provider_root: Path, output: Path) -> dict[str, object]:
    import pytest
    import pssolver

    provider_root = provider_root.expanduser().resolve()
    if not provider_root.is_dir() or provider_root.is_symlink():
        raise ValueError("provider root must be an ordinary directory")
    before = {
        "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    after = {
        "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    recorder = OutcomeRecorder()
    node_paths = [str(provider_root / node) for node in NODE_IDS]
    rootdir = output.expanduser().resolve().parent / "pytest_root"
    rootdir.mkdir(parents=True, exist_ok=False)
    argv = [
        "-q",
        "-p",
        "no:cacheprovider",
        "--import-mode=importlib",
        f"--rootdir={rootdir}",
        *node_paths,
    ]
    returncode = int(pytest.main(argv, plugins=[recorder]))
    passed, problems = classify(returncode, recorder.collected, recorder.outcomes)
    import_path = Path(pssolver.__file__).resolve()
    source_shadow = import_path.is_relative_to(provider_root)
    if source_shadow:
        problems.append("pssolver imported from Provider source root")
        passed = False
    environment = {
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "visible_device_count": int(torch.cuda.device_count()),
        "allocated_device": (
            str(torch.empty((), device="cuda").device)
            if torch.cuda.is_available()
            else None
        ),
        "device_name": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "tf32_before_policy": before,
        "tf32_after_policy": after,
        "pssolver_import": str(import_path),
        "source_shadow_import": source_shadow,
    }
    report = {
        "schema": SCHEMA,
        "classification": PASS if passed else "FAIL_RC4_4_7_CUDA_ONLY_TESTS",
        "passed": passed,
        "environment": environment,
        "requested_node_ids": list(NODE_IDS),
        "collected_node_ids": recorder.collected,
        "outcomes": recorder.outcomes,
        "pytest_returncode": returncode,
        "problems": problems,
    }
    _atomic_json(output, report)
    if not passed:
        raise RuntimeError(f"CUDA-only test gate failed: {problems}")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    run(args.provider_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
