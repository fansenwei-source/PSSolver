#!/usr/bin/env python3
"""Run current, historical, restart, and tamper checkpoint gates on H100."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.plan_v0_3_s09_h100 import CANDIDATE_COMMIT, CANDIDATE_WHEEL_SHA256
from benchmarks.run_rc426_cuda_tests import Results
from benchmarks import run_rc5_g1_checkpoints as legacy


SCHEMA = "pssolver.v0_3.s09.checkpoints.v1"
TESTS = (
    "tests/test_rc5_legacy_functional_checkpoints.py",
    "tests/test_rc5_r53_legacy_channel_checkpoint.py",
    "tests/test_v0_3_s08_historical_cpu_coverage.py",
    "tests/test_v0_3_s05_checkpoint_metadata_integrity.py",
)


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
            raise RuntimeError("checkpoint report round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _cross_device(root: Path) -> list[dict[str, object]]:
    cells: list[dict[str, object]] = []
    for family in legacy.FAMILIES:
        for source, target in legacy.DIRECTIONS:
            cell_root = root / "cells" / f"{family}_{source}_to_{target}"
            cell_root.mkdir(parents=True)
            report = (
                legacy._legacy_cell(cell_root, source, target)
                if family == "legacy_channel"
                else legacy._functional_cell(cell_root, family, source, target)
            )
            if report.get("passed") is not True:
                raise RuntimeError(f"checkpoint cell failed: {family}/{source}->{target}")
            _atomic_json(root / "reports" / f"{family}_{source}_to_{target}.json", report)
            cells.append(report)
    return cells


def execute(
    test_root: Path,
    junit: Path,
    output: Path,
    *,
    device: str = "cuda",
    include_cross_device: bool = True,
) -> dict[str, object]:
    import importlib.metadata
    import pssolver
    import pytest
    import torch

    legacy._tf32_off()
    if importlib.metadata.version("pssolver") != "0.3.0.dev0":
        raise RuntimeError("candidate package version differs")
    if "site-packages" not in str(Path(pssolver.__file__).resolve()):
        raise RuntimeError("candidate is not imported from an installed wheel")
    if device == "cuda" and (
        not torch.cuda.is_available() or torch.cuda.device_count() != 1
    ):
        raise RuntimeError("exactly one visible CUDA device is required")
    if device == "cuda" and (
        str(torch.empty((), device="cuda").device) != "cuda:0"
        or "H100" not in torch.cuda.get_device_name(0)
    ):
        raise RuntimeError("allocated device is not cuda:0 H100")
    support = test_root.expanduser().resolve()
    if not support.is_dir() or support.is_symlink() or (support / "pssolver").exists():
        raise RuntimeError("ordinary non-shadowing support root required")
    collector = Results()
    previous = Path.cwd()
    try:
        os.chdir(support)
        code = int(pytest.main([
            "-q", "-p", "no:cacheprovider", "--import-mode=importlib",
            f"--junitxml={junit.expanduser().absolute()}", *TESTS,
        ], plugins=[collector]))
    finally:
        os.chdir(previous)
    outcomes = collector.metadata()
    if code != 0 or len(outcomes["passed"]) != 78:
        raise RuntimeError("checkpoint installed-wheel pytest outcome differs")
    for name in ("failed", "skipped", "xfailed", "xpassed", "collection_errors"):
        if outcomes[name]:
            raise RuntimeError(f"checkpoint pytest contains {name}")
    tmpdir = Path(os.environ.get("TMPDIR", ""))
    if not tmpdir.is_absolute() or not tmpdir.is_dir() or tmpdir.is_symlink() or tmpdir.parent.parent != Path("/tmp"):
        raise RuntimeError("checkpoint TMPDIR is not an allocated node-local directory")
    cells: list[dict[str, object]] = []
    if include_cross_device:
        with tempfile.TemporaryDirectory(prefix="pssolver-s09-checkpoints-", dir=tmpdir) as temporary:
            cells = _cross_device(Path(temporary))
    if include_cross_device and len(cells) != 8:
        raise RuntimeError("cross-device checkpoint cell count differs")
    result = {
        "schema": SCHEMA,
        "name": "checkpoint_restart_and_released_fixture_matrix",
        "provider_candidate_commit": CANDIDATE_COMMIT,
        "candidate_wheel_sha256": CANDIDATE_WHEEL_SHA256,
        "current_matrix": True,
        "released_fixtures": True,
        "execution_device": device,
        "cross_device_executed": include_cross_device,
        "restart_exact": all(row["continuous_split_restart_equal"] is True for row in cells),
        "negative_gates": all(
            row[name]["rejected"] is True and row[name]["target_unchanged"] is True
            for row in cells for name in ("payload_tamper", "identity_mismatch")
        ),
        "sources_unchanged": all(row["source_checkpoint_unchanged"] is True for row in cells),
        "pytest_passed_count": len(outcomes["passed"]),
        "cross_device_cell_count": len(cells),
        "passed": True,
    }
    _atomic_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--junit", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--skip-cross-device", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    execute(
        args.test_root,
        args.junit,
        args.output,
        device=args.device,
        include_cross_device=not args.skip_cross_device,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
