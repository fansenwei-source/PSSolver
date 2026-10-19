#!/usr/bin/env python3
"""Run the RC6 current cross-device checkpoint matrix on one H100."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile
from collections.abc import Iterator, Mapping

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.plan_rc6_single_h100_qualification import (
    CONSUMER_SOURCE,
    PROVIDER_CANDIDATE,
)
from benchmarks import run_rc5_g1_checkpoints as rc5


SCHEMA = "pssolver.rc6.current_checkpoint_h100.v1"
CHECKPOINT_WORK_ROOT_POLICY = "node_local_tmpdir"
NODE_LOCAL_TMP_ROOT = Path("/tmp")


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
            raise RuntimeError("checkpoint JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def summarize(cells: list[dict[str, object]], restart_output_absent: bool) -> dict[str, object]:
    expected = {
        (family, source, target)
        for family in rc5.FAMILIES
        for source, target in rc5.DIRECTIONS
    }
    observed = {
        (str(cell.get("family")), str(cell.get("source_device")), str(cell.get("target_device")))
        for cell in cells
    }
    if observed != expected or len(cells) != 8:
        raise ValueError("current checkpoint cell set differs")
    for cell in cells:
        if cell.get("passed") is not True:
            raise ValueError(f"checkpoint cell failed: {cell.get('family')}")
        if cell.get("continuous_split_restart_equal") is not True:
            raise ValueError("checkpoint replay differs")
        if cell.get("source_checkpoint_unchanged") is not True:
            raise ValueError("source checkpoint changed")
        if cell.get("fallback_used") is not False:
            raise ValueError("checkpoint runtime fallback used")
        for name in ("payload_tamper", "identity_mismatch"):
            gate = cell.get(name)
            if not isinstance(gate, dict) or not all(
                gate.get(key) is True
                for key in ("rejected", "guard_reached", "target_unchanged")
            ):
                raise ValueError(f"checkpoint negative gate failed: {name}")
    if restart_output_absent is not True:
        raise ValueError("rejected restart materialized output")
    return {
        "schema": SCHEMA,
        "provider_commit": PROVIDER_CANDIDATE,
        "consumer_source_commit": CONSUMER_SOURCE,
        "passed": True,
        "family_count": 4,
        "direction_count": 2,
        "cell_count": 8,
        "continuous_split_restart_passed": True,
        "transactional_restore_passed": True,
        "negative_gates_passed": True,
        "rejected_restart_output_remains_absent": True,
        "source_checkpoints_unchanged": True,
        "cells": cells,
    }


def _node_local_tmpdir(environment: Mapping[str, str] | None = None) -> Path:
    values = os.environ if environment is None else environment
    job_id = values.get("SLURM_JOB_ID", "").strip()
    if not job_id or not job_id.isdecimal():
        raise RuntimeError("numeric SLURM_JOB_ID is required")
    raw_tmpdir = values.get("TMPDIR", "").strip()
    if not raw_tmpdir:
        raise RuntimeError("TMPDIR is required")
    tmpdir = Path(raw_tmpdir)
    if not tmpdir.is_absolute():
        raise RuntimeError("TMPDIR must be absolute")
    expected = NODE_LOCAL_TMP_ROOT / f"pssolver_rc6_{job_id}" / "pytest_tmp"
    if tmpdir != expected:
        raise RuntimeError("TMPDIR does not match the allocated RC6 node-local path")
    if not tmpdir.is_dir() or tmpdir.is_symlink():
        raise RuntimeError("TMPDIR must be an existing ordinary directory")
    if tmpdir.resolve() != expected.resolve():
        raise RuntimeError("TMPDIR resolves outside the allocated RC6 node-local path")
    return tmpdir


@contextmanager
def _checkpoint_work_root(
    environment: Mapping[str, str] | None = None,
) -> Iterator[tuple[Path, Path]]:
    tmpdir = _node_local_tmpdir(environment)
    root: Path | None = None
    try:
        with tempfile.TemporaryDirectory(
            prefix="rc6_current_checkpoints_",
            dir=tmpdir,
        ) as temporary_root:
            root = Path(temporary_root)
            if root.parent != tmpdir or root.is_symlink():
                raise RuntimeError("checkpoint work root escaped node-local TMPDIR")
            yield tmpdir, root
    finally:
        if root is not None and root.exists():
            raise RuntimeError("checkpoint work root was not removed")


def execute(
    test_root: Path,
    output: Path,
    checkpoint_work_root_policy: str,
) -> dict[str, object]:
    import importlib.metadata
    import torch

    rc5._tf32_off()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("exactly one CUDA device is required")
    if str(torch.empty((), device="cuda").device) != "cuda:0":
        raise RuntimeError("allocated CUDA device differs")
    if "H100" not in torch.cuda.get_device_name(0):
        raise RuntimeError("allocated GPU is not H100")
    if importlib.metadata.version("pssolver") != "0.2.0rc6":
        raise RuntimeError("candidate package version differs")
    if checkpoint_work_root_policy != CHECKPOINT_WORK_ROOT_POLICY:
        raise ValueError("checkpoint work-root policy differs")
    with _checkpoint_work_root() as (tmpdir, root):
        cells: list[dict[str, object]] = []
        for family in rc5.FAMILIES:
            for source, target in rc5.DIRECTIONS:
                cell_root = root / "cells" / f"{family}_{source}_to_{target}"
                cell_root.mkdir(parents=True)
                cell = (
                    rc5._legacy_cell(cell_root, source, target)
                    if family == "legacy_channel"
                    else rc5._functional_cell(cell_root, family, source, target)
                )
                _atomic_json(
                    root / "reports" / f"{family}_{source}_to_{target}.json",
                    cell,
                )
                cells.append(cell)

        import pytest
        from benchmarks.run_rc426_cuda_tests import Results

        support = test_root.expanduser().resolve()
        if not support.is_dir() or support.is_symlink() or (support / "pssolver").exists():
            raise RuntimeError("ordinary non-shadowing test root required")
        collector = Results()
        previous = Path.cwd()
        try:
            os.chdir(support)
            exit_code = int(
                pytest.main(
                    [
                        "-q", "-p", "no:cacheprovider", "--import-mode=importlib",
                        "tests/test_phase2_plane_workflow_migration.py::test_restart_integrity_rejection_does_not_create_output",
                    ],
                    plugins=[collector],
                )
            )
        finally:
            os.chdir(previous)
        outcomes = collector.metadata()
        restart_output_absent = exit_code == 0 and len(outcomes["passed"]) == 1
        result = summarize(cells, restart_output_absent)
        result["checkpoint_work_root"] = {
            "policy": CHECKPOINT_WORK_ROOT_POLICY,
            "tmpdir": str(tmpdir),
            "temporary_root": str(root),
            "temporary_root_parent_matches_tmpdir": root.parent == tmpdir,
        }
    result["checkpoint_work_root"]["removed_after_stage"] = True
    _atomic_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-work-root-policy",
        choices=(CHECKPOINT_WORK_ROOT_POLICY,),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    execute(args.test_root, args.output, args.checkpoint_work_root_policy)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
