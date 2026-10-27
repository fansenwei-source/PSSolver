#!/usr/bin/env python3
"""Collect reference-only CPU memory traces for the three diagnostic roles.

Repeated identical runs vary by one or two pressure-field allocations, so this
probe cannot prove parity or identify the source of a one-field CUDA peak.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform

import torch
from torch.profiler import profile, ProfilerActivity

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.profile_rc458_runtime_family import _build_case, _tensor_sha256
from benchmarks.run_rc5_g1_numerical import _config
from benchmarks.v0_3_s09_revert_pressure_proxy import use_revert_pressure_proxy
from pssolver.linear_solvers.stokes.channel_no_slip import (
    ChannelNoSlipModalStokesSolver,
)


SCHEMA = "pssolver.v0_3.s09.ap_cache_cpu_memory_probe.v1"
ROLES = ("candidate_cache_on", "candidate_revert_proxy", "historical_source")
MEMORY_INTERPRETATION = {
    "reference_only": True,
    "reproducible_at_one_pressure_field_resolution": False,
    "used_as_revert_proxy_fidelity_proof": False,
    "used_for_qualification_or_v41_attribution": False,
    "decisive_evidence": "untimed H100 CUDA allocator snapshots",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    finally:
        temporary.unlink(missing_ok=True)


def _triplet(value: str, cast):
    result = tuple(cast(item) for item in value.split(","))
    if len(result) != 3:
        raise argparse.ArgumentTypeError("expected three comma-separated values")
    return result


def run_probe(
    *,
    role: str,
    shape: tuple[int, int, int],
    lengths: tuple[float, float, float],
    work_root: Path,
    timeline_path: Path,
) -> dict[str, object]:
    if role not in ROLES:
        raise ValueError(f"role must be one of {ROLES}")
    if work_root.exists() or work_root.is_symlink():
        raise FileExistsError(f"work root already exists: {work_root}")
    if timeline_path.exists() or timeline_path.is_symlink():
        raise FileExistsError(f"timeline already exists: {timeline_path}")
    work_root.mkdir(parents=True)
    timeline_path.parent.mkdir(parents=True, exist_ok=True)

    calls = {"count": 0}
    original_operator = ChannelNoSlipModalStokesSolver._pressure_operator

    def counted_operator(self, value):
        calls["count"] += 1
        return original_operator(self, value)

    ChannelNoSlipModalStokesSolver._pressure_operator = counted_operator
    diagnostic_role = (
        "candidate_revert_proxy" if role == "candidate_revert_proxy" else role
    )
    try:
        with use_revert_pressure_proxy(diagnostic_role) as proof:
            config = dataclasses.replace(
                _config("channel_legacy", "candidate", "cpu-probe", "0" * 64),
                device="cpu",
                shape=shape,
                lengths=lengths,
            )
            with torch.no_grad():
                case = _build_case(work_root, config)
                for _ in range(3):
                    case.step()
                calls["count"] = 0
                with profile(
                    activities=[ProfilerActivity.CPU],
                    profile_memory=True,
                    record_shapes=True,
                    with_stack=True,
                ) as profiler:
                    for _ in range(3):
                        case.step()
                operator_calls = calls["count"]
                final_state_sha256 = _tensor_sha256(case.state())
    finally:
        ChannelNoSlipModalStokesSolver._pressure_operator = original_operator

    profiler.export_memory_timeline(str(timeline_path), device="cpu")
    timeline = json.loads(timeline_path.read_text(encoding="utf-8"))
    if not isinstance(timeline, list) or len(timeline) != 2:
        raise RuntimeError("unexpected CPU memory timeline schema")
    totals = [sum(row) for row in timeline[1]]
    if not totals:
        raise RuntimeError("CPU memory timeline is empty")
    if role == "candidate_revert_proxy" and not proof["proved_revert_proxy"]:
        raise RuntimeError(f"revert proxy proof failed: {proof!r}")
    return {
        "schema": SCHEMA,
        "role": role,
        "shape": list(shape),
        "lengths": list(lengths),
        "warmup_steps": 3,
        "profiled_steps": 3,
        "peak_live_bytes": max(totals),
        "memory_interpretation": dict(MEMORY_INTERPRETATION),
        "pressure_operator_calls_in_profiled_steps": operator_calls,
        "final_state_sha256": final_state_sha256,
        "revert_proxy": proof,
        "timeline": {
            "path": str(timeline_path),
            "sha256": _sha256(timeline_path),
            "events": len(totals),
        },
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "pssolver_import": str(Path(__import__("pssolver").__file__).resolve()),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--shape", type=lambda value: _triplet(value, int), required=True)
    parser.add_argument("--lengths", type=lambda value: _triplet(value, float), required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--timeline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_probe(
        role=args.role,
        shape=args.shape,
        lengths=args.lengths,
        work_root=args.work_root,
        timeline_path=args.timeline,
    )
    _atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
