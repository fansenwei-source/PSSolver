#!/usr/bin/env python3
"""Frozen 6x60 CPU preflight for the S-09 Channel pressure paths.

Each isolated profile retains all sixty timestep samples.  The profile center
is the median of six fixed, consecutive ten-step block means.  Six paired
trials are run in alternating baseline/candidate order.  This is the same
estimator frozen for the S-09 H100 profiles; no sample may be removed or
replaced.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmarks.benchmark_v0_3_s09_channel_cpu import (  # noqa: E402
    Cell,
    FAMILIES,
    MEMORY_LIMIT,
    MODES,
    TIME_CENTER_LIMIT,
    TIME_PAIRED_LIMIT,
    _atomic_json,
    run_cell,
)


SCHEMA = "pssolver.v0_3.s09.channel_cpu_frozen_6x60.v1"
TRIALS = 6
WARMUP_STEPS = 3
MEASURED_STEPS = 60
FIXED_BLOCK_COUNT = 6
FIXED_BLOCK_SIZE = 10
SHAPE = (128, 20, 20)
LENGTHS = (40.0, 6.0, 6.0)


def fixed_block_means(samples: list[float]) -> list[float]:
    if len(samples) != MEASURED_STEPS:
        raise ValueError(f"expected exactly {MEASURED_STEPS} retained samples")
    blocks = [
        statistics.fmean(samples[offset : offset + FIXED_BLOCK_SIZE])
        for offset in range(0, MEASURED_STEPS, FIXED_BLOCK_SIZE)
    ]
    if len(blocks) != FIXED_BLOCK_COUNT:
        raise AssertionError("fixed block partition is inconsistent")
    return blocks


def decorate_cell(report: dict[str, object]) -> dict[str, object]:
    timing = report["timing"]
    assert isinstance(timing, dict)
    samples = [float(value) for value in timing["samples_seconds"]]
    blocks = fixed_block_means(samples)
    timing.update(
        {
            "fixed_block_count": FIXED_BLOCK_COUNT,
            "fixed_block_size": FIXED_BLOCK_SIZE,
            "fixed_block_means_seconds": blocks,
            "profile_center": "median_of_six_fixed_block_means",
            "profile_center_step_seconds": statistics.median(blocks),
            "all_samples_retained": True,
        }
    )
    report["schema"] = SCHEMA
    return report


def analyze(cells: list[dict[str, object]]) -> dict[str, object]:
    analyses: list[dict[str, object]] = []
    passed = True
    for family in FAMILIES:
        for mode in MODES:
            subset = [
                value
                for value in cells
                if value["config"]["family"] == family
                and value["config"]["mode"] == mode
            ]
            trials = sorted({int(value["config"]["trial"]) for value in subset})
            if trials != list(range(1, TRIALS + 1)):
                raise ValueError(f"expected trials 1..{TRIALS} for {family}/{mode}")
            paired_time: list[float] = []
            paired_memory: list[float] = []
            for trial in trials:
                by_role = {
                    str(value["config"]["role"]): value
                    for value in subset
                    if int(value["config"]["trial"]) == trial
                }
                if set(by_role) != {"baseline", "candidate"}:
                    raise ValueError(
                        f"missing paired role for {family}/{mode}/trial{trial}"
                    )
                baseline = by_role["baseline"]
                candidate = by_role["candidate"]
                paired_time.append(
                    float(candidate["timing"]["profile_center_step_seconds"])
                    / float(baseline["timing"]["profile_center_step_seconds"])
                )
                paired_memory.append(
                    float(candidate["memory"]["peak_rss_during_window_bytes"])
                    / float(baseline["memory"]["peak_rss_during_window_bytes"])
                )
            center = statistics.median(paired_time)
            memory_center = statistics.median(paired_memory)
            gate = (
                center <= TIME_CENTER_LIMIT
                and max(paired_time) <= TIME_PAIRED_LIMIT
                and memory_center <= MEMORY_LIMIT
                and max(paired_memory) <= MEMORY_LIMIT
            )
            passed = passed and gate
            analyses.append(
                {
                    "family": family,
                    "mode": mode,
                    "paired_time_ratios": paired_time,
                    "median_time_ratio": center,
                    "maximum_time_ratio": max(paired_time),
                    "paired_peak_rss_ratios": paired_memory,
                    "median_peak_rss_ratio": memory_center,
                    "maximum_peak_rss_ratio": max(paired_memory),
                    "passed": gate,
                }
            )
    return {
        "thresholds": {
            "median_time_ratio_max": TIME_CENTER_LIMIT,
            "paired_time_ratio_max": TIME_PAIRED_LIMIT,
            "median_peak_rss_ratio_max": MEMORY_LIMIT,
            "paired_peak_rss_ratio_max": MEMORY_LIMIT,
            "relaxation_allowed": False,
        },
        "estimator": {
            "trials_per_cell": TRIALS,
            "samples_per_profile": MEASURED_STEPS,
            "fixed_block_count": FIXED_BLOCK_COUNT,
            "fixed_block_size": FIXED_BLOCK_SIZE,
            "profile_center": "median_of_six_fixed_block_means",
            "alternating_pair_order": True,
            "all_samples_retained": True,
            "trial_replacement_allowed": False,
        },
        "cells": analyses,
        "passed": passed,
        "eligible_for_h100_preparation": passed,
    }


def _cell_command(
    args: argparse.Namespace,
    *,
    family: str,
    mode: str,
    role: str,
    trial: int,
    output: Path,
) -> list[str]:
    root = args.baseline_root if role == "baseline" else args.candidate_root
    commit = args.baseline_commit if role == "baseline" else args.candidate_commit
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        "--cell",
        "--family",
        family,
        "--mode",
        mode,
        "--role",
        role,
        "--trial",
        str(trial),
        "--provider-root",
        str(root),
        "--provider-commit",
        commit,
        "--output",
        str(output),
    ]


def run_matrix(args: argparse.Namespace) -> dict[str, object]:
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite {output}")
    cell_root = output.parent / f"{output.stem}_cells"
    if cell_root.exists() or cell_root.is_symlink():
        raise FileExistsError(f"refusing to overwrite {cell_root}")
    cell_root.mkdir(parents=True)
    cells: list[dict[str, object]] = []
    sequence: list[dict[str, object]] = []
    env = os.environ.copy()
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    for family in FAMILIES:
        for mode in MODES:
            for trial in range(1, TRIALS + 1):
                roles = ("baseline", "candidate") if trial % 2 else (
                    "candidate",
                    "baseline",
                )
                for role in roles:
                    cell_output = (
                        cell_root / f"{family}_{mode}_{role}_trial{trial}.json"
                    )
                    command = _cell_command(
                        args,
                        family=family,
                        mode=mode,
                        role=role,
                        trial=trial,
                        output=cell_output,
                    )
                    completed = subprocess.run(
                        command,
                        env=env,
                        cwd=cell_root,
                        text=True,
                        capture_output=True,
                        check=False,
                    )
                    sequence.append(
                        {
                            "command": command,
                            "returncode": completed.returncode,
                            "stdout": completed.stdout,
                            "stderr": completed.stderr,
                        }
                    )
                    if completed.returncode != 0:
                        raise RuntimeError(
                            f"cell failed: {family}/{mode}/{role}/trial{trial}: "
                            f"{completed.stderr}"
                        )
                    cells.append(json.loads(cell_output.read_text(encoding="utf-8")))
    result = {
        "schema": SCHEMA,
        "kind": "matrix",
        "baseline": {
            "root": str(args.baseline_root.resolve()),
            "commit": args.baseline_commit,
        },
        "candidate": {
            "root": str(args.candidate_root.resolve()),
            "commit": args.candidate_commit,
        },
        "contract": {
            "families": list(FAMILIES),
            "modes": list(MODES),
            "trials": TRIALS,
            "warmup_steps": WARMUP_STEPS,
            "measured_steps": MEASURED_STEPS,
            "fixed_block_count": FIXED_BLOCK_COUNT,
            "fixed_block_size": FIXED_BLOCK_SIZE,
            "profile_center": "median_of_six_fixed_block_means",
            "shape": list(SHAPE),
            "lengths": list(LENGTHS),
            "alternating_pair_order": True,
            "all_samples_retained": True,
            "trial_replacement_allowed": False,
        },
        "sequence": sequence,
        "raw_cells": cells,
        "analysis": analyze(cells),
    }
    _atomic_json(output, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--cell", action="store_true", help=argparse.SUPPRESS)
    modes.add_argument("--matrix", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--family", choices=FAMILIES, help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=MODES, help=argparse.SUPPRESS)
    parser.add_argument("--role", choices=("baseline", "candidate"), help=argparse.SUPPRESS)
    parser.add_argument("--trial", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--provider-root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--provider-commit", help=argparse.SUPPRESS)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--baseline-commit")
    parser.add_argument("--candidate-root", type=Path)
    parser.add_argument("--candidate-commit")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.cell:
        required = (
            args.family,
            args.mode,
            args.role,
            args.trial,
            args.provider_root,
            args.provider_commit,
        )
        if any(value is None for value in required):
            raise ValueError("internal cell invocation is incomplete")
        report = decorate_cell(
            run_cell(
                Cell(
                    family=args.family,
                    mode=args.mode,
                    role=args.role,
                    trial=args.trial,
                    provider_root=args.provider_root,
                    provider_commit=args.provider_commit,
                    warmup_steps=WARMUP_STEPS,
                    measured_steps=MEASURED_STEPS,
                    shape=SHAPE,
                    lengths=LENGTHS,
                )
            )
        )
        _atomic_json(args.output, report)
        return 0
    required = (
        args.baseline_root,
        args.baseline_commit,
        args.candidate_root,
        args.candidate_commit,
    )
    if any(value is None for value in required):
        raise ValueError("matrix mode requires both provider roots and commits")
    report = run_matrix(args)
    print(json.dumps(report["analysis"], allow_nan=False, indent=2, sort_keys=True))
    return 0 if report["analysis"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
