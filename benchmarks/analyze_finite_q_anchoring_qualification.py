#!/usr/bin/env python3
"""Fail-closed analyzer for the P8.5.6 H100 profile matrix."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
import math
from pathlib import Path
import statistics


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"JSON root is not an object: {path}")
    return value


def _seconds(report: Mapping[str, object], name: str) -> float:
    timing = report.get("timing")
    _require(isinstance(timing, Mapping), "timing metadata is missing")
    value = float(timing[f"{name}_timestep_seconds"])
    _require(math.isfinite(value) and value > 0.0, "invalid timing value")
    return value


def analyze(
    reports: list[Mapping[str, object]],
    *,
    expected_commit: str,
    expected_package_root: Path,
) -> dict[str, object]:
    _require(len(reports) == 12, "exactly 12 profile reports are required")
    index = {}
    for report in reports:
        _require(report.get("phase") == "P8.5.6", "profile phase differs")
        _require(
            report.get("kind") == "finite_q_anchoring_profile",
            "profile kind differs",
        )
        config = report.get("config")
        _require(isinstance(config, Mapping), "profile config is missing")
        key = (
            tuple(config["shape"]),
            int(config["trial"]),
            str(config["mode"]),
        )
        _require(key not in index, "duplicate profile identity")
        index[key] = report
        _require(report.get("finite") is True, "profile state is not finite")
        runtime = report.get("runtime")
        _require(isinstance(runtime, Mapping), "runtime metadata is missing")
        _require(runtime.get("fallback_allowed") is False, "fallback was allowed")
        _require(runtime.get("fallback_used") is False, "fallback was used")
        _require(runtime.get("graph_breaks") == 0, "graph break detected")
        _require(
            runtime.get("construction_in_timed_loop") is False,
            "construction entered the timed loop",
        )
        _require(
            runtime.get("root_solve_in_timed_loop") is False,
            "root solve entered the timed loop",
        )
        _require(
            runtime.get("matrix_factorization_in_timed_loop") is False,
            "factorization entered the timed loop",
        )
        counts = report.get("operation_counts")
        _require(isinstance(counts, Mapping), "operation counts are missing")
        per_step = counts.get("per_step")
        _require(isinstance(per_step, Mapping), "per-step counts are missing")
        _require(per_step.get("forward_transform") == 10.0, "forward count differs")
        _require(per_step.get("inverse_transform") == 5.0, "inverse count differs")
        _require(per_step.get("helmholtz_solve") == 5.0, "solve count differs")
        _require(per_step.get("helmholtz_apply") == 0.0, "apply count differs")
        conditioning = report.get("conditioning")
        _require(isinstance(conditioning, Mapping), "conditioning is missing")
        _require(
            float(conditioning["maximum_basis_condition_number"]) <= 1.0e10,
            "conditioning limit exceeded",
        )
        cuda = report.get("cuda")
        _require(isinstance(cuda, Mapping), "CUDA identity is missing")
        _require(cuda.get("available") is True, "CUDA is unavailable")
        _require(cuda.get("name") == "NVIDIA H100 PCIe", "GPU model differs")
        torch_metadata = report.get("torch")
        _require(isinstance(torch_metadata, Mapping), "torch metadata is missing")
        _require(torch_metadata.get("tf32_matmul") is False, "matmul TF32 enabled")
        _require(torch_metadata.get("tf32_cudnn") is False, "cuDNN TF32 enabled")
        git = report.get("git")
        _require(isinstance(git, Mapping), "git metadata is missing")
        _require(git.get("head") == expected_commit, "profile commit differs")
        _require(git.get("dirty") is False, "profile worktree is dirty")
        imported = Path(str(report.get("pssolver_import"))).resolve()
        _require(
            imported.is_relative_to(expected_package_root.resolve()),
            "profile did not import the installed package",
        )

    shapes = ((128, 128, 32), (320, 320, 80))
    modes = ("scalar_reference", "aggregate_runtime")
    expected = {
        (shape, trial, mode)
        for shape in shapes
        for trial in (1, 2, 3)
        for mode in modes
    }
    _require(set(index) == expected, "profile matrix is incomplete")
    performance = {}
    for shape in shapes:
        mean_ratios = []
        median_ratios = []
        allocated_ratios = []
        active_ratios = []
        reserved_ratios = []
        for trial in (1, 2, 3):
            reference = index[(shape, trial, "scalar_reference")]
            aggregate = index[(shape, trial, "aggregate_runtime")]
            _require(
                reference["initial_q_sha256"]
                == aggregate["initial_q_sha256"],
                "paired initial Q identity differs",
            )
            _require(
                reference["final_q_sha256"] == aggregate["final_q_sha256"],
                "paired final Q identity differs",
            )
            mean_ratios.append(
                _seconds(aggregate, "mean") / _seconds(reference, "mean")
            )
            median_ratios.append(
                _seconds(aggregate, "median") / _seconds(reference, "median")
            )
            reference_memory = reference["memory"]
            aggregate_memory = aggregate["memory"]
            for values, name in (
                (allocated_ratios, "peak_allocated_bytes"),
                (active_ratios, "peak_active_bytes"),
                (reserved_ratios, "peak_reserved_bytes"),
            ):
                denominator = int(reference_memory[name])
                _require(denominator > 0, f"reference {name} is unavailable")
                values.append(int(aggregate_memory[name]) / denominator)
        aggregate_mean_ratio = statistics.fmean(mean_ratios)
        aggregate_median_ratio = statistics.median(median_ratios)
        _require(aggregate_mean_ratio <= 1.15, "mean timestep gate failed")
        _require(aggregate_median_ratio <= 1.15, "median timestep gate failed")
        _require(max(allocated_ratios) <= 1.10, "allocated memory gate failed")
        _require(max(active_ratios) <= 1.10, "active memory gate failed")
        _require(max(reserved_ratios) <= 1.10, "reserved memory gate failed")
        performance["x".join(str(value) for value in shape)] = {
            "paired_mean_ratios": mean_ratios,
            "paired_median_ratios": median_ratios,
            "aggregate_mean_ratio": aggregate_mean_ratio,
            "aggregate_median_ratio": aggregate_median_ratio,
            "maximum_peak_allocated_ratio": max(allocated_ratios),
            "maximum_peak_active_ratio": max(active_ratios),
            "maximum_peak_reserved_ratio": max(reserved_ratios),
        }
    return {
        "schema_version": 1,
        "phase": "P8.5.6",
        "kind": "finite_q_anchoring_profile_analysis",
        "classification": "PASS_P8_5_6_PROFILE_MATRIX",
        "profile_count": len(reports),
        "performance": performance,
        "eligible_for_final_scientific_gates": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-package-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = sorted(args.profile_dir.glob("*.json"))
    result = analyze(
        [_load(path) for path in paths],
        expected_commit=args.expected_commit,
        expected_package_root=args.expected_package_root,
    )
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
