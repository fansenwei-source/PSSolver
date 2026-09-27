#!/usr/bin/env python3
"""Fail-closed analysis of the frozen P8.4.5 H100 evidence bundle."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Mapping


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _objects(paths: Iterable[Path]) -> list[dict[str, object]]:
    values = []
    for path in paths:
        value = json.loads(path.read_text(encoding="utf-8"))
        _require(isinstance(value, dict), f"JSON root must be an object: {path}")
        values.append(value)
    return values


def _counter_named(value: object, needle: str) -> int:
    total = 0
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if needle in str(key).lower() and isinstance(nested, int):
                total += nested
            else:
                total += _counter_named(nested, needle)
    elif isinstance(value, list):
        total += sum(_counter_named(item, needle) for item in value)
    return total


def _profile_key(report: Mapping[str, object]) -> tuple[tuple[int, ...], int, str]:
    config = report["config"]
    _require(isinstance(config, Mapping), "profile config is missing")
    return (
        tuple(int(value) for value in config["shape"]),
        int(config["trial"]),
        str(config["variant"]),
    )


def _profile_seconds(report: Mapping[str, object], statistic: str) -> float:
    timing = report["timing"]
    _require(isinstance(timing, Mapping), "profile timing is missing")
    value = float(timing[f"{statistic}_timestep_seconds"])
    _require(math.isfinite(value) and value > 0.0, "invalid timestep result")
    return value


def analyze(
    plan: Mapping[str, object],
    reports: Iterable[Mapping[str, object]],
    *,
    expected_commit: str,
) -> dict[str, object]:
    """Validate a complete P8.4.5 bundle and return its compact verdict."""

    _require(plan.get("phase") == "P8.4.5", "qualification plan phase mismatch")
    contract = plan.get("qualification_contract")
    _require(isinstance(contract, Mapping), "qualification contract is missing")
    profiles = []
    manufactured = []
    restarts = []
    for report in reports:
        _require(report.get("phase") == "P8.4.5", "report phase mismatch")
        kind = report.get("kind")
        if kind == "plane_static_lifting_profile":
            profiles.append(report)
        elif kind == "plane_static_lifting_manufactured_convergence":
            manufactured.append(report)
        elif kind == "plane_static_lifting_restart":
            restarts.append(report)
        else:
            raise RuntimeError(f"unexpected report kind: {kind!r}")

    expected_profiles = int(contract["profile_report_count"])
    _require(len(profiles) == expected_profiles, "profile report count mismatch")
    _require(len(manufactured) == 1, "one manufactured report is required")
    _require(len(restarts) == 1, "one restart report is required")

    profile_index = {_profile_key(report): report for report in profiles}
    _require(len(profile_index) == len(profiles), "duplicate profile identity")
    expected_shapes = tuple(tuple(value) for value in contract["profile_shapes"])
    expected_trials = int(contract["profile_trials"])
    expected_keys = {
        (shape, trial, variant)
        for shape in expected_shapes
        for trial in range(1, expected_trials + 1)
        for variant in ("homogeneous_control", "strong_planar_lifting")
    }
    _require(set(profile_index) == expected_keys, "profile matrix is incomplete")

    wall_tolerance = float(contract["wall_residual_linf_max"])
    expected_forward = float(contract["forward_transforms_per_step"])
    expected_inverse = float(contract["inverse_transforms_per_step"])
    graph_breaks = 0
    for key, report in profile_index.items():
        runtime = report["runtime_identity"]
        _require(isinstance(runtime, Mapping), "runtime identity is missing")
        _require(runtime.get("requested") == "legacy_production", "wrong runtime")
        _require(runtime.get("effective") == "legacy_production", "runtime fallback")
        _require(runtime.get("fallback_used") is False, "runtime fallback used")
        _require(report.get("finite") is True, "profile state is non-finite")
        git = report["git"]
        _require(isinstance(git, Mapping), "git provenance is missing")
        _require(git.get("head") == expected_commit, "profile commit mismatch")
        transforms = report["transform_calls"]
        _require(isinstance(transforms, Mapping), "transform counts are missing")
        _require(
            float(transforms["forward_per_step"]) == expected_forward,
            "forward transform count changed",
        )
        _require(
            float(transforms["inverse_per_step"]) == expected_inverse,
            "inverse transform count changed",
        )
        pointwise = report.get("pointwise_compile")
        _require(isinstance(pointwise, Mapping), "pointwise metadata is missing")
        execution = pointwise.get("execution")
        _require(isinstance(execution, Mapping), "pointwise execution is missing")
        _require(execution.get("requested") == "compile", "compile was not requested")
        _require(execution.get("effective") == "compile", "compile fallback detected")
        _require(
            execution.get("fallback_allowed") is False,
            "pointwise fallback was allowed",
        )
        graph_breaks += _counter_named(pointwise, "graph_break")
        if key[2] == "strong_planar_lifting":
            _require(report.get("lifting") is not None, "lifting metadata is absent")
            wall = report.get("wall_residual")
            _require(isinstance(wall, Mapping), "wall residual is absent")
            _require(
                float(wall["max_linf"]) <= wall_tolerance,
                "wall residual exceeds its frozen tolerance",
            )
        else:
            _require(report.get("lifting") is None, "control unexpectedly uses lifting")
            _require(report.get("wall_residual") is None, "control has wall residual")

    _require(graph_breaks == 0, "pointwise execution reported graph breaks")
    performance: dict[str, object] = {}
    timestep_limit = float(contract["lifting_control_timestep_ratio_max"])
    allocated_limit = float(contract["lifting_control_peak_allocated_ratio_max"])
    reserved_limit = float(contract["lifting_control_peak_reserved_ratio_max"])
    for shape in expected_shapes:
        ratios: dict[str, list[float]] = defaultdict(list)
        for trial in range(1, expected_trials + 1):
            control = profile_index[(shape, trial, "homogeneous_control")]
            lifting = profile_index[(shape, trial, "strong_planar_lifting")]
            _require(
                control["initial_q_sha256"] == lifting["initial_q_sha256"],
                "paired profiles use different initial Q",
            )
            for statistic in ("mean", "median"):
                ratios[statistic].append(
                    _profile_seconds(lifting, statistic)
                    / _profile_seconds(control, statistic)
                )
            for memory_name in ("peak_allocated_bytes", "peak_reserved_bytes"):
                control_memory = int(control["memory"][memory_name])
                lifting_memory = int(lifting["memory"][memory_name])
                _require(control_memory > 0, "CUDA memory evidence is required")
                ratios[memory_name].append(lifting_memory / control_memory)
        mean_ratio = statistics.fmean(ratios["mean"])
        median_ratio = statistics.median(ratios["median"])
        allocated_ratio = max(ratios["peak_allocated_bytes"])
        reserved_ratio = max(ratios["peak_reserved_bytes"])
        _require(mean_ratio <= timestep_limit, "mean timestep non-regression failed")
        _require(
            median_ratio <= timestep_limit,
            "median timestep non-regression failed",
        )
        _require(allocated_ratio <= allocated_limit, "allocated-memory gate failed")
        _require(reserved_ratio <= reserved_limit, "reserved-memory gate failed")
        performance["x".join(str(value) for value in shape)] = {
            "paired_mean_timestep_ratios": ratios["mean"],
            "paired_median_timestep_ratios": ratios["median"],
            "mean_timestep_ratio": mean_ratio,
            "median_timestep_ratio": median_ratio,
            "max_peak_allocated_ratio": allocated_ratio,
            "max_peak_reserved_ratio": reserved_ratio,
        }

    manufactured_report = manufactured[0]
    _require(
        manufactured_report.get("device") == "cuda",
        "manufactured device mismatch",
    )
    _require(
        manufactured_report.get("finite") is True,
        "manufactured result is non-finite",
    )
    _require(
        float(manufactured_report["minimum_rate"])
        >= float(contract["manufactured_minimum_rate"]),
        "manufactured convergence rate failed",
    )
    restart = restarts[0]
    _require(restart.get("device") == "cuda", "restart device mismatch")
    _require(
        restart.get("pointwise_execution") == "compile",
        "restart did not exercise compiled pointwise kernels",
    )
    _require(restart.get("finite") is True, "restart state is non-finite")
    _require(restart.get("all_byte_identical") is True, "restart is not exact")
    _require(
        restart.get("continuous_physical_sha256")
        == restart.get("resumed_physical_sha256"),
        "restart physical-state identities differ",
    )
    lifting_restart = restart.get("lifting_restart")
    _require(isinstance(lifting_restart, Mapping), "restart lifting identity is absent")
    _require(
        lifting_restart.get("representation") == "homogeneous_remainder",
        "checkpoint representation changed",
    )

    return {
        "schema_version": 1,
        "phase": "P8.4.5",
        "classification": "PASS_P8_4_5_PLANE_STATIC_LIFTING_H100_CLOSURE",
        "qualification_complete": True,
        "expected_commit": expected_commit,
        "report_count": len(profiles) + len(manufactured) + len(restarts),
        "performance": performance,
        "wall_residual_passed": True,
        "manufactured_convergence_passed": True,
        "restart_byte_identity_passed": True,
        "transform_call_non_regression_passed": True,
        "graph_breaks": graph_breaks,
        "production_default_changed": False,
        "compiled_runtime_promoted": False,
        "nonhomogeneous_neumann_supported": False,
        "eligible_for_p8_5_planning": True,
        "p8_5_authorized": False,
        "phase_9_authorized": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--reports-directory", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"output already exists: {args.output}")
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    reports = _objects(sorted(args.reports_directory.glob("*.json")))
    result = analyze(plan, reports, expected_commit=args.expected_commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
