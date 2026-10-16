#!/usr/bin/env python3
"""Diagnose, without re-adjudicating, rc5 G-1 timing variability.

This analyzer is deliberately analysis-only.  It preserves every raw sample,
does not replace trials, and never emits a qualification PASS classification.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
from pathlib import Path
import statistics

from benchmarks.plan_rc5_g1_focused_performance_h100 import build_plan


SCHEMA = "pssolver.rc5.g1.timing_variability_analysis.v1"
CLASSIFICATION = (
    "HARNESS_FIRST_SAMPLE_CONTAMINATION_AND_RESIDUAL_IN_WINDOW_VARIABILITY"
)
SPIKE_OVER_MEDIAN_RATIO = 1.10


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _number(value: object, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ValueError(f"{name} must be positive")
    return result


def _close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=1.0e-15)


def _sample_statistics(samples: list[float]) -> dict[str, float]:
    _require(len(samples) >= 3, "at least three timing samples are required")
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    stdev = statistics.stdev(samples)
    return {
        "mean_timestep_seconds": mean,
        "median_timestep_seconds": median,
        "sample_standard_deviation_seconds": stdev,
        "coefficient_of_variation": stdev / mean,
    }


def _correlation(left: list[float], right: list[float]) -> float | None:
    _require(len(left) == len(right), "correlation inputs differ in length")
    _require(len(left) >= 2, "correlation requires two samples")
    left_mean = statistics.fmean(left)
    right_mean = statistics.fmean(right)
    left_delta = [value - left_mean for value in left]
    right_delta = [value - right_mean for value in right]
    left_norm = math.sqrt(sum(value * value for value in left_delta))
    right_norm = math.sqrt(sum(value * value for value in right_delta))
    if left_norm == 0.0 or right_norm == 0.0:
        return None
    return sum(a * b for a, b in zip(left_delta, right_delta, strict=True)) / (
        left_norm * right_norm
    )


def _profiler_source_audit(path: Path) -> dict[str, object]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    function = next(
        (
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_profile"
        ),
        None,
    )
    _require(function is not None, "run_profile is missing from profiler source")
    calls: dict[str, list[int]] = {
        "gc.collect": [],
        "torch.cuda.empty_cache": [],
        "torch.cuda.reset_peak_memory_stats": [],
        "timer.reset": [],
        "_timed_steps": [],
    }
    for node in ast.walk(function):
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func)
        if name in calls:
            calls[name].append(node.lineno)
    for name, lines in calls.items():
        _require(len(lines) == 1, f"expected one {name} call in run_profile")
    ordered = [
        calls["gc.collect"][0],
        calls["torch.cuda.empty_cache"][0],
        calls["torch.cuda.reset_peak_memory_stats"][0],
        calls["timer.reset"][0],
        calls["_timed_steps"][0],
    ]
    _require(ordered == sorted(ordered), "profiler reset/timing call order differs")
    return {
        "path": str(path.resolve()),
        "calls": {name: lines[0] for name, lines in calls.items()},
        "empty_cache_after_warmup_before_timing": True,
        "untimed_rewarm_between_empty_cache_and_timing": False,
        "first_post_empty_cache_step_is_timed": True,
    }


def _profile_record(
    path: Path, entry: dict[str, object], cv_max: float
) -> dict[str, object]:
    profile = _json(path)
    config = profile.get("config")
    timing = profile.get("timing")
    _require(isinstance(config, dict), f"{path.name} config is missing")
    _require(isinstance(timing, dict), f"{path.name} timing is missing")
    _require(config.get("family") == entry["family"], f"{path.name} family differs")
    _require(
        config.get("provider_role") == entry["provider_role"],
        f"{path.name} provider role differs",
    )
    _require(config.get("trial") == entry["trial"], f"{path.name} trial differs")
    raw = timing.get("samples_seconds")
    expected_count = int(entry["timed_steps"])
    _require(
        isinstance(raw, list) and len(raw) == expected_count,
        f"{path.name} timing sample count differs",
    )
    samples = [
        _number(value, f"{path.name} sample {index}", positive=True)
        for index, value in enumerate(raw)
    ]
    full = _sample_statistics(samples)
    post_first = _sample_statistics(samples[1:])
    for key, computed in full.items():
        reported = _number(timing.get(key), f"{path.name} reported {key}")
        _require(
            _close(reported, computed),
            f"{path.name} reported {key} does not match raw samples",
        )
    median = full["median_timestep_seconds"]
    maximum = max(samples)
    maximum_index = samples.index(maximum)
    spike_indices = [
        index
        for index, value in enumerate(samples)
        if value / median > SPIKE_OVER_MEDIAN_RATIO
    ]
    return {
        "key": entry["key"],
        "family": entry["family"],
        "grid": entry["grid"],
        "role": entry["provider_role"],
        "trial": entry["trial"],
        "within_pair_order": entry["within_pair_order"],
        "full_window": full,
        "full_window_cv_failed": full["coefficient_of_variation"] > cv_max,
        "diagnostic_excluding_first_sample": post_first,
        "diagnostic_excluding_first_sample_cv_failed": (
            post_first["coefficient_of_variation"] > cv_max
        ),
        "first_sample_seconds": samples[0],
        "first_sample_over_median_ratio": samples[0] / median,
        "maximum_sample_seconds": maximum,
        "maximum_sample_index": maximum_index,
        "maximum_over_median_ratio": maximum / median,
        "diagnostic_spike_over_median_ratio": SPIKE_OVER_MEDIAN_RATIO,
        "diagnostic_spike_indices": spike_indices,
        "raw_samples_preserved": True,
    }


def analyze(profile_root: Path, profiler_source: Path) -> dict[str, object]:
    plan = build_plan()
    expected = plan["profile_matrix"]
    _require(profile_root.is_dir(), "profile root is not a directory")
    expected_names = sorted(f"{entry['key']}.json" for entry in expected)
    actual_names = sorted(path.name for path in profile_root.glob("*.json"))
    _require(actual_names == expected_names, "profile file set differs")
    cv_max = float(plan["timing_validity"]["coefficient_of_variation_max"])
    records: dict[str, dict[str, object]] = {}
    for entry in expected:
        key = str(entry["key"])
        records[key] = _profile_record(profile_root / f"{key}.json", entry, cv_max)

    families: list[dict[str, object]] = []
    for cell in plan["focused_cells"]:
        family = str(cell["family"])
        grid = str(cell["grid"])
        selected = [
            record
            for record in records.values()
            if record["family"] == family and record["grid"] == grid
        ]
        full_failures = sum(bool(record["full_window_cv_failed"]) for record in selected)
        residual_failures = sum(
            bool(record["diagnostic_excluding_first_sample_cv_failed"])
            for record in selected
        )
        first_overhead = sum(
            float(record["first_sample_over_median_ratio"])
            > SPIKE_OVER_MEDIAN_RATIO
            for record in selected
        )
        pairs: list[dict[str, object]] = []
        for trial in range(1, 7):
            baseline = records[f"baseline_{family}_{grid}_trial{trial}"]
            candidate = records[f"candidate_{family}_{grid}_trial{trial}"]
            baseline_profile = _json(profile_root / f"{baseline['key']}.json")
            candidate_profile = _json(profile_root / f"{candidate['key']}.json")
            baseline_samples = [
                float(value)
                for value in baseline_profile["timing"]["samples_seconds"]
            ]
            candidate_samples = [
                float(value)
                for value in candidate_profile["timing"]["samples_seconds"]
            ]
            pairs.append(
                {
                    "trial": trial,
                    "execution_order": [
                        role
                        for _, role in sorted(
                            (
                                (baseline["within_pair_order"], "baseline"),
                                (candidate["within_pair_order"], "candidate"),
                            )
                        )
                    ],
                    "full_sample_correlation": _correlation(
                        baseline_samples, candidate_samples
                    ),
                    "diagnostic_excluding_first_sample_correlation": _correlation(
                        baseline_samples[1:], candidate_samples[1:]
                    ),
                    "full_mean_ratio": (
                        statistics.fmean(candidate_samples)
                        / statistics.fmean(baseline_samples)
                    ),
                    "diagnostic_excluding_first_sample_mean_ratio": (
                        statistics.fmean(candidate_samples[1:])
                        / statistics.fmean(baseline_samples[1:])
                    ),
                }
            )
        families.append(
            {
                "family": family,
                "grid": grid,
                "profile_count": len(selected),
                "full_window_cv_failure_count": full_failures,
                "diagnostic_excluding_first_sample_cv_failure_count": residual_failures,
                "profiles_with_first_sample_overhead": first_overhead,
                "diagnostic_spike_count": sum(
                    len(record["diagnostic_spike_indices"]) for record in selected
                ),
                "maximum_cv": max(
                    float(record["full_window"]["coefficient_of_variation"])
                    for record in selected
                ),
                "pairs": pairs,
            }
        )

    full_failure_count = sum(
        bool(record["full_window_cv_failed"]) for record in records.values()
    )
    residual_failure_count = sum(
        bool(record["diagnostic_excluding_first_sample_cv_failed"])
        for record in records.values()
    )
    first_overhead_count = sum(
        float(record["first_sample_over_median_ratio"])
        > SPIKE_OVER_MEDIAN_RATIO
        for record in records.values()
    )
    source_audit = _profiler_source_audit(profiler_source)
    _require(full_failure_count > 0, "input evidence does not contain a CV failure")
    _require(first_overhead_count > 0, "first-sample overhead was not observed")
    _require(residual_failure_count > 0, "no residual in-window variability was observed")
    return {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "analysis_only": True,
        "formal_v7_classification_preserved": (
            "FAIL_RC5_G1_FOCUSED_PERFORMANCE_MEMORY_GATE"
        ),
        "profile_count": len(records),
        "coefficient_of_variation_max": cv_max,
        "full_window_cv_failure_count": full_failure_count,
        "diagnostic_excluding_first_sample_cv_failure_count": residual_failure_count,
        "profiles_with_first_sample_overhead": first_overhead_count,
        "first_sample_contamination_supported": True,
        "first_sample_contamination_is_complete_explanation": False,
        "residual_in_window_variability_supported": True,
        "profiler_source_audit": source_audit,
        "families": families,
        "profiles": [records[key] for key in sorted(records)],
        "raw_samples_modified": False,
        "outlier_removal_performed": False,
        "trial_replacement_performed": False,
        "thresholds_changed": False,
        "qualification_gate_change_authorized": False,
        "new_h100_submission_count": 0,
        "eligible_for_cumulative_reassembly": False,
        "g1_complete": False,
        "recommended_next_step": (
            "pre-register a measurement-window correction that separates "
            "post-empty-cache allocation recovery from timed steps, then validate "
            "the remaining sequential-step variability without deleting samples"
        ),
    }


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
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
        _require(
            json.loads(path.read_text(encoding="utf-8")) == expected,
            "analysis JSON round-trip differs",
        )
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--profiler-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, analyze(args.profile_root, args.profiler_source))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
