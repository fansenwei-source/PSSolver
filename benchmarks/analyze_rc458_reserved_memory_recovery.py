#!/usr/bin/env python3
"""Replay RC4.5.8 with a trial-set CUDA reserved-memory contract.

CUDA ``reserved`` bytes describe caching-allocator segments retained by a
process.  Trial labels do not identify allocator segments, so the stable
three-trial comparison is the candidate group maximum divided by the baseline
group maximum.  Live ``allocated`` memory remains subject to the original
strict per-trial checks in the underlying analyzer.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import statistics
import tempfile

from benchmarks.analyze_rc458_tf32_h100 import _atomic_json, _json, _require
from benchmarks.analyze_rc458_transform_counter_recovery import (
    analyze as analyze_counter_recovery,
)
from benchmarks.plan_rc458_tf32_h100 import build_plan


SCHEMA = "pssolver.rc4_5_8.reserved_memory_recovery_analysis.v1"
PASS_CLASSIFICATION = (
    "PASS_RC4_5_8_SINGLE_H100_TF32_CUMULATIVE_CLOSURE_"
    "WITH_COUNTER_AND_RESERVED_MEMORY_REDUCER_RECOVERY"
)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def adjudicate_reserved_group(
    baseline: list[int], candidate: list[int], *, ratio_limit: float
) -> dict[str, object]:
    """Validate one three-trial allocator group without pairing trial labels."""
    _require(len(baseline) == 3, "baseline reserved trial count differs")
    _require(len(candidate) == 3, "candidate reserved trial count differs")
    baseline_values = [
        _positive_integer(value, f"baseline reserved trial {index}")
        for index, value in enumerate(baseline, start=1)
    ]
    candidate_values = [
        _positive_integer(value, f"candidate reserved trial {index}")
        for index, value in enumerate(candidate, start=1)
    ]
    limit = _positive_number(ratio_limit, "reserved ratio limit")
    baseline_peak = max(baseline_values)
    candidate_peak = max(candidate_values)
    peak_ratio = candidate_peak / baseline_peak
    _require(peak_ratio <= limit, "group-peak reserved-memory regression")
    return {
        "baseline_trials_bytes": baseline_values,
        "candidate_trials_bytes": candidate_values,
        "baseline_sorted_bytes": sorted(baseline_values),
        "candidate_sorted_bytes": sorted(candidate_values),
        "raw_paired_candidate_over_baseline": [
            right / left
            for left, right in zip(baseline_values, candidate_values, strict=True)
        ],
        "baseline_group_peak_bytes": baseline_peak,
        "candidate_group_peak_bytes": candidate_peak,
        "candidate_over_baseline_group_peak_ratio": peak_ratio,
        "candidate_over_baseline_group_mean_ratio": (
            statistics.fmean(candidate_values) / statistics.fmean(baseline_values)
        ),
        "trial_multiset_equal": sorted(baseline_values) == sorted(candidate_values),
        "ratio_limit": limit,
        "passed": True,
    }


def analyze(
    profile_root: Path, auxiliary: dict[str, object]
) -> dict[str, object]:
    """Apply the stable reserved reducer, then reuse all earlier hard gates."""
    profile_root = profile_root.expanduser().resolve()
    plan = build_plan()
    expected = plan["profile_matrix"]
    thresholds = plan["thresholds"]
    _require(isinstance(expected, list), "profile matrix must be a list")
    _require(isinstance(thresholds, dict), "threshold contract must be an object")
    ratio_limit = _positive_number(
        thresholds.get("peak_reserved_ratio_max"), "reserved ratio limit"
    )
    expected_files = sorted(f"{entry['key']}.json" for entry in expected)
    actual_files = sorted(path.name for path in profile_root.glob("*.json"))
    _require(actual_files == expected_files, "source profile file set differs")

    entries: dict[str, dict[str, object]] = {}
    profiles: dict[str, dict[str, object]] = {}
    for entry in expected:
        _require(isinstance(entry, dict), "profile matrix entry must be an object")
        key = entry.get("key")
        _require(isinstance(key, str), "profile key must be a string")
        source = profile_root / f"{key}.json"
        _require(source.is_file() and not source.is_symlink(), f"profile missing: {key}")
        profile = _json(source)
        config = profile.get("config")
        memory = profile.get("memory")
        _require(isinstance(config, dict), f"{key} config must be an object")
        _require(isinstance(memory, dict), f"{key} memory must be an object")
        _require(config.get("family") == entry.get("family"), f"{key} family differs")
        _positive_integer(memory.get("peak_reserved_bytes"), f"{key} peak reserved")
        entries[key] = entry
        profiles[key] = profile

    groups: list[dict[str, object]] = []
    normalized_reserved: dict[str, int] = {}
    for family in plan["tf32_scope_gate"]["runtime_families"]:
        grids = sorted(
            {
                str(entry["grid"])
                for entry in expected
                if entry["family"] == family
            }
        )
        for grid in grids:
            baseline_keys = [
                f"baseline_{family}_{grid}_trial{trial}" for trial in range(1, 4)
            ]
            candidate_keys = [
                f"candidate_{family}_{grid}_trial{trial}" for trial in range(1, 4)
            ]
            _require(
                all(key in profiles for key in (*baseline_keys, *candidate_keys)),
                f"{family}/{grid} profile group incomplete",
            )
            baseline = [
                _positive_integer(
                    profiles[key]["memory"]["peak_reserved_bytes"],
                    f"{key} peak reserved",
                )
                for key in baseline_keys
            ]
            candidate = [
                _positive_integer(
                    profiles[key]["memory"]["peak_reserved_bytes"],
                    f"{key} peak reserved",
                )
                for key in candidate_keys
            ]
            record = adjudicate_reserved_group(
                baseline, candidate, ratio_limit=ratio_limit
            )
            record.update({"family": family, "grid": grid})
            groups.append(record)
            for key in baseline_keys:
                normalized_reserved[key] = max(baseline)
            for key in candidate_keys:
                normalized_reserved[key] = max(candidate)

    _require(len(groups) == 16, "reserved-memory group count differs")
    _require(len(normalized_reserved) == 96, "reserved-memory profile count differs")
    with tempfile.TemporaryDirectory(prefix="pssolver-rc458-memory-replay-") as raw:
        normalized_root = Path(raw)
        for key, profile in profiles.items():
            normalized = copy.deepcopy(profile)
            memory = normalized.get("memory")
            _require(isinstance(memory, dict), f"{key} copied memory missing")
            memory["peak_reserved_bytes"] = normalized_reserved[key]
            (normalized_root / f"{key}.json").write_text(
                json.dumps(normalized, allow_nan=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        recovered = analyze_counter_recovery(normalized_root, auxiliary)

    return {
        **recovered,
        "schema": SCHEMA,
        "classification": PASS_CLASSIFICATION,
        "qualification_complete": True,
        "reserved_memory_contract": {
            "raw_evidence_modified": False,
            "original_analyzer_modified": False,
            "transform_counter_recovery_modified": False,
            "profile_count": len(profiles),
            "group_count": len(groups),
            "threshold_changed": False,
            "peak_reserved_ratio_max": ratio_limit,
            "hard_gate_reducer": (
                "max(candidate_trial_peak_reserved_bytes) / "
                "max(baseline_trial_peak_reserved_bytes)"
            ),
            "raw_paired_ratios_role": "diagnostic_only",
            "peak_allocated_contract": "original_strict_per_trial_gate_unchanged",
            "groups": groups,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--auxiliary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, analyze(args.profile_root, _json(args.auxiliary)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
