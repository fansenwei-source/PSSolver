#!/usr/bin/env python3
"""Replay RC4.5.8 with an explicit raw/refresh/base transform contract."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import tempfile

from benchmarks.analyze_rc458_tf32_h100 import (
    _atomic_json,
    _json,
    _require,
    analyze as analyze_original,
)
from benchmarks.plan_rc458_tf32_h100 import build_plan


SCHEMA = "pssolver.rc4_5_8.transform_counter_recovery_analysis.v1"
PASS_CLASSIFICATION = (
    "PASS_RC4_5_8_SINGLE_H100_TF32_CUMULATIVE_CLOSURE_"
    "WITH_TRANSFORM_COUNTER_LAYER_RECOVERY"
)
REFRESH_FAMILIES = frozenset({"channel_legacy", "channel_compiled_v2"})
REFRESH_INTERVAL = 20
DYNAMIC_TRANSFORM_GROUPS = 1


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


def _scheduled_refreshes(*, warmup_steps: int, timed_steps: int) -> int:
    _require(warmup_steps >= 0, "warmup steps must be non-negative")
    _require(timed_steps > 0, "timed steps must be positive")
    return (warmup_steps + timed_steps) // REFRESH_INTERVAL - (
        warmup_steps // REFRESH_INTERVAL
    )


def adjudicate_transform_counters(
    profile: dict[str, object], entry: dict[str, object]
) -> tuple[dict[str, object], dict[str, object]]:
    """Validate raw evidence and return an analyzer-only base-layer view."""
    key = entry.get("key")
    _require(isinstance(key, str), "profile key must be a string")
    config = profile.get("config")
    calls = profile.get("transform_calls")
    _require(isinstance(config, dict), f"{key} config must be an object")
    _require(isinstance(calls, dict), f"{key} transform_calls must be an object")
    family = config.get("family")
    _require(family == entry.get("family"), f"{key} family differs")
    warmup_steps = _integer(config.get("warmup_steps"), f"{key} warmup_steps")
    timed_steps = _integer(
        profile.get("completed_profile_steps"), f"{key} completed_profile_steps"
    )
    _require(warmup_steps == entry.get("warmup_steps"), f"{key} warmup differs")
    _require(timed_steps == entry.get("timed_steps"), f"{key} timed steps differ")
    _require(
        profile.get("runtime_completed_steps") == warmup_steps + timed_steps,
        f"{key} runtime clock differs",
    )
    raw_forward = _integer(calls.get("forward"), f"{key} raw forward")
    raw_inverse = _integer(calls.get("inverse"), f"{key} raw inverse")
    _require(
        calls.get("forward_per_step") == raw_forward / timed_steps,
        f"{key} raw forward rate differs",
    )
    _require(
        calls.get("inverse_per_step") == raw_inverse / timed_steps,
        f"{key} raw inverse rate differs",
    )
    expected_forward = _integer(
        entry.get("forward_transforms_per_step"), f"{key} expected forward"
    )
    expected_inverse = _integer(
        entry.get("inverse_transforms_per_step"), f"{key} expected inverse"
    )
    scheduled_refreshes = (
        _scheduled_refreshes(warmup_steps=warmup_steps, timed_steps=timed_steps)
        if family in REFRESH_FAMILIES
        else 0
    )
    scheduled_forward = scheduled_refreshes * DYNAMIC_TRANSFORM_GROUPS
    base_forward = raw_forward - scheduled_forward
    _require(base_forward >= 0, f"{key} refresh calls exceed raw forward calls")
    _require(
        base_forward == expected_forward * timed_steps,
        f"{key} base forward transform count differs",
    )
    _require(
        raw_inverse == expected_inverse * timed_steps,
        f"{key} inverse transform count differs",
    )
    normalized = copy.deepcopy(profile)
    normalized_calls = normalized["transform_calls"]
    _require(isinstance(normalized_calls, dict), "copied transform calls missing")
    normalized_calls["forward_per_step"] = base_forward / timed_steps
    record = {
        "key": key,
        "family": family,
        "provider_role": entry.get("provider_role"),
        "grid": entry.get("grid"),
        "trial": entry.get("trial"),
        "warmup_steps": warmup_steps,
        "timed_steps": timed_steps,
        "refresh_interval": REFRESH_INTERVAL if family in REFRESH_FAMILIES else None,
        "dynamic_transform_group_count": (
            DYNAMIC_TRANSFORM_GROUPS if family in REFRESH_FAMILIES else 0
        ),
        "scheduled_refreshes": scheduled_refreshes,
        "scheduled_refresh_forward_calls": scheduled_forward,
        "raw_forward_calls": raw_forward,
        "raw_forward_per_step": raw_forward / timed_steps,
        "base_forward_calls": base_forward,
        "base_forward_per_step": base_forward / timed_steps,
        "raw_inverse_calls": raw_inverse,
        "inverse_per_step": raw_inverse / timed_steps,
        "passed": True,
    }
    return normalized, record


def analyze(
    profile_root: Path, auxiliary: dict[str, object]
) -> dict[str, object]:
    profile_root = profile_root.expanduser().resolve()
    plan = build_plan()
    expected = plan["profile_matrix"]
    _require(isinstance(expected, list), "profile matrix must be a list")
    expected_files = sorted(f"{entry['key']}.json" for entry in expected)
    actual_files = sorted(path.name for path in profile_root.glob("*.json"))
    _require(actual_files == expected_files, "source profile file set differs")
    records: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="pssolver-rc458-counter-replay-") as raw:
        normalized_root = Path(raw)
        for entry in expected:
            _require(isinstance(entry, dict), "profile matrix entry must be an object")
            key = entry.get("key")
            _require(isinstance(key, str), "profile key must be a string")
            source = profile_root / f"{key}.json"
            _require(source.is_file() and not source.is_symlink(), f"profile missing: {key}")
            normalized, record = adjudicate_transform_counters(_json(source), entry)
            (normalized_root / source.name).write_text(
                json.dumps(normalized, allow_nan=False, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            records.append(record)
        original = analyze_original(normalized_root, auxiliary)
    refresh_records = [
        record for record in records if record["family"] in REFRESH_FAMILIES
    ]
    _require(len(records) == 96, "counter replay profile count differs")
    _require(len(refresh_records) == 24, "refresh profile count differs")
    _require(
        all(record["scheduled_refreshes"] == 3 for record in refresh_records),
        "scheduled refresh count differs",
    )
    _require(
        all(record["raw_forward_calls"] == 663 for record in refresh_records),
        "raw Channel forward count differs",
    )
    _require(
        all(record["base_forward_calls"] == 660 for record in refresh_records),
        "base Channel forward count differs",
    )
    _require(
        all(record["raw_inverse_calls"] == 2160 for record in refresh_records),
        "raw Channel inverse count differs",
    )
    return {
        **original,
        "schema": SCHEMA,
        "classification": PASS_CLASSIFICATION,
        "qualification_complete": True,
        "analysis_execution": "cpu_only_replay_of_frozen_h100_profiles",
        "h100_profiles_rerun": False,
        "scientific_timestep_executed_by_recovery": False,
        "transform_counter_contract": {
            "raw_evidence_modified": False,
            "original_analyzer_modified": False,
            "profile_schema": "pssolver.rc4_5_8.runtime_family_profile.v1",
            "profile_count": len(records),
            "refresh_profile_count": len(refresh_records),
            "refresh_families": sorted(REFRESH_FAMILIES),
            "refresh_interval": REFRESH_INTERVAL,
            "dynamic_transform_group_count": DYNAMIC_TRANSFORM_GROUPS,
            "scheduled_refreshes_per_profile": 3,
            "raw_forward_calls_per_refresh_profile": 663,
            "scheduled_refresh_forward_calls_per_profile": 3,
            "base_forward_calls_per_refresh_profile": 660,
            "base_forward_per_step": 11.0,
            "inverse_calls_per_refresh_profile": 2160,
            "inverse_per_step": 36.0,
            "records": records,
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
