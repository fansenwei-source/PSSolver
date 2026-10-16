#!/usr/bin/env python3
"""Aggregate rc5 G-1 instrumentation without issuing a qualification PASS."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

from benchmarks.diagnose_rc5_g1_timing_instrumentation import SCHEMA as PROFILE_SCHEMA
from benchmarks.plan_rc5_g1_timing_instrumentation import build_plan


SCHEMA = "pssolver.rc5.g1.timing_instrumentation_analysis.v1"
CLASSIFICATION = "ANALYSIS_ONLY_INSTRUMENTATION_EVIDENCE_COMPLETE"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _cv(values: list[float]) -> float:
    _require(len(values) >= 3, "CV requires at least three values")
    mean = statistics.fmean(values)
    _require(math.isfinite(mean) and mean > 0.0, "timing mean is invalid")
    return statistics.stdev(values) / mean


def _window_summary(
    window: dict[str, object], *, synchronized: bool
) -> dict[str, object]:
    records = window.get("records")
    _require(
        isinstance(records, list) and len(records) == 60,
        "window must contain 60 steps",
    )
    cuda = [float(record["cuda_event_seconds"]) for record in records]
    host = [float(record["host_duration_seconds"]) for record in records]
    _require(
        all(math.isfinite(value) and value > 0.0 for value in cuda + host),
        "timing contains an invalid value",
    )
    median_cuda = statistics.median(cuda)
    median_host = statistics.median(host)
    cuda_spikes = [
        index for index, value in enumerate(cuda) if value / median_cuda > 1.10
    ]
    host_spikes = [
        index for index, value in enumerate(host) if value / median_host > 1.10
    ]
    gc_steps = sorted(
        {
            int(event["step_index"])
            for event in window["gc"]["events"]
            if event["step_index"] is not None
        }
    )
    refresh_steps = [
        int(record["step_index_zero_based"])
        for record in records
        if record["scheduled_refresh"]["due"] is True
    ]
    diagnostics_applicable = sum(
        record["public_diagnostics"].get("applicable") is True for record in records
    )
    telemetry = window["background_gpu_telemetry"]
    _require(isinstance(telemetry, dict), "telemetry record is missing")
    return {
        "synchronize_each_step": synchronized,
        "cuda_event_cv": _cv(cuda),
        "host_duration_cv": _cv(host),
        "cuda_spike_indices_zero_based": cuda_spikes,
        "host_spike_indices_zero_based": host_spikes,
        "gc_step_indices_zero_based": gc_steps,
        "refresh_step_indices_zero_based": refresh_steps,
        "cuda_spikes_coincident_with_gc": sorted(set(cuda_spikes) & set(gc_steps)),
        "cuda_spikes_coincident_with_refresh": sorted(
            set(cuda_spikes) & set(refresh_steps)
        ),
        "host_spikes_coincident_with_gc": sorted(set(host_spikes) & set(gc_steps)),
        "public_diagnostics_applicable_step_count": diagnostics_applicable,
        "telemetry_row_count": int(telemetry["row_count"]),
        "telemetry_parse_error_count": len(telemetry["parse_errors"]),
        "telemetry_visible_device_binding": telemetry["binding"],
    }


def analyze(profile_root: Path) -> dict[str, object]:
    plan = build_plan()
    expected = plan["profile_matrix"]
    expected_names = sorted(f"{entry['key']}.json" for entry in expected)
    actual_names = sorted(path.name for path in profile_root.glob("*.json"))
    _require(actual_names == expected_names, "instrumentation profile set differs")
    records: list[dict[str, object]] = []
    for entry in expected:
        path = profile_root / f"{entry['key']}.json"
        profile = _json(path)
        _require(
            profile.get("schema") == PROFILE_SCHEMA,
            f"{path.name} schema differs",
        )
        _require(
            profile.get("analysis_only") is True,
            f"{path.name} is not analysis-only",
        )
        _require(
            profile.get("used_for_formal_performance_adjudication") is False,
            f"{path.name} claims formal use",
        )
        config = profile.get("config")
        windows = profile.get("windows")
        identity = profile.get("identity")
        _require(
            isinstance(config, dict)
            and isinstance(windows, dict)
            and isinstance(identity, dict),
            f"{path.name} structure differs",
        )
        _require(
            config.get("family") == entry["family"],
            f"{path.name} family differs",
        )
        _require(
            config.get("provider_commit") == entry["provider_commit"],
            f"{path.name} provider differs",
        )
        _require(
            identity.get("initial_state_sha256_equal") is True,
            f"{path.name} initial state differs",
        )
        _require(
            identity.get("final_state_sha256_equal") is True,
            f"{path.name} final state differs",
        )
        asynchronous = _window_summary(
            windows["asynchronous_compatibility"], synchronized=False
        )
        synchronized = _window_summary(
            windows["synchronized_attribution_replay"], synchronized=True
        )
        _require(
            asynchronous["telemetry_parse_error_count"] == 0,
            f"{path.name} async telemetry parse failed",
        )
        _require(
            synchronized["telemetry_parse_error_count"] == 0,
            f"{path.name} sync telemetry parse failed",
        )
        _require(
            asynchronous["telemetry_row_count"] > 0,
            f"{path.name} async telemetry is empty",
        )
        _require(
            synchronized["telemetry_row_count"] > 0,
            f"{path.name} sync telemetry is empty",
        )
        records.append(
            {
                "key": entry["key"],
                "family": entry["family"],
                "trial": entry["trial"],
                "asynchronous": asynchronous,
                "synchronized": synchronized,
            }
        )
    families = []
    for family in plan["families"]:
        selected = [record for record in records if record["family"] == family]
        _require(len(selected) == 3, f"{family} diagnostic trial count differs")
        families.append(
            {
                "family": family,
                "profile_count": 3,
                "asynchronous_cuda_cv": [
                    record["asynchronous"]["cuda_event_cv"]
                    for record in selected
                ],
                "synchronized_cuda_cv": [
                    record["synchronized"]["cuda_event_cv"]
                    for record in selected
                ],
                "asynchronous_gc_spike_coincidence_count": sum(
                    len(record["asynchronous"]["cuda_spikes_coincident_with_gc"])
                    for record in selected
                ),
                "synchronized_gc_spike_coincidence_count": sum(
                    len(record["synchronized"]["cuda_spikes_coincident_with_gc"])
                    for record in selected
                ),
                "asynchronous_refresh_spike_coincidence_count": sum(
                    len(record["asynchronous"]["cuda_spikes_coincident_with_refresh"])
                    for record in selected
                ),
            }
        )
    return {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "analysis_only": True,
        "instrumentation_evidence_complete": True,
        "profile_count": len(records),
        "formal_v10_classification_preserved": (
            "FAIL_RC5_G1_CORRECTED_MEASUREMENT_WINDOW_GATE"
        ),
        "formal_requalification_performed": False,
        "root_cause_automatically_adjudicated": False,
        "correlation_is_causation": False,
        "thresholds_changed": False,
        "outlier_removal_performed": False,
        "trial_replacement_performed": False,
        "production_default_changed": False,
        "families": families,
        "profiles": records,
    }


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    payload = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, analyze(args.profile_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
