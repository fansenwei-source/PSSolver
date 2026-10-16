#!/usr/bin/env python3
"""Fail-closed analyzer for the rc5 G-1 focused H100 matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

from benchmarks.plan_rc5_g1_focused_performance_h100 import build_plan
from benchmarks.plan_rc5_g1_h100 import REPAIRED_PERIODIC_FAMILIES
from benchmarks.profile_rc458_runtime_family import SCHEMA as PROFILE_SCHEMA


SCHEMA = "pssolver.rc5.g1.focused_performance_analysis.v1"
PASS_CLASSIFICATION = "PASS_RC5_G1_FOCUSED_PERFORMANCE_MEMORY_EVIDENCE"
_TF32_OFF = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


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


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ValueError(f"{name} must be positive")
    return value


def _close(observed: float, expected: float) -> bool:
    return math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=1.0e-15)


def _sha256(value: object, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256",
    )
    return value


def _timing(profile: dict[str, object], key: str, plan: dict[str, object]) -> dict[str, float]:
    timing = profile.get("timing")
    _require(isinstance(timing, dict), f"{key} timing is missing")
    samples = timing.get("samples_seconds")
    expected_count = int(plan["timing_validity"]["raw_sample_count"])
    _require(
        isinstance(samples, list) and len(samples) == expected_count,
        f"{key} timing sample count differs",
    )
    values = [
        _number(value, f"{key} timing sample {index}", positive=True)
        for index, value in enumerate(samples)
    ]
    mean = statistics.fmean(values)
    median = statistics.median(values)
    stdev = statistics.stdev(values)
    cv = stdev / mean
    reported = {
        "mean": _number(
            timing.get("mean_timestep_seconds"), f"{key} reported mean", positive=True
        ),
        "median": _number(
            timing.get("median_timestep_seconds"),
            f"{key} reported median",
            positive=True,
        ),
        "stdev": _number(
            timing.get("sample_standard_deviation_seconds"),
            f"{key} reported sample standard deviation",
        ),
        "cv": _number(
            timing.get("coefficient_of_variation"), f"{key} reported CV"
        ),
    }
    for name, recomputed in (
        ("mean", mean),
        ("median", median),
        ("stdev", stdev),
        ("cv", cv),
    ):
        _require(
            _close(reported[name], recomputed),
            f"{key} reported {name} does not match raw timing samples",
        )
    cv_max = float(plan["timing_validity"]["coefficient_of_variation_max"])
    _require(
        cv <= cv_max,
        f"{key} timing validity failed: CV {cv!r} exceeds {cv_max!r}",
    )
    return {
        "mean_timestep_seconds": mean,
        "median_timestep_seconds": median,
        "sample_standard_deviation_seconds": stdev,
        "coefficient_of_variation": cv,
    }


def _validate_measurement_window(
    profile: dict[str, object], entry: dict[str, object], key: str
) -> None:
    if "post_flush_rewarm_steps" not in entry:
        return
    window = profile.get("measurement_window")
    replay = profile.get("diagnostic_replay")
    _require(isinstance(window, dict), f"{key} measurement window is missing")
    _require(isinstance(replay, dict), f"{key} diagnostic replay is missing")
    expected_rewarm = int(entry["post_flush_rewarm_steps"])
    expected_start = int(entry["measurement_start_completed_steps"])
    expected_runtime = int(entry["expected_runtime_completed_steps"])
    _require(window.get("warmup_steps") == entry["warmup_steps"], f"{key} window warmup differs")
    _require(window.get("post_flush_rewarm_steps") == expected_rewarm, f"{key} window rewarm differs")
    _require(window.get("timed_steps") == entry["timed_steps"], f"{key} window timed steps differ")
    _require(window.get("steps_after_warmup") == entry["warmup_steps"], f"{key} warmup clock differs")
    _require(window.get("measurement_start_completed_steps") == expected_start, f"{key} measurement start differs")
    for name in (
        "rewarm_timed",
        "rewarm_in_formal_transform_counters",
        "rewarm_in_formal_compiler_counters",
        "rewarm_in_formal_memory_peaks",
    ):
        _require(window.get(name) is False, f"{key} {name} must be false")
    _require(
        window.get("formal_memory_semantics")
        == "steady_state_after_allocation_recovery",
        f"{key} formal memory semantics differ",
    )
    snapshots = window.get("memory_snapshots")
    _require(isinstance(snapshots, dict), f"{key} memory snapshots are missing")
    _require(
        set(snapshots)
        == {
            "after_warmup",
            "after_empty_cache",
            "after_post_flush_rewarm",
            "after_formal_timed_window",
        },
        f"{key} memory snapshot phases differ",
    )
    memory_fields = {
        "allocated_bytes",
        "reserved_bytes",
        "active_bytes",
        "peak_allocated_bytes",
        "peak_reserved_bytes",
        "peak_active_bytes",
    }
    for phase, snapshot in snapshots.items():
        _require(isinstance(snapshot, dict), f"{key} {phase} memory snapshot differs")
        _require(set(snapshot) == memory_fields, f"{key} {phase} memory fields differ")
        for field, value in snapshot.items():
            _require(
                isinstance(value, int) and not isinstance(value, bool) and value >= 0,
                f"{key} {phase}.{field} must be a non-negative integer",
            )
    cold = window.get("cold_recovery_memory")
    _require(isinstance(cold, dict), f"{key} cold recovery memory is missing")
    _require(
        set(cold)
        == {
            "allocated_delta_bytes",
            "reserved_delta_bytes",
            "active_delta_bytes",
        },
        f"{key} cold recovery memory fields differ",
    )
    _require(
        all(isinstance(value, int) and not isinstance(value, bool) for value in cold.values()),
        f"{key} cold recovery memory values must be integers",
    )
    telemetry = window.get("device_telemetry_after_formal_window")
    _require(isinstance(telemetry, dict), f"{key} device telemetry is missing")
    _require(isinstance(telemetry.get("applicable"), bool), f"{key} telemetry applicability differs")
    if telemetry["applicable"]:
        rows = telemetry.get("rows")
        _require(isinstance(rows, list) and rows, f"{key} telemetry rows are missing")
        for index, row in enumerate(rows):
            _require(isinstance(row, dict), f"{key} telemetry row {index} differs")
            _require(isinstance(row.get("pstate"), str) and row["pstate"], f"{key} telemetry pstate differs")
            for field in (
                "sm_clock_mhz",
                "memory_clock_mhz",
                "power_draw_watts",
                "temperature_celsius",
            ):
                _number(row.get(field), f"{key} telemetry {field}")
    else:
        _require(
            isinstance(telemetry.get("reason"), str) and telemetry["reason"],
            f"{key} unavailable telemetry reason is missing",
        )

    _require(replay.get("separate_fresh_runtime") is True, f"{key} replay is not fresh")
    _require(
        replay.get("used_for_performance_or_memory_adjudication") is False,
        f"{key} replay entered formal adjudication",
    )
    _require(replay.get("finite") is True, f"{key} replay state is non-finite")
    _require(replay.get("completed_steps") == expected_runtime, f"{key} replay clock differs")
    _require(
        replay.get("initial_state_sha256") == profile.get("initial_state_sha256"),
        f"{key} replay initial state differs",
    )
    _require(
        replay.get("final_state_sha256") == profile.get("final_state_sha256"),
        f"{key} replay final state differs",
    )
    records = replay.get("step_records")
    _require(
        isinstance(records, list) and len(records) == entry["timed_steps"],
        f"{key} replay step-record count differs",
    )
    for index, record in enumerate(records, start=1):
        _require(isinstance(record, dict), f"{key} replay record {index} differs")
        _require(record.get("step_index") == index, f"{key} replay step index differs")
        _require(
            record.get("runtime_completed_steps") == expected_start + index,
            f"{key} replay completed-step sequence differs",
        )
        diagnostics = record.get("diagnostics")
        _require(isinstance(diagnostics, dict), f"{key} replay diagnostics are missing")
        _require(
            isinstance(diagnostics.get("applicable"), bool),
            f"{key} replay diagnostics applicability differs",
        )
        if diagnostics["applicable"]:
            observed = diagnostics.get("observed_fields")
            _require(
                isinstance(observed, list) and observed,
                f"{key} applicable diagnostics contain no observed fields",
            )
        else:
            _require(
                isinstance(diagnostics.get("reason"), str) and diagnostics["reason"],
                f"{key} unavailable diagnostics reason is missing",
            )


def _validate_profile(
    profile: dict[str, object], entry: dict[str, object], plan: dict[str, object]
) -> dict[str, object]:
    key = str(entry["key"])
    _require(profile.get("schema") == PROFILE_SCHEMA, f"{key} profile schema differs")
    config = profile.get("config")
    provider = profile.get("provider_identity")
    runtime = profile.get("runtime_identity")
    environment = profile.get("environment")
    memory = profile.get("memory")
    transforms = profile.get("transform_calls")
    compiler = profile.get("compiler")
    for name, value in (
        ("config", config),
        ("provider identity", provider),
        ("runtime identity", runtime),
        ("environment", environment),
        ("memory", memory),
        ("transform calls", transforms),
        ("compiler", compiler),
    ):
        _require(isinstance(value, dict), f"{key} {name} is missing")

    grid = plan["grid_contracts"][entry["grid"]]
    expected_config = {
        "family": entry["family"],
        "provider_role": entry["provider_role"],
        "provider_commit": entry["provider_commit"],
        "provider_wheel_sha256": entry["provider_wheel_sha256"],
        "trial": entry["trial"],
        "shape": grid["shape"],
        "lengths": grid["lengths"],
        "device": "cuda",
        "dtype": entry["dtype"],
        "warmup_steps": entry["warmup_steps"],
        "profile_steps": entry["timed_steps"],
    }
    if "post_flush_rewarm_steps" in entry:
        expected_config["post_flush_rewarm_steps"] = entry[
            "post_flush_rewarm_steps"
        ]
    for name, expected in expected_config.items():
        _require(config.get(name) == expected, f"{key} config {name} differs")
    _require(provider.get("role") == entry["provider_role"], f"{key} role differs")
    _require(
        provider.get("commit") == entry["provider_commit"], f"{key} commit differs"
    )
    _require(
        provider.get("wheel_sha256") == entry["provider_wheel_sha256"],
        f"{key} wheel differs",
    )
    _require(runtime.get("requested") == entry["runtime_path"], f"{key} requested runtime differs")
    _require(runtime.get("effective") == entry["runtime_path"], f"{key} effective runtime differs")
    _require(runtime.get("fallback_used") is False, f"{key} runtime fallback")
    _require(environment.get("allocated_device") == "cuda:0", f"{key} allocated device differs")
    _require("H100" in str(environment.get("device_name")), f"{key} is not an H100 profile")
    _require(environment.get("tf32_caller_before") == _TF32_OFF, f"{key} TF32 before differs")
    _require(environment.get("tf32_caller_after") == _TF32_OFF, f"{key} TF32 after differs")
    _require(profile.get("finite") is True, f"{key} state is non-finite")
    _require(
        _integer(profile.get("completed_profile_steps"), f"{key} completed steps")
        == entry["timed_steps"],
        f"{key} completed profile steps differ",
    )
    _require(
        _integer(profile.get("runtime_completed_steps"), f"{key} runtime steps")
        == entry.get(
            "expected_runtime_completed_steps",
            entry["warmup_steps"] + entry["timed_steps"],
        ),
        f"{key} runtime completed steps differ",
    )
    _require(compiler.get("graph_breaks") == 0, f"{key} graph break")
    _require(compiler.get("compile_fallback") is False, f"{key} compile fallback")
    _validate_measurement_window(profile, entry, key)

    raw_forward = _integer(transforms.get("forward"), f"{key} raw forward")
    raw_inverse = _integer(transforms.get("inverse"), f"{key} raw inverse")
    timed_steps = int(entry["timed_steps"])
    _require(
        raw_forward == entry["expected_forward_raw_calls"],
        f"{key} raw forward transform count differs",
    )
    _require(
        raw_inverse == entry["expected_inverse_raw_calls"],
        f"{key} raw inverse transform count differs",
    )
    _require(
        transforms.get("forward_per_step") == raw_forward / timed_steps,
        f"{key} forward transform rate differs",
    )
    _require(
        transforms.get("inverse_per_step") == raw_inverse / timed_steps,
        f"{key} inverse transform rate differs",
    )
    steady_forward = raw_forward - int(
        entry["scheduled_forward_calls_in_timed_window"]
    )
    steady_inverse = raw_inverse - int(
        entry["scheduled_inverse_calls_in_timed_window"]
    )
    _require(
        steady_forward == entry["forward_transforms_per_step"] * timed_steps,
        f"{key} steady forward transform count differs",
    )
    _require(
        steady_inverse == entry["inverse_transforms_per_step"] * timed_steps,
        f"{key} steady inverse transform count differs",
    )

    timing = _timing(profile, key, plan)
    allocated = _integer(
        memory.get("peak_allocated_bytes"), f"{key} peak allocated", positive=True
    )
    active = _integer(
        memory.get("peak_active_bytes"), f"{key} peak active", positive=True
    )
    reserved = _integer(
        memory.get("peak_reserved_bytes"), f"{key} peak reserved", positive=True
    )
    if "post_flush_rewarm_steps" in entry:
        formal_snapshot = profile["measurement_window"]["memory_snapshots"][
            "after_formal_timed_window"
        ]
        _require(
            formal_snapshot["peak_allocated_bytes"] == allocated,
            f"{key} formal allocated peak differs from memory gate",
        )
        _require(
            formal_snapshot["peak_active_bytes"] == active,
            f"{key} formal active peak differs from memory gate",
        )
        _require(
            formal_snapshot["peak_reserved_bytes"] == reserved,
            f"{key} formal reserved peak differs from memory gate",
        )
    initial = _sha256(profile.get("initial_state_sha256"), f"{key} initial state")
    final = _sha256(profile.get("final_state_sha256"), f"{key} final state")
    return {
        "key": key,
        "family": entry["family"],
        "grid": entry["grid"],
        "trial": entry["trial"],
        "role": entry["provider_role"],
        "within_pair_order": entry["within_pair_order"],
        "timing": timing,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "initial_state_sha256": initial,
        "final_state_sha256": final,
    }


def _analyze_with_plan(
    profile_root: Path,
    plan: dict[str, object],
    *,
    analysis_schema: str,
    pass_classification: str,
) -> dict[str, object]:
    expected = plan["profile_matrix"]
    _require(profile_root.is_dir(), "profile root is not a directory")
    records: dict[str, dict[str, object]] = {}
    for entry in expected:
        key = str(entry["key"])
        path = profile_root / f"{key}.json"
        _require(path.is_file() and not path.is_symlink(), f"profile missing: {key}")
        profile = _json(path)
        records[key] = _validate_profile(profile, entry, plan)
    actual_files = sorted(path.name for path in profile_root.glob("*.json"))
    expected_files = sorted(f"{entry['key']}.json" for entry in expected)
    _require(actual_files == expected_files, "profile file set differs")

    thresholds = plan["thresholds"]
    groups: list[dict[str, object]] = []
    for cell in plan["focused_cells"]:
        family = str(cell["family"])
        grid = str(cell["grid"])
        baseline_means: list[float] = []
        candidate_means: list[float] = []
        paired_ratios: list[float] = []
        baseline_allocated: list[int] = []
        candidate_allocated: list[int] = []
        baseline_reserved: list[int] = []
        candidate_reserved: list[int] = []
        pair_records: list[dict[str, object]] = []
        trial_count = int(
            plan["execution_order"]["paired_trials_per_cell"]
        )
        for trial in range(1, trial_count + 1):
            baseline = records[f"baseline_{family}_{grid}_trial{trial}"]
            candidate = records[f"candidate_{family}_{grid}_trial{trial}"]
            _require(
                baseline["initial_state_sha256"] == candidate["initial_state_sha256"],
                f"{family}/{grid}/trial{trial} initial state differs",
            )
            if family not in REPAIRED_PERIODIC_FAMILIES:
                _require(
                    baseline["final_state_sha256"]
                    == candidate["final_state_sha256"],
                    f"{family}/{grid}/trial{trial} final state differs",
                )
            baseline_mean = float(baseline["timing"]["mean_timestep_seconds"])
            candidate_mean = float(candidate["timing"]["mean_timestep_seconds"])
            paired_ratio = candidate_mean / baseline_mean
            baseline_means.append(baseline_mean)
            candidate_means.append(candidate_mean)
            paired_ratios.append(paired_ratio)
            baseline_allocated.append(int(baseline["peak_allocated_bytes"]))
            candidate_allocated.append(int(candidate["peak_allocated_bytes"]))
            baseline_reserved.append(int(baseline["peak_reserved_bytes"]))
            candidate_reserved.append(int(candidate["peak_reserved_bytes"]))
            pair_records.append(
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
                    "baseline_mean_timestep_seconds": baseline_mean,
                    "candidate_mean_timestep_seconds": candidate_mean,
                    "candidate_over_baseline_mean_ratio": paired_ratio,
                    "baseline_cv": baseline["timing"]["coefficient_of_variation"],
                    "candidate_cv": candidate["timing"]["coefficient_of_variation"],
                }
            )
        mean_ratio = statistics.fmean(candidate_means) / statistics.fmean(
            baseline_means
        )
        median_ratio = statistics.median(candidate_means) / statistics.median(
            baseline_means
        )
        allocated_ratio = max(candidate_allocated) / max(baseline_allocated)
        reserved_ratio = max(candidate_reserved) / max(baseline_reserved)
        _require(
            mean_ratio <= thresholds["mean_timestep_ratio_max"],
            f"{family}/{grid} mean timestep regression",
        )
        _require(
            median_ratio <= thresholds["median_of_profile_means_ratio_max"],
            f"{family}/{grid} median timestep regression",
        )
        _require(
            max(paired_ratios)
            <= thresholds["every_paired_mean_timestep_ratio_max"],
            f"{family}/{grid} paired timestep regression",
        )
        _require(
            allocated_ratio <= thresholds["peak_allocated_group_ratio_max"],
            f"{family}/{grid} allocated memory regression",
        )
        _require(
            reserved_ratio <= thresholds["peak_reserved_group_ratio_max"],
            f"{family}/{grid} reserved memory regression",
        )
        groups.append(
            {
                "family": family,
                "grid": grid,
                "replaces_v6_failed_gate": cell["v6_failed_gate"],
                "v6_failed_observed": cell["v6_observed"],
                "mean_timestep_ratio": mean_ratio,
                "median_of_profile_means_ratio": median_ratio,
                "paired_mean_timestep_ratios": paired_ratios,
                "peak_allocated_group_ratio": allocated_ratio,
                "peak_reserved_group_ratio": reserved_ratio,
                "pairs": pair_records,
                "passed": True,
            }
        )
    return {
        "schema": analysis_schema,
        "classification": pass_classification,
        "focused_evidence_complete": True,
        "focused_cell_count": len(groups),
        "profile_count": len(records),
        "groups": groups,
        "timing_validity": plan["timing_validity"],
        "thresholds": thresholds,
        "qualification_complete": False,
        "g1_complete": False,
        "eligible_for_cumulative_reassembly": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }


def analyze(profile_root: Path) -> dict[str, object]:
    return _analyze_with_plan(
        profile_root,
        build_plan(),
        analysis_schema=SCHEMA,
        pass_classification=PASS_CLASSIFICATION,
    )


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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, analyze(args.profile_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
