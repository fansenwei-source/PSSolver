#!/usr/bin/env python3
"""Fail-closed analyzer for the v0.3 S-09 single-H100 qualification."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

from benchmarks.plan_v0_3_s09_h100 import (
    BASELINE_COMMIT,
    BASELINE_WHEEL_SHA256,
    CANDIDATE_COMMIT,
    CANDIDATE_WHEEL_SHA256,
    CONSUMER_RUNTIME_COMMIT,
    CONSUMER_WHEEL_SHA256,
    build_plan,
)


PROFILE_SCHEMA = "pssolver.rc4_5_8.runtime_family_profile.v1"
SCHEMA = "pssolver.v0_3.s09.single_h100_analysis.v1"
AUXILIARY_SCHEMA = "pssolver.v0_3.s09.h100_auxiliary.v1"
PASS_CLASSIFICATION = "PASS_V0_3_S09_SINGLE_H100_CUMULATIVE_QUALIFICATION"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"regular JSON file required: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    return value


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
        if json.loads(path.read_text(encoding="utf-8")) != expected:
            raise RuntimeError("analysis JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _validate_auxiliary(value: dict[str, object], plan: dict[str, object]) -> None:
    _require(value.get("schema") == AUXILIARY_SCHEMA, "auxiliary schema mismatch")
    _require(value.get("formal_h100_submission_count") == 1, "submission count differs")
    _require(value.get("automatic_retry") is False, "automatic retry was used")
    _require(value.get("second_job_submitted") is False, "second H100 job submitted")
    stages = value.get("stages")
    expected = [name for name in plan["job_stage_order"] if name != "paired_performance_memory_profiles" and name != "cumulative_analyzer_and_checksums"]
    _require(isinstance(stages, list), "auxiliary stages missing")
    indexed = {row.get("name"): row for row in stages if isinstance(row, dict)}
    _require(list(indexed) == expected, "auxiliary stage order differs")
    for name, row in indexed.items():
        _require(row.get("passed") is True, f"auxiliary stage failed: {name}")
        _require(row.get("provider_candidate_commit") == CANDIDATE_COMMIT, f"candidate identity differs: {name}")
        _require(row.get("candidate_wheel_sha256") == CANDIDATE_WHEEL_SHA256, f"candidate wheel differs: {name}")
    environment = indexed["node_local_tmp_environment_and_host_compiler"]
    _require(environment.get("baseline_wheel_sha256") == BASELINE_WHEEL_SHA256, "baseline wheel differs")
    _require(environment.get("consumer_wheel_sha256") == CONSUMER_WHEEL_SHA256, "Consumer wheel differs")
    for key in ("cuda_available", "single_visible_h100", "allocated_cuda_0", "installed_wheels_only", "source_shadow_import_false", "pip_check", "node_local_tmp", "host_compiler", "empty_compile_caches"):
        _require(environment.get(key) is True, f"environment gate failed: {key}")
    long_horizon = indexed["cuda_long_horizon_hermitian"]
    _require(long_horizon.get("cases") == plan["long_horizon"]["cases"], "long-horizon case set differs")
    _require(long_horizon.get("all_finite") is True, "long-horizon non-finite state")
    _require(long_horizon.get("all_within_bound") is True, "Hermitian bound failed")
    tf32 = indexed["tf32_forward_backward_scope"]
    _require(tf32.get("caller_state_count") == 6, "TF32 caller-state count differs")
    _require(tf32.get("functional_case_count") == 12, "TF32 functional case count differs")
    for key in ("forward_observed_policy", "backward_observed_policy", "success_restored", "exception_restored", "nested_lifo", "outside_scope_preserved"):
        _require(tf32.get(key) is True, f"TF32 gate failed: {key}")
    checkpoints = indexed["checkpoint_restart_and_released_fixture_matrix"]
    for key in ("current_matrix", "released_fixtures", "restart_exact", "negative_gates", "sources_unchanged"):
        _require(checkpoints.get(key) is True, f"checkpoint gate failed: {key}")
    numerical = indexed["numerical_gradient_taylor_and_replay"]
    for key in ("standard_regression", "gradient", "finite_difference", "taylor", "replay"):
        _require(numerical.get(key) is True, f"numerical gate failed: {key}")
    _require(
        numerical.get("gradient_taylor_evidence_stage")
        == "independent_control_consumer",
        "numerical gradient/Taylor evidence binding differs",
    )
    consumer = indexed["independent_control_consumer"]
    _require(consumer.get("consumer_source_commit") == CONSUMER_RUNTIME_COMMIT, "Consumer source differs")
    _require(consumer.get("consumer_wheel_sha256") == CONSUMER_WHEEL_SHA256, "Consumer wheel differs")
    for key in ("installed_wheels_only", "functional_api_1_1", "provider_identity_migrated", "periodic_gradient", "channel_gradient", "checkpointed_adjoint", "negative_controls", "finite_cuda_smokes"):
        _require(consumer.get(key) is True, f"Consumer gate failed: {key}")


def _profile_center(samples: list[float], block_size: int) -> float:
    blocks = [statistics.fmean(samples[index:index + block_size]) for index in range(0, len(samples), block_size)]
    return statistics.median(blocks)


def analyze(profile_root: Path, auxiliary: dict[str, object]) -> dict[str, object]:
    plan = build_plan()
    thresholds = plan["thresholds"]
    _validate_auxiliary(auxiliary, plan)
    off = {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    observed: dict[str, dict[str, object]] = {}
    for entry in plan["profile_matrix"]:
        key = str(entry["key"])
        path = profile_root / f"{key}.json"
        profile = _json(path)
        _require(profile.get("schema") == PROFILE_SCHEMA, f"profile schema: {key}")
        config = profile.get("config")
        provider = profile.get("provider_identity")
        runtime = profile.get("runtime_identity")
        environment = profile.get("environment")
        timing = profile.get("timing")
        memory = profile.get("memory")
        transforms = profile.get("transform_calls")
        compiler = profile.get("compiler")
        for name, value in (("config", config), ("provider", provider), ("runtime", runtime), ("environment", environment), ("timing", timing), ("memory", memory), ("transforms", transforms), ("compiler", compiler)):
            _require(isinstance(value, dict), f"{key} {name} missing")
        grid = plan["grid_contracts"][entry["grid"]]
        expected_config = {
            "family": entry["family"], "trial": entry["trial"],
            "shape": grid["shape"], "lengths": grid["lengths"],
            "device": "cuda", "dtype": entry["dtype"],
        }
        for name, value in expected_config.items():
            _require(config.get(name) == value, f"{key} config {name} differs")
        _require(provider.get("role") == entry["provider_role"], f"{key} role differs")
        _require(provider.get("commit") == entry["provider_commit"], f"{key} commit differs")
        _require(provider.get("wheel_sha256") == entry["provider_wheel_sha256"], f"{key} wheel differs")
        _require(runtime.get("requested") == runtime.get("effective"), f"{key} runtime mismatch")
        _require(runtime.get("fallback_used") is False, f"{key} runtime fallback")
        _require(environment.get("allocated_device") == "cuda:0", f"{key} device differs")
        _require("H100" in str(environment.get("device_name")), f"{key} not H100")
        _require(environment.get("tf32_caller_before") == off, f"{key} TF32 before differs")
        _require(environment.get("tf32_caller_after") == off, f"{key} TF32 after differs")
        _require(profile.get("finite") is True, f"{key} non-finite")
        timed_steps = _integer(profile.get("completed_profile_steps"), f"{key} completed steps")
        _require(timed_steps == entry["timed_steps"], f"{key} step count differs")
        _require(profile.get("runtime_completed_steps") == entry["warmup_steps"] + timed_steps, f"{key} runtime count differs")
        _require(compiler.get("graph_breaks") == 0, f"{key} graph break")
        _require(compiler.get("compile_fallback") is False, f"{key} compile fallback")
        raw_forward = _integer(transforms.get("forward"), f"{key} raw forward")
        raw_inverse = _integer(transforms.get("inverse"), f"{key} raw inverse")
        _require(raw_forward == entry["expected_forward_raw_calls"], f"{key} raw forward differs")
        _require(raw_inverse == entry["expected_inverse_raw_calls"], f"{key} raw inverse differs")
        _require(transforms.get("forward_per_step") == raw_forward / timed_steps, f"{key} forward rate differs")
        _require(transforms.get("inverse_per_step") == raw_inverse / timed_steps, f"{key} inverse rate differs")
        _require(raw_forward - entry["scheduled_refreshes_in_timed_window"] == entry["steady_forward_transforms_per_step"] * timed_steps, f"{key} steady forward differs")
        _require(raw_inverse == entry["steady_inverse_transforms_per_step"] * timed_steps, f"{key} steady inverse differs")
        samples = timing.get("samples_seconds")
        _require(isinstance(samples, list) and len(samples) == 60, f"{key} timing samples")
        clean_samples = [_positive(value, f"{key} sample {index}") for index, value in enumerate(samples)]
        _require(_positive(timing.get("mean_timestep_seconds"), f"{key} mean") == statistics.fmean(clean_samples), f"{key} mean is not raw-sample mean")
        _positive(timing.get("median_timestep_seconds"), f"{key} median")
        _require(_integer(memory.get("peak_allocated_bytes"), f"{key} allocated") > 0, f"{key} allocated invalid")
        _require(_integer(memory.get("peak_reserved_bytes"), f"{key} reserved") > 0, f"{key} reserved invalid")
        for name in ("initial_state_sha256", "final_state_sha256"):
            value = profile.get(name)
            _require(isinstance(value, str) and len(value) == 64, f"{key} {name}")
        observed[key] = profile
    actual = sorted(path.name for path in profile_root.glob("*.json"))
    expected = sorted(f"{entry['key']}.json" for entry in plan["profile_matrix"])
    _require(actual == expected, "profile file set differs")

    groups: list[dict[str, object]] = []
    families = plan["profile_contract"]["runtime_families"]
    for family in families:
        grids = sorted({entry["grid"] for entry in plan["profile_matrix"] if entry["family"] == family})
        for grid in grids:
            baseline_samples: list[float] = []
            candidate_samples: list[float] = []
            baseline_means: list[float] = []
            candidate_means: list[float] = []
            ratios: list[float] = []
            allocated: dict[str, list[int]] = {"baseline": [], "candidate": []}
            reserved: dict[str, list[int]] = {"baseline": [], "candidate": []}
            for trial in range(1, 7):
                baseline = observed[f"baseline_{family}_{grid}_trial{trial}"]
                candidate = observed[f"candidate_{family}_{grid}_trial{trial}"]
                _require(baseline["initial_state_sha256"] == candidate["initial_state_sha256"], f"{family}/{grid}/trial{trial} initial state differs")
                bs = [float(value) for value in baseline["timing"]["samples_seconds"]]
                cs = [float(value) for value in candidate["timing"]["samples_seconds"]]
                baseline_samples.extend(bs)
                candidate_samples.extend(cs)
                baseline_means.append(statistics.fmean(bs))
                candidate_means.append(statistics.fmean(cs))
                ratios.append(_profile_center(cs, 10) / _profile_center(bs, 10))
                allocated["baseline"].append(baseline["memory"]["peak_allocated_bytes"])
                allocated["candidate"].append(candidate["memory"]["peak_allocated_bytes"])
                reserved["baseline"].append(baseline["memory"]["peak_reserved_bytes"])
                reserved["candidate"].append(candidate["memory"]["peak_reserved_bytes"])
            aggregate_ratio = statistics.fmean(candidate_samples) / statistics.fmean(baseline_samples)
            median_means_ratio = statistics.median(candidate_means) / statistics.median(baseline_means)
            median_ratio = statistics.median(ratios)
            odd_median = statistics.median(ratios[0::2])
            even_median = statistics.median(ratios[1::2])
            imbalance = max(odd_median, even_median) / min(odd_median, even_median)
            allocated_ratio = max(allocated["candidate"]) / max(allocated["baseline"])
            reserved_ratio = max(reserved["candidate"]) / max(reserved["baseline"])
            checks = {
                "aggregate_all_sample_mean": aggregate_ratio <= thresholds["aggregate_all_sample_mean_ratio_max"],
                "median_of_profile_means": median_means_ratio <= thresholds["median_of_profile_means_ratio_max"],
                "median_of_six_trial_ratios": median_ratio <= thresholds["median_of_six_trial_ratios_max"],
                "baseline_then_candidate_stratum": odd_median <= thresholds["baseline_then_candidate_stratum_median_ratio_max"],
                "candidate_then_baseline_stratum": even_median <= thresholds["candidate_then_baseline_stratum_median_ratio_max"],
                "order_stratum_imbalance": imbalance <= thresholds["order_stratum_imbalance_ratio_max"],
                "minimum_reference_trials": sum(value <= thresholds["individual_trial_ratio_reference_max"] for value in ratios) >= thresholds["minimum_trials_at_or_below_individual_reference"],
                "peak_allocated": allocated_ratio <= thresholds["peak_allocated_group_ratio_max"],
                "peak_reserved": reserved_ratio <= thresholds["peak_reserved_group_ratio_max"],
            }
            _require(all(checks.values()), f"{family}/{grid} performance or memory gate failed: {checks}")
            groups.append(
                {
                    "family": family, "grid": grid,
                    "aggregate_all_sample_mean_ratio": aggregate_ratio,
                    "median_of_profile_means_ratio": median_means_ratio,
                    "median_of_six_trial_ratios": median_ratio,
                    "baseline_then_candidate_stratum_median_ratio": odd_median,
                    "candidate_then_baseline_stratum_median_ratio": even_median,
                    "order_stratum_imbalance_ratio": imbalance,
                    "trial_ratios": ratios,
                    "trials_at_or_below_individual_reference": sum(value <= thresholds["individual_trial_ratio_reference_max"] for value in ratios),
                    "peak_allocated_group_ratio": allocated_ratio,
                    "peak_reserved_group_ratio": reserved_ratio,
                    "checks": checks, "passed": True,
                }
            )
    _require(len(groups) == 16, "profile cell count differs")
    return {
        "schema": SCHEMA,
        "classification": PASS_CLASSIFICATION,
        "qualification_complete": True,
        "provider_baseline_commit": BASELINE_COMMIT,
        "provider_candidate_commit": CANDIDATE_COMMIT,
        "consumer_source_commit": CONSUMER_RUNTIME_COMMIT,
        "profile_count": len(observed),
        "group_count": len(groups),
        "groups": groups,
        "auxiliary": auxiliary,
        "eligible_for_release_integration_review": True,
        "independent_post_release_verification_required": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
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
