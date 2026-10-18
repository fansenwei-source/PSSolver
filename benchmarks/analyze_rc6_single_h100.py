#!/usr/bin/env python3
"""Fail-closed analyzer for the RC6 installed-wheel single-H100 closure."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

from benchmarks.plan_rc6_single_h100_qualification import (
    CONSUMER_SOURCE,
    CONSUMER_WHEEL_SHA256,
    PROVIDER_BASELINE,
    PROVIDER_BASELINE_WHEEL_SHA256,
    PROVIDER_CANDIDATE,
    PROVIDER_CANDIDATE_WHEEL_SHA256,
    build_plan,
)
from benchmarks.profile_rc458_runtime_family import SCHEMA as PROFILE_SCHEMA


SCHEMA = "pssolver.rc6.single_h100_analysis.v1"
AUXILIARY_SCHEMA = "pssolver.rc6.h100_auxiliary.v1"
PASS_CLASSIFICATION = "PASS_RC6_SINGLE_H100_CUMULATIVE_QUALIFICATION"


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


def _validate_environment(report: object) -> None:
    _require(isinstance(report, dict), "environment report missing")
    _require(report.get("schema") == "pssolver.rc6.environment.v1", "environment schema")
    _require(report.get("passed") is True, "environment gate failed")
    _require(report.get("cuda_available") is True, "CUDA unavailable")
    _require(report.get("visible_device_count") == 1, "CUDA device count differs")
    _require(report.get("allocated_device") == "cuda:0", "allocated device differs")
    _require("H100" in str(report.get("device_name")), "GPU is not H100")
    _require(report.get("tf32_matmul") is False, "matmul TF32 enabled")
    _require(report.get("tf32_cudnn") is False, "cuDNN TF32 enabled")
    _require(report.get("source_shadow_import") is False, "source shadow import")
    _require(report.get("pip_check") == "PASS", "pip check failed")
    _require(report.get("imports_from_site_packages") is True, "installed imports missing")
    _require(
        report.get("installed_wheels")
        == {
            "baseline": PROVIDER_BASELINE_WHEEL_SHA256,
            "candidate": PROVIDER_CANDIDATE_WHEEL_SHA256,
            "consumer": CONSUMER_WHEEL_SHA256,
        },
        "installed wheel identities differ",
    )
    preflight = report.get("preflight")
    _require(isinstance(preflight, dict), "environment preflight missing")
    for name in (
        "node_local_tmp_rename_noreplace",
        "pytest_tmpdir_is_unique_node_local_tmp",
        "host_compiler_stdatomic_and_cuda_compile",
        "empty_triton_cache_compile",
        "empty_torchinductor_cache_compile",
    ):
        _require(preflight.get(name) is True, f"environment preflight failed: {name}")


def _validate_cuda(report: object, plan: dict[str, object]) -> None:
    _require(isinstance(report, dict), "CUDA-only report missing")
    _require(report.get("schema") == "pssolver.rc6.cuda_only.v1", "CUDA schema")
    _require(report.get("provider_commit") == PROVIDER_CANDIDATE, "CUDA commit differs")
    _require(report.get("provider_wheel_sha256") == PROVIDER_CANDIDATE_WHEEL_SHA256, "CUDA wheel differs")
    _require(report.get("passed") is True, "CUDA-only gate failed")
    _require(report.get("passed_count") == 7, "CUDA pass count differs")
    for name in ("failed_count", "skipped_count", "deselected_count"):
        _require(report.get(name) == 0, f"CUDA {name} differs")
    _require(report.get("node_ids") == plan["cuda_only_tests"], "CUDA node IDs differ")


def _validate_contract_smokes(report: object, plan: dict[str, object]) -> None:
    _require(isinstance(report, dict), "contract-smoke report missing")
    _require(report.get("schema") == "pssolver.rc6.contract_smokes.v1", "contract-smoke schema")
    _require(report.get("provider_commit") == PROVIDER_CANDIDATE, "contract-smoke commit differs")
    _require(report.get("provider_wheel_sha256") == PROVIDER_CANDIDATE_WHEEL_SHA256, "contract-smoke wheel differs")
    _require(report.get("passed") is True, "contract-smoke gate failed")
    checks = report.get("checks")
    _require(isinstance(checks, dict), "contract-smoke checks missing")
    _require(checks == plan["rc6_contract_smokes"], "contract-smoke checks differ")


def _validate_tf32(report: object) -> None:
    _require(isinstance(report, dict), "TF32 report missing")
    _require(report.get("schema") == "pssolver.rc4_5_8.tf32_scope_h100.v1", "TF32 schema")
    _require(report.get("provider_commit") == PROVIDER_CANDIDATE, "TF32 commit differs")
    _require(report.get("provider_wheel_sha256") == PROVIDER_CANDIDATE_WHEEL_SHA256, "TF32 wheel differs")
    _require(report.get("caller_state_count") == 6, "TF32 caller count differs")
    _require(report.get("runtime_family_case_count") == 48, "TF32 case count differs")
    _require(report.get("original_state_restored") is True, "TF32 state not restored")
    _require(report.get("passed") is True, "TF32 gate failed")


def _validate_checkpoints(report: object) -> None:
    _require(isinstance(report, dict), "checkpoint report missing")
    _require(report.get("schema") == "pssolver.rc6.checkpoint_h100.v1", "checkpoint schema")
    _require(report.get("provider_commit") == PROVIDER_CANDIDATE, "checkpoint commit differs")
    _require(report.get("consumer_source_commit") == CONSUMER_SOURCE, "checkpoint Consumer differs")
    _require(report.get("passed") is True, "checkpoint gate failed")
    _require(report.get("current_family_count") == 4, "checkpoint family count differs")
    _require(report.get("direction_count") == 2, "checkpoint direction count differs")
    for name in (
        "continuous_split_restart_passed",
        "transactional_restore_passed",
        "negative_gates_passed",
        "rejected_restart_output_remains_absent",
        "released_dispositions_match_cpu_record",
        "source_checkpoints_unchanged",
    ):
        _require(report.get(name) is True, f"checkpoint gate failed: {name}")
    _require(report.get("released_primary_case_count") == 13, "released case count differs")


def _validate_numerical(report: object, plan: dict[str, object]) -> None:
    _require(isinstance(report, dict), "numerical report missing")
    _require(report.get("schema") == "pssolver.rc6.numerical_h100.v1", "numerical schema")
    _require(report.get("provider_baseline_commit") == PROVIDER_BASELINE, "numerical baseline differs")
    _require(report.get("provider_candidate_commit") == PROVIDER_CANDIDATE, "numerical candidate differs")
    _require(report.get("passed") is True, "numerical gate failed")
    entries = report.get("families")
    _require(isinstance(entries, list) and len(entries) == 8, "numerical family count differs")
    expected = set(plan["numerical_gate"]["runtime_families"])
    indexed = {entry.get("family"): entry for entry in entries if isinstance(entry, dict)}
    _require(set(indexed) == expected, "numerical family set differs")
    for family, entry in indexed.items():
        _require(entry.get("completed_steps") == 2, f"{family} step count differs")
        _require(entry.get("finite") is True, f"{family} non-finite")
        _require(entry.get("fallback_used") is False, f"{family} fallback")
        _require(entry.get("byte_identical") is True, f"{family} not byte-identical")
        _require(
            _nonnegative(entry.get("relative_l2"), f"{family} relative L2")
            <= plan["numerical_gate"]["physical_relative_l2_max"],
            f"{family} relative L2 failed",
        )
        _require(
            _nonnegative(entry.get("linf"), f"{family} Linf")
            <= plan["numerical_gate"]["physical_linf_max"],
            f"{family} Linf failed",
        )
        _require(
            _nonnegative(entry.get("candidate_hermitian_violation"), f"{family} Hermitian violation")
            <= plan["numerical_gate"]["candidate_hermitian_violation_max"],
            f"{family} Hermitian violation failed",
        )


def _validate_consumer(report: object, plan: dict[str, object]) -> None:
    contract = plan["consumer_h100_gate"]
    identity = plan["consumer"]
    _require(isinstance(report, dict), "Consumer report missing")
    _require(report.get("schema") == "pssolver.rc6.consumer_h100.v1", "Consumer schema")
    _require(report.get("passed") is True, "Consumer gate failed")
    _require(report.get("source_commit") == CONSUMER_SOURCE, "Consumer source differs")
    _require(report.get("wheel_sha256") == CONSUMER_WHEEL_SHA256, "Consumer wheel differs")
    expected = {
        "passed_count": contract["expected_passed"],
        "skipped_count": contract["expected_skipped"],
        "failed_count": contract["failed_allowed"],
        "xfailed_count": contract["xfailed_allowed"],
        "deselected_count": contract["deselected_allowed"],
        "skip_node_id": contract["allowed_skip"],
    }
    for name, value in expected.items():
        _require(report.get(name) == value, f"Consumer {name} differs")
    identities = report.get("identities")
    _require(isinstance(identities, dict), "Consumer identities missing")
    _require(identities.get("periodic") == identity["periodic_identity_sha256"], "Periodic identity differs")
    _require(identities.get("channel_fixed") == identity["channel_fixed_identity_sha256"], "fixed Channel identity differs")
    _require(identities.get("channel_tolerance") == identity["channel_tolerance_identity_sha256"], "tolerance Channel identity differs")
    for name in (
        "guard_order_regression_passed",
        "periodic_and_channel_cuda_execution_finite",
        "checkpoint_replay_and_gradient_gates_passed",
    ):
        _require(report.get(name) is True, f"Consumer gate failed: {name}")
    _require(report.get("scientific_control_result") is False, "Consumer scientific claim out of scope")


def _validate_auxiliary(value: dict[str, object], plan: dict[str, object]) -> None:
    _require(value.get("schema") == AUXILIARY_SCHEMA, "auxiliary schema mismatch")
    _require(value.get("formal_h100_submission_count") == 1, "submission count differs")
    _require(value.get("automatic_retry") is False, "automatic retry was used")
    _require(value.get("second_job_submitted") is False, "second H100 job submitted")
    _validate_environment(value.get("environment"))
    _validate_cuda(value.get("cuda_only"), plan)
    _validate_contract_smokes(value.get("contract_smokes"), plan)
    _validate_tf32(value.get("tf32_scope"))
    _validate_checkpoints(value.get("checkpoints"))
    _validate_numerical(value.get("numerical"), plan)
    _validate_consumer(value.get("consumer"), plan)


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
    families = plan["numerical_gate"]["runtime_families"]
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
                _require(baseline["final_state_sha256"] == candidate["final_state_sha256"], f"{family}/{grid}/trial{trial} final state differs")
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
        "provider_baseline_commit": PROVIDER_BASELINE,
        "provider_candidate_commit": PROVIDER_CANDIDATE,
        "consumer_source_commit": CONSUMER_SOURCE,
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
