#!/usr/bin/env python3
"""Fail-closed cumulative analyzer for the rc5 G-1 H100 qualification."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

from benchmarks.plan_rc5_g1_h100 import (
    CONSUMER_SOURCE,
    CONSUMER_WHEEL_SHA256,
    PROVIDER_BASELINE,
    PROVIDER_BASELINE_WHEEL_SHA256,
    PROVIDER_CANDIDATE,
    PROVIDER_CANDIDATE_WHEEL_SHA256,
    REPAIRED_PERIODIC_FAMILIES,
    build_plan,
)
from benchmarks.profile_rc458_runtime_family import SCHEMA as PROFILE_SCHEMA


SCHEMA = "pssolver.rc5.g1.single_h100_analysis.v1"
AUXILIARY_SCHEMA = "pssolver.rc5.g1.h100_auxiliary.v1"
PASS_CLASSIFICATION = "PASS_RC5_G1_SINGLE_H100_CUMULATIVE_QUALIFICATION"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return value


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be positive and finite")
    return result


def _nonnegative(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
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
    _require(report.get("schema") == "pssolver.rc5.g1.environment.v1", "environment schema")
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


def _validate_cuda(report: object, expected: list[str]) -> None:
    _require(isinstance(report, dict), "CUDA-only report missing")
    _require(report.get("schema") == "pssolver.rc5.g1.cuda_only.v1", "CUDA schema")
    _require(report.get("provider_commit") == PROVIDER_CANDIDATE, "CUDA commit differs")
    _require(report.get("provider_wheel_sha256") == PROVIDER_CANDIDATE_WHEEL_SHA256, "CUDA wheel differs")
    _require(report.get("passed") is True, "CUDA-only gate failed")
    _require(report.get("passed_count") == 7, "CUDA pass count differs")
    _require(report.get("failed_count") == 0, "CUDA failure recorded")
    _require(report.get("skipped_count") == 0, "CUDA skip recorded")
    _require(report.get("deselected_count") == 0, "CUDA deselection recorded")
    _require(report.get("node_ids") == expected, "CUDA node IDs differ")


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
    _require(report.get("schema") == "pssolver.rc5.g1.checkpoint_h100.v1", "checkpoint schema")
    _require(report.get("provider_commit") == PROVIDER_CANDIDATE, "checkpoint commit differs")
    _require(report.get("consumer_source_commit") == CONSUMER_SOURCE, "checkpoint Consumer differs")
    _require(report.get("passed") is True, "checkpoint gate failed")
    _require(report.get("current_family_count") == 4, "checkpoint family count differs")
    _require(report.get("direction_count") == 2, "checkpoint direction count differs")
    _require(report.get("continuous_split_restart_passed") is True, "restart gate failed")
    _require(report.get("transactional_restore_passed") is True, "transaction gate failed")
    _require(report.get("negative_gates_passed") is True, "checkpoint negative gate failed")
    _require(report.get("released_primary_case_count") == 13, "released case count differs")
    _require(report.get("released_dispositions_match_cpu_record") is True, "released disposition differs")
    _require(report.get("source_checkpoints_unchanged") is True, "source checkpoint mutated")


def _validate_numerical(report: object, plan: dict[str, object]) -> None:
    _require(isinstance(report, dict), "numerical report missing")
    _require(report.get("schema") == "pssolver.rc5.g1.numerical_h100.v1", "numerical schema")
    _require(report.get("provider_baseline_commit") == PROVIDER_BASELINE, "numerical baseline differs")
    _require(report.get("provider_candidate_commit") == PROVIDER_CANDIDATE, "numerical candidate differs")
    _require(report.get("passed") is True, "numerical gate failed")
    entries = report.get("families")
    _require(isinstance(entries, list) and len(entries) == 8, "numerical family count differs")
    indexed = {entry.get("family"): entry for entry in entries if isinstance(entry, dict)}
    expected = set(plan["tf32_scope_gate"]["runtime_families"])
    _require(set(indexed) == expected, "numerical family set differs")
    repaired = set(REPAIRED_PERIODIC_FAMILIES)
    thresholds = plan["thresholds"]
    for family, entry in indexed.items():
        _require(entry.get("finite") is True, f"{family} non-finite")
        _require(entry.get("fallback_used") is False, f"{family} fallback")
        if family not in repaired:
            _require(entry.get("comparison") == "byte_identity", f"{family} comparison differs")
            _require(entry.get("byte_identical") is True, f"{family} not byte-identical")
            continue
        _require(entry.get("comparison") == "physical_tolerance", f"{family} comparison differs")
        components = entry.get("components")
        _require(isinstance(components, dict), f"{family} components missing")
        _require(set(components) == {"Q", "velocity", "pressure"}, f"{family} components differ")
        for component, metrics in components.items():
            _require(isinstance(metrics, dict), f"{family}/{component} metrics missing")
            _require(
                _nonnegative(metrics.get("relative_l2"), f"{family}/{component} relative L2")
                <= thresholds["physical_relative_l2_max"],
                f"{family}/{component} relative L2 failed",
            )
            _require(
                _nonnegative(metrics.get("linf"), f"{family}/{component} Linf")
                <= thresholds["physical_linf_max"],
                f"{family}/{component} Linf failed",
            )
        _require(
            _nonnegative(entry.get("candidate_hermitian_violation"), f"{family} Hermitian violation")
            == 0.0,
            f"{family} Hermitian violation is not exactly zero",
        )


def _validate_consumer(report: object, contract: dict[str, object]) -> None:
    _require(isinstance(report, dict), "Consumer report missing")
    _require(report.get("schema") == "pssolver.rc5.g1.consumer_h100.v1", "Consumer schema")
    _require(report.get("passed") is True, "Consumer gate failed")
    _require(report.get("source_commit") == CONSUMER_SOURCE, "Consumer source differs")
    _require(report.get("wheel_sha256") == CONSUMER_WHEEL_SHA256, "Consumer wheel differs")
    for name in ("passed", "skipped", "failed", "xfailed", "deselected"):
        expected_name = {
            "passed": "expected_passed",
            "skipped": "expected_skipped",
            "failed": "failed_allowed",
            "xfailed": "xfailed_allowed",
            "deselected": "deselected_allowed",
        }[name]
        _require(report.get(f"{name}_count") == contract[expected_name], f"Consumer {name} count differs")
    _require(report.get("skip_node_id") == contract["allowed_skip"], "Consumer skip differs")
    identities = report.get("identities")
    _require(isinstance(identities, dict), "Consumer identities missing")
    _require(identities.get("periodic") == contract["periodic_identity_sha256"], "Periodic identity differs")
    _require(identities.get("channel_fixed") == contract["channel_fixed_identity_sha256"], "fixed Channel identity differs")
    _require(identities.get("channel_tolerance") == contract["channel_tolerance_identity_sha256"], "tolerance Channel identity differs")
    _require(report.get("old_periodic_pin_rejected_before_mutation") is True, "old pin negative gate failed")


def _validate_auxiliary(value: dict[str, object], plan: dict[str, object]) -> None:
    _require(value.get("schema") == AUXILIARY_SCHEMA, "auxiliary schema mismatch")
    _require(value.get("formal_h100_submission_count") == 1, "submission count differs")
    _require(value.get("automatic_retry") is False, "automatic retry was used")
    _require(value.get("second_job_submitted") is False, "second H100 job submitted")
    _validate_environment(value.get("environment"))
    _validate_cuda(value.get("cuda_only"), plan["cuda_only_tests"])
    _validate_tf32(value.get("tf32_scope"))
    _validate_checkpoints(value.get("checkpoints"))
    _validate_numerical(value.get("numerical"), plan)
    _validate_consumer(value.get("consumer"), plan["consumer_h100_gate"])


def analyze(profile_root: Path, auxiliary: dict[str, object]) -> dict[str, object]:
    plan = build_plan()
    thresholds = plan["thresholds"]
    _validate_auxiliary(auxiliary, plan)
    expected = plan["profile_matrix"]
    observed: dict[str, dict[str, object]] = {}
    off = {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    }
    for entry in expected:
        key = str(entry["key"])
        path = profile_root / f"{key}.json"
        _require(path.is_file() and not path.is_symlink(), f"profile missing: {key}")
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
        for name, value in (
            ("config", config), ("provider", provider), ("runtime", runtime),
            ("environment", environment), ("timing", timing), ("memory", memory),
            ("transforms", transforms), ("compiler", compiler),
        ):
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
        _require(profile.get("completed_profile_steps") == 60, f"{key} step count differs")
        _require(profile.get("runtime_completed_steps") == 70, f"{key} runtime count differs")
        _require(compiler.get("graph_breaks") == 0, f"{key} graph break")
        _require(compiler.get("compile_fallback") is False, f"{key} compile fallback")
        _require(transforms.get("forward_per_step") == entry["forward_transforms_per_step"], f"{key} forward transforms")
        _require(transforms.get("inverse_per_step") == entry["inverse_transforms_per_step"], f"{key} inverse transforms")
        samples = timing.get("samples_seconds")
        _require(isinstance(samples, list) and len(samples) == 60, f"{key} timing samples")
        for index, sample in enumerate(samples):
            _positive(sample, f"{key} sample {index}")
        _positive(timing.get("mean_timestep_seconds"), f"{key} mean")
        _positive(timing.get("median_timestep_seconds"), f"{key} median")
        _require(_integer(memory.get("peak_allocated_bytes"), f"{key} allocated") > 0, f"{key} allocated invalid")
        _require(_integer(memory.get("peak_reserved_bytes"), f"{key} reserved") > 0, f"{key} reserved invalid")
        for name in ("initial_state_sha256", "final_state_sha256"):
            value = profile.get(name)
            _require(isinstance(value, str) and len(value) == 64, f"{key} {name}")
        observed[key] = profile
    actual_files = sorted(path.name for path in profile_root.glob("*.json"))
    expected_files = sorted(f"{entry['key']}.json" for entry in expected)
    _require(actual_files == expected_files, "profile file set differs")

    groups: list[dict[str, object]] = []
    for family in plan["tf32_scope_gate"]["runtime_families"]:
        grids = sorted({entry["grid"] for entry in expected if entry["family"] == family})
        for grid in grids:
            baseline_times: list[float] = []
            candidate_times: list[float] = []
            paired: list[float] = []
            baseline_allocated: list[int] = []
            candidate_allocated: list[int] = []
            baseline_reserved: list[int] = []
            candidate_reserved: list[int] = []
            for trial in range(1, 4):
                baseline = observed[f"baseline_{family}_{grid}_trial{trial}"]
                candidate = observed[f"candidate_{family}_{grid}_trial{trial}"]
                _require(baseline["initial_state_sha256"] == candidate["initial_state_sha256"], f"{family}/{grid}/trial{trial} initial state differs")
                if family not in REPAIRED_PERIODIC_FAMILIES:
                    _require(baseline["final_state_sha256"] == candidate["final_state_sha256"], f"{family}/{grid}/trial{trial} final state differs")
                bt = _positive(baseline["timing"]["mean_timestep_seconds"], "baseline time")
                ct = _positive(candidate["timing"]["mean_timestep_seconds"], "candidate time")
                baseline_times.append(bt)
                candidate_times.append(ct)
                paired.append(ct / bt)
                baseline_allocated.append(baseline["memory"]["peak_allocated_bytes"])
                candidate_allocated.append(candidate["memory"]["peak_allocated_bytes"])
                baseline_reserved.append(baseline["memory"]["peak_reserved_bytes"])
                candidate_reserved.append(candidate["memory"]["peak_reserved_bytes"])
            mean_ratio = statistics.fmean(candidate_times) / statistics.fmean(baseline_times)
            median_ratio = statistics.median(candidate_times) / statistics.median(baseline_times)
            allocated_ratio = max(candidate_allocated) / max(baseline_allocated)
            reserved_ratio = max(candidate_reserved) / max(baseline_reserved)
            _require(mean_ratio <= thresholds["mean_timestep_ratio_max"], f"{family}/{grid} mean regression")
            _require(median_ratio <= thresholds["median_timestep_ratio_max"], f"{family}/{grid} median regression")
            _require(max(paired) <= thresholds["paired_timestep_ratio_max"], f"{family}/{grid} paired regression")
            _require(allocated_ratio <= thresholds["peak_allocated_group_ratio_max"], f"{family}/{grid} allocated regression")
            _require(reserved_ratio <= thresholds["peak_reserved_group_ratio_max"], f"{family}/{grid} reserved regression")
            groups.append({
                "family": family, "grid": grid, "mean_timestep_ratio": mean_ratio,
                "median_timestep_ratio": median_ratio, "paired_timestep_ratios": paired,
                "peak_allocated_group_ratio": allocated_ratio,
                "peak_reserved_group_ratio": reserved_ratio,
                "final_state_policy": "physical_tolerance" if family in REPAIRED_PERIODIC_FAMILIES else "byte_identity",
                "passed": True,
            })
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
        "g1_complete": False,
        "post_release_verification_required": True,
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
