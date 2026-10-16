#!/usr/bin/env python3
"""Analysis-only replay of the proposed rc5 G-1 paired-block contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.plan_rc5_g1_h100 import REPAIRED_PERIODIC_FAMILIES
from benchmarks.plan_rc5_g1_paired_block_contract import (
    BLOCK_COUNT,
    BLOCK_SIZE,
    build_plan,
)


SCHEMA = "pssolver.rc5.g1.paired_block_replay.v1"
PROFILE_SCHEMA = "pssolver.rc4_5_8.runtime_family_profile.v1"
PASS_CLASSIFICATION = "ANALYSIS_ONLY_V10_REPLAY_PASSES_PROPOSED_V13_CONTRACT"
FAIL_CLASSIFICATION = "ANALYSIS_ONLY_V10_REPLAY_FAILS_PROPOSED_V13_CONTRACT"
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
    _require(isinstance(value, dict), f"{path} must contain an object")
    return value


def _number(value: object, name: str, *, positive: bool = False) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{name} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    if positive:
        _require(result > 0.0, f"{name} must be positive")
    return result


def _integer(value: object, name: str, *, positive: bool = False) -> int:
    _require(
        not isinstance(value, bool) and isinstance(value, int),
        f"{name} must be an integer",
    )
    if positive:
        _require(value > 0, f"{name} must be positive")
    return value


def _close(left: float, right: float) -> bool:
    return math.isclose(left, right, rel_tol=1.0e-12, abs_tol=1.0e-15)


def _sha256(value: object, name: str) -> str:
    _require(
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value),
        f"{name} must be a lowercase SHA-256",
    )
    return value


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_progress(
    profile_root: Path,
    progress_root: Path,
    expected: list[dict[str, object]],
) -> None:
    _require(progress_root.is_dir(), "progress root is not a directory")
    expected_names = [f"{index:03d}.json" for index in range(1, len(expected) + 1)]
    actual_names = sorted(path.name for path in progress_root.glob("*.json"))
    _require(actual_names == expected_names, "progress file set differs")
    completed: list[dict[str, object]] = []
    for index, entry in enumerate(expected, start=1):
        profile_path = profile_root / f"{entry['key']}.json"
        completed.append(
            {
                "key": entry["key"],
                "sequence": index,
                "sha256": _file_sha256(profile_path),
            }
        )
        path = progress_root / f"{index:03d}.json"
        _require(path.is_file() and not path.is_symlink(), f"progress missing: {index}")
        record = _json(path)
        _require(
            record.get("schema") == "pssolver.rc5.g1.measurement_window_progress.v1",
            f"progress {index} schema differs",
        )
        _require(record.get("completed") == completed, f"progress {index} differs")


def _timing(profile: dict[str, object], key: str) -> dict[str, object]:
    timing = profile.get("timing")
    _require(isinstance(timing, dict), f"{key} timing is missing")
    raw = timing.get("samples_seconds")
    _require(isinstance(raw, list) and len(raw) == 60, f"{key} requires 60 samples")
    samples = [
        _number(value, f"{key} sample {index}", positive=True)
        for index, value in enumerate(raw)
    ]
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    stdev = statistics.stdev(samples)
    cv = stdev / mean
    reported = {
        "mean": _number(timing.get("mean_timestep_seconds"), f"{key} mean", positive=True),
        "median": _number(timing.get("median_timestep_seconds"), f"{key} median", positive=True),
        "stdev": _number(timing.get("sample_standard_deviation_seconds"), f"{key} stdev"),
        "cv": _number(timing.get("coefficient_of_variation"), f"{key} CV"),
    }
    for name, recomputed in (
        ("mean", mean),
        ("median", median),
        ("stdev", stdev),
        ("cv", cv),
    ):
        _require(_close(reported[name], recomputed), f"{key} reported {name} differs")
    blocks = [
        statistics.fmean(samples[start : start + BLOCK_SIZE])
        for start in range(0, len(samples), BLOCK_SIZE)
    ]
    _require(len(blocks) == BLOCK_COUNT, f"{key} block count differs")
    return {
        "samples_seconds": samples,
        "mean_timestep_seconds": mean,
        "median_timestep_seconds": median,
        "sample_standard_deviation_seconds": stdev,
        "coefficient_of_variation": cv,
        "cv_at_or_below_reference": cv <= 0.05,
        "fixed_block_means_seconds": blocks,
    }


def _validate_profile(
    profile: dict[str, object], entry: dict[str, object], plan: dict[str, object]
) -> dict[str, object]:
    key = str(entry["key"])
    _require(profile.get("schema") == PROFILE_SCHEMA, f"{key} schema differs")
    for section in (
        "config",
        "provider_identity",
        "runtime_identity",
        "environment",
        "memory",
        "transform_calls",
        "compiler",
    ):
        _require(isinstance(profile.get(section), dict), f"{key} {section} is missing")
    config = profile["config"]
    provider = profile["provider_identity"]
    runtime = profile["runtime_identity"]
    environment = profile["environment"]
    memory = profile["memory"]
    transforms = profile["transform_calls"]
    compiler = profile["compiler"]
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
        "post_flush_rewarm_steps": entry["post_flush_rewarm_steps"],
    }
    for name, expected in expected_config.items():
        _require(config.get(name) == expected, f"{key} config {name} differs")
    _require(provider.get("role") == entry["provider_role"], f"{key} role differs")
    _require(provider.get("commit") == entry["provider_commit"], f"{key} commit differs")
    _require(provider.get("wheel_sha256") == entry["provider_wheel_sha256"], f"{key} wheel differs")
    _require(runtime.get("requested") == entry["runtime_path"], f"{key} requested runtime differs")
    _require(runtime.get("effective") == entry["runtime_path"], f"{key} effective runtime differs")
    _require(runtime.get("fallback_used") is False, f"{key} runtime fallback")
    _require(environment.get("allocated_device") == "cuda:0", f"{key} device differs")
    _require("H100" in str(environment.get("device_name")), f"{key} is not H100 evidence")
    _require(environment.get("tf32_caller_before") == _TF32_OFF, f"{key} TF32-before differs")
    _require(environment.get("tf32_caller_after") == _TF32_OFF, f"{key} TF32-after differs")
    _require(profile.get("finite") is True, f"{key} state is non-finite")
    _require(profile.get("completed_profile_steps") == 60, f"{key} completed steps differ")
    _require(profile.get("runtime_completed_steps") == 73, f"{key} runtime clock differs")
    _require(compiler.get("graph_breaks") == 0, f"{key} graph break")
    _require(compiler.get("compile_fallback") is False, f"{key} compile fallback")
    _require(
        _integer(transforms.get("forward"), f"{key} forward")
        == entry["expected_forward_raw_calls"],
        f"{key} forward count differs",
    )
    _require(
        _integer(transforms.get("inverse"), f"{key} inverse")
        == entry["expected_inverse_raw_calls"],
        f"{key} inverse count differs",
    )
    timing = _timing(profile, key)
    return {
        "key": key,
        "family": entry["family"],
        "grid": entry["grid"],
        "trial": entry["trial"],
        "role": entry["provider_role"],
        "within_pair_order": entry["within_pair_order"],
        "timing": timing,
        "peak_allocated_bytes": _integer(
            memory.get("peak_allocated_bytes"), f"{key} allocated", positive=True
        ),
        "peak_reserved_bytes": _integer(
            memory.get("peak_reserved_bytes"), f"{key} reserved", positive=True
        ),
        "initial_state_sha256": _sha256(profile.get("initial_state_sha256"), f"{key} initial"),
        "final_state_sha256": _sha256(profile.get("final_state_sha256"), f"{key} final"),
    }


def analyze(profile_root: Path, progress_root: Path) -> dict[str, object]:
    plan = build_plan()
    expected = plan["profile_matrix"]
    _require(profile_root.is_dir(), "profile root is not a directory")
    expected_names = sorted(f"{entry['key']}.json" for entry in expected)
    actual_names = sorted(path.name for path in profile_root.glob("*.json"))
    _require(actual_names == expected_names, "profile file set differs")
    _validate_progress(profile_root, progress_root, expected)
    records: dict[str, dict[str, object]] = {}
    for entry in expected:
        key = str(entry["key"])
        path = profile_root / f"{key}.json"
        _require(path.is_file() and not path.is_symlink(), f"profile missing: {key}")
        records[key] = _validate_profile(_json(path), entry, plan)

    thresholds = plan["thresholds"]
    groups: list[dict[str, object]] = []
    for cell in plan["focused_cells"]:
        family = str(cell["family"])
        grid = str(cell["grid"])
        baseline_samples: list[float] = []
        candidate_samples: list[float] = []
        baseline_means: list[float] = []
        candidate_means: list[float] = []
        all_block_ratios: list[float] = []
        trial_block_medians: list[float] = []
        baseline_allocated: list[int] = []
        candidate_allocated: list[int] = []
        baseline_reserved: list[int] = []
        candidate_reserved: list[int] = []
        pairs: list[dict[str, object]] = []
        for trial in range(1, 7):
            baseline = records[f"baseline_{family}_{grid}_trial{trial}"]
            candidate = records[f"candidate_{family}_{grid}_trial{trial}"]
            _require(
                baseline["initial_state_sha256"] == candidate["initial_state_sha256"],
                f"{family}/{grid}/trial{trial} initial state differs",
            )
            if family not in REPAIRED_PERIODIC_FAMILIES:
                _require(
                    baseline["final_state_sha256"] == candidate["final_state_sha256"],
                    f"{family}/{grid}/trial{trial} final state differs",
                )
            expected_order = ["baseline", "candidate"] if trial % 2 else ["candidate", "baseline"]
            observed_order = [
                role
                for _, role in sorted(
                    (
                        (baseline["within_pair_order"], "baseline"),
                        (candidate["within_pair_order"], "candidate"),
                    )
                )
            ]
            _require(observed_order == expected_order, f"{family}/{grid}/trial{trial} order differs")
            baseline_timing = baseline["timing"]
            candidate_timing = candidate["timing"]
            block_ratios = [
                candidate_value / baseline_value
                for candidate_value, baseline_value in zip(
                    candidate_timing["fixed_block_means_seconds"],
                    baseline_timing["fixed_block_means_seconds"],
                    strict=True,
                )
            ]
            trial_median = statistics.median(block_ratios)
            all_block_ratios.extend(block_ratios)
            trial_block_medians.append(trial_median)
            baseline_samples.extend(baseline_timing["samples_seconds"])
            candidate_samples.extend(candidate_timing["samples_seconds"])
            baseline_means.append(baseline_timing["mean_timestep_seconds"])
            candidate_means.append(candidate_timing["mean_timestep_seconds"])
            baseline_allocated.append(baseline["peak_allocated_bytes"])
            candidate_allocated.append(candidate["peak_allocated_bytes"])
            baseline_reserved.append(baseline["peak_reserved_bytes"])
            candidate_reserved.append(candidate["peak_reserved_bytes"])
            pairs.append(
                {
                    "trial": trial,
                    "execution_order": observed_order,
                    "baseline_cv": baseline_timing["coefficient_of_variation"],
                    "candidate_cv": candidate_timing["coefficient_of_variation"],
                    "baseline_cv_at_or_below_reference": baseline_timing["cv_at_or_below_reference"],
                    "candidate_cv_at_or_below_reference": candidate_timing["cv_at_or_below_reference"],
                    "paired_all_sample_mean_ratio": (
                        candidate_timing["mean_timestep_seconds"]
                        / baseline_timing["mean_timestep_seconds"]
                    ),
                    "fixed_block_ratios": block_ratios,
                    "median_fixed_block_ratio": trial_median,
                }
            )
        metrics = {
            "aggregate_all_sample_mean_ratio": (
                statistics.fmean(candidate_samples) / statistics.fmean(baseline_samples)
            ),
            "median_of_profile_means_ratio": (
                statistics.median(candidate_means) / statistics.median(baseline_means)
            ),
            "median_of_all_fixed_block_ratios": statistics.median(all_block_ratios),
            "maximum_trial_median_fixed_block_ratio": max(trial_block_medians),
            "peak_allocated_group_ratio": max(candidate_allocated) / max(baseline_allocated),
            "peak_reserved_group_ratio": max(candidate_reserved) / max(baseline_reserved),
        }
        gates = {
            "aggregate_all_sample_mean": metrics["aggregate_all_sample_mean_ratio"]
            <= thresholds["aggregate_all_sample_mean_ratio_max"],
            "median_of_profile_means": metrics["median_of_profile_means_ratio"]
            <= thresholds["median_of_profile_means_ratio_max"],
            "median_of_all_fixed_block_ratios": metrics["median_of_all_fixed_block_ratios"]
            <= thresholds["median_of_all_fixed_block_ratios_max"],
            "every_trial_median_fixed_block_ratio": metrics["maximum_trial_median_fixed_block_ratio"]
            <= thresholds["every_trial_median_fixed_block_ratio_max"],
            "peak_allocated": metrics["peak_allocated_group_ratio"]
            <= thresholds["peak_allocated_group_ratio_max"],
            "peak_reserved": metrics["peak_reserved_group_ratio"]
            <= thresholds["peak_reserved_group_ratio_max"],
        }
        groups.append(
            {
                "family": family,
                "grid": grid,
                "metrics": metrics,
                "gates": gates,
                "passed": all(gates.values()),
                "cv_reference_exceedance_profile_count": sum(
                    not pair[name]
                    for pair in pairs
                    for name in (
                        "baseline_cv_at_or_below_reference",
                        "candidate_cv_at_or_below_reference",
                    )
                ),
                "pairs": pairs,
            }
        )
    passed = all(group["passed"] for group in groups)
    return {
        "schema": SCHEMA,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "analysis_only": True,
        "post_hoc_v10_replay": True,
        "proposed_contract_passed": passed,
        "formal_v10_classification_preserved": (
            "FAIL_RC5_G1_CORRECTED_MEASUREMENT_WINDOW_GATE"
        ),
        "formal_requalification_performed": False,
        "qualification_complete": False,
        "g1_complete": False,
        "profile_count": len(records),
        "progress_record_count": len(expected),
        "execution_sequence_validated": True,
        "focused_cell_count": len(groups),
        "thresholds": thresholds,
        "raw_sample_contract": plan["raw_sample_contract"],
        "timing_diagnostics": plan["timing_diagnostics"],
        "groups": groups,
        "production_default_changed": False,
    }


def _atomic_json(path: Path, payload: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    data = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _require(json.loads(path.read_text(encoding="utf-8")) == json.loads(data), "JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--progress-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, analyze(args.profile_root, args.progress_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
