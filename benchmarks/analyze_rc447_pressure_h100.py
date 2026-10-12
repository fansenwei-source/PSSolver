#!/usr/bin/env python3
"""Fail-closed cumulative analyzer for RC4.4.7 pressure qualification."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

from benchmarks.profile_rc447_channel_pressure import SCHEMA as PROFILE_SCHEMA
from benchmarks.run_rc447_cuda_tests import (
    PASS as CUDA_TEST_PASS,
    SCHEMA as CUDA_TEST_SCHEMA,
)


SCHEMA = "pssolver.rc4_4_7.channel_pressure_h100_analysis.v1"
CORRECTNESS_SCHEMA = "pssolver.rc4_4_7.channel_pressure_cuda.v1"
PASS = "PASS_RC4_4_7_CHANNEL_PRESSURE_SINGLE_H100_CUMULATIVE_CLOSURE"
CORRECTNESS_PASS = "PASS_RC4_4_7_CHANNEL_PRESSURE_CUDA_CORRECTNESS"
GRIDS = {
    "R128x20x20": (128, 20, 20),
    "R512x40x40": (512, 40, 40),
}
TRIALS = (1, 2, 3)
COMMITS = {
    "baseline": "5071d73a00e0d19be62ebd39edf9918818d1267a",
    "candidate": "dc8c87abc7b216231d09ca12fcc7e76b2249a7ab",
}
WHEELS = {
    "baseline": "9c99ebcdc2e03b2c61b5bc3a5d84d97e99775f488409e2e61f5c06e009695e61",
    "candidate": "814c8a22252b290f7ebd3a249ebaa63152ec978d2726a7eaf82946ec64901ca7",
}
THRESHOLDS = {
    "mean_timestep_ratio_max": 1.05,
    "median_timestep_ratio_max": 1.05,
    "every_paired_timestep_ratio_max": 1.10,
    "peak_allocated_ratio_max": 1.05,
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _number(value: object, name: str, *, positive: bool = False) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} is not numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} is not finite")
    if positive:
        _require(result > 0.0, f"{name} must be positive")
    return result


def _validate_correctness(report: dict[str, Any]) -> None:
    _require(report.get("schema") == CORRECTNESS_SCHEMA, "correctness schema differs")
    _require(
        report.get("classification") == CORRECTNESS_PASS,
        "candidate correctness classification differs",
    )
    _require(report.get("qualification_complete") is True, "correctness incomplete")
    environment = report.get("environment", {})
    _require(environment.get("allocated_device") == "cuda:0", "correctness device differs")
    _require("H100" in str(environment.get("device_name")), "correctness did not use H100")
    tf32 = environment.get("tf32_after_policy", {})
    _require(tf32 == {"cudnn": False, "matmul": False}, "correctness TF32 policy differs")
    identities = report.get("identities", {})
    _require(identities.get("provider_commit") == COMMITS["candidate"], "correctness Provider commit differs")
    _require(
        identities.get("consumer_source_commit")
        == "0cf55b4fb699ff40433000f5b4efe52b1d33546b",
        "correctness Consumer source differs",
    )
    gates = report.get("gates", {})
    expected = {
        "public_identity",
        "scale_matrix",
        "transpose_identity",
        "custom_adjoint",
        "finite_difference",
        "resolution_aware_taylor",
        "checkpointed_full_history",
        "same_version_restart",
        "negative_identity_rejection",
    }
    _require(set(gates) == expected, "correctness gate set differs")
    _require(all(gates[name].get("passed") is True for name in expected), "correctness subgate failed")
    _require(report.get("runtime_fallback_used") is False, "correctness runtime fallback")
    _require(report.get("compile_fallback_used") is False, "correctness compile fallback")
    _require(report.get("graph_breaks") == 0, "correctness graph break")


def _validate_cuda_tests(report: dict[str, Any]) -> None:
    _require(report.get("schema") == CUDA_TEST_SCHEMA, "CUDA-test schema differs")
    _require(report.get("classification") == CUDA_TEST_PASS, "CUDA-test classification differs")
    _require(report.get("passed") is True, "CUDA-only tests failed")
    _require(report.get("pytest_returncode") == 0, "CUDA-test pytest return code differs")
    outcomes = report.get("outcomes", {})
    _require(len(outcomes) == 7, "CUDA-test outcome count differs")
    _require(set(outcomes.values()) == {"passed"}, "CUDA tests contain a non-pass outcome")
    environment = report.get("environment", {})
    _require(environment.get("allocated_device") == "cuda:0", "CUDA-test device differs")
    _require("H100" in str(environment.get("device_name")), "CUDA tests did not use H100")
    _require(environment.get("tf32_after_policy") == {"cudnn": False, "matmul": False}, "CUDA-test TF32 policy differs")
    _require(environment.get("source_shadow_import") is False, "CUDA-test source shadow import")


def _profile_key(report: dict[str, Any], role: str) -> tuple[str, int]:
    _require(report.get("schema") == PROFILE_SCHEMA, f"{role} profile schema differs")
    _require(report.get("role") == role, f"{role} role differs")
    provider = report.get("provider_identity", {})
    _require(provider.get("commit") == COMMITS[role], f"{role} Provider commit differs")
    _require(provider.get("wheel_sha256") == WHEELS[role], f"{role} Provider wheel differs")
    config = report.get("config", {})
    shape = tuple(config.get("shape", ()))
    grid = next((name for name, wanted in GRIDS.items() if shape == wanted), None)
    _require(grid is not None, f"{role} profile grid differs")
    trial = config.get("trial")
    _require(trial in TRIALS, f"{role} profile trial differs")
    expected = {
        "runtime_path": "legacy_channel",
        "lengths": [32.0, 5.0, 5.0],
        "device": "cuda",
        "warmup_steps": 10,
        "profile_steps": 50,
    }
    for key, wanted in expected.items():
        _require(config.get(key) == wanted, f"{role} {grid} {key} differs")
    runtime = report.get("runtime_identity", {})
    _require(
        runtime.get("requested") == runtime.get("effective") == "legacy_channel",
        f"{role} runtime identity differs",
    )
    _require(runtime.get("fallback_used") is False, f"{role} runtime fallback")
    environment = report.get("environment", {})
    _require(environment.get("allocated_device") == "cuda:0", f"{role} allocated device differs")
    _require("H100" in str(environment.get("device_name")), f"{role} did not use H100")
    _require(
        environment.get("tf32_after_policy") == {"cudnn": False, "matmul": False},
        f"{role} TF32 policy differs",
    )
    _require(report.get("completed_steps") == 60, f"{role} completed steps differ")
    _require(report.get("finite") is True, f"{role} state is non-finite")
    calls = report.get("transform_calls", {})
    _require(calls.get("forward_per_step") == 11.0, f"{role} forward transform count differs")
    _require(calls.get("inverse_per_step") == 36.0, f"{role} inverse transform count differs")
    compiler = report.get("compiler", {})
    _require(compiler.get("graph_breaks_during_profile") == 0, f"{role} graph break")
    pointwise = compiler.get("pointwise", {})
    _require(
        pointwise.get("requested") == pointwise.get("effective"),
        f"{role} pointwise execution fallback",
    )
    _require(
        pointwise.get("fallback_allowed") is False
        and pointwise.get("fallback_reason") is None,
        f"{role} compile fallback contract differs",
    )
    for name in ("mean_timestep_seconds", "median_timestep_seconds"):
        _number(report.get("timing", {}).get(name), f"{role} {name}", positive=True)
    samples = report.get("timing", {}).get("samples_seconds")
    _require(isinstance(samples, list) and len(samples) == 50, f"{role} sample count differs")
    for index, sample in enumerate(samples):
        _number(sample, f"{role} sample {index}", positive=True)
    for name in ("peak_allocated_bytes", "peak_active_bytes", "peak_reserved_bytes"):
        _number(report.get("memory", {}).get(name), f"{role} {name}", positive=True)
    history = report.get("pressure_history")
    _require(isinstance(history, list) and len(history) == 50, f"{role} pressure history differs")
    for item in history:
        _require(
            isinstance(item.get("achieved_iteration_count"), int)
            and item["achieved_iteration_count"] > 0,
            f"{role} invalid pressure iteration count",
        )
        _number(item.get("achieved_normalized_residual"), f"{role} pressure residual")
        if role == "candidate":
            _require(
                item.get("schema") == "channel_pressure_diagnostics.v2",
                "candidate pressure diagnostics schema differs",
            )
            _require(item.get("termination_reason_observed") is True, "candidate pressure termination is not observed")
            _require(item.get("acceptable_observed") is True, "candidate pressure acceptability is not observed")
            _require(item.get("acceptable") is True, "candidate pressure solve is unacceptable")
        else:
            _require(
                item.get("schema") == "legacy_public_flow_diagnostics.v1",
                "baseline pressure diagnostics schema differs",
            )
            _require(item.get("termination_reason_observed") is False, "baseline invented a legacy termination reason")
            _require(item.get("acceptable_observed") is False, "baseline invented legacy acceptability")
            _require(item.get("acceptable") is None, "baseline legacy acceptability must be null")
    for name in ("initial_q_sha256", "final_state_sha256"):
        value = report.get(name)
        _require(isinstance(value, str) and len(value) == 64, f"{role} {name} invalid")
    return str(grid), int(trial)


def _index(reports: Iterable[dict[str, Any]], role: str) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for report in reports:
        key = _profile_key(report, role)
        _require(key not in result, f"duplicate {role} profile {key}")
        result[key] = report
    expected = {(grid, trial) for grid in GRIDS for trial in TRIALS}
    _require(set(result) == expected, f"{role} profile matrix incomplete")
    return result


def _ratio(numerator: float, denominator: float, name: str) -> float:
    _require(denominator > 0.0, f"{name} denominator is not positive")
    result = numerator / denominator
    _require(math.isfinite(result), f"{name} ratio is non-finite")
    return result


def analyze(
    baseline_reports: Iterable[dict[str, Any]],
    candidate_reports: Iterable[dict[str, Any]],
    correctness_report: dict[str, Any],
    cuda_test_report: dict[str, Any],
) -> dict[str, Any]:
    _validate_correctness(correctness_report)
    _validate_cuda_tests(cuda_test_report)
    baseline = _index(baseline_reports, "baseline")
    candidate = _index(candidate_reports, "candidate")
    grids: dict[str, object] = {}
    for grid in GRIDS:
        a = [baseline[(grid, trial)] for trial in TRIALS]
        b = [candidate[(grid, trial)] for trial in TRIALS]
        paired: list[dict[str, object]] = []
        for trial, left, right in zip(TRIALS, a, b, strict=True):
            _require(left["initial_q_sha256"] == right["initial_q_sha256"], f"{grid} trial {trial} initial state differs")
            pressure_paths_match = left["pressure_history"] == right["pressure_history"]
            if pressure_paths_match:
                _require(left["final_state_sha256"] == right["final_state_sha256"], f"{grid} trial {trial} final state differs")
            ratio = _ratio(
                float(right["timing"]["mean_timestep_seconds"]),
                float(left["timing"]["mean_timestep_seconds"]),
                f"{grid} trial {trial}",
            )
            _require(ratio <= THRESHOLDS["every_paired_timestep_ratio_max"], f"{grid} trial {trial} timestep regression")
            paired.append(
                {
                    "trial": trial,
                    "candidate_over_baseline_timestep": ratio,
                    "pressure_paths_match": pressure_paths_match,
                    "final_state_sha_equal": left["final_state_sha256"] == right["final_state_sha256"],
                }
            )
        a_mean = [float(item["timing"]["mean_timestep_seconds"]) for item in a]
        b_mean = [float(item["timing"]["mean_timestep_seconds"]) for item in b]
        a_median = [float(item["timing"]["median_timestep_seconds"]) for item in a]
        b_median = [float(item["timing"]["median_timestep_seconds"]) for item in b]
        mean_ratio = _ratio(statistics.fmean(b_mean), statistics.fmean(a_mean), f"{grid} mean")
        median_ratio = _ratio(statistics.median(b_median), statistics.median(a_median), f"{grid} median")
        allocated_ratio = _ratio(
            max(float(item["memory"]["peak_allocated_bytes"]) for item in b),
            max(float(item["memory"]["peak_allocated_bytes"]) for item in a),
            f"{grid} allocated",
        )
        reserved_ratio = _ratio(
            max(float(item["memory"]["peak_reserved_bytes"]) for item in b),
            max(float(item["memory"]["peak_reserved_bytes"]) for item in a),
            f"{grid} reserved",
        )
        _require(mean_ratio <= THRESHOLDS["mean_timestep_ratio_max"], f"{grid} mean timestep regression")
        _require(median_ratio <= THRESHOLDS["median_timestep_ratio_max"], f"{grid} median timestep regression")
        _require(allocated_ratio <= THRESHOLDS["peak_allocated_ratio_max"], f"{grid} allocated-memory regression")
        grids[grid] = {
            "baseline_mean_timestep_seconds": a_mean,
            "candidate_mean_timestep_seconds": b_mean,
            "baseline_median_timestep_seconds": a_median,
            "candidate_median_timestep_seconds": b_median,
            "mean_candidate_over_baseline": mean_ratio,
            "median_candidate_over_baseline": median_ratio,
            "peak_allocated_candidate_over_baseline": allocated_ratio,
            "peak_reserved_candidate_over_baseline_diagnostic": reserved_ratio,
            "paired": paired,
        }
    return {
        "schema": SCHEMA,
        "classification": PASS,
        "qualification_complete": True,
        "rc4_4_7_complete": True,
        "rc4_4_complete": True,
        "B25_closed_verified": True,
        "B28_closed_verified": True,
        "B29_closed_verified": True,
        "provider_commits": COMMITS,
        "provider_wheels": WHEELS,
        "correctness_classification": CORRECTNESS_PASS,
        "cuda_test_classification": CUDA_TEST_PASS,
        "thresholds": THRESHOLDS,
        "grids": grids,
        "eligible_for_rc4_5_planning": True,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
        "larger_batch_qualified": False,
        "optimizer_campaign_authorized": False,
        "scientific_control_result": False,
    }


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _load(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-profile", action="append", type=Path, required=True)
    parser.add_argument("--candidate-profile", action="append", type=Path, required=True)
    parser.add_argument("--correctness-report", type=Path, required=True)
    parser.add_argument("--cuda-test-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(
        args.output,
        analyze(
            (_load(path) for path in args.baseline_profile),
            (_load(path) for path in args.candidate_profile),
            _load(args.correctness_report),
            _load(args.cuda_test_report),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
