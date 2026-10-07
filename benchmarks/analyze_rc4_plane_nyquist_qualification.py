#!/usr/bin/env python3
"""Adjudicate the RC4.1 Plane Nyquist single-H100 profile matrix."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import math
import os
from pathlib import Path
import statistics
import tempfile
from typing import Any


SCHEMA = "pssolver.rc4_1_2.plane_nyquist_h100_analysis.v1"
GRIDS = {
    "R128": (128, 128, 32),
    "R320": (320, 320, 80),
}
FROZEN_INPUT_PATHS = {
    "R128": "/scratch1/vincent/PSSolver/data_pssolver_phase9_p95_h100_69bd077_20260928_v1/inputs/R128/Q_0.npy",
    "R320": "/scratch1/vincent/PSSolver/data_pssolver_phase9_p95_h100_69bd077_20260928_v1/inputs/R320/Q_0.npy",
}
TRIALS = (1, 2, 3)
MEAN_RATIO_LIMIT = 1.03
MEDIAN_RATIO_LIMIT = 1.03
PAIRED_RATIO_LIMIT = 1.05
MEMORY_RATIO_LIMIT = 1.03


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _finite_positive(value: object, name: str) -> float:
    _require(
        isinstance(value, (int, float)) and not isinstance(value, bool),
        f"{name} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result) and result > 0.0, f"{name} must be positive")
    return result


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(payload, dict), f"{path} must contain a JSON object")
    return payload


def _grid_id(report: dict[str, Any]) -> str:
    shape = tuple(report.get("config", {}).get("shape", ()))
    matches = [name for name, expected in GRIDS.items() if shape == expected]
    _require(len(matches) == 1, f"unexpected profile shape: {shape}")
    return matches[0]


def _validate_profile(
    report: dict[str, Any],
    *,
    expected_commit: str,
    role: str,
) -> tuple[str, int]:
    _require(report.get("schema_version") == 1, f"{role} profile schema mismatch")
    config = report.get("config", {})
    grid = _grid_id(report)
    trial = config.get("trial")
    _require(trial in TRIALS, f"{role} {grid} trial is invalid")
    expected_config = {
        "runtime_path": "legacy_production",
        "lengths": [100.0, 100.0, 20.0],
        "device": "cuda",
        "dtype": "float64",
        "dt": 0.005,
        "activity_number": 18.0,
        "dealias_rule": "cubic_half",
        "projected_transform_execution": "truncated",
        "transform_execution_order": "real_first",
        "spectral_storage": "hermitian_half",
        "molecular_field_linear_space": "spectral",
        "stress_divergence_sum_space": "spectral",
        "pointwise_execution": "compile",
        "reuse_q_gradients": True,
        "spectral_refresh_interval": None,
        "warmup_steps": 10,
        "profile_steps": 50,
        "seed": 24,
    }
    for key, expected in expected_config.items():
        _require(config.get(key) == expected, f"{role} {grid} {key} mismatch")
    _require(
        config.get("initial_q_path") == FROZEN_INPUT_PATHS[grid],
        f"{role} {grid} frozen input path mismatch",
    )

    runtime = report.get("runtime_identity", {})
    _require(runtime.get("requested") == "legacy_production", f"{role} runtime request mismatch")
    _require(runtime.get("effective") == "legacy_production", f"{role} runtime effective mismatch")
    _require(runtime.get("fallback_used") is False, f"{role} runtime fallback")
    _require(report.get("finite") is True, f"{role} {grid} state is not finite")
    _require(report.get("completed_steps") == 60, f"{role} {grid} step count mismatch")

    environment = report.get("environment", {})
    git = environment.get("git", {})
    _require(git.get("head") == expected_commit, f"{role} Git HEAD mismatch")
    _require(git.get("dirty") is False and git.get("status") == "", f"{role} worktree is dirty")
    _require(environment.get("device") == "cuda", f"{role} device token mismatch")
    _require("H100" in str(environment.get("device_name")), f"{role} did not use H100")
    _require(environment.get("cuda_matmul_allow_tf32") is False, f"{role} TF32 is enabled")

    calls = report.get("transform_calls", {})
    _require(calls.get("forward_per_step") == 7.0, f"{role} forward calls changed")
    _require(calls.get("inverse_per_step") == 32.0, f"{role} inverse calls changed")
    compile_record = report.get("pointwise_compile", {})
    for phase in ("dynamo_during_build", "dynamo_during_warmup", "dynamo_during_profile"):
        _require(
            compile_record.get(phase, {}).get("graph_breaks") == 0,
            f"{role} graph break in {phase}",
        )
    _finite_positive(report.get("throughput", {}).get("mean_timestep_seconds"), f"{role} mean timestep")
    _finite_positive(report.get("memory", {}).get("peak_allocated_bytes"), f"{role} peak allocated")
    _finite_positive(report.get("memory", {}).get("peak_reserved_bytes"), f"{role} peak reserved")
    for key in ("initial_q_sha256", "final_state_sha256"):
        value = report.get(key)
        _require(isinstance(value, str) and len(value) == 64, f"{role} {key} is invalid")
    return grid, int(trial)


def _index_profiles(
    reports: Iterable[dict[str, Any]],
    *,
    expected_commit: str,
    role: str,
) -> dict[tuple[str, int], dict[str, Any]]:
    indexed: dict[tuple[str, int], dict[str, Any]] = {}
    for report in reports:
        key = _validate_profile(report, expected_commit=expected_commit, role=role)
        _require(key not in indexed, f"duplicate {role} profile {key}")
        indexed[key] = report
    expected = {(grid, trial) for grid in GRIDS for trial in TRIALS}
    _require(set(indexed) == expected, f"{role} profile matrix is incomplete")
    return indexed


def _validate_cuda_report(report: dict[str, Any]) -> None:
    _require(
        report.get("schema") == "pssolver.rc4_1_1.plane_nyquist_storage_diagnostic.v2",
        "CUDA diagnostic schema mismatch",
    )
    _require(
        report.get("classification") == "PASS_PLANE_PERIODIC_NYQUIST_STORAGE_EQUIVALENCE",
        "CUDA storage-equivalence diagnostic failed",
    )
    _require(report.get("configuration", {}).get("device") == "cuda", "CUDA diagnostic device mismatch")
    environment = report.get("environment", {})
    _require(environment.get("allocated_device") == "cuda:0", "allocated CUDA identity mismatch")
    _require("H100" in str(environment.get("device_name")), "CUDA diagnostic did not use H100")
    _require(environment.get("cuda_matmul_allow_tf32") is False, "CUDA diagnostic matmul TF32 enabled")
    _require(environment.get("cudnn_allow_tf32") is False, "CUDA diagnostic cuDNN TF32 enabled")
    cases = report.get("cases")
    _require(isinstance(cases, list) and len(cases) == 5, "CUDA diagnostic case count mismatch")
    for case in cases:
        name = str(case.get("name"))
        for storage in ("full_complex", "hermitian_half"):
            _require(case.get("raw", {}).get(storage, {}).get("finite") is True, f"{name} {storage} is not finite")
        summary = case.get("summary", {})
        _require(float(summary.get("raw_max_velocity_relative_l2", math.inf)) <= 1.0e-12, f"{name} velocity storage mismatch")
        _require(float(summary.get("raw_pressure_relative_l2", math.inf)) <= 1.0e-12, f"{name} pressure storage mismatch")


def _ratio(numerator: float, denominator: float, name: str) -> float:
    _require(denominator > 0.0, f"{name} denominator must be positive")
    value = numerator / denominator
    _require(math.isfinite(value), f"{name} ratio is not finite")
    return value


def analyze(
    baseline_reports: Iterable[dict[str, Any]],
    candidate_reports: Iterable[dict[str, Any]],
    cuda_report: dict[str, Any],
    *,
    baseline_commit: str,
    candidate_commit: str,
) -> dict[str, Any]:
    _require(len(baseline_commit) == 40, "baseline commit must be a full hash")
    _require(len(candidate_commit) == 40, "candidate commit must be a full hash")
    _validate_cuda_report(cuda_report)
    baseline = _index_profiles(baseline_reports, expected_commit=baseline_commit, role="baseline")
    candidate = _index_profiles(candidate_reports, expected_commit=candidate_commit, role="candidate")
    grids: dict[str, Any] = {}
    for grid in GRIDS:
        baseline_grid = [baseline[(grid, trial)] for trial in TRIALS]
        candidate_grid = [candidate[(grid, trial)] for trial in TRIALS]
        baseline_initial = {item["initial_q_sha256"] for item in baseline_grid}
        candidate_initial = {item["initial_q_sha256"] for item in candidate_grid}
        _require(len(baseline_initial) == 1, f"{grid} baseline initial identity drift")
        _require(candidate_initial == baseline_initial, f"{grid} paired initial identity mismatch")
        _require(len({item["final_state_sha256"] for item in baseline_grid}) == 1, f"{grid} baseline final state is nondeterministic")
        _require(len({item["final_state_sha256"] for item in candidate_grid}) == 1, f"{grid} candidate final state is nondeterministic")

        baseline_times = [float(item["throughput"]["mean_timestep_seconds"]) for item in baseline_grid]
        candidate_times = [float(item["throughput"]["mean_timestep_seconds"]) for item in candidate_grid]
        paired_ratios = [
            _ratio(candidate_time, baseline_time, f"{grid} paired trial {trial}")
            for trial, baseline_time, candidate_time in zip(
                TRIALS, baseline_times, candidate_times, strict=True
            )
        ]
        mean_ratio = _ratio(statistics.fmean(candidate_times), statistics.fmean(baseline_times), f"{grid} mean")
        median_ratio = _ratio(statistics.median(candidate_times), statistics.median(baseline_times), f"{grid} median")
        allocated_ratio = _ratio(
            max(int(item["memory"]["peak_allocated_bytes"]) for item in candidate_grid),
            max(int(item["memory"]["peak_allocated_bytes"]) for item in baseline_grid),
            f"{grid} allocated memory",
        )
        reserved_ratio = _ratio(
            max(int(item["memory"]["peak_reserved_bytes"]) for item in candidate_grid),
            max(int(item["memory"]["peak_reserved_bytes"]) for item in baseline_grid),
            f"{grid} reserved memory",
        )
        _require(mean_ratio <= MEAN_RATIO_LIMIT, f"{grid} mean timestep regression")
        _require(median_ratio <= MEDIAN_RATIO_LIMIT, f"{grid} median timestep regression")
        _require(max(paired_ratios) <= PAIRED_RATIO_LIMIT, f"{grid} paired timestep regression")
        _require(allocated_ratio <= MEMORY_RATIO_LIMIT, f"{grid} allocated-memory regression")
        _require(reserved_ratio <= MEMORY_RATIO_LIMIT, f"{grid} reserved-memory regression")
        grids[grid] = {
            "baseline_timestep_seconds": baseline_times,
            "candidate_timestep_seconds": candidate_times,
            "paired_candidate_over_baseline": paired_ratios,
            "mean_candidate_over_baseline": mean_ratio,
            "median_candidate_over_baseline": median_ratio,
            "peak_allocated_candidate_over_baseline": allocated_ratio,
            "peak_reserved_candidate_over_baseline": reserved_ratio,
            "initial_q_sha256": next(iter(baseline_initial)),
            "baseline_final_state_sha256": baseline_grid[0]["final_state_sha256"],
            "candidate_final_state_sha256": candidate_grid[0]["final_state_sha256"],
        }
    return {
        "schema": SCHEMA,
        "classification": "PASS_RC4_1_2_PLANE_NYQUIST_SINGLE_H100_NON_REGRESSION",
        "qualification_complete": True,
        "baseline_commit": baseline_commit,
        "candidate_commit": candidate_commit,
        "cuda_correctness": {
            "classification": cuda_report["classification"],
            "case_count": len(cuda_report["cases"]),
            "allocated_device": cuda_report["environment"]["allocated_device"],
            "device_name": cuda_report["environment"]["device_name"],
        },
        "thresholds": {
            "mean_timestep_ratio_max": MEAN_RATIO_LIMIT,
            "median_timestep_ratio_max": MEDIAN_RATIO_LIMIT,
            "paired_timestep_ratio_max": PAIRED_RATIO_LIMIT,
            "peak_memory_ratio_max": MEMORY_RATIO_LIMIT,
        },
        "grids": grids,
        "production_default_changed": False,
        "pssolver_control_modified": False,
        "nematics3d_modified": False,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-profile", type=Path, action="append", required=True)
    parser.add_argument("--candidate-profile", type=Path, action="append", required=True)
    parser.add_argument("--cuda-report", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"output exists: {args.output}")
    return args


def main() -> int:
    args = parse_args()
    result = analyze(
        (_load(path) for path in args.baseline_profile),
        (_load(path) for path in args.candidate_profile),
        _load(args.cuda_report),
        baseline_commit=args.baseline_commit,
        candidate_commit=args.candidate_commit,
    )
    write_json_atomic(args.output, result)
    print(args.output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
