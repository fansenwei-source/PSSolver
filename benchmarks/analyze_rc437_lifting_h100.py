#!/usr/bin/env python3
"""Fail-closed RC4.3.7 H100 lifting-closure analyzer."""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping
import json
import math
import os
from pathlib import Path
import statistics
from typing import Any

from benchmarks.profile_rc437_plane_runtime import SCHEMA as PROFILE_SCHEMA
from benchmarks.qualify_rc437_lifting_cuda import PASS as CUDA_PASS, SCHEMA as CUDA_SCHEMA


SCHEMA = "pssolver.rc4_3_7.lifting_h100_analysis.v1"
PASS = "PASS_RC4_3_7_INTERNAL_LIFTING_SINGLE_H100_CLOSURE"
GRIDS = {"R128": (128, 128, 32), "R320": (320, 320, 80)}
TRIALS = (1, 2, 3)
INPUTS = {
    "R128": "/scratch1/vincent/PSSolver/data_pssolver_phase9_p95_h100_69bd077_20260928_v1/inputs/R128/Q_0.npy",
    "R320": "/scratch1/vincent/PSSolver/data_pssolver_phase9_p95_h100_69bd077_20260928_v1/inputs/R320/Q_0.npy",
}
MEAN_RATIO_MAX = 1.15
MEDIAN_RATIO_MAX = 1.15
ALLOCATED_RATIO_MAX = 1.10
ACTIVE_RATIO_MAX = 1.10


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _finite_positive(value: object, name: str) -> float:
    _require(isinstance(value, (int, float)) and not isinstance(value, bool), f"{name} is not numeric")
    result = float(value)
    _require(math.isfinite(result) and result > 0.0, f"{name} must be finite and positive")
    return result


def _graph_breaks(value: object) -> int:
    if isinstance(value, Mapping):
        return sum(
            int(nested) if key == "graph_breaks" and isinstance(nested, int) else _graph_breaks(nested)
            for key, nested in value.items()
        )
    if isinstance(value, list):
        return sum(_graph_breaks(item) for item in value)
    return 0


def _profile_identity(
    report: dict[str, Any],
    *,
    role: str,
    commit: str,
    expected_root: Path,
) -> tuple[str, int]:
    _require(report.get("schema") == PROFILE_SCHEMA, f"{role} profile schema mismatch")
    config = report.get("config")
    _require(isinstance(config, Mapping), f"{role} config missing")
    shape = tuple(config.get("shape", ()))
    matching = [name for name, expected in GRIDS.items() if shape == expected]
    _require(len(matching) == 1, f"{role} profile shape differs")
    grid = matching[0]
    trial = config.get("trial")
    _require(trial in TRIALS, f"{role} {grid} trial differs")
    expected = {
        "runtime_path": "legacy_production", "lengths": [100.0, 100.0, 20.0],
        "device": "cuda", "dtype": "float64", "dt": 0.005,
        "activity_number": 18.0, "dealias_rule": "cubic_half",
        "projected_transform_execution": "truncated", "transform_execution_order": "real_first",
        "spectral_storage": "hermitian_half", "molecular_field_linear_space": "spectral",
        "stress_divergence_sum_space": "spectral", "pointwise_execution": "compile",
        "reuse_q_gradients": True, "spectral_refresh_interval": None,
        "warmup_steps": 10, "profile_steps": 50, "seed": 24,
        "initial_q_path": INPUTS[grid],
    }
    for key, wanted in expected.items():
        _require(config.get(key) == wanted, f"{role} {grid} {key} differs")
    runtime = report.get("runtime_identity", {})
    _require(runtime.get("requested") == runtime.get("effective") == "legacy_production", f"{role} runtime identity differs")
    _require(runtime.get("fallback_used") is False, f"{role} runtime fallback")
    environment = report.get("environment", {})
    git = environment.get("git", {})
    _require(git.get("head") == commit, f"{role} Git HEAD differs")
    _require(git.get("dirty") is False and git.get("status") == "", f"{role} worktree dirty")
    _require(environment.get("allocated_device") == "cuda:0", f"{role} allocated device differs")
    _require("H100" in str(environment.get("device_name")), f"{role} did not use H100")
    _require(environment.get("cuda_matmul_allow_tf32") is False, f"{role} matmul TF32 enabled")
    _require(environment.get("cudnn_allow_tf32") is False, f"{role} cuDNN TF32 enabled")
    imported = Path(str(environment.get("pssolver_import"))).resolve()
    _require(
        imported.is_relative_to(expected_root.resolve()),
        f"{role} profile imported the wrong source tree",
    )
    _require(report.get("completed_steps") == 60, f"{role} completed steps differ")
    _require(report.get("finite") is True, f"{role} non-finite state")
    calls = report.get("transform_calls", {})
    _require(calls.get("forward_per_step") == 7.0, f"{role} forward transform count differs")
    _require(calls.get("inverse_per_step") == 32.0, f"{role} inverse transform count differs")
    pointwise = report.get("pointwise_compile", {})
    execution = pointwise.get("execution", {})
    _require(execution.get("requested") == execution.get("effective") == "compile", f"{role} compile not effective")
    _require(execution.get("fallback_allowed") is False, f"{role} compile fallback allowed")
    _require(_graph_breaks(pointwise) == 0, f"{role} graph break")
    for key in ("mean_timestep_seconds", "median_timestep_seconds"):
        _finite_positive(report.get("timing", {}).get(key), f"{role} {key}")
    for key in ("peak_allocated_bytes", "peak_active_bytes", "peak_reserved_bytes"):
        _finite_positive(report.get("memory", {}).get(key), f"{role} {key}")
    for key in ("initial_q_sha256", "final_state_sha256"):
        value = report.get(key)
        _require(isinstance(value, str) and len(value) == 64, f"{role} {key} invalid")
    return grid, int(trial)


def _index(
    reports: Iterable[dict[str, Any]],
    *,
    role: str,
    commit: str,
    expected_root: Path,
) -> dict[tuple[str, int], dict[str, Any]]:
    result: dict[tuple[str, int], dict[str, Any]] = {}
    for report in reports:
        key = _profile_identity(
            report,
            role=role,
            commit=commit,
            expected_root=expected_root,
        )
        _require(key not in result, f"duplicate {role} profile {key}")
        result[key] = report
    expected = {(grid, trial) for grid in GRIDS for trial in TRIALS}
    _require(set(result) == expected, f"{role} profile matrix incomplete")
    return result


def _ratio(numerator: float, denominator: float, name: str) -> float:
    _require(denominator > 0.0, f"{name} denominator is not positive")
    result = numerator / denominator
    _require(math.isfinite(result), f"{name} ratio non-finite")
    return result


def _validate_cuda(report: dict[str, Any], candidate_root: Path) -> None:
    _require(report.get("schema") == CUDA_SCHEMA, "CUDA correctness schema differs")
    _require(report.get("classification") == CUDA_PASS and report.get("passed") is True, "CUDA correctness failed")
    environment = report.get("environment", {})
    _require(environment.get("device") == "cuda:0", "CUDA correctness device differs")
    _require("H100" in str(environment.get("device_name")), "CUDA correctness did not use H100")
    _require(environment.get("cuda_matmul_allow_tf32") is False, "CUDA correctness matmul TF32 enabled")
    _require(environment.get("cudnn_allow_tf32") is False, "CUDA correctness cuDNN TF32 enabled")
    imported = Path(str(environment.get("pssolver_import"))).resolve()
    _require(imported.is_relative_to(candidate_root.resolve()), "CUDA correctness imported wrong candidate")
    fresh = report.get("fresh_conditioning", {})
    _require(fresh.get("policy") == "localized_quintic_wall_transition_v1", "fresh policy differs")
    _require(float(fresh.get("cross_device_relative_l2", math.inf)) <= 1.0e-15, "CPU/CUDA conditioning differs")
    _require(fresh.get("interior_byte_identical_to_unconditioned") is True, "localized interior changed")
    _require(fresh.get("transition_region_changed") is True, "transition was not applied")
    _require(float(fresh.get("wall_residual_linf", math.inf)) <= 1.0e-12, "wall residual failed")
    restart = report.get("restart", {})
    _require(restart.get("restored_steps") == 2, "restart progress differs")
    _require(restart.get("all_byte_identical") is True, "restart is not byte-identical")
    _require(restart.get("finite") is True, "restart state non-finite")
    provenance = restart.get("fresh_construction_provenance", {})
    _require(provenance.get("conditioning_applied") is True, "fresh provenance differs")
    boundary = report.get("qualification_boundary", {})
    _require(boundary == {
        "internal_lifting_infrastructure_only": True,
        "public_lifted_plane_reopened": False,
        "localized_policy_publicly_qualified": False,
        "production_default_changed": False,
    }, "qualification boundary expanded")


def analyze(
    baseline_reports: Iterable[dict[str, Any]],
    candidate_reports: Iterable[dict[str, Any]],
    cuda_report: dict[str, Any],
    *,
    baseline_commit: str,
    candidate_commit: str,
    baseline_root: Path,
    candidate_root: Path,
) -> dict[str, Any]:
    _require(len(baseline_commit) == len(candidate_commit) == 40, "full commit hashes required")
    _validate_cuda(cuda_report, candidate_root)
    baseline = _index(
        baseline_reports,
        role="baseline",
        commit=baseline_commit,
        expected_root=baseline_root,
    )
    candidate = _index(
        candidate_reports,
        role="candidate",
        commit=candidate_commit,
        expected_root=candidate_root,
    )
    grids: dict[str, object] = {}
    for grid in GRIDS:
        a = [baseline[(grid, trial)] for trial in TRIALS]
        b = [candidate[(grid, trial)] for trial in TRIALS]
        _require({item["initial_q_sha256"] for item in a} == {item["initial_q_sha256"] for item in b} and len({item["initial_q_sha256"] for item in a}) == 1, f"{grid} initial identity differs")
        _require({item["final_state_sha256"] for item in a} == {item["final_state_sha256"] for item in b} and len({item["final_state_sha256"] for item in a}) == 1, f"{grid} supported dynamics changed")
        a_mean = [float(item["timing"]["mean_timestep_seconds"]) for item in a]
        b_mean = [float(item["timing"]["mean_timestep_seconds"]) for item in b]
        a_median = [float(item["timing"]["median_timestep_seconds"]) for item in a]
        b_median = [float(item["timing"]["median_timestep_seconds"]) for item in b]
        mean_ratio = _ratio(statistics.fmean(b_mean), statistics.fmean(a_mean), f"{grid} mean")
        median_ratio = _ratio(statistics.median(b_median), statistics.median(a_median), f"{grid} median")
        allocated_ratio = _ratio(max(int(item["memory"]["peak_allocated_bytes"]) for item in b), max(int(item["memory"]["peak_allocated_bytes"]) for item in a), f"{grid} allocated")
        active_ratio = _ratio(max(int(item["memory"]["peak_active_bytes"]) for item in b), max(int(item["memory"]["peak_active_bytes"]) for item in a), f"{grid} active")
        reserved_ratio = _ratio(max(int(item["memory"]["peak_reserved_bytes"]) for item in b), max(int(item["memory"]["peak_reserved_bytes"]) for item in a), f"{grid} reserved")
        _require(mean_ratio <= MEAN_RATIO_MAX, f"{grid} mean timestep regression")
        _require(median_ratio <= MEDIAN_RATIO_MAX, f"{grid} median timestep regression")
        _require(allocated_ratio <= ALLOCATED_RATIO_MAX, f"{grid} allocated-memory regression")
        _require(active_ratio <= ACTIVE_RATIO_MAX, f"{grid} active-memory regression")
        grids[grid] = {
            "baseline_mean_timestep_seconds": a_mean,
            "candidate_mean_timestep_seconds": b_mean,
            "baseline_median_timestep_seconds": a_median,
            "candidate_median_timestep_seconds": b_median,
            "mean_candidate_over_baseline": mean_ratio,
            "median_candidate_over_baseline": median_ratio,
            "peak_allocated_candidate_over_baseline": allocated_ratio,
            "peak_active_candidate_over_baseline": active_ratio,
            "peak_reserved_candidate_over_baseline_diagnostic": reserved_ratio,
            "final_state_sha256": a[0]["final_state_sha256"],
        }
    return {
        "schema": SCHEMA,
        "classification": PASS,
        "qualification_complete": True,
        "rc4_3_7_complete": True,
        "rc4_3_complete": True,
        "baseline_commit": baseline_commit,
        "candidate_commit": candidate_commit,
        "cuda_correctness": {"classification": CUDA_PASS, "internal_lifting_infrastructure_only": True},
        "thresholds": {
            "mean_timestep_ratio_max": MEAN_RATIO_MAX,
            "median_timestep_ratio_max": MEDIAN_RATIO_MAX,
            "peak_allocated_ratio_max": ALLOCATED_RATIO_MAX,
            "peak_active_ratio_max": ACTIVE_RATIO_MAX,
            "peak_reserved_role": "diagnostic_only",
        },
        "grids": grids,
        "public_lifted_plane_reopened": False,
        "localized_policy_publicly_qualified": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
        "pssolver_control_modified": False,
        "nematics3d_modified": False,
    }


def _atomic_json(path: Path, value: object) -> None:
    path = path.resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            json.dump(value, stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
        _load(path)
    finally:
        if temporary.exists(): temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-profile", action="append", type=Path, required=True)
    parser.add_argument("--candidate-profile", action="append", type=Path, required=True)
    parser.add_argument("--cuda-report", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    result = analyze(
        (_load(path) for path in args.baseline_profile),
        (_load(path) for path in args.candidate_profile),
        _load(args.cuda_report),
        baseline_commit=args.baseline_commit,
        candidate_commit=args.candidate_commit,
        baseline_root=args.baseline_root,
        candidate_root=args.candidate_root,
    )
    _atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
