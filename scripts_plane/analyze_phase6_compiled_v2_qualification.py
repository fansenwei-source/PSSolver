#!/usr/bin/env python3
"""Aggregate the frozen Phase 6 H100 profile and workflow evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
import statistics


RUNTIMES = ("legacy_production", "compiled_v2")
GRIDS = ((128, 128, 32), (320, 320, 80))
TRIALS = (1, 2, 3)
EXPECTED_COMPARISON_ROLES = {
    "R128_cross_100",
    "R320_cross_100",
    "R128_legacy_restart",
    "R128_compiled_restart",
    "R320_legacy_restart",
    "R320_compiled_restart",
}
REQUIRED_AUXILIARY_GATES = {
    "cuda_only_tests_passed",
    "continuous_complete_markers_valid",
    "diagnostics_byte_identity_R128",
    "diagnostics_byte_identity_R320",
    "normalized_metadata_identity_R128",
    "normalized_metadata_identity_R320",
    "output_filename_shape_dtype_identity_R128",
    "output_filename_shape_dtype_identity_R320",
    "legacy_to_compiled_restart_rejected_before_mutation",
    "compiled_to_legacy_restart_rejected_before_mutation",
    "legacy_tampered_identity_rejected_before_mutation",
    "compiled_tampered_identity_rejected_before_mutation",
    "qualification_worktree_clean_before_and_after",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load(path: str | Path, description: str) -> tuple[Path, dict[str, object]]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{description} is missing: {source}")
    try:
        value = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{description} is not valid JSON: {source}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"{description} must contain a JSON object")
    return source, dict(value)


def _profile_record(
    path: str | Path,
    *,
    expected_commit: str,
    expected_gpu_name: str,
) -> dict[str, object]:
    source, value = _load(path, "runtime profile")
    try:
        config = value["config"]
        runtime = value["runtime_identity"]
        environment = value["environment"]
        git = environment["git"]
        calls = value["transform_calls"]
        throughput = value["throughput"]
        memory = value["memory"]
        compile_profile = value["pointwise_compile"]["dynamo_during_profile"]
        runtime_path = config["runtime_path"]
        shape = tuple(config["shape"])
        trial = config["trial"]
    except (KeyError, TypeError) as exc:
        raise ValueError("runtime profile lacks the Phase 6 schema") from exc
    if runtime_path not in RUNTIMES or shape not in GRIDS or trial not in TRIALS:
        raise ValueError("runtime profile identity is outside the frozen matrix")
    if runtime["requested"] != runtime_path or runtime["effective"] != runtime_path:
        raise ValueError("runtime profile requested/effective identity differs")
    if runtime.get("fallback_used") is not False:
        raise ValueError("runtime profile used fallback")
    if git.get("head") != expected_commit or git.get("dirty") is not False:
        raise ValueError("runtime profile Git identity is invalid")
    if environment.get("cuda_available") is not True:
        raise ValueError("runtime profile was not executed with CUDA")
    if expected_gpu_name not in str(environment.get("device_name")):
        raise ValueError("runtime profile GPU model is invalid")
    if environment.get("cuda_matmul_allow_tf32") is not False:
        raise ValueError("runtime profile enabled TF32")
    frozen = {
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
    if any(config.get(name) != expected for name, expected in frozen.items()):
        raise ValueError("runtime profile configuration differs from the contract")
    if value.get("finite") is not True or value.get("completed_steps") != 60:
        raise ValueError("runtime profile is incomplete or non-finite")
    if calls.get("forward_per_step") != 7 or calls.get("inverse_per_step") != 32:
        raise ValueError("runtime profile transform-call contract failed")
    if compile_profile.get("graph_breaks") != 0:
        raise ValueError("runtime profile recorded a graph break")
    mean = float(throughput["mean_timestep_seconds"])
    allocated = int(memory["peak_allocated_bytes"])
    reserved = int(memory["peak_reserved_bytes"])
    if not math.isfinite(mean) or mean <= 0.0 or allocated <= 0 or reserved <= 0:
        raise ValueError("runtime profile timing or memory is invalid")
    return {
        "path": str(source),
        "sha256": _sha256(source),
        "runtime": runtime_path,
        "shape": list(shape),
        "trial": trial,
        "mean_timestep_seconds": mean,
        "peak_allocated_bytes": allocated,
        "peak_reserved_bytes": reserved,
        "initial_q_sha256": value["initial_q_sha256"],
        "final_state_sha256": value["final_state_sha256"],
    }


def _comparison_record(path: str | Path) -> dict[str, object]:
    source, value = _load(path, "workflow comparison")
    role = value.get("comparison_role")
    if role not in EXPECTED_COMPARISON_ROLES:
        raise ValueError("workflow comparison role is invalid")
    if value.get("classification") != "PASS":
        raise ValueError(f"workflow comparison failed: {role}")
    if value.get("require_byte_identity") is not True:
        raise ValueError("workflow comparison did not require byte identity")
    if value.get("byte_identity_gate") is not True:
        raise ValueError("workflow comparison byte identity failed")
    if role.endswith("cross_100"):
        if value.get("require_initial_q_identity") is not True:
            raise ValueError("cross-runtime comparison omitted initial-Q identity")
        if value.get("initial_q_identity_gate") is not True:
            raise ValueError("cross-runtime initial-Q identity failed")
    return {
        "path": str(source),
        "sha256": _sha256(source),
        "role": role,
        "array_count": value.get("array_count"),
        "maximum_gate_relative_l2": value.get("maximum_gate_relative_l2"),
    }


def _auxiliary_record(
    path: str | Path,
    *,
    expected_commit: str,
    expected_gpu_name: str,
) -> dict[str, object]:
    source, value = _load(path, "auxiliary gate report")
    try:
        gates = value["gates"]
        slurm = value["slurm"]
        environment = value["environment"]
    except (KeyError, TypeError) as exc:
        raise ValueError("auxiliary report lacks the Phase 6 schema") from exc
    if not isinstance(gates, Mapping) or set(gates) != REQUIRED_AUXILIARY_GATES:
        raise ValueError("auxiliary report gate set differs from the contract")
    if any(value is not True for value in gates.values()):
        raise ValueError("one or more Phase 6 auxiliary gates failed")
    if value.get("classification") != "PASS_PHASE6_AUXILIARY_GATES":
        raise ValueError("auxiliary report classification is not PASS")
    if value.get("expected_commit") != expected_commit:
        raise ValueError("auxiliary report commit identity is invalid")
    if value.get("formal_h100_submission_count") != 1:
        raise ValueError("auxiliary report submission count is invalid")
    if value.get("automatic_retry") is not False:
        raise ValueError("auxiliary report recorded an automatic retry")
    if slurm.get("state") != "COMPLETED" or slurm.get("exit_code") != "0:0":
        raise ValueError("auxiliary report Slurm result is not successful")
    if expected_gpu_name not in str(environment.get("gpu_name")):
        raise ValueError("auxiliary report GPU model is invalid")
    if environment.get("tf32") is not False:
        raise ValueError("auxiliary report enabled TF32")
    if value.get("production_default_changed") is not False:
        raise ValueError("auxiliary report changed the production default")
    return {
        "path": str(source),
        "sha256": _sha256(source),
        "classification": value["classification"],
        "gates": dict(gates),
        "slurm": dict(slurm),
        "environment": dict(environment),
    }


def analyze_phase6_qualification(
    profile_paths: Sequence[str | Path],
    comparison_paths: Sequence[str | Path],
    auxiliary_gate_path: str | Path,
    *,
    expected_commit: str,
    expected_gpu_name: str = "H100",
) -> dict[str, object]:
    if len(expected_commit) != 40 or any(
        character not in "0123456789abcdef" for character in expected_commit
    ):
        raise ValueError("expected_commit must be a lowercase full SHA-1")
    if len(profile_paths) != 12:
        raise ValueError("Phase 6 requires exactly 12 runtime profiles")
    if len(comparison_paths) != len(EXPECTED_COMPARISON_ROLES):
        raise ValueError("Phase 6 requires exactly six workflow comparisons")

    profiles = [
        _profile_record(
            path,
            expected_commit=expected_commit,
            expected_gpu_name=expected_gpu_name,
        )
        for path in profile_paths
    ]
    keys = {
        (tuple(item["shape"]), item["runtime"], item["trial"])
        for item in profiles
    }
    expected_keys = {
        (shape, runtime, trial)
        for shape in GRIDS
        for runtime in RUNTIMES
        for trial in TRIALS
    }
    if keys != expected_keys:
        raise ValueError("runtime profile matrix is incomplete or duplicated")
    comparisons = [_comparison_record(path) for path in comparison_paths]
    if {item["role"] for item in comparisons} != EXPECTED_COMPARISON_ROLES:
        raise ValueError("workflow comparison matrix is incomplete or duplicated")
    auxiliary = _auxiliary_record(
        auxiliary_gate_path,
        expected_commit=expected_commit,
        expected_gpu_name=expected_gpu_name,
    )

    grid_reports: dict[str, object] = {}
    all_gates = []
    for shape in GRIDS:
        label = f"R{shape[0]}"
        selected = [item for item in profiles if tuple(item["shape"]) == shape]
        by_runtime = {
            runtime: sorted(
                (item for item in selected if item["runtime"] == runtime),
                key=lambda item: item["trial"],
            )
            for runtime in RUNTIMES
        }
        baseline = by_runtime["legacy_production"]
        candidate = by_runtime["compiled_v2"]
        if len({item["initial_q_sha256"] for item in selected}) != 1:
            raise ValueError(f"{label} profile initial-Q identities differ")
        if any(
            baseline[index]["final_state_sha256"]
            != candidate[index]["final_state_sha256"]
            for index in range(3)
        ):
            raise ValueError(f"{label} paired profile final-state identities differ")
        paired = [
            candidate[index]["mean_timestep_seconds"]
            / baseline[index]["mean_timestep_seconds"]
            for index in range(3)
        ]
        mean_ratio = statistics.mean(
            item["mean_timestep_seconds"] for item in candidate
        ) / statistics.mean(item["mean_timestep_seconds"] for item in baseline)
        median_ratio = statistics.median(
            item["mean_timestep_seconds"] for item in candidate
        ) / statistics.median(
            item["mean_timestep_seconds"] for item in baseline
        )
        allocated_ratio = max(
            item["peak_allocated_bytes"] for item in candidate
        ) / max(item["peak_allocated_bytes"] for item in baseline)
        reserved_ratio = max(
            item["peak_reserved_bytes"] for item in candidate
        ) / max(item["peak_reserved_bytes"] for item in baseline)
        faster_count = sum(value < 1.0 for value in paired)
        gates = {
            "mean_timestep_non_regression": mean_ratio <= 1.02,
            "median_timestep_non_regression": median_ratio <= 1.02,
            "peak_allocated_non_regression": allocated_ratio <= 1.02,
            "peak_reserved_non_regression": reserved_ratio <= 1.02,
            "paired_candidate_faster": faster_count >= 2,
        }
        all_gates.extend(gates.values())
        grid_reports[label] = {
            "shape": list(shape),
            "paired_candidate_over_baseline": paired,
            "candidate_over_baseline_mean_timestep": mean_ratio,
            "candidate_over_baseline_median_timestep": median_ratio,
            "candidate_over_baseline_peak_allocated": allocated_ratio,
            "candidate_over_baseline_peak_reserved": reserved_ratio,
            "candidate_faster_count": faster_count,
            "gates": gates,
        }

    passed = all(all_gates)
    return {
        "schema_version": 1,
        "classification": (
            "PASS_PHASE6_H100_CANDIDATE_GATE"
            if passed
            else "FAIL_PHASE6_H100_CANDIDATE_GATE"
        ),
        "expected_commit": expected_commit,
        "profile_count": len(profiles),
        "comparison_count": len(comparisons),
        "grids": grid_reports,
        "profiles": profiles,
        "comparisons": comparisons,
        "auxiliary_gates": auxiliary,
        "all_gates_passed": passed,
        "eligible_for_long_run_stage": passed,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }


def _write_new(path: Path, value: Mapping[str, object]) -> None:
    target = path.expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite output: {target}")
    target.write_text(
        json.dumps(dict(value), allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, action="append", required=True)
    parser.add_argument("--comparison", type=Path, action="append", required=True)
    parser.add_argument("--auxiliary-gates", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--expected-gpu-name", default="H100")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = analyze_phase6_qualification(
        args.profile,
        args.comparison,
        args.auxiliary_gates,
        expected_commit=args.expected_commit,
        expected_gpu_name=args.expected_gpu_name,
    )
    _write_new(args.output, report)
    print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    return 0 if report["all_gates_passed"] is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
