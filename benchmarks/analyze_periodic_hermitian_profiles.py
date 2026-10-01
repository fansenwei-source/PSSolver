#!/usr/bin/env python3
"""Fail-closed analyzer for Periodic Hermitian-repair A/B profiles."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

from benchmarks.profile_periodic_hermitian_qualification import (
    EAGER_INAPPLICABLE_REASON,
    PROFILE_ROLES,
    QUALIFICATION_PROFILE_KIND,
)


PASS_CLASSIFICATION = "PASS_PERIODIC_HERMITIAN_PROFILE_NON_REGRESSION"
FROZEN_GRIDS = {
    "R128": (128, 128, 32),
    "R320": (320, 320, 80),
}
FROZEN_INPUT_SHA256 = {
    "R128": "28c70b72118c6d55b7646919ef2157959f40f52a1585753bb0732565cc6895c6",
    "R320": "84a934a828794a3976b4af897f85d60c79e0c5f545b5f8265a9a4abed1e2cb4a",
}
FROZEN_LENGTHS = [100.0, 100.0, 20.0]
FROZEN_TRIALS = 3
FROZEN_WARMUP_STEPS = 5
FROZEN_PROFILE_STEPS = 20
FROZEN_DT = 0.001
FROZEN_BASE_ACTIVITY = 0.013
MEAN_RATIO_MAX = 1.03
MEDIAN_RATIO_MAX = 1.03
MEMORY_RATIO_MAX = 1.05
PROFILE_CV_MAX = 0.20
H100_NAME = "NVIDIA H100 PCIe"
CANDIDATE_PROJECTION = "self_conjugate_planes_each_step"


class HermitianProfileEvidenceError(RuntimeError):
    """Raised when qualification evidence violates the frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HermitianProfileEvidenceError(message)


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise HermitianProfileEvidenceError(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _finite(value: object, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise HermitianProfileEvidenceError(
            f"{description} is not numeric"
        ) from exc
    _require(math.isfinite(result), f"{description} is not finite")
    return result


def _positive(value: object, description: str) -> float:
    result = _finite(value, description)
    _require(result > 0.0, f"{description} must be positive")
    return result


def _positive_integer(value: object, description: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{description} must be a positive integer",
    )
    return value


def _validate_inapplicable(value: object, description: str) -> None:
    _require(isinstance(value, dict), f"{description} capability is absent")
    _require(value.get("applicable") is False, f"{description} must be inapplicable")
    _require(value.get("value") is None, f"{description} must not invent a value")
    _require(
        value.get("reason") == EAGER_INAPPLICABLE_REASON,
        f"{description} inapplicability reason differs",
    )


def validate_capability_audit(
    audit: object,
    *,
    role: str,
    candidate: bool,
) -> tuple[int, int]:
    _require(isinstance(audit, dict), "capability audit is absent")
    _require(
        audit.get("measurement_scope") == "one_fresh_untimed_operation",
        "transform audit scope differs",
    )
    _require(audit.get("timing_contaminated") is False, "timing was contaminated")
    _require(audit.get("role") == role, "capability role differs")

    compilation = audit.get("compilation")
    _require(isinstance(compilation, dict), "compilation capability is absent")
    pointwise = compilation.get("pointwise_execution")
    _require(
        pointwise == {"requested": "eager", "effective": "eager"},
        "pointwise execution is not the frozen eager path",
    )
    _validate_inapplicable(compilation.get("graph_breaks"), "graph breaks")
    _validate_inapplicable(compilation.get("compile_fallback"), "compile fallback")

    runtime = audit.get("runtime_selection")
    _require(isinstance(runtime, dict), "runtime selection is absent")
    _require(runtime.get("fallback_used") is False, "runtime fallback was used")
    metadata = runtime.get("metadata")
    _require(isinstance(metadata, dict), "runtime metadata is absent")
    if role == "production_forward":
        _require(
            runtime.get("source") == "PeriodicRuntimeAdapter.to_metadata",
            "production fallback source differs",
        )
        _require(
            runtime.get("requested") == runtime.get("effective"),
            "production runtime requested/effective mismatch",
        )
    else:
        _require(
            runtime.get("source")
            == "FunctionalRuntimeIdentity.execution.functional_runtime",
            "functional fallback source differs",
        )
        _require(runtime.get("fallback_allowed") is False, "fallback was allowed")
        _require(
            runtime.get("kind") == "periodic_activity_batch_one",
            "functional runtime kind differs",
        )
    if candidate:
        _require(
            metadata.get("hermitian_state_projection") == CANDIDATE_PROJECTION,
            "candidate runtime lacks the Hermitian projection identity",
        )

    dispatch = audit.get("transform_dispatch")
    _require(isinstance(dispatch, dict), "transform dispatch audit is absent")
    _require(
        dispatch.get("semantics")
        == "python_backend_forward_inverse_dispatches",
        "transform dispatch semantics differ",
    )
    forward = _positive_integer(dispatch.get("forward_calls"), "forward calls")
    inverse = _positive_integer(dispatch.get("inverse_calls"), "inverse calls")
    result = audit.get("result")
    _require(isinstance(result, dict), "audit result is absent")
    _require(result.get("finite") is True, "audit result is non-finite")
    sha = result.get("sha256")
    _require(isinstance(sha, str) and len(sha) == 64, "audit SHA-256 is absent")
    return forward, inverse


def _grid_id(shape: object) -> str:
    matches = [name for name, frozen in FROZEN_GRIDS.items() if shape == list(frozen)]
    _require(len(matches) == 1, "profile shape is not a frozen grid")
    return matches[0]


def _validate_profile(
    report: dict[str, object],
    *,
    baseline_commit: str,
    candidate_commit: str,
) -> tuple[str, str, str, int, tuple[int, int]]:
    _require(report.get("schema_version") == 1, "profile schema differs")
    _require(report.get("kind") == QUALIFICATION_PROFILE_KIND, "profile kind differs")
    _require(
        report.get("qualification") == "periodic_hermitian_state_repair",
        "qualification identity differs",
    )
    variant = report.get("variant")
    _require(variant in {"baseline", "candidate"}, "profile variant differs")
    expected_commit = baseline_commit if variant == "baseline" else candidate_commit

    base = report.get("base_profile")
    _require(isinstance(base, dict), "base P9.5 profile is absent")
    _require(base.get("schema_version") == 1, "base profile schema differs")
    _require(
        base.get("kind") == "p95_periodic_functional_profile",
        "base profile kind differs",
    )
    _require(base.get("phase") == "P9.5", "base profile phase differs")
    config = base.get("config")
    _require(isinstance(config, dict), "base profile config is absent")
    grid = _grid_id(config.get("shape"))
    role = config.get("role")
    _require(role in PROFILE_ROLES, "profile role differs")
    trial = _positive_integer(config.get("trial"), "trial")
    _require(trial <= FROZEN_TRIALS, "trial is outside the frozen matrix")
    _require(config.get("device") == "cuda:0", "profile device differs")
    _require(config.get("dtype") == "float64", "profile dtype differs")
    _require(config.get("lengths") == FROZEN_LENGTHS, "profile lengths differ")
    _require(config.get("dt") == FROZEN_DT, "profile timestep differs")
    _require(
        config.get("base_activity") == FROZEN_BASE_ACTIVITY,
        "profile activity differs",
    )
    _require(
        config.get("warmup_steps") == FROZEN_WARMUP_STEPS,
        "profile warmup differs",
    )
    _require(
        config.get("profile_steps") == FROZEN_PROFILE_STEPS,
        "profile window differs",
    )
    initial_sha = config.get("initial_q_sha256")
    _require(
        initial_sha == FROZEN_INPUT_SHA256[grid],
        "initial Q SHA-256 differs from the frozen input",
    )

    environment = base.get("environment")
    _require(isinstance(environment, dict), "profile environment is absent")
    _require(environment.get("cuda_available") is True, "CUDA is unavailable")
    _require(environment.get("device") == "cuda:0", "allocated device differs")
    _require(environment.get("device_name") == H100_NAME, "GPU is not the frozen H100")
    _require(environment.get("tf32_matmul") is False, "matmul TF32 is enabled")
    _require(environment.get("tf32_cudnn") is False, "cuDNN TF32 is enabled")
    git = environment.get("git")
    _require(isinstance(git, dict), "Git provenance is absent")
    _require(git.get("head") == expected_commit, "profile commit differs")
    _require(git.get("status_porcelain") == "", "profile worktree is dirty")

    timing = base.get("timing")
    _require(isinstance(timing, dict), "timing evidence is absent")
    samples = timing.get("samples_seconds")
    _require(
        isinstance(samples, list) and len(samples) == FROZEN_PROFILE_STEPS,
        "timing sample count differs",
    )
    measured = [_positive(value, "timing sample") for value in samples]
    mean = statistics.fmean(measured)
    median = statistics.median(measured)
    deviation = statistics.stdev(measured) if len(measured) > 1 else 0.0
    cv = deviation / mean
    _require(cv <= PROFILE_CV_MAX, "profile CV exceeds the frozen gate")
    _require(
        math.isclose(
            _positive(timing.get("mean_seconds"), "mean time"),
            mean,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ),
        "reported mean differs from samples",
    )
    _require(
        math.isclose(
            _positive(timing.get("median_seconds"), "median time"),
            median,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ),
        "reported median differs from samples",
    )

    memory = base.get("memory")
    _require(isinstance(memory, dict), "memory evidence is absent")
    allocated = _positive_integer(memory.get("peak_allocated_bytes"), "peak allocated")
    reserved = _positive_integer(memory.get("peak_reserved_bytes"), "peak reserved")
    _require(allocated <= reserved, "allocated memory exceeds reserved memory")
    result = base.get("result")
    _require(isinstance(result, dict) and result.get("finite") is True, "result is non-finite")
    if role == "functional_vjp":
        _require(result.get("nonzero") is True, "VJP has a zero gradient path")
        _require(
            result.get("replay_bitwise_equal") is True,
            "VJP replay is not byte-identical",
        )
        result_sha = result.get("gradient_sha256")
    else:
        result_sha = result.get("state_sha256")
    _require(
        isinstance(result_sha, str) and len(result_sha) == 64,
        "profile result SHA-256 is absent",
    )

    counts = validate_capability_audit(
        report.get("capability_audit"),
        role=role,
        candidate=variant == "candidate",
    )
    return variant, grid, role, trial, counts


def analyze(
    reports: list[dict[str, object]],
    *,
    baseline_commit: str,
    candidate_commit: str,
) -> dict[str, object]:
    expected_count = 2 * len(FROZEN_GRIDS) * len(PROFILE_ROLES) * FROZEN_TRIALS
    _require(
        len(reports) == expected_count,
        f"expected {expected_count} profiles, found {len(reports)}",
    )
    indexed: dict[tuple[str, str, str, int], dict[str, object]] = {}
    counts: dict[tuple[str, str, str, int], tuple[int, int]] = {}
    for report in reports:
        variant, grid, role, trial, transform_counts = _validate_profile(
            report,
            baseline_commit=baseline_commit,
            candidate_commit=candidate_commit,
        )
        key = (variant, grid, role, trial)
        _require(key not in indexed, f"duplicate profile key: {key!r}")
        indexed[key] = report
        counts[key] = transform_counts

    summary: dict[str, object] = {}
    for grid in FROZEN_GRIDS:
        grid_summary: dict[str, object] = {}
        for role in PROFILE_ROLES:
            baseline = [
                indexed[("baseline", grid, role, trial)]
                for trial in range(1, FROZEN_TRIALS + 1)
            ]
            candidate = [
                indexed[("candidate", grid, role, trial)]
                for trial in range(1, FROZEN_TRIALS + 1)
            ]
            initial_hashes = {
                item["base_profile"]["config"]["initial_q_sha256"]
                for item in (*baseline, *candidate)
            }
            _require(len(initial_hashes) == 1, f"initial Q differs for {grid} {role}")
            for trial in range(1, FROZEN_TRIALS + 1):
                _require(
                    counts[("baseline", grid, role, trial)]
                    == counts[("candidate", grid, role, trial)],
                    f"transform calls changed for {grid} {role} trial {trial}",
                )

            baseline_means = [item["base_profile"]["timing"]["mean_seconds"] for item in baseline]
            candidate_means = [item["base_profile"]["timing"]["mean_seconds"] for item in candidate]
            baseline_medians = [item["base_profile"]["timing"]["median_seconds"] for item in baseline]
            candidate_medians = [item["base_profile"]["timing"]["median_seconds"] for item in candidate]
            mean_ratio = statistics.fmean(candidate_means) / statistics.fmean(baseline_means)
            median_ratio = statistics.median(candidate_medians) / statistics.median(baseline_medians)
            allocated_ratio = max(item["base_profile"]["memory"]["peak_allocated_bytes"] for item in candidate) / max(item["base_profile"]["memory"]["peak_allocated_bytes"] for item in baseline)
            reserved_ratio = max(item["base_profile"]["memory"]["peak_reserved_bytes"] for item in candidate) / max(item["base_profile"]["memory"]["peak_reserved_bytes"] for item in baseline)
            _require(mean_ratio <= MEAN_RATIO_MAX, f"mean regression for {grid} {role}")
            _require(median_ratio <= MEDIAN_RATIO_MAX, f"median regression for {grid} {role}")
            _require(allocated_ratio <= MEMORY_RATIO_MAX, f"allocated-memory regression for {grid} {role}")
            _require(reserved_ratio <= MEMORY_RATIO_MAX, f"reserved-memory regression for {grid} {role}")
            grid_summary[role] = {
                "mean_timestep_ratio": mean_ratio,
                "median_timestep_ratio": median_ratio,
                "peak_allocated_ratio": allocated_ratio,
                "peak_reserved_ratio": reserved_ratio,
                "transform_calls": {
                    "forward": counts[("candidate", grid, role, 1)][0],
                    "inverse": counts[("candidate", grid, role, 1)][1],
                },
            }
        for variant in ("baseline", "candidate"):
            production = [
                indexed[(variant, grid, "production_forward", trial)]
                for trial in range(1, FROZEN_TRIALS + 1)
            ]
            functional = [
                indexed[(variant, grid, "functional_forward", trial)]
                for trial in range(1, FROZEN_TRIALS + 1)
            ]
            vjp = [
                indexed[(variant, grid, "functional_vjp", trial)]
                for trial in range(1, FROZEN_TRIALS + 1)
            ]
            for trial, left, right in zip(
                range(1, FROZEN_TRIALS + 1),
                production,
                functional,
                strict=True,
            ):
                _require(
                    left["base_profile"]["result"]["state_sha256"]
                    == right["base_profile"]["result"]["state_sha256"],
                    f"production/functional state differs for {variant} {grid} trial {trial}",
                )
            _require(
                len(
                    {
                        item["base_profile"]["result"]["state_sha256"]
                        for item in functional
                    }
                )
                == 1,
                f"functional replay differs for {variant} {grid}",
            )
            _require(
                len(
                    {
                        item["base_profile"]["result"]["gradient_sha256"]
                        for item in vjp
                    }
                )
                == 1,
                f"VJP replay differs for {variant} {grid}",
            )
            for role_reports in (functional, vjp):
                r12 = role_reports[0]["base_profile"]["correctness"].get("r12")
                _require(
                    isinstance(r12, dict) and r12.get("passed") is True,
                    f"production/functional consistency failed for {variant} {grid}",
                )
            gradient = vjp[0]["base_profile"]["correctness"].get("gradient")
            _require(
                isinstance(gradient, dict) and gradient.get("passed") is True,
                f"gradient validation failed for {variant} {grid}",
            )
        summary[grid] = grid_summary

    return {
        "schema_version": 1,
        "classification": PASS_CLASSIFICATION,
        "passed": True,
        "baseline_commit": baseline_commit,
        "candidate_commit": candidate_commit,
        "profile_count": len(reports),
        "gates": {
            "mean_ratio_max": MEAN_RATIO_MAX,
            "median_ratio_max": MEDIAN_RATIO_MAX,
            "memory_ratio_max": MEMORY_RATIO_MAX,
            "profile_cv_max": PROFILE_CV_MAX,
            "transform_calls_must_match": True,
            "eager_compile_metrics_must_be_inapplicable": True,
        },
        "summary": summary,
    }


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--baseline-commit", required=True)
    parser.add_argument("--candidate-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    paths = sorted(args.profiles.expanduser().resolve().glob("*.json"))
    reports = [_load(path) for path in paths]
    result = analyze(
        reports,
        baseline_commit=args.baseline_commit,
        candidate_commit=args.candidate_commit,
    )
    _atomic_json(args.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
