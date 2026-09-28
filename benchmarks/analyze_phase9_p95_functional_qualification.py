#!/usr/bin/env python3
"""Fail-closed analyzer for the frozen P9.5 H100 profile matrix."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics


PASS_CLASSIFICATION = "PASS_P9_5_H100_BATCH_ONE_FORWARD_VJP_MEMORY"
FROZEN_ROLES = (
    "production_forward",
    "functional_forward",
    "functional_vjp",
)
FROZEN_GRIDS = (
    ("R128", [128, 128, 32], [100.0, 100.0, 20.0]),
    ("R320", [320, 320, 80], [100.0, 100.0, 20.0]),
)
FROZEN_GATES = {
    "profile_cv_max": 0.2,
    "functional_over_production_mean_max": 1.2,
    "functional_over_production_median_max": 1.2,
    "forward_peak_allocated_ratio_max": 1.75,
    "forward_peak_reserved_ratio_max": 1.75,
    "vjp_over_functional_forward_mean_max": 12.0,
    "vjp_peak_allocated_device_fraction_max": 0.85,
    "vjp_peak_reserved_device_fraction_max": 0.95,
    "production_functional_state": "paired_sha256_exact",
    "functional_replay": "sha256_exact_across_trials",
    "vjp_replay": "sha256_exact_across_trials",
    "r12": "pass_on_each_grid",
    "gradient_validation": "pass_on_each_grid",
    "finite": True,
}


class P95EvidenceError(RuntimeError):
    """Raised when the P9.5 evidence matrix violates its frozen contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise P95EvidenceError(message)


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise P95EvidenceError(f"cannot read JSON {path}: {exc}") from exc
    _require(isinstance(value, dict), f"JSON root must be an object: {path}")
    return value


def _positive(value: object, description: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise P95EvidenceError(f"{description} is not numeric") from exc
    _require(
        math.isfinite(result) and result > 0.0,
        f"{description} must be finite and positive",
    )
    return result


def _integer(value: object, description: str) -> int:
    _require(
        isinstance(value, int) and not isinstance(value, bool) and value > 0,
        f"{description} must be a positive integer",
    )
    return value


def _grid_id(report: dict[str, object], plan: dict[str, object]) -> str:
    config = report["config"]
    matches = [
        item["id"]
        for item in plan["grids"]
        if config.get("shape") == item["shape"]
        and config.get("lengths") == item["lengths"]
    ]
    _require(len(matches) == 1, "profile grid does not match the frozen plan")
    return matches[0]


def _validate_plan(plan: dict[str, object]) -> None:
    _require(plan.get("schema_version") == 1, "unexpected plan schema")
    _require(plan.get("phase") == "P9.5", "plan is not P9.5")
    _require(tuple(plan.get("roles", ())) == FROZEN_ROLES, "plan roles changed")
    grids = plan.get("grids")
    _require(isinstance(grids, list), "plan grids are absent")
    observed = tuple(
        (item.get("id"), item.get("shape"), item.get("lengths")) for item in grids
    )
    _require(observed == FROZEN_GRIDS, "plan grids changed")
    _require(plan.get("trials") == 3, "plan trial count changed")
    _require(plan.get("warmup_steps") == 5, "plan warmup changed")
    _require(plan.get("profile_steps") == 20, "plan profile window changed")
    _require(plan.get("dt") == 0.001, "plan timestep changed")
    _require(plan.get("base_activity") == 0.013, "plan activity changed")
    _require(plan.get("gpu_model") == "NVIDIA H100 PCIe", "plan GPU changed")
    _require(plan.get("gates") == FROZEN_GATES, "plan gates changed")


def _validate_profile(
    report: dict[str, object],
    *,
    plan: dict[str, object],
    expected_commit: str,
) -> tuple[str, str, int]:
    _require(report.get("schema_version") == 1, "unexpected profile schema")
    _require(
        report.get("kind") == "p95_periodic_functional_profile",
        "unexpected profile kind",
    )
    _require(report.get("phase") == "P9.5", "unexpected profile phase")
    config = report.get("config")
    _require(isinstance(config, dict), "profile config is absent")
    grid = _grid_id(report, plan)
    role = config.get("role")
    _require(role in plan["roles"], "profile role is not frozen")
    trial = _integer(config.get("trial"), "trial")
    _require(1 <= trial <= plan["trials"], "trial is outside the frozen matrix")
    _require(config.get("device") == "cuda:0", "profile device is not cuda:0")
    _require(config.get("dtype") == "float64", "profile dtype is not float64")
    _require(config.get("dt") == plan["dt"], "profile dt differs from plan")
    _require(
        config.get("base_activity") == plan["base_activity"],
        "profile activity differs from plan",
    )
    _require(
        config.get("warmup_steps") == plan["warmup_steps"],
        "profile warmup differs from plan",
    )
    _require(
        config.get("profile_steps") == plan["profile_steps"],
        "profile step count differs from plan",
    )
    sha = config.get("initial_q_sha256")
    _require(
        isinstance(sha, str) and len(sha) == 64,
        "initial Q SHA-256 is absent",
    )

    environment = report.get("environment")
    _require(isinstance(environment, dict), "environment is absent")
    git = environment.get("git")
    _require(isinstance(git, dict), "Git provenance is absent")
    _require(git.get("head") == expected_commit, "profile commit differs")
    _require(git.get("status_porcelain") == "", "profile worktree is dirty")
    _require(environment.get("cuda_available") is True, "CUDA is unavailable")
    _require(environment.get("device") == "cuda:0", "allocated device differs")
    _require(
        environment.get("device_name") == plan["gpu_model"],
        "profile did not run on the frozen H100 model",
    )
    _require(environment.get("tf32_matmul") is False, "matmul TF32 is enabled")
    _require(environment.get("tf32_cudnn") is False, "cuDNN TF32 is enabled")

    timing = report.get("timing")
    _require(isinstance(timing, dict), "timing evidence is absent")
    samples = timing.get("samples_seconds")
    _require(
        isinstance(samples, list) and len(samples) == plan["profile_steps"],
        "timing sample count differs from plan",
    )
    for index, value in enumerate(samples):
        _positive(value, f"timing sample {index}")
    mean = statistics.fmean(samples)
    median = statistics.median(samples)
    deviation = statistics.stdev(samples) if len(samples) > 1 else 0.0
    computed_cv = deviation / mean
    _require(
        math.isclose(
            _positive(timing.get("mean_seconds"), "mean time"),
            mean,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ),
        "reported mean does not match timing samples",
    )
    _require(
        math.isclose(
            _positive(timing.get("median_seconds"), "median time"),
            median,
            rel_tol=1.0e-12,
            abs_tol=0.0,
        ),
        "reported median does not match timing samples",
    )
    cv = float(timing.get("coefficient_of_variation", math.inf))
    _require(math.isfinite(cv) and cv >= 0.0, "timing CV is invalid")
    _require(
        math.isclose(cv, computed_cv, rel_tol=1.0e-12, abs_tol=1.0e-15),
        "reported CV does not match timing samples",
    )
    _require(cv <= plan["gates"]["profile_cv_max"], "profile CV exceeds gate")

    memory = report.get("memory")
    _require(isinstance(memory, dict), "memory evidence is absent")
    allocated = _integer(memory.get("peak_allocated_bytes"), "peak allocated")
    active = _integer(memory.get("peak_active_bytes"), "peak active")
    reserved = _integer(memory.get("peak_reserved_bytes"), "peak reserved")
    total = _integer(memory.get("device_total_bytes"), "device total memory")
    _require(active <= reserved and allocated <= reserved, "memory peaks disagree")
    _require(reserved <= total, "reserved memory exceeds device total")

    result = report.get("result")
    _require(isinstance(result, dict), "profile result is absent")
    _require(result.get("finite") is True, "profile produced non-finite values")
    if role == "functional_vjp":
        _require(result.get("nonzero") is True, "VJP has a zero gradient path")
        _require(
            result.get("replay_bitwise_equal") is True,
            "VJP replay is not bitwise identical",
        )
        norms = result.get("gradient_norms")
        _require(isinstance(norms, list) and len(norms) == 3, "VJP norms absent")
        for index, value in enumerate(norms):
            _positive(value, f"gradient norm {index}")
        _require(
            isinstance(result.get("gradient_sha256"), str)
            and len(result["gradient_sha256"]) == 64,
            "gradient SHA-256 is absent",
        )
    else:
        _require(
            isinstance(result.get("state_sha256"), str)
            and len(result["state_sha256"]) == 64,
            "state SHA-256 is absent",
        )

    if role != "production_forward":
        identity = report.get("functional_identity")
        _require(isinstance(identity, dict), "functional identity is absent")
        execution = identity.get("execution")
        _require(isinstance(execution, dict), "functional execution is absent")
        functional = execution.get("functional_runtime")
        _require(isinstance(functional, dict), "functional runtime is absent")
        _require(functional["fallback_used"] is False, "functional fallback used")
        _require(
            functional["deterministic_replay"] == "not_qualified",
            "pre-P9.5 CUDA replay claim changed before qualification",
        )
    return grid, role, trial


def analyze(
    plan: dict[str, object],
    reports: list[dict[str, object]],
    *,
    expected_commit: str,
) -> dict[str, object]:
    _validate_plan(plan)
    expected_count = len(plan["grids"]) * len(plan["roles"]) * plan["trials"]
    _require(
        len(reports) == expected_count,
        f"expected {expected_count} profiles, found {len(reports)}",
    )
    indexed = {}
    for report in reports:
        key = _validate_profile(
            report,
            plan=plan,
            expected_commit=expected_commit,
        )
        _require(key not in indexed, f"duplicate profile key: {key!r}")
        indexed[key] = report

    summary = {}
    for grid_record in plan["grids"]:
        grid = grid_record["id"]
        production = [
            indexed[(grid, "production_forward", trial)]
            for trial in range(1, plan["trials"] + 1)
        ]
        functional = [
            indexed[(grid, "functional_forward", trial)]
            for trial in range(1, plan["trials"] + 1)
        ]
        vjp = [
            indexed[(grid, "functional_vjp", trial)]
            for trial in range(1, plan["trials"] + 1)
        ]
        initial_hashes = {
            report["config"]["initial_q_sha256"]
            for report in (*production, *functional, *vjp)
        }
        _require(len(initial_hashes) == 1, f"initial Q differs for {grid}")
        for trial, left, right in zip(
            range(1, plan["trials"] + 1),
            production,
            functional,
            strict=True,
        ):
            _require(
                left["result"]["state_sha256"]
                == right["result"]["state_sha256"],
                f"production/functional final state differs for {grid} trial {trial}",
            )
        _require(
            len({item["result"]["state_sha256"] for item in functional}) == 1,
            f"functional CUDA replay differs across trials for {grid}",
        )
        _require(
            len({item["result"]["gradient_sha256"] for item in vjp}) == 1,
            f"VJP replay differs across trials for {grid}",
        )
        for role_reports in (functional, vjp):
            r12 = role_reports[0]["correctness"].get("r12")
            _require(
                isinstance(r12, dict) and r12.get("passed") is True,
                f"R12 H100 gate failed for {grid}",
            )
        gradient = vjp[0]["correctness"].get("gradient")
        _require(
            isinstance(gradient, dict) and gradient.get("passed") is True,
            f"H100 gradient gate failed for {grid}",
        )

        production_times = [item["timing"]["mean_seconds"] for item in production]
        functional_times = [item["timing"]["mean_seconds"] for item in functional]
        vjp_times = [item["timing"]["mean_seconds"] for item in vjp]
        forward_ratio = statistics.fmean(functional_times) / statistics.fmean(
            production_times
        )
        forward_median_ratio = statistics.median(functional_times) / statistics.median(
            production_times
        )
        vjp_ratio = statistics.fmean(vjp_times) / statistics.fmean(functional_times)
        forward_allocated_ratio = max(
            item["memory"]["peak_allocated_bytes"] for item in functional
        ) / max(item["memory"]["peak_allocated_bytes"] for item in production)
        forward_reserved_ratio = max(
            item["memory"]["peak_reserved_bytes"] for item in functional
        ) / max(item["memory"]["peak_reserved_bytes"] for item in production)
        vjp_peak_allocated = max(
            item["memory"]["peak_allocated_bytes"] for item in vjp
        )
        vjp_peak_reserved = max(
            item["memory"]["peak_reserved_bytes"] for item in vjp
        )
        device_total = min(item["memory"]["device_total_bytes"] for item in vjp)
        gates = plan["gates"]
        _require(
            forward_ratio <= gates["functional_over_production_mean_max"],
            f"functional mean forward non-regression failed for {grid}",
        )
        _require(
            forward_median_ratio
            <= gates["functional_over_production_median_max"],
            f"functional median forward non-regression failed for {grid}",
        )
        _require(
            forward_allocated_ratio <= gates["forward_peak_allocated_ratio_max"],
            f"functional allocated-memory gate failed for {grid}",
        )
        _require(
            forward_reserved_ratio <= gates["forward_peak_reserved_ratio_max"],
            f"functional reserved-memory gate failed for {grid}",
        )
        _require(
            vjp_ratio <= gates["vjp_over_functional_forward_mean_max"],
            f"VJP time gate failed for {grid}",
        )
        _require(
            vjp_peak_allocated / device_total
            <= gates["vjp_peak_allocated_device_fraction_max"],
            f"VJP allocated-memory fraction failed for {grid}",
        )
        _require(
            vjp_peak_reserved / device_total
            <= gates["vjp_peak_reserved_device_fraction_max"],
            f"VJP reserved-memory fraction failed for {grid}",
        )
        summary[grid] = {
            "production_mean_seconds": statistics.fmean(production_times),
            "functional_forward_mean_seconds": statistics.fmean(functional_times),
            "functional_vjp_mean_seconds": statistics.fmean(vjp_times),
            "functional_over_production_mean_ratio": forward_ratio,
            "functional_over_production_median_ratio": forward_median_ratio,
            "vjp_over_functional_forward_mean_ratio": vjp_ratio,
            "forward_peak_allocated_ratio": forward_allocated_ratio,
            "forward_peak_reserved_ratio": forward_reserved_ratio,
            "vjp_peak_allocated_bytes": vjp_peak_allocated,
            "vjp_peak_reserved_bytes": vjp_peak_reserved,
            "device_total_bytes": device_total,
            "state_replay": "bitwise",
            "gradient_replay": "bitwise",
            "r12": "pass",
            "gradient_validation": "pass",
        }

    return {
        "schema_version": 1,
        "phase": "P9.5",
        "classification": PASS_CLASSIFICATION,
        "qualification_complete": True,
        "expected_commit": expected_commit,
        "profile_count": len(reports),
        "batch_sizes_qualified": [1],
        "larger_batch_qualified": False,
        "cuda_deterministic_replay": "bitwise",
        "summary": summary,
        "gates": plan["gates"],
        "production_default_changed": False,
        "stable_package_root_changed": False,
        "eligible_for_p9_6": True,
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
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    plan = _load(args.plan)
    reports = [
        _load(path)
        for path in sorted(args.profiles.glob("*.json"))
    ]
    _atomic_json(
        args.output,
        analyze(plan, reports, expected_commit=args.expected_commit),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
