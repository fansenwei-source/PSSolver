#!/usr/bin/env python3
"""Analyze the P9.5 validator-v2 diagnostic without qualifying P9.5."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from benchmarks.diagnose_phase9_p95_gradient_validator_v2 import (
    ACTIVITY_EPSILONS,
    DIAGNOSTIC_KIND,
    DIRECTION_KINDS,
    METHODS,
    REFERENCE_RELATIVE_TOLERANCE,
    STATE_EPSILONS,
)


class GradientValidatorV2EvidenceError(RuntimeError):
    """Raised when validator-v2 diagnostic evidence is incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GradientValidatorV2EvidenceError(message)


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GradientValidatorV2EvidenceError(
            f"cannot read validator-v2 JSON {path}: {exc}"
        ) from exc
    _require(isinstance(value, dict), "validator-v2 JSON root must be an object")
    return value


def _validate_report(
    report: dict[str, object], *, expected_commit: str
) -> tuple[
    str,
    dict[tuple[str, str, str, float], float],
    dict[tuple[str, str], bool],
]:
    _require(report.get("schema_version") == 1, "unexpected diagnostic schema")
    _require(report.get("kind") == DIAGNOSTIC_KIND, "unexpected diagnostic kind")
    _require(
        report.get("phase") == "P9.5-gradient-validator-v2-diagnostic",
        "unexpected diagnostic phase",
    )
    config = report.get("config")
    _require(isinstance(config, dict), "diagnostic config is absent")
    grid = config.get("grid_id")
    _require(grid in {"R128", "R320"}, "unexpected diagnostic grid")
    expected_shape = {"R128": [128, 128, 32], "R320": [320, 320, 80]}[grid]
    _require(config.get("shape") == expected_shape, "diagnostic shape differs")
    _require(config.get("lengths") == [100.0, 100.0, 20.0], "lengths differ")
    _require(config.get("device") == "cuda:0", "device is not cuda:0")
    _require(config.get("dtype") == "float64", "dtype is not float64")
    _require(config.get("dt") == 0.001, "timestep differs")
    _require(config.get("base_activity") == 0.013, "activity differs")
    _require(
        config.get("state_epsilons") == list(STATE_EPSILONS),
        "state epsilon ladder differs",
    )
    _require(
        config.get("activity_epsilons") == list(ACTIVITY_EPSILONS),
        "activity epsilon ladder differs",
    )
    _require(
        config.get("direction_kinds") == list(DIRECTION_KINDS),
        "direction set differs",
    )
    _require(config.get("methods") == list(METHODS), "method set differs")
    _require(
        config.get("reference_relative_tolerance")
        == REFERENCE_RELATIVE_TOLERANCE,
        "reference tolerance differs",
    )
    sha = config.get("initial_q_sha256")
    _require(isinstance(sha, str) and len(sha) == 64, "initial Q hash absent")

    environment = report.get("environment")
    _require(isinstance(environment, dict), "environment is absent")
    git = environment.get("git")
    _require(isinstance(git, dict), "Git provenance is absent")
    _require(git.get("head") == expected_commit, "diagnostic commit differs")
    _require(git.get("status_porcelain") == "", "diagnostic worktree is dirty")
    _require(environment.get("cuda_available") is True, "CUDA is unavailable")
    _require(environment.get("device") == "cuda:0", "allocated device differs")
    _require(
        environment.get("device_name") == "NVIDIA H100 PCIe",
        "diagnostic did not use the frozen H100 model",
    )
    _require(environment.get("tf32_matmul") is False, "matmul TF32 is enabled")
    _require(environment.get("tf32_cudnn") is False, "cuDNN TF32 is enabled")

    frozen = report.get("frozen_validation")
    _require(isinstance(frozen, dict), "frozen validation report is absent")
    paths = frozen.get("gradient_paths")
    _require(isinstance(paths, list) and len(paths) == 6, "gradient paths absent")
    _require(
        all(
            item.get("finite") is True
            and item.get("nonzero") is True
            and item.get("passed") is True
            for item in paths
        ),
        "one or more differentiable paths failed",
    )
    derivatives = frozen.get("directional_derivatives")
    _require(
        isinstance(derivatives, list) and len(derivatives) == 2,
        "frozen directional derivatives are absent",
    )
    base_objective = float(report.get("base_objective", math.nan))
    _require(math.isfinite(base_objective), "base objective is non-finite")
    _require(report.get("all_finite") is True, "diagnostic contains NaN or Inf")
    _require(
        report.get("objective_replay_bitwise") is True,
        "objective replay is not bitwise exact",
    )

    sweeps = report.get("sweeps")
    _require(isinstance(sweeps, list) and len(sweeps) == 6, "sweep matrix differs")
    indexed: dict[tuple[str, str, str, float], float] = {}
    usable_directions: dict[tuple[str, str], bool] = {}
    expected_pairs = {
        (target, direction)
        for target in ("state", "activity")
        for direction in DIRECTION_KINDS
    }
    observed_pairs = set()
    for sweep in sweeps:
        target = sweep.get("target")
        direction = sweep.get("direction")
        _require((target, direction) in expected_pairs, "unexpected sweep identity")
        _require((target, direction) not in observed_pairs, "duplicate sweep")
        observed_pairs.add((target, direction))
        autograd_value = float(sweep.get("autograd", math.nan))
        _require(math.isfinite(autograd_value), "autograd derivative is non-finite")
        _require(
            sweep.get("autograd_nonzero") is (autograd_value != 0.0),
            "autograd nonzero flag disagrees with derivative",
        )
        direction_cosine = float(
            sweep.get("absolute_direction_cosine", math.nan)
        )
        _require(
            math.isfinite(direction_cosine)
            and 0.0 <= direction_cosine <= 1.0,
            "direction cosine is invalid",
        )
        usable_directions[(target, direction)] = autograd_value != 0.0
        epsilons = STATE_EPSILONS if target == "state" else ACTIVITY_EPSILONS
        rows = sweep.get("rows")
        _require(
            isinstance(rows, list) and len(rows) == len(epsilons),
            "epsilon sweep is incomplete",
        )
        for expected_epsilon, row in zip(epsilons, rows, strict=True):
            _require(row.get("epsilon") == expected_epsilon, "epsilon order differs")
            _require(
                all(
                    math.isfinite(float(row.get(name, math.nan)))
                    for name in ("objective_plus", "objective_minus")
                ),
                "perturbed objective is non-finite",
            )
            remainder = row.get("one_sided_linearization_remainder")
            _require(isinstance(remainder, dict), "Taylor remainder is absent")
            _require(
                all(
                    math.isfinite(float(remainder.get(name, math.nan)))
                    and float(remainder.get(name, math.nan)) >= 0.0
                    for name in ("plus", "minus", "normalized_max")
                ),
                "Taylor remainder is invalid",
            )
            methods = row.get("methods")
            _require(isinstance(methods, dict), "method evidence is absent")
            _require(set(methods) == set(METHODS), "method evidence differs")
            for method in METHODS:
                evidence = methods[method]
                _require(evidence.get("finite") is True, "method result is non-finite")
                error = float(evidence.get("relative_error", math.nan))
                _require(
                    math.isfinite(error) and error >= 0.0,
                    "relative error invalid",
                )
                _require(
                    evidence.get("passes_reference_tolerance")
                    is (error <= REFERENCE_RELATIVE_TOLERANCE),
                    "reference pass flag disagrees with error",
                )
                indexed[(target, direction, method, expected_epsilon)] = error
    _require(observed_pairs == expected_pairs, "sweep identity matrix is incomplete")
    _require(report.get("qualification_changed") is False, "qualification changed")
    _require(report.get("p9_5_pass_claimed") is False, "P9.5 pass was claimed")
    _require(report.get("p9_6_authorized") is False, "P9.6 was authorized")
    return grid, indexed, usable_directions


def _error_orders(
    errors: dict[float, float],
) -> list[dict[str, float | None]]:
    ordered = sorted(errors)
    result = []
    for lower, upper in zip(ordered, ordered[1:]):
        lower_error = errors[lower]
        upper_error = errors[upper]
        order = None
        if lower_error > 0.0 and upper_error > 0.0:
            order = math.log(upper_error / lower_error) / math.log(upper / lower)
        result.append(
            {
                "epsilon_lower": lower,
                "epsilon_upper": upper,
                "error_lower": lower_error,
                "error_upper": upper_error,
                "empirical_error_order": order,
            }
        )
    return result


def analyze(
    reports: list[dict[str, object]], *, expected_commit: str
) -> dict[str, object]:
    _require(len(reports) == 2, "exactly two grid diagnostics are required")
    grids: dict[str, dict[tuple[str, str, str, float], float]] = {}
    usability: dict[str, dict[tuple[str, str], bool]] = {}
    input_hashes = {}
    for report in reports:
        grid, indexed, usable = _validate_report(
            report, expected_commit=expected_commit
        )
        _require(grid not in grids, f"duplicate diagnostic grid {grid}")
        grids[grid] = indexed
        usability[grid] = usable
        input_hashes[grid] = report["config"]["initial_q_sha256"]
    _require(set(grids) == {"R128", "R320"}, "grid diagnostic set differs")
    _require(
        input_hashes["R128"] != input_hashes["R320"],
        "cross-grid input identities unexpectedly match",
    )

    common = []
    best = []
    convergence = {}
    for target in ("state", "activity"):
        epsilons = STATE_EPSILONS if target == "state" else ACTIVITY_EPSILONS
        for direction in DIRECTION_KINDS:
            for method in METHODS:
                candidates = []
                errors_by_grid = {}
                for grid in ("R128", "R320"):
                    values = {
                        epsilon: grids[grid][
                            (target, direction, method, epsilon)
                        ]
                        for epsilon in epsilons
                    }
                    errors_by_grid[grid] = values
                    convergence[f"{grid}:{target}:{direction}:{method}"] = (
                        _error_orders(values)
                    )
                for epsilon in epsilons:
                    maximum = max(
                        errors_by_grid[grid][epsilon]
                        for grid in ("R128", "R320")
                    )
                    candidates.append((maximum, epsilon))
                    direction_usable = all(
                        usability[grid][(target, direction)]
                        for grid in ("R128", "R320")
                    )
                    if (
                        direction_usable
                        and maximum <= REFERENCE_RELATIVE_TOLERANCE
                    ):
                        common.append(
                            {
                                "target": target,
                                "direction": direction,
                                "method": method,
                                "epsilon": epsilon,
                                "max_cross_grid_relative_error": maximum,
                            }
                        )
                maximum, epsilon = min(candidates)
                best.append(
                    {
                        "target": target,
                        "direction": direction,
                        "method": method,
                        "epsilon": epsilon,
                        "max_cross_grid_relative_error": maximum,
                        "direction_usable_cross_grid": all(
                            usability[grid][(target, direction)]
                            for grid in ("R128", "R320")
                        ),
                        "relative_errors": {
                            grid: errors_by_grid[grid][epsilon]
                            for grid in ("R128", "R320")
                        },
                    }
                )

    independent_directions = {"frozen_oscillatory", "low_mode"}
    targets_with_independent_paired_candidate = {
        item["target"]
        for item in common
        if item["method"] == "paired_quadratic"
        and item["direction"] in independent_directions
    }
    targets_with_any_paired_candidate = {
        item["target"]
        for item in common
        if item["method"] == "paired_quadratic"
    }
    if targets_with_independent_paired_candidate == {"state", "activity"}:
        classification = "PAIRED_FD_INDEPENDENT_CROSS_GRID_CANDIDATE_FOUND"
    elif targets_with_any_paired_candidate == {"state", "activity"}:
        classification = "PAIRED_FD_ALIGNED_ONLY_CROSS_GRID_CANDIDATE_FOUND"
    else:
        classification = "NO_PAIRED_FD_CROSS_GRID_CANDIDATE_AT_REFERENCE_TOLERANCE"

    return {
        "schema_version": 1,
        "phase": "P9.5-gradient-validator-v2-diagnostic",
        "classification": classification,
        "diagnostic_complete": True,
        "qualification_complete": False,
        "p9_5_complete": False,
        "eligible_for_p9_6": False,
        "expected_commit": expected_commit,
        "reference_relative_tolerance": REFERENCE_RELATIVE_TOLERANCE,
        "common_cross_grid_candidates": common,
        "best_cross_grid_candidates": best,
        "empirical_error_orders": convergence,
        "contract_change_authorized": False,
        "production_default_changed": False,
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
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = [_load(path) for path in sorted(args.diagnostics.glob("*.json"))]
    _atomic_json(args.output, analyze(reports, expected_commit=args.expected_commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
