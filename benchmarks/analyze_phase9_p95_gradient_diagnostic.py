#!/usr/bin/env python3
"""Analyze P9.5 H100 epsilon sweeps without changing qualification status."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

from benchmarks.diagnose_phase9_p95_gradient_epsilon import (
    ACTIVITY_EPSILONS,
    DIAGNOSTIC_KIND,
    FROZEN_RELATIVE_TOLERANCE,
)


class GradientDiagnosticEvidenceError(RuntimeError):
    """Raised when the diagnostic evidence is malformed or incomplete."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GradientDiagnosticEvidenceError(message)


def _load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GradientDiagnosticEvidenceError(
            f"cannot read diagnostic JSON {path}: {exc}"
        ) from exc
    _require(isinstance(value, dict), "diagnostic JSON root must be an object")
    return value


def _validate_report(
    report: dict[str, object], *, expected_commit: str
) -> tuple[str, int, dict[float, float]]:
    _require(report.get("schema_version") == 1, "unexpected diagnostic schema")
    _require(report.get("kind") == DIAGNOSTIC_KIND, "unexpected diagnostic kind")
    _require(
        report.get("phase") == "P9.5-gradient-diagnostic",
        "unexpected diagnostic phase",
    )
    config = report.get("config")
    _require(isinstance(config, dict), "diagnostic config is absent")
    grid = config.get("grid_id")
    _require(grid in {"R128", "R320"}, "unexpected diagnostic grid")
    trial = config.get("trial")
    _require(
        isinstance(trial, int) and not isinstance(trial, bool) and 1 <= trial <= 3,
        "unexpected diagnostic trial",
    )
    expected_shape = {"R128": [128, 128, 32], "R320": [320, 320, 80]}[grid]
    _require(config.get("shape") == expected_shape, "diagnostic shape differs")
    _require(config.get("lengths") == [100.0, 100.0, 20.0], "lengths differ")
    _require(config.get("device") == "cuda:0", "device is not cuda:0")
    _require(config.get("dtype") == "float64", "dtype is not float64")
    _require(config.get("dt") == 0.001, "timestep differs")
    _require(config.get("base_activity") == 0.013, "activity differs")
    _require(
        config.get("activity_epsilons") == list(ACTIVITY_EPSILONS),
        "epsilon sweep differs",
    )
    _require(
        config.get("relative_tolerance") == FROZEN_RELATIVE_TOLERANCE,
        "relative tolerance differs",
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
    indexed_derivatives = {item.get("input"): item for item in derivatives}
    _require(set(indexed_derivatives) == {"state", "activity"}, "inputs differ")
    _require(
        indexed_derivatives["state"].get("passed") is True,
        "state directional derivative failed",
    )

    rows = report.get("activity_sweep")
    _require(
        isinstance(rows, list) and len(rows) == len(ACTIVITY_EPSILONS),
        "activity sweep is incomplete",
    )
    errors = {}
    for expected_epsilon, row in zip(ACTIVITY_EPSILONS, rows, strict=True):
        _require(row.get("epsilon") == expected_epsilon, "epsilon order differs")
        _require(row.get("finite") is True, "epsilon row is non-finite")
        _require(
            row.get("objective_replay_bitwise") is True,
            "objective replay is not bitwise exact",
        )
        error = float(row.get("relative_error", math.inf))
        _require(math.isfinite(error) and error >= 0.0, "relative error invalid")
        _require(
            row.get("passes_frozen_relative_tolerance")
            is (error <= FROZEN_RELATIVE_TOLERANCE),
            "epsilon pass flag disagrees with its error",
        )
        errors[expected_epsilon] = error
    _require(report.get("qualification_changed") is False, "qualification changed")
    _require(report.get("p9_5_pass_claimed") is False, "P9.5 pass was claimed")
    return grid, trial, errors


def analyze(
    reports: list[dict[str, object]], *, expected_commit: str
) -> dict[str, object]:
    _require(len(reports) == 6, "exactly six grid/trial diagnostics are required")
    indexed = {}
    for report in reports:
        grid, trial, errors = _validate_report(
            report, expected_commit=expected_commit
        )
        key = (grid, trial)
        _require(key not in indexed, f"duplicate diagnostic key {key}")
        indexed[key] = errors
    expected_keys = {
        (grid, trial)
        for grid in ("R128", "R320")
        for trial in range(1, 4)
    }
    _require(set(indexed) == expected_keys, "grid/trial diagnostic set differs")
    for grid in ("R128", "R320"):
        hashes = {
            report["config"]["initial_q_sha256"]
            for report in reports
            if report["config"]["grid_id"] == grid
        }
        _require(len(hashes) == 1, f"{grid} input identity differs across trials")
    common = [
        epsilon
        for epsilon in ACTIVITY_EPSILONS
        if all(
            indexed[key][epsilon] <= FROZEN_RELATIVE_TOLERANCE
            for key in sorted(expected_keys)
        )
    ]
    ranked = sorted(
        ACTIVITY_EPSILONS,
        key=lambda epsilon: max(
            indexed[key][epsilon] for key in sorted(expected_keys)
        ),
    )
    classification = (
        "DIAGNOSED_ACTIVITY_FD_SCALE_SENSITIVITY_WITH_COMMON_CANDIDATE"
        if common
        else "DIAGNOSTIC_COMPLETE_WITHOUT_COMMON_EPSILON_CANDIDATE"
    )
    return {
        "schema_version": 1,
        "phase": "P9.5-gradient-diagnostic",
        "classification": classification,
        "diagnostic_complete": True,
        "qualification_complete": False,
        "p9_5_complete": False,
        "eligible_for_p9_6": False,
        "expected_commit": expected_commit,
        "frozen_relative_tolerance": FROZEN_RELATIVE_TOLERANCE,
        "relative_errors": {
            f"{grid}:trial{trial}": {
                str(epsilon): value for epsilon, value in errors.items()
            }
            for (grid, trial), errors in sorted(indexed.items())
        },
        "common_passing_epsilons": common,
        "lowest_max_error_epsilon": ranked[0],
        "lowest_max_error": max(
            indexed[key][ranked[0]] for key in sorted(expected_keys)
        ),
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
    reports = [
        _load(path) for path in sorted(args.diagnostics.glob("*.json"))
    ]
    _atomic_json(args.output, analyze(reports, expected_commit=args.expected_commit))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
