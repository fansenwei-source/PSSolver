#!/usr/bin/env python3
"""Analyze the stopped S-09 CPU gate without running new profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


SCHEMA = "pssolver.v0_3.s09.cpu_6x60_stop_analysis.v1"
EXPECTED_SELF_SHA256 = "5f652cdc9c79093bbae4862fc3080a839cec7c6969e0e3b4bc76d7c4c0a710d2"
EXPECTED_CANDIDATE_SHA256 = "f4e42828e2c5dc1caad9729fb1b2b005d08a4f53cc4b3e82274dae436ae71085"
EXPECTED_CAUSAL_SHA256 = "ecf2638b45ab5e4febd61d0b5c25fa2f13936aace40085704dbf078777506ea2"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_bound(path: Path, expected: str) -> dict[str, object]:
    observed = _sha256(path)
    if observed != expected:
        raise ValueError(f"matrix hash mismatch for {path}: {observed}")
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, value: object) -> None:
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        if json.loads(path.read_text(encoding="utf-8")) != json.loads(payload):
            raise RuntimeError("atomic JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def _profile(
    matrix: dict[str, object],
    *,
    family: str,
    mode: str,
    trial: int,
    role: str,
) -> dict[str, object]:
    matches = [
        value
        for value in matrix["raw_cells"]
        if value["config"]["family"] == family
        and value["config"]["mode"] == mode
        and int(value["config"]["trial"]) == trial
        and value["config"]["role"] == role
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one profile for {family}/{mode}/{trial}/{role}")
    return matches[0]


def _analysis_cell(matrix: dict[str, object], family: str, mode: str) -> dict[str, object]:
    matches = [
        value
        for value in matrix["analysis"]["cells"]
        if value["family"] == family and value["mode"] == mode
    ]
    if len(matches) != 1:
        raise ValueError(f"expected one analysis cell for {family}/{mode}")
    return matches[0]


def _pressure_iterations(profile: dict[str, object]) -> int:
    return int(
        profile["pressure_diagnostics"]["payload"]["pressure"]["primal"][
            "achieved_iteration_count"
        ]
    )


def analyze_existing_evidence(
    self_control: dict[str, object],
    candidate: dict[str, object],
    causal: dict[str, object],
) -> dict[str, object]:
    failed = [value for value in candidate["analysis"]["cells"] if not value["passed"]]
    if len(failed) != 1:
        raise ValueError("expected the preserved candidate matrix to have one failed cell")
    failed_cell = failed[0]
    if (failed_cell["family"], failed_cell["mode"]) != (
        "channel_functional",
        "relative_tolerance",
    ):
        raise ValueError("unexpected failed cell")
    paired = [float(value) for value in failed_cell["paired_time_ratios"]]
    failed_trial = paired.index(max(paired)) + 1
    baseline = _profile(
        candidate,
        family="channel_functional",
        mode="relative_tolerance",
        trial=failed_trial,
        role="baseline",
    )
    measured = _profile(
        candidate,
        family="channel_functional",
        mode="relative_tolerance",
        trial=failed_trial,
        role="candidate",
    )
    baseline_samples = [float(value) for value in baseline["timing"]["samples_seconds"]]
    candidate_samples = [float(value) for value in measured["timing"]["samples_seconds"]]
    baseline_blocks = [
        float(value) for value in baseline["timing"]["fixed_block_means_seconds"]
    ]
    candidate_blocks = [
        float(value) for value in measured["timing"]["fixed_block_means_seconds"]
    ]
    if len(baseline_samples) != 60 or len(candidate_samples) != 60:
        raise ValueError("the preserved profile must retain all sixty samples")
    block_ratios = [
        candidate_value / baseline_value
        for baseline_value, candidate_value in zip(baseline_blocks, candidate_blocks)
    ]
    causal_cell = _analysis_cell(causal, "channel_functional", "relative_tolerance")
    self_max = max(
        float(value["maximum_time_ratio"])
        for value in self_control["analysis"]["cells"]
    )
    return {
        "schema": SCHEMA,
        "classification": "TRANSIENT_PROFILE_SLOWDOWN_SUPPORTED_CAUSE_UNRESOLVED",
        "new_profile_execution_count": 0,
        "input_verdicts_preserved": {
            "self_control_passed": bool(self_control["analysis"]["passed"]),
            "candidate_bound_passed": bool(candidate["analysis"]["passed"]),
            "failed_cell_count": len(failed),
            "self_control_maximum_paired_time_ratio": self_max,
        },
        "failed_profile": {
            "family": failed_cell["family"],
            "mode": failed_cell["mode"],
            "trial": failed_trial,
            "paired_profile_center_ratio": max(paired),
            "six_trial_median_ratio": float(failed_cell["median_time_ratio"]),
            "baseline_block_means_seconds": baseline_blocks,
            "candidate_block_means_seconds": candidate_blocks,
            "paired_block_ratios": block_ratios,
            "baseline_first_half_mean_seconds": statistics.fmean(baseline_samples[:30]),
            "baseline_second_half_mean_seconds": statistics.fmean(baseline_samples[30:]),
            "candidate_first_half_mean_seconds": statistics.fmean(candidate_samples[:30]),
            "candidate_second_half_mean_seconds": statistics.fmean(candidate_samples[30:]),
            "first_half_ratio": statistics.fmean(candidate_samples[:30])
            / statistics.fmean(baseline_samples[:30]),
            "second_half_ratio": statistics.fmean(candidate_samples[30:])
            / statistics.fmean(baseline_samples[30:]),
            "candidate_max_sample_seconds": max(candidate_samples),
            "first_three_candidate_blocks_strictly_decrease": all(
                left > right
                for left, right in zip(candidate_blocks[:2], candidate_blocks[1:3])
            ),
            "baseline_pressure_iterations": _pressure_iterations(baseline),
            "candidate_pressure_iterations": _pressure_iterations(measured),
            "same_pressure_iteration_count": _pressure_iterations(baseline)
            == _pressure_iterations(measured),
            "remaining_trial_ratios": paired[1:],
            "remaining_trials_all_below_1_05": all(value <= 1.05 for value in paired[1:]),
        },
        "prior_intervention": {
            "description": "5x30 candidate-with-acceptance-recompute divided by diagnostic-without-acceptance-recompute",
            "channel_functional_tolerance_median_ratio": float(
                causal_cell["median_time_ratio"]
            ),
            "channel_functional_tolerance_maximum_ratio": float(
                causal_cell["maximum_time_ratio"]
            ),
            "sufficient_to_explain_1_070771_failure": float(
                causal_cell["median_time_ratio"]
            )
            >= max(paired),
        },
        "adjudication": {
            "stable_true_residual_recompute_overhead_supported": True,
            "extra_pressure_iterations_supported": False,
            "single_deleted_outlier_explanation_supported": False,
            "transient_first_half_slowdown_supported": True,
            "unique_runtime_root_cause_supported": False,
            "qualification_failure_remains_binding": True,
            "new_measurement_required_to_distinguish_scheduler_frequency_cache_or_runtime_warmup": True,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--causal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze_existing_evidence(
        _read_bound(args.self_control, EXPECTED_SELF_SHA256),
        _read_bound(args.candidate, EXPECTED_CANDIDATE_SHA256),
        _read_bound(args.causal, EXPECTED_CAUSAL_SHA256),
    )
    if not all(
        math.isfinite(float(value))
        for value in result["failed_profile"]["paired_block_ratios"]
    ):
        raise ValueError("analysis produced a non-finite ratio")
    _atomic_json(args.output, result)
    print(json.dumps(result, allow_nan=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
