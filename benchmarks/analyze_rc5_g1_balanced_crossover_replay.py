#!/usr/bin/env python3
"""Replay frozen v10/v13 evidence under the proposed balanced-crossover contract.

This is an analysis-only tool.  It first runs the existing paired-block input,
identity, progress, timing, numerical, transform, and memory validation.  It
then replaces only the maximum-single-trial adjudication with the proposed
order-balanced replicated-crossover estimator.  It cannot grant G-1
qualification or change either historical classification.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.analyze_rc5_g1_paired_block_replay import (
    _json,
    _timing,
    analyze as validate_paired_block_evidence,
)
from benchmarks.plan_rc5_g1_balanced_crossover_contract import build_plan


SCHEMA = "pssolver.rc5.g1.balanced_crossover_dual_replay.v1"
PASS_CLASSIFICATION = (
    "ANALYSIS_ONLY_V10_V13_REPLAY_PASSES_PROPOSED_V14_BALANCED_CROSSOVER_CONTRACT"
)
FAIL_CLASSIFICATION = (
    "ANALYSIS_ONLY_V10_V13_REPLAY_FAILS_PROPOSED_V14_BALANCED_CROSSOVER_CONTRACT"
)
RUN_PASS_CLASSIFICATION = "ANALYSIS_ONLY_REPLAY_PASSES_PROPOSED_BALANCED_CROSSOVER_CONTRACT"
RUN_FAIL_CLASSIFICATION = "ANALYSIS_ONLY_REPLAY_FAILS_PROPOSED_BALANCED_CROSSOVER_CONTRACT"
V10_FORMAL_CLASSIFICATION = "FAIL_RC5_G1_CORRECTED_MEASUREMENT_WINDOW_GATE"
V13_FORMAL_CLASSIFICATION = "FAIL_RC5_G1_PAIRED_BLOCK_PERFORMANCE_MEMORY_GATE"
RUNS = {
    "v10": {
        "progress_schema": "pssolver.rc5.g1.measurement_window_progress.v1",
        "formal_classification": V10_FORMAL_CLASSIFICATION,
    },
    "v13": {
        "progress_schema": "pssolver.rc5.g1.paired_block_progress.v1",
        "formal_classification": V13_FORMAL_CLASSIFICATION,
    },
}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


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


def adjudicate_cell(
    *,
    trial_ratios: list[float],
    aggregate_all_sample_mean_ratio: float,
    median_of_profile_means_ratio: float,
    peak_allocated_group_ratio: float,
    peak_reserved_group_ratio: float,
    thresholds: dict[str, object] | None = None,
) -> dict[str, object]:
    """Apply the proposed cell-level estimator to six ordered trial ratios."""

    if thresholds is None:
        thresholds = build_plan()["thresholds"]
    _require(len(trial_ratios) == 6, "exactly six trial ratios are required")
    ratios = [
        _number(value, f"trial ratio {index}", positive=True)
        for index, value in enumerate(trial_ratios, start=1)
    ]
    aggregate = _number(
        aggregate_all_sample_mean_ratio,
        "aggregate all-sample mean ratio",
        positive=True,
    )
    median_profile = _number(
        median_of_profile_means_ratio,
        "median of profile means ratio",
        positive=True,
    )
    allocated = _number(
        peak_allocated_group_ratio, "peak allocated group ratio", positive=True
    )
    reserved = _number(
        peak_reserved_group_ratio, "peak reserved group ratio", positive=True
    )
    baseline_then_candidate = [ratios[index] for index in (0, 2, 4)]
    candidate_then_baseline = [ratios[index] for index in (1, 3, 5)]
    baseline_then_candidate_median = statistics.median(baseline_then_candidate)
    candidate_then_baseline_median = statistics.median(candidate_then_baseline)
    order_imbalance = max(
        baseline_then_candidate_median, candidate_then_baseline_median
    ) / min(baseline_then_candidate_median, candidate_then_baseline_median)
    individual_limit = _number(
        thresholds["individual_trial_ratio_reference_max"],
        "individual trial reference",
        positive=True,
    )
    passing_trial_count = sum(value <= individual_limit for value in ratios)
    metrics = {
        "aggregate_all_sample_mean_ratio": aggregate,
        "median_of_profile_means_ratio": median_profile,
        "median_of_six_trial_ratios": statistics.median(ratios),
        "baseline_then_candidate_stratum_median_ratio": baseline_then_candidate_median,
        "candidate_then_baseline_stratum_median_ratio": candidate_then_baseline_median,
        "order_stratum_imbalance_ratio": order_imbalance,
        "individual_trial_ratios": ratios,
        "trials_at_or_below_individual_reference": passing_trial_count,
        "peak_allocated_group_ratio": allocated,
        "peak_reserved_group_ratio": reserved,
    }
    gates = {
        "aggregate_all_sample_mean": aggregate
        <= _number(thresholds["aggregate_all_sample_mean_ratio_max"], "aggregate limit"),
        "median_of_profile_means": median_profile
        <= _number(thresholds["median_of_profile_means_ratio_max"], "profile-median limit"),
        "median_of_six_trial_ratios": metrics["median_of_six_trial_ratios"]
        <= _number(thresholds["median_of_six_trial_ratios_max"], "trial-median limit"),
        "baseline_then_candidate_stratum": baseline_then_candidate_median
        <= _number(
            thresholds["baseline_then_candidate_stratum_median_ratio_max"],
            "baseline-then-candidate limit",
        ),
        "candidate_then_baseline_stratum": candidate_then_baseline_median
        <= _number(
            thresholds["candidate_then_baseline_stratum_median_ratio_max"],
            "candidate-then-baseline limit",
        ),
        "order_stratum_imbalance": order_imbalance
        <= _number(
            thresholds["order_stratum_imbalance_ratio_max"],
            "order imbalance limit",
        ),
        "minimum_individual_trial_count": passing_trial_count
        >= int(thresholds["minimum_trials_at_or_below_individual_reference"]),
        "peak_allocated": allocated
        <= _number(thresholds["peak_allocated_group_ratio_max"], "allocated limit"),
        "peak_reserved": reserved
        <= _number(thresholds["peak_reserved_group_ratio_max"], "reserved limit"),
    }
    return {"metrics": metrics, "gates": gates, "passed": all(gates.values())}


def _replay_run(
    name: str,
    profile_root: Path,
    progress_root: Path,
) -> dict[str, object]:
    metadata = RUNS[name]
    legacy = validate_paired_block_evidence(
        profile_root,
        progress_root,
        evidence_mode="post_hoc_v10_replay",
        progress_schema=str(metadata["progress_schema"]),
    )
    _require(legacy["profile_count"] == 48, f"{name} profile count differs")
    _require(legacy["execution_sequence_validated"] is True, f"{name} sequence differs")

    plan = build_plan()
    groups: list[dict[str, object]] = []
    for cell in plan["focused_cells"]:
        family = str(cell["family"])
        grid = str(cell["grid"])
        baseline_samples: list[float] = []
        candidate_samples: list[float] = []
        baseline_means: list[float] = []
        candidate_means: list[float] = []
        baseline_allocated: list[int] = []
        candidate_allocated: list[int] = []
        baseline_reserved: list[int] = []
        candidate_reserved: list[int] = []
        trial_ratios: list[float] = []
        pairs: list[dict[str, object]] = []
        for trial in range(1, 7):
            baseline_key = f"baseline_{family}_{grid}_trial{trial}"
            candidate_key = f"candidate_{family}_{grid}_trial{trial}"
            baseline_profile = _json(profile_root / f"{baseline_key}.json")
            candidate_profile = _json(profile_root / f"{candidate_key}.json")
            baseline_timing = _timing(baseline_profile, baseline_key)
            candidate_timing = _timing(candidate_profile, candidate_key)
            baseline_center = statistics.median(
                baseline_timing["fixed_block_means_seconds"]
            )
            candidate_center = statistics.median(
                candidate_timing["fixed_block_means_seconds"]
            )
            ratio = candidate_center / baseline_center
            trial_ratios.append(ratio)
            baseline_samples.extend(baseline_timing["samples_seconds"])
            candidate_samples.extend(candidate_timing["samples_seconds"])
            baseline_means.append(baseline_timing["mean_timestep_seconds"])
            candidate_means.append(candidate_timing["mean_timestep_seconds"])
            baseline_memory = baseline_profile["memory"]
            candidate_memory = candidate_profile["memory"]
            baseline_allocated.append(int(baseline_memory["peak_allocated_bytes"]))
            candidate_allocated.append(int(candidate_memory["peak_allocated_bytes"]))
            baseline_reserved.append(int(baseline_memory["peak_reserved_bytes"]))
            candidate_reserved.append(int(candidate_memory["peak_reserved_bytes"]))
            pairs.append(
                {
                    "trial": trial,
                    "execution_order": (
                        ["baseline", "candidate"]
                        if trial % 2
                        else ["candidate", "baseline"]
                    ),
                    "baseline_profile_center_seconds": baseline_center,
                    "candidate_profile_center_seconds": candidate_center,
                    "candidate_over_baseline_trial_ratio": ratio,
                    "baseline_fixed_block_means_seconds": baseline_timing[
                        "fixed_block_means_seconds"
                    ],
                    "candidate_fixed_block_means_seconds": candidate_timing[
                        "fixed_block_means_seconds"
                    ],
                }
            )
        adjudication = adjudicate_cell(
            trial_ratios=trial_ratios,
            aggregate_all_sample_mean_ratio=(
                statistics.fmean(candidate_samples) / statistics.fmean(baseline_samples)
            ),
            median_of_profile_means_ratio=(
                statistics.median(candidate_means) / statistics.median(baseline_means)
            ),
            peak_allocated_group_ratio=max(candidate_allocated) / max(baseline_allocated),
            peak_reserved_group_ratio=max(candidate_reserved) / max(baseline_reserved),
            thresholds=plan["thresholds"],
        )
        groups.append(
            {
                "family": family,
                "grid": grid,
                **adjudication,
                "pairs": pairs,
            }
        )
    passed = all(group["passed"] for group in groups)
    return {
        "run": name,
        "classification": RUN_PASS_CLASSIFICATION if passed else RUN_FAIL_CLASSIFICATION,
        "formal_historical_classification_preserved": metadata["formal_classification"],
        "legacy_paired_block_replay_classification": legacy["classification"],
        "analysis_only": True,
        "proposed_contract_passed": passed,
        "profile_count": 48,
        "progress_record_count": 48,
        "execution_sequence_validated": True,
        "groups": groups,
    }


def analyze(
    *,
    v10_profile_root: Path,
    v10_progress_root: Path,
    v13_profile_root: Path,
    v13_progress_root: Path,
) -> dict[str, object]:
    runs = [
        _replay_run("v10", v10_profile_root, v10_progress_root),
        _replay_run("v13", v13_profile_root, v13_progress_root),
    ]
    passed = all(run["proposed_contract_passed"] for run in runs)
    plan = build_plan()
    return {
        "schema": SCHEMA,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "analysis_only": True,
        "proposed_contract_passed_by_both_frozen_runs": passed,
        "formal_v10_classification_preserved": V10_FORMAL_CLASSIFICATION,
        "formal_v13_classification_preserved": V13_FORMAL_CLASSIFICATION,
        "formal_v13_pass_granted": False,
        "formal_requalification_performed": False,
        "qualification_complete": False,
        "g1_complete": False,
        "thresholds": plan["thresholds"],
        "raw_sample_contract": plan["raw_sample_contract"],
        "replicated_crossover_contract": plan["replicated_crossover_contract"],
        "runs": runs,
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
        _require(json.loads(path.read_text(encoding="utf-8")) == payload, "JSON round-trip differs")
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v10-profile-root", type=Path, required=True)
    parser.add_argument("--v10-progress-root", type=Path, required=True)
    parser.add_argument("--v13-profile-root", type=Path, required=True)
    parser.add_argument("--v13-progress-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(
        args.output,
        analyze(
            v10_profile_root=args.v10_profile_root,
            v10_progress_root=args.v10_progress_root,
            v13_profile_root=args.v13_profile_root,
            v13_progress_root=args.v13_progress_root,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
