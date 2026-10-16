#!/usr/bin/env python3
"""Diagnose the rc5 G-1 v13 paired-block failure without re-adjudicating it.

The formal v13 result remains a failure.  This analysis compares the formal
v13 result with the pre-existing v10 replay and the v11/v12 timing diagnostics
to determine whether the failed cell is persistent across evidence runs.  It
does not remove samples, replace trials, change a threshold, or grant a
qualification pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics


SCHEMA = "pssolver.rc5.g1.v13_failure_analysis.v1"
CLASSIFICATION = (
    "ANALYSIS_ONLY_V13_MAX_TRIAL_FAILURE_CELL_MIGRATION_SUPPORTED"
)
V10_CLASSIFICATION = "ANALYSIS_ONLY_V10_REPLAY_FAILS_PROPOSED_V13_CONTRACT"
V13_CLASSIFICATION = "FAIL_RC5_G1_PAIRED_BLOCK_PERFORMANCE_MEMORY_GATE"
V11_CLASSIFICATION = (
    "MIXED_PHASE_SHIFTED_PERIODIC_AND_POSITION_DRIFTING_TIMING_VARIABILITY"
)
V12_CLASSIFICATION = "NO_SINGLE_GC_REFRESH_OR_PRESSURE_TERMINATION_CAUSE_SUPPORTED"
MAX_TRIAL_GATE = "every_trial_median_fixed_block_ratio"
PERSISTENT_GATES = (
    "aggregate_all_sample_mean",
    "median_of_profile_means",
    "median_of_all_fixed_block_ratios",
    "peak_allocated",
    "peak_reserved",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), f"{path} must contain a JSON object")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: object, name: str) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)),
        f"{name} must be numeric",
    )
    result = float(value)
    _require(math.isfinite(result), f"{name} must be finite")
    return result


def _cell(group: dict[str, object]) -> str:
    family = group.get("family")
    grid = group.get("grid")
    _require(isinstance(family, str) and family, "group family is invalid")
    _require(isinstance(grid, str) and grid, "group grid is invalid")
    return f"{family}/{grid}"


def _groups(payload: dict[str, object], name: str) -> dict[str, dict[str, object]]:
    raw = payload.get("groups")
    _require(isinstance(raw, list) and len(raw) == 4, f"{name} requires four groups")
    result: dict[str, dict[str, object]] = {}
    for value in raw:
        _require(isinstance(value, dict), f"{name} group must be an object")
        key = _cell(value)
        _require(key not in result, f"{name} contains a duplicate group: {key}")
        gates = value.get("gates")
        _require(isinstance(gates, dict), f"{name} {key} gates are missing")
        _require(
            set(gates) == set(PERSISTENT_GATES) | {MAX_TRIAL_GATE},
            f"{name} {key} gate set differs",
        )
        _require(
            all(isinstance(item, bool) for item in gates.values()),
            f"{name} {key} gate values must be booleans",
        )
        passed = value.get("passed")
        _require(isinstance(passed, bool), f"{name} {key} passed is invalid")
        _require(passed == all(gates.values()), f"{name} {key} passed differs")
        result[key] = value
    return result


def _failed_cells(groups: dict[str, dict[str, object]]) -> list[str]:
    return sorted(key for key, value in groups.items() if value["passed"] is False)


def _pair_records(group: dict[str, object], name: str) -> list[dict[str, object]]:
    pairs = group.get("pairs")
    _require(isinstance(pairs, list) and len(pairs) == 6, f"{name} requires six pairs")
    result: list[dict[str, object]] = []
    for expected_trial, value in enumerate(pairs, start=1):
        _require(isinstance(value, dict), f"{name} pair must be an object")
        _require(value.get("trial") == expected_trial, f"{name} trial order differs")
        ratios = value.get("fixed_block_ratios")
        _require(
            isinstance(ratios, list) and len(ratios) == 6,
            f"{name} trial {expected_trial} requires six block ratios",
        )
        finite_ratios = [
            _number(item, f"{name} trial {expected_trial} block ratio")
            for item in ratios
        ]
        _require(
            all(item > 0.0 for item in finite_ratios),
            f"{name} trial {expected_trial} block ratio must be positive",
        )
        reported = _number(
            value.get("median_fixed_block_ratio"),
            f"{name} trial {expected_trial} median",
        )
        _require(
            math.isclose(
                reported,
                statistics.median(finite_ratios),
                rel_tol=1.0e-12,
                abs_tol=1.0e-15,
            ),
            f"{name} trial {expected_trial} median differs",
        )
        result.append(value)
    return result


def _failed_only_at_max_trial(group: dict[str, object], name: str) -> bool:
    gates = group["gates"]
    _require(isinstance(gates, dict), f"{name} gates are missing")
    return gates[MAX_TRIAL_GATE] is False and all(
        gates[key] is True for key in PERSISTENT_GATES
    )


def _group_snapshot(group: dict[str, object]) -> dict[str, object]:
    metrics = group.get("metrics")
    gates = group.get("gates")
    _require(isinstance(metrics, dict), "group metrics are missing")
    _require(isinstance(gates, dict), "group gates are missing")
    return {
        "cell": _cell(group),
        "passed": group["passed"],
        "maximum_trial_median_fixed_block_ratio": _number(
            metrics.get("maximum_trial_median_fixed_block_ratio"),
            "maximum trial median fixed-block ratio",
        ),
        "maximum_trial_gate_passed": gates[MAX_TRIAL_GATE],
        "all_persistent_gates_passed": all(gates[key] for key in PERSISTENT_GATES),
    }


def adjudicate(
    v10: dict[str, object],
    v13: dict[str, object],
    v11: dict[str, object],
    v12: dict[str, object],
) -> dict[str, object]:
    """Compare frozen evidence and return a non-qualifying diagnosis."""

    _require(v10.get("classification") == V10_CLASSIFICATION, "v10 classification differs")
    _require(v10.get("analysis_only") is True, "v10 must be analysis-only")
    _require(v10.get("proposed_contract_passed") is False, "v10 must remain failed")
    _require(v13.get("classification") == V13_CLASSIFICATION, "v13 classification differs")
    _require(v13.get("analysis_only") is False, "v13 must be formal evidence")
    _require(v13.get("proposed_contract_passed") is False, "v13 must remain failed")
    _require(v13.get("formal_requalification_performed") is True, "v13 must be formal")
    _require(v11.get("classification") == V11_CLASSIFICATION, "v11 classification differs")
    _require(v11.get("analysis_only") is True, "v11 must be analysis-only")
    _require(v12.get("classification") == V12_CLASSIFICATION, "v12 classification differs")
    _require(v12.get("analysis_only") is True, "v12 must be analysis-only")

    v10_groups = _groups(v10, "v10")
    v13_groups = _groups(v13, "v13")
    _require(set(v10_groups) == set(v13_groups), "v10/v13 cell sets differ")
    v10_failed = _failed_cells(v10_groups)
    v13_failed = _failed_cells(v13_groups)
    _require(len(v10_failed) == 1, "v10 must contain exactly one failed cell")
    _require(len(v13_failed) == 1, "v13 must contain exactly one failed cell")
    _require(v10_failed != v13_failed, "failed cell did not migrate across runs")
    _require(
        _failed_only_at_max_trial(v10_groups[v10_failed[0]], "v10 failed group"),
        "v10 failure must be isolated to the maximum-trial gate",
    )
    _require(
        _failed_only_at_max_trial(v13_groups[v13_failed[0]], "v13 failed group"),
        "v13 failure must be isolated to the maximum-trial gate",
    )
    _require(
        all(
            group["gates"][key] is True
            for groups in (v10_groups, v13_groups)
            for group in groups.values()
            for key in PERSISTENT_GATES
        ),
        "a persistent aggregate or memory gate failed",
    )

    periodic_key = "periodic_functional/periodic_R128"
    _require(v13_failed == [periodic_key], "v13 failed cell is not Periodic R128")
    v13_periodic_pairs = _pair_records(v13_groups[periodic_key], "v13 Periodic")
    v10_periodic_pairs = _pair_records(v10_groups[periodic_key], "v10 Periodic")
    threshold = _number(
        v13.get("thresholds", {}).get("every_trial_median_fixed_block_ratio_max"),
        "v13 maximum-trial threshold",
    )
    trial_medians = [
        _number(pair["median_fixed_block_ratio"], "v13 Periodic trial median")
        for pair in v13_periodic_pairs
    ]
    failing_trials = [
        int(pair["trial"])
        for pair in v13_periodic_pairs
        if _number(pair["median_fixed_block_ratio"], "v13 Periodic trial median")
        > threshold
    ]
    _require(failing_trials == [3], "v13 Periodic failing-trial identity differs")
    trial3 = v13_periodic_pairs[2]
    trial5 = v13_periodic_pairs[4]
    trial3_ratios = [float(value) for value in trial3["fixed_block_ratios"]]
    trial5_ratios = [float(value) for value in trial5["fixed_block_ratios"]]
    _require(
        sum(value > threshold for value in trial3_ratios) == 4,
        "v13 Periodic trial 3 exceedance pattern differs",
    )
    _require(
        all(value < 0.95 for value in trial5_ratios),
        "v13 Periodic trial 5 reverse-sign pattern differs",
    )
    _require(
        trial5.get("baseline_cv_at_or_below_reference") is True
        and trial5.get("candidate_cv_at_or_below_reference") is True,
        "v13 Periodic trial 5 must be internally stable",
    )

    odd_medians = [trial_medians[index] for index in (0, 2, 4)]
    even_medians = [trial_medians[index] for index in (1, 3, 5)]
    v10_periodic_maximum = max(
        _number(pair["median_fixed_block_ratio"], "v10 Periodic trial median")
        for pair in v10_periodic_pairs
    )
    v10_failed_group = v10_groups[v10_failed[0]]
    v13_failed_group = v13_groups[v13_failed[0]]

    return {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "analysis_only": True,
        "formal_v13_classification_preserved": V13_CLASSIFICATION,
        "formal_v13_pass_granted": False,
        "qualification_complete": False,
        "g1_complete": False,
        "root_cause_resolved": False,
        "persistent_provider_regression_supported_by_current_evidence": False,
        "paired_block_contract_repeatability_concern_supported": True,
        "automatic_retry_supported": False,
        "thresholds_changed": False,
        "samples_deleted": False,
        "trials_replaced": False,
        "production_default_changed": False,
        "cross_run": {
            "same_cell_set": True,
            "v10_failed_cells": v10_failed,
            "v13_failed_cells": v13_failed,
            "failed_cell_migrated": True,
            "both_failures_isolated_to_maximum_trial_gate": True,
            "all_persistent_aggregate_and_memory_gates_passed": True,
            "v10_failed_group": _group_snapshot(v10_failed_group),
            "v13_failed_group": _group_snapshot(v13_failed_group),
            "v10_periodic_maximum_trial_median_fixed_block_ratio": v10_periodic_maximum,
            "v10_periodic_maximum_trial_gate_passed": (
                v10_groups[periodic_key]["gates"][MAX_TRIAL_GATE]
            ),
        },
        "v13_periodic": {
            "threshold": threshold,
            "failing_trials": failing_trials,
            "trial_median_fixed_block_ratios": trial_medians,
            "minimum_trial_median_fixed_block_ratio": min(trial_medians),
            "maximum_trial_median_fixed_block_ratio": max(trial_medians),
            "trial_median_span": max(trial_medians) - min(trial_medians),
            "baseline_then_candidate_trial_median": statistics.median(odd_medians),
            "candidate_then_baseline_trial_median": statistics.median(even_medians),
            "trial3_fixed_block_ratios": trial3_ratios,
            "trial3_blocks_above_threshold": sum(
                value > threshold for value in trial3_ratios
            ),
            "trial5_fixed_block_ratios": trial5_ratios,
            "trial5_both_profiles_cv_at_or_below_reference": True,
            "trial5_candidate_faster_in_every_block": True,
        },
        "prior_diagnostics": {
            "v11_classification": V11_CLASSIFICATION,
            "v12_classification": V12_CLASSIFICATION,
            "position_drifting_variability_observed": v12.get("adjudication", {}).get(
                "position_drifting_timing_variability_observed"
            ),
            "single_root_cause_resolved": v12.get("adjudication", {}).get(
                "single_root_cause_resolved"
            ),
        },
        "interpretation": {
            "supported": (
                "The maximum-per-trial failure is not repeatable in one cell: "
                "v10 and v13 fail different cells, while every aggregate, group-median, "
                "and memory gate passes in both runs."
            ),
            "not_supported": (
                "The evidence does not establish either a persistent candidate slowdown "
                "or a unique environmental root cause."
            ),
            "next_action": (
                "Review the experimental pairing and maximum-per-trial contract before "
                "authorizing another H100 qualification; do not retry v13 unchanged."
            ),
        },
    }


def analyze_files(
    v10_path: Path,
    v13_path: Path,
    v11_path: Path,
    v12_path: Path,
) -> dict[str, object]:
    result = adjudicate(
        _json(v10_path),
        _json(v13_path),
        _json(v11_path),
        _json(v12_path),
    )
    result["source_evidence"] = {
        name: {"path": str(path.resolve()), "sha256": _sha256(path)}
        for name, path in (
            ("v10_paired_block_replay", v10_path),
            ("v13_formal_analysis", v13_path),
            ("v11_timing_stability", v11_path),
            ("v12_attribution", v12_path),
        )
    }
    return result


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
        _require(
            json.loads(path.read_text(encoding="utf-8")) == json.loads(data),
            "JSON round-trip differs",
        )
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v10-analysis", type=Path, required=True)
    parser.add_argument("--v13-analysis", type=Path, required=True)
    parser.add_argument("--v11-analysis", type=Path, required=True)
    parser.add_argument("--v12-analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(
        args.output,
        analyze_files(
            args.v10_analysis,
            args.v13_analysis,
            args.v11_analysis,
            args.v12_analysis,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
