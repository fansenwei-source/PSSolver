#!/usr/bin/env python3
"""Formally adjudicate new G-1 balanced-crossover H100 evidence."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics

if not __package__:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.analyze_rc5_g1_balanced_crossover_replay import adjudicate_cell
from benchmarks.analyze_rc5_g1_paired_block_replay import (
    _json,
    _timing,
    analyze as validate_paired_block_evidence,
)
from benchmarks.plan_rc5_g1_balanced_crossover_qualification import build_plan


SCHEMA = "pssolver.rc5.g1.balanced_crossover_qualification.v1"
PASS_CLASSIFICATION = "PASS_RC5_G1_BALANCED_CROSSOVER_PERFORMANCE_MEMORY_EVIDENCE"
FAIL_CLASSIFICATION = "FAIL_RC5_G1_BALANCED_CROSSOVER_PERFORMANCE_MEMORY_GATE"
PROGRESS_SCHEMA = "pssolver.rc5.g1.balanced_crossover_progress.v1"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _balanced_groups(
    profile_root: Path, plan: dict[str, object]
) -> list[dict[str, object]]:
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
    return groups


def analyze(profile_root: Path, progress_root: Path) -> dict[str, object]:
    """Validate all formal evidence and apply only the adopted v14 estimator."""

    superseded = validate_paired_block_evidence(
        profile_root,
        progress_root,
        evidence_mode="new_pre_registered_evidence",
        progress_schema=PROGRESS_SCHEMA,
    )
    _require(superseded["profile_count"] == 48, "formal profile count differs")
    _require(
        superseded["execution_sequence_validated"] is True,
        "formal execution sequence differs",
    )
    plan = build_plan()
    groups = _balanced_groups(profile_root, plan)
    _require(len(groups) == 4, "balanced focused-cell count differs")
    passed = all(group["passed"] for group in groups)
    return {
        "schema": SCHEMA,
        "classification": PASS_CLASSIFICATION if passed else FAIL_CLASSIFICATION,
        "analysis_only": False,
        "evidence_mode": "new_pre_registered_evidence",
        "formal_contract_schema": plan["schema"],
        "formal_contract_classification": plan["classification"],
        "formal_contract_adoption_authorized": True,
        "proposed_contract_passed": passed,
        "formal_requalification_performed": True,
        "profile_count": 48,
        "progress_record_count": 48,
        "execution_sequence_validated": True,
        "thresholds": plan["thresholds"],
        "raw_sample_contract": plan["raw_sample_contract"],
        "replicated_crossover_contract": plan["replicated_crossover_contract"],
        "groups": groups,
        "superseded_maximum_trial_estimator_diagnostic": {
            "classification": superseded["classification"],
            "passed": superseded["proposed_contract_passed"],
            "used_for_formal_adjudication": False,
            "reason": "the maximum-single-trial estimator was superseded by the adopted v14 contract",
        },
        "qualification_complete": False,
        "g1_complete": False,
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
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--progress-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    _atomic_json(args.output, analyze(args.profile_root, args.progress_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
