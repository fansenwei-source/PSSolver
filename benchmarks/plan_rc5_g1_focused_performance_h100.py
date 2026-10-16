#!/usr/bin/env python3
"""Build the pre-registered rc5 G-1 focused performance/memory plan.

The plan remeasures only the four cells that failed the immutable v6 replay.
It does not relax a G-1 threshold and it does not itself authorize an H100 job.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

from benchmarks.plan_rc5_g1_h100 import (
    FAMILIES,
    GRID_CONTRACTS,
    PROVIDER_BASELINE,
    PROVIDER_BASELINE_WHEEL_SHA256,
    PROVIDER_CANDIDATE,
    PROVIDER_CANDIDATE_WHEEL_SHA256,
    _scheduled_refreshes,
)


SCHEMA = "pssolver.rc5.g1.focused_performance_plan.v1"
V5_CONTROL_DIRECTORY = (
    "/home/fansenwei/pssolver_rc5_g1_h100_dabeb9a_20261015_recovery_v5"
)
V5_MANIFEST_SHA256 = (
    "049f7422b62618416745ca665f82ab2370f6a5f2e662ce296f332f461e3b7f71"
)
V6_CONTROL_DIRECTORY = (
    "/home/fansenwei/pssolver_rc5_g1_transform_window_analysis_"
    "dabeb9a_20261016_v6"
)
V6_MANIFEST_SHA256 = (
    "74ef98002f8bd6b37f0d9e6cc001d17d9653cd6cb274afbd726273bd02eaeac2"
)
FOCUSED_CELLS = (
    {
        "family": "plane_compiled_v2",
        "grid": "plane_R128",
        "v6_failed_gate": "paired_timestep_ratio",
        "v6_observed": 1.0563119777324725,
    },
    {
        "family": "channel_compiled_v2",
        "grid": "channel_R128x20x20",
        "v6_failed_gate": "paired_timestep_ratio",
        "v6_observed": 1.0584790919925313,
    },
    {
        "family": "periodic_functional",
        "grid": "periodic_R128",
        "v6_failed_gate": "paired_timestep_ratio",
        "v6_observed": 1.066582662553552,
    },
    {
        "family": "channel_functional",
        "grid": "channel_R512x40x40",
        "v6_failed_gate": "peak_reserved_group_ratio",
        "v6_observed": 1.0402010050251256,
    },
)
TRIAL_COUNT = 6
WARMUP_STEPS = 10
TIMED_STEPS = 60
TIMING_CV_MAX = 0.05


def _family(key: str):
    matches = [family for family in FAMILIES if family.key == key]
    if len(matches) != 1:
        raise ValueError(f"unknown or duplicate runtime family: {key}")
    return matches[0]


def _role_order(trial: int) -> tuple[str, str]:
    if trial < 1 or trial > TRIAL_COUNT:
        raise ValueError("trial is outside the focused plan")
    return (
        ("baseline", "candidate")
        if trial % 2 == 1
        else ("candidate", "baseline")
    )


def build_profile_matrix() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    sequence = 0
    identities = {
        "baseline": (PROVIDER_BASELINE, PROVIDER_BASELINE_WHEEL_SHA256),
        "candidate": (PROVIDER_CANDIDATE, PROVIDER_CANDIDATE_WHEEL_SHA256),
    }
    for cell_index, cell in enumerate(FOCUSED_CELLS, start=1):
        family = _family(str(cell["family"]))
        grid = str(cell["grid"])
        if grid not in family.grids:
            raise ValueError(f"{family.key} does not support {grid}")
        for trial in range(1, TRIAL_COUNT + 1):
            for within_pair_order, role in enumerate(_role_order(trial), start=1):
                sequence += 1
                commit, wheel = identities[role]
                refreshes = _scheduled_refreshes(
                    family=family.key,
                    warmup_steps=WARMUP_STEPS,
                    timed_steps=TIMED_STEPS,
                )
                scheduled_forward = refreshes if family.key in {
                    "channel_legacy",
                    "channel_compiled_v2",
                } else 0
                entries.append(
                    {
                        "sequence": sequence,
                        "cell_sequence": cell_index,
                        "within_pair_order": within_pair_order,
                        "key": f"{role}_{family.key}_{grid}_trial{trial}",
                        "provider_role": role,
                        "provider_commit": commit,
                        "provider_wheel_sha256": wheel,
                        "family": family.key,
                        "application": family.application,
                        "runtime_path": family.runtime_path,
                        "grid": grid,
                        "trial": trial,
                        "warmup_steps": WARMUP_STEPS,
                        "timed_steps": TIMED_STEPS,
                        "device": "cuda",
                        "dtype": family.dtype,
                        "tf32": "off",
                        "forward_transforms_per_step": (
                            family.forward_transforms_per_step
                        ),
                        "inverse_transforms_per_step": (
                            family.inverse_transforms_per_step
                        ),
                        "scheduled_refreshes_in_timed_window": refreshes,
                        "scheduled_forward_calls_in_timed_window": (
                            scheduled_forward
                        ),
                        "scheduled_inverse_calls_in_timed_window": 0,
                        "expected_forward_raw_calls": (
                            family.forward_transforms_per_step * TIMED_STEPS
                            + scheduled_forward
                        ),
                        "expected_inverse_raw_calls": (
                            family.inverse_transforms_per_step * TIMED_STEPS
                        ),
                    }
                )
    return entries


def build_plan() -> dict[str, object]:
    profiles = build_profile_matrix()
    return {
        "schema": SCHEMA,
        "purpose": (
            "replace only the four failed v6 performance/memory cell decisions "
            "with fresh, order-balanced evidence"
        ),
        "provider": {
            "baseline_commit": PROVIDER_BASELINE,
            "baseline_wheel_sha256": PROVIDER_BASELINE_WHEEL_SHA256,
            "candidate_commit": PROVIDER_CANDIDATE,
            "candidate_wheel_sha256": PROVIDER_CANDIDATE_WHEEL_SHA256,
        },
        "immutable_inputs": {
            "v5_control_directory": V5_CONTROL_DIRECTORY,
            "v5_manifest_sha256": V5_MANIFEST_SHA256,
            "v6_control_directory": V6_CONTROL_DIRECTORY,
            "v6_manifest_sha256": V6_MANIFEST_SHA256,
            "v6_historical_classification_must_remain_failed": True,
            "source_profiles_must_remain_unchanged": True,
        },
        "focused_cells": list(FOCUSED_CELLS),
        "grid_contracts": {
            cell["grid"]: GRID_CONTRACTS[cell["grid"]]
            for cell in FOCUSED_CELLS
        },
        "profile_matrix": profiles,
        "profile_count": len(profiles),
        "execution_order": {
            "paired_trials_per_cell": TRIAL_COUNT,
            "odd_trial_roles": ["baseline", "candidate"],
            "even_trial_roles": ["candidate", "baseline"],
            "baseline_first_pair_count_per_cell": 3,
            "candidate_first_pair_count_per_cell": 3,
            "isolated_empty_compiler_caches_per_profile": True,
            "warmup_steps": WARMUP_STEPS,
            "timed_steps": TIMED_STEPS,
        },
        "timing_validity": {
            "raw_sample_count": TIMED_STEPS,
            "sample_standard_deviation": "sample",
            "coefficient_of_variation_max": TIMING_CV_MAX,
            "reported_summary_must_match_raw_samples": True,
            "invalid_profile_disposition": "fail_closed_stop",
            "outlier_removal_allowed": False,
            "trial_replacement_allowed": False,
        },
        "thresholds": {
            "mean_timestep_ratio_max": 1.03,
            "median_of_profile_means_ratio_max": 1.03,
            "every_paired_mean_timestep_ratio_max": 1.05,
            "peak_allocated_group_ratio_max": 1.03,
            "peak_reserved_group_ratio_max": 1.03,
            "completed_steps": TIMED_STEPS,
            "finite": True,
            "runtime_fallback": False,
            "compile_fallback": False,
            "graph_breaks": 0,
            "paired_initial_sha_equal": True,
            "transform_counts_exact": True,
        },
        "adjudication": {
            "all_four_cells_must_pass": True,
            "no_threshold_is_relaxed": True,
            "new_evidence_replaces_only_failed_v6_cell_decisions": True,
            "focused_pass_is_not_g1_complete": True,
            "cumulative_reassembly_required_after_focused_pass": True,
        },
        "authorization": {
            "push_authorized": False,
            "h100_submission_authorized": False,
            "automatic_retry_authorized": False,
            "production_default_changed": False,
        },
    }


def _atomic_json(path: Path, payload: object) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)
    _atomic_json(args.output, build_plan())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
