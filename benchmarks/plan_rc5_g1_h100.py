#!/usr/bin/env python3
"""Build the frozen rc5 G-1 installed-wheel single-H100 plan."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable

from benchmarks.plan_rc458_tf32_h100 import (
    CALLER_STATES,
    CUDA_ONLY_NODE_IDS,
    FAMILIES,
    GRID_CONTRACTS,
)


SCHEMA = "pssolver.rc5.g1.single_h100_plan.v2"
PROVIDER_BASELINE = "658869456d762c917ca4409b091539c3733df2fb"
PROVIDER_CANDIDATE = "dabeb9a487a82700ddd364c4233941a175808a3a"
PROVIDER_BASELINE_WHEEL_SHA256 = (
    "1bf31f324180dac58777a23785fb4e1424d3cc589099cf8c6328e2f6c74d8cc5"
)
PROVIDER_CANDIDATE_WHEEL_SHA256 = (
    "954072040dc1681529c29e09d80f2fe9112d73442b98932978e1739c90212b22"
)
CONSUMER_SOURCE = "6d3558b1367da9a3e7d6b3f8fbc5584cf384bfe3"
CONSUMER_WHEEL_SHA256 = (
    "4e112df06770a572d3395c71c3e6073b8e4ea458229857243ed91845f2f0bf75"
)
REPAIRED_PERIODIC_FAMILIES = (
    "periodic_complete_stress",
    "periodic_functional",
)
SCHEDULED_REFRESH_FAMILIES = frozenset(
    {
        "channel_legacy",
        "channel_compiled_v2",
    }
)
SCHEDULED_REFRESH_INTERVAL = 20
SCHEDULED_REFRESH_FORWARD_CALLS = 1


def _scheduled_refreshes(*, family: str, warmup_steps: int, timed_steps: int) -> int:
    """Count refresh events that fall inside the timed profile window."""
    if family not in SCHEDULED_REFRESH_FAMILIES:
        return 0
    if warmup_steps < 0 or timed_steps <= 0:
        raise ValueError("profile step counts must define a non-empty timed window")
    return (warmup_steps + timed_steps) // SCHEDULED_REFRESH_INTERVAL - (
        warmup_steps // SCHEDULED_REFRESH_INTERVAL
    )


def build_profile_matrix() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    sequence = 0
    for family in FAMILIES:
        for grid in family.grids:
            for trial in range(1, 4):
                for role, commit, wheel in (
                    (
                        "baseline",
                        PROVIDER_BASELINE,
                        PROVIDER_BASELINE_WHEEL_SHA256,
                    ),
                    (
                        "candidate",
                        PROVIDER_CANDIDATE,
                        PROVIDER_CANDIDATE_WHEEL_SHA256,
                    ),
                ):
                    sequence += 1
                    warmup_steps = 10
                    timed_steps = 60
                    scheduled_refreshes = _scheduled_refreshes(
                        family=family.key,
                        warmup_steps=warmup_steps,
                        timed_steps=timed_steps,
                    )
                    scheduled_forward_calls = (
                        scheduled_refreshes * SCHEDULED_REFRESH_FORWARD_CALLS
                    )
                    entries.append(
                        {
                            "sequence": sequence,
                            "key": f"{role}_{family.key}_{grid}_trial{trial}",
                            "provider_role": role,
                            "provider_commit": commit,
                            "provider_wheel_sha256": wheel,
                            "family": family.key,
                            "application": family.application,
                            "runtime_path": family.runtime_path,
                            "grid": grid,
                            "trial": trial,
                            "warmup_steps": warmup_steps,
                            "timed_steps": timed_steps,
                            "device": "cuda",
                            "dtype": family.dtype,
                            "tf32": "off",
                            "forward_transforms_per_step": (
                                family.forward_transforms_per_step
                            ),
                            "inverse_transforms_per_step": (
                                family.inverse_transforms_per_step
                            ),
                            "scheduled_refreshes_in_timed_window": (
                                scheduled_refreshes
                            ),
                            "scheduled_forward_calls_in_timed_window": (
                                scheduled_forward_calls
                            ),
                            "scheduled_inverse_calls_in_timed_window": 0,
                            "expected_forward_raw_calls": (
                                family.forward_transforms_per_step * timed_steps
                                + scheduled_forward_calls
                            ),
                            "expected_inverse_raw_calls": (
                                family.inverse_transforms_per_step * timed_steps
                            ),
                        }
                    )
    return entries


def build_plan() -> dict[str, object]:
    profiles = build_profile_matrix()
    families = [family.key for family in FAMILIES]
    unchanged = [
        family for family in families if family not in REPAIRED_PERIODIC_FAMILIES
    ]
    return {
        "schema": SCHEMA,
        "provider": {
            "baseline_commit": PROVIDER_BASELINE,
            "baseline_tag": "v0.2.0rc4",
            "baseline_wheel_sha256": PROVIDER_BASELINE_WHEEL_SHA256,
            "candidate_commit": PROVIDER_CANDIDATE,
            "candidate_version": "0.2.0rc5",
            "candidate_wheel_sha256": PROVIDER_CANDIDATE_WHEEL_SHA256,
        },
        "consumer": {
            "qualified_source_commit": CONSUMER_SOURCE,
            "wheel_sha256": CONSUMER_WHEEL_SHA256,
            "functional_api_version": "1.0",
            "periodic_adapter_version": 2,
            "channel_adapter_version": 3,
        },
        "environment": {
            "gpu_name_contains": "H100",
            "visible_cuda_device_count": 1,
            "allocated_device": "cuda:0",
            "device_argument": "cuda",
            "tf32_policy": "off",
            "source_shadow_import": False,
            "installed_wheels_only": True,
            "isolated_compiler_caches_per_profile": True,
        },
        "cuda_only_tests": list(CUDA_ONLY_NODE_IDS),
        "tf32_scope_gate": {
            "caller_states": list(CALLER_STATES),
            "runtime_families": families,
            "case_count": len(CALLER_STATES) * len(FAMILIES),
            "success_restores_exact_state": True,
            "exception_restores_exact_state": True,
            "nested_scope_restores_lifo": True,
            "invalid_policy_does_not_mutate": True,
        },
        "checkpoint_gate": {
            "current_families": [
                "periodic_functional",
                "channel_functional_fixed_iteration",
                "channel_functional_tolerance",
                "legacy_channel",
            ],
            "directions": ["cpu_to_cuda", "cuda_to_cpu"],
            "released_primary_case_count": 13,
            "transactional_restore": True,
            "source_checkpoints_unchanged": True,
        },
        "grid_contracts": GRID_CONTRACTS,
        "profile_matrix": profiles,
        "profile_count": len(profiles),
        "transform_measurement_window": {
            "raw_counts_are_authoritative": True,
            "reported_per_step_is_raw_count_divided_by_timed_steps": True,
            "steady_state_counts_exclude_scheduled_refresh_calls": True,
            "scheduled_refresh_families": sorted(SCHEDULED_REFRESH_FAMILIES),
            "scheduled_refresh_interval": SCHEDULED_REFRESH_INTERVAL,
            "forward_calls_per_scheduled_refresh": (
                SCHEDULED_REFRESH_FORWARD_CALLS
            ),
            "inverse_calls_per_scheduled_refresh": 0,
            "source_profiles_are_immutable": True,
        },
        "thresholds": {
            "completed_steps": 60,
            "finite": True,
            "runtime_fallback": False,
            "compile_fallback": False,
            "graph_breaks": 0,
            "paired_initial_sha_equal": True,
            "mean_timestep_ratio_max": 1.03,
            "median_timestep_ratio_max": 1.03,
            "paired_timestep_ratio_max": 1.05,
            "peak_allocated_group_ratio_max": 1.03,
            "peak_reserved_group_ratio_max": 1.03,
            "transform_counts_exact": True,
            "physical_relative_l2_max": 1.0e-12,
            "physical_linf_max": 1.0e-12,
            "candidate_hermitian_violation_max": 0.0,
        },
        "numerical_comparison": {
            "byte_identity_families": unchanged,
            "repaired_periodic_families": list(REPAIRED_PERIODIC_FAMILIES),
            "repaired_periodic_components": ["Q", "velocity", "pressure"],
            "global_final_state_sha_equality": False,
        },
        "consumer_h100_gate": {
            "source_commit": CONSUMER_SOURCE,
            "expected_passed": 342,
            "expected_skipped": 1,
            "allowed_skip": "tests/test_warmup_experiments.py::test_r5_benchmark",
            "failed_allowed": 0,
            "xfailed_allowed": 0,
            "deselected_allowed": 0,
            "periodic_identity_sha256": (
                "d85f6540494696fddb4549cd4d431d362fe4a55bc53cb9cfd8158fae7c09530d"
            ),
            "channel_fixed_identity_sha256": (
                "b95aa0ce22035df610282f1066cce23b47819807f3d942138b0d5038b727a905"
            ),
            "channel_tolerance_identity_sha256": (
                "b18000feec39cdcdb8e8d371ad80b5037a10ba844c21db0dcddd7b162d324908"
            ),
        },
        "submission": {
            "formal_h100_submission_count": 1,
            "automatic_retry": False,
            "second_job_without_new_authorization": False,
            "fail_fast": True,
        },
    }


def _atomic_json(path: Path, value: object) -> None:
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
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
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
