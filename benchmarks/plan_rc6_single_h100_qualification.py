#!/usr/bin/env python3
"""Build the frozen RC6 installed-wheel single-H100 qualification contract.

This module plans one fail-fast job.  It does not authorize a submission and
does not create an output directory unless the caller explicitly requests an
output file.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable


SCHEMA = "pssolver.rc6.single_h100_qualification_plan.v1"
CLASSIFICATION = "READY_RC6_SINGLE_H100_CUMULATIVE_QUALIFICATION_SUPPORT"

PROVIDER_BASELINE = "674a8121779be72c95aa5dc9e2babca0ce7b93a9"
PROVIDER_BASELINE_WHEEL_SHA256 = (
    "3097b08b08ba7b2d488fd8af348ec48986d69007a640bd806a225ec9963bb65f"
)
PROVIDER_CANDIDATE = "ea801ec08f5b2465661334e1c126133e55b68e96"
PROVIDER_CANDIDATE_WHEEL_SHA256 = (
    "8840ebfaa44a20a828928256c319edfa2aa4a9b8adf4f1bac014c33964ba0dd6"
)
CONSUMER_SOURCE = "208edd03e6b26a0e788c07d295fddd7a67237189"
CONSUMER_WHEEL_SHA256 = (
    "eca0c2c1e9b39de1f653eb1d0c1dc69a6f49c72736da758eea814205f115be0f"
)

CUDA_ONLY_NODE_IDS = (
    "tests/test_benchmark_bounded_axis_applicability.py::"
    "test_cuda_smoke_records_actual_device_and_memory",
    "tests/test_bounded_axis_execution_plan.py::"
    "test_cuda_plan_key_uses_the_allocated_device_identity",
    "tests/test_phase4_combined_sbdf2_modal_block.py::"
    "test_cuda_canary_binds_concrete_device_and_remains_finite",
    "tests/test_phase8_p845_h100_qualification.py::"
    "test_unindexed_cuda_request_binds_lifting_to_allocated_device_identity",
    "tests/test_phase8_p856_finite_q_h100_qualification.py::"
    "test_cuda_runtime_matches_cpu_and_binds_allocated_device",
    "tests/test_phase8_p856_finite_q_h100_qualification.py::"
    "test_cuda_continuous_split_and_file_restart_are_exact",
    "tests/test_rc4_plane_nyquist_cuda.py::"
    "test_plane_nyquist_repair_binds_allocated_cuda_device_and_storage_equivalence",
)

CALLER_STATES = (
    {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    },
    {
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": True,
    },
    {
        "float32_matmul_precision": "high",
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": False,
    },
    {
        "float32_matmul_precision": "high",
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": True,
    },
    {
        "float32_matmul_precision": "medium",
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": False,
    },
    {
        "float32_matmul_precision": "medium",
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": True,
    },
)

GRIDS = {
    "plane_R128": {"shape": [128, 128, 32], "lengths": [40.0, 40.0, 10.0]},
    "plane_R320": {"shape": [320, 320, 80], "lengths": [100.0, 100.0, 20.0]},
    "periodic_R128": {"shape": [128, 128, 32], "lengths": [40.0, 40.0, 10.0]},
    "periodic_R320": {"shape": [320, 320, 80], "lengths": [100.0, 100.0, 20.0]},
    "channel_R128x20x20": {"shape": [128, 20, 20], "lengths": [40.0, 6.0, 6.0]},
    "channel_R512x40x40": {"shape": [512, 40, 40], "lengths": [160.0, 12.0, 12.0]},
}

FAMILIES = (
    {
        "key": "plane_legacy_production",
        "grids": ["plane_R128", "plane_R320"],
        "dtype": "float64",
        "forward": 7,
        "inverse": 32,
    },
    {
        "key": "plane_compiled_v2",
        "grids": ["plane_R128", "plane_R320"],
        "dtype": "float64",
        "forward": 7,
        "inverse": 32,
    },
    {
        "key": "periodic_complete_stress",
        "grids": ["periodic_R128", "periodic_R320"],
        "dtype": "float64",
        "forward": 7,
        "inverse": 46,
    },
    {
        "key": "channel_legacy",
        "grids": ["channel_R128x20x20", "channel_R512x40x40"],
        "dtype": "float32",
        "forward": 11,
        "inverse": 36,
    },
    {
        "key": "channel_compiled_v2",
        "grids": ["channel_R128x20x20", "channel_R512x40x40"],
        "dtype": "float32",
        "forward": 11,
        "inverse": 36,
    },
    {
        "key": "channel_complete_stress",
        "grids": ["channel_R128x20x20", "channel_R512x40x40"],
        "dtype": "float64",
        "forward": 15,
        "inverse": 42,
    },
    {
        "key": "periodic_functional",
        "grids": ["periodic_R128", "periodic_R320"],
        "dtype": "float64",
        "forward": 7,
        "inverse": 46,
    },
    {
        "key": "channel_functional",
        "grids": ["channel_R128x20x20", "channel_R512x40x40"],
        "dtype": "float64",
        "forward": 15,
        "inverse": 57,
    },
)

SCHEDULED_REFRESH_FAMILIES = frozenset(("channel_legacy", "channel_compiled_v2"))
SCHEDULED_REFRESH_INTERVAL = 20


def _profile_order(trial: int) -> tuple[tuple[str, str, str], ...]:
    baseline = ("baseline", PROVIDER_BASELINE, PROVIDER_BASELINE_WHEEL_SHA256)
    candidate = ("candidate", PROVIDER_CANDIDATE, PROVIDER_CANDIDATE_WHEEL_SHA256)
    return (baseline, candidate) if trial % 2 else (candidate, baseline)


def build_profile_matrix() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    sequence = 0
    for family in FAMILIES:
        for grid in family["grids"]:
            for trial in range(1, 7):
                for order_position, (role, commit, wheel) in enumerate(
                    _profile_order(trial), start=1
                ):
                    sequence += 1
                    warmup_steps = 10
                    timed_steps = 60
                    refreshes = 0
                    if family["key"] in SCHEDULED_REFRESH_FAMILIES:
                        refreshes = (
                            (warmup_steps + timed_steps) // SCHEDULED_REFRESH_INTERVAL
                            - warmup_steps // SCHEDULED_REFRESH_INTERVAL
                        )
                    entries.append(
                        {
                            "sequence": sequence,
                            "key": (
                                f"{role}_{family['key']}_{grid}_trial{trial}"
                            ),
                            "family": family["key"],
                            "grid": grid,
                            "trial": trial,
                            "order_position": order_position,
                            "provider_role": role,
                            "provider_commit": commit,
                            "provider_wheel_sha256": wheel,
                            "device": "cuda",
                            "dtype": family["dtype"],
                            "warmup_steps": warmup_steps,
                            "timed_steps": timed_steps,
                            "samples_per_profile": 60,
                            "fixed_block_size": 10,
                            "fixed_block_count": 6,
                            "steady_forward_transforms_per_step": family["forward"],
                            "steady_inverse_transforms_per_step": family["inverse"],
                            "scheduled_refreshes_in_timed_window": refreshes,
                            "expected_forward_raw_calls": (
                                family["forward"] * timed_steps + refreshes
                            ),
                            "expected_inverse_raw_calls": family["inverse"] * timed_steps,
                            "isolated_empty_triton_cache": True,
                            "isolated_empty_torchinductor_cache": True,
                        }
                    )
    return entries


def build_plan() -> dict[str, object]:
    profiles = build_profile_matrix()
    family_names = [str(family["key"]) for family in FAMILIES]
    return {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "provider": {
            "baseline_commit": PROVIDER_BASELINE,
            "baseline_version": "0.2.0rc5",
            "baseline_wheel_sha256": PROVIDER_BASELINE_WHEEL_SHA256,
            "candidate_commit": PROVIDER_CANDIDATE,
            "candidate_version": "0.2.0rc6",
            "candidate_wheel_sha256": PROVIDER_CANDIDATE_WHEEL_SHA256,
        },
        "consumer": {
            "qualified_source_commit": CONSUMER_SOURCE,
            "wheel_sha256": CONSUMER_WHEEL_SHA256,
            "functional_api_version": "1.0",
            "provider_identity_migration_required": False,
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
        "scheduler": {
            "partition": "hagan-gpu",
            "account": "hagan-lab",
            "qos": "medium",
            "gres": "gpu:H100:1",
            "cpus_per_task": 8,
            "memory": "64G",
            "time_limit": "02:00:00",
            "requeue": False,
        },
        "environment": {
            "python": "3.10.18",
            "torch": "2.5.1+cu121",
            "torch_cuda_runtime": "12.1",
            "numpy": "1.22.3",
            "scipy": "1.8.1",
            "gpu_name_contains": "H100",
            "visible_cuda_device_count": 1,
            "allocated_device": "cuda:0",
            "device_argument": "cuda",
            "installed_wheels_only": True,
            "source_shadow_import": False,
            "pip_check": "PASS",
        },
        "preflight": {
            "node_local_tmp": True,
            "rename_noreplace_required": True,
            "pytest_tmpdir_is_unique_node_local_tmp": True,
            "decomposed_observations_before_enforcement": True,
            "tf32_before_and_after_each_stage_persisted": True,
            "host_compiler_stdatomic_and_cuda_compile_smoke": True,
            "empty_triton_cache_compile_smoke": True,
            "empty_torchinductor_cache_compile_smoke": True,
        },
        "stage_order": [
            "node_local_tmp_and_environment",
            "provider_cuda_only",
            "rc6_device_applicable_contract_smokes",
            "tf32_scope",
            "current_checkpoint_cross_device",
            "released_checkpoint_matrix",
            "eight_family_numerical",
            "balanced_crossover_profiles",
            "consumer_installed_wheel_h100",
            "cumulative_analyzer",
        ],
        "cuda_only_tests": list(CUDA_ONLY_NODE_IDS),
        "rc6_contract_smokes": {
            "plane_compiled_nonfinite_public_error_on_cuda": True,
            "pressure_floor_fixed_and_tolerance_modes_on_cuda": True,
            "complete_stress_compile_preflight_before_allocation": True,
            "channel_public_failure_hierarchy_and_metadata": True,
            "checkpoint_early_stop_not_translated": True,
            "rejected_restart_does_not_materialize_output": True,
        },
        "tf32_scope": {
            "caller_states": list(CALLER_STATES),
            "runtime_families": family_names,
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
            "continuous_split_restart_exact": True,
            "transactional_restore": True,
            "rejected_restart_output_remains_absent": True,
            "source_checkpoints_unchanged": True,
            "released_rc1_through_rc5_primary_case_count": 13,
            "released_dispositions_must_match_cpu_record": True,
        },
        "numerical_gate": {
            "runtime_families": family_names,
            "family_count": len(family_names),
            "completed_steps": 2,
            "finite": True,
            "fallback_used": False,
            "unchanged_paths_byte_identical": True,
            "physical_relative_l2_max": 1.0e-12,
            "physical_linf_max": 1.0e-12,
            "candidate_hermitian_violation_max": 0.0,
        },
        "grid_contracts": GRIDS,
        "profile_matrix": profiles,
        "profile_contract": {
            "cell_count": len(FAMILIES) * 2,
            "trials_per_cell": 6,
            "profiles_per_cell": 12,
            "profile_count": len(profiles),
            "samples_per_profile": 60,
            "fixed_block_size": 10,
            "fixed_block_count": 6,
            "profile_center": "median_of_six_fixed_block_means",
            "trial_ratio": "candidate_profile_center_over_baseline_profile_center",
            "baseline_then_candidate_trials": [1, 3, 5],
            "candidate_then_baseline_trials": [2, 4, 6],
            "all_samples_retained": True,
            "outlier_removal_allowed": False,
            "sample_deletion_allowed": False,
            "trial_replacement_allowed": False,
        },
        "thresholds": {
            "aggregate_all_sample_mean_ratio_max": 1.03,
            "median_of_profile_means_ratio_max": 1.03,
            "median_of_six_trial_ratios_max": 1.03,
            "baseline_then_candidate_stratum_median_ratio_max": 1.05,
            "candidate_then_baseline_stratum_median_ratio_max": 1.05,
            "order_stratum_imbalance_ratio_max": 1.05,
            "individual_trial_ratio_reference_max": 1.05,
            "minimum_trials_at_or_below_individual_reference": 4,
            "peak_allocated_group_ratio_max": 1.03,
            "peak_reserved_group_ratio_max": 1.03,
            "finite_required": True,
            "runtime_fallback": False,
            "compile_fallback": False,
            "graph_breaks": 0,
            "paired_initial_sha_equal": True,
            "transform_counts_exact": True,
        },
        "consumer_h100_gate": {
            "expected_passed": 344,
            "expected_skipped": 1,
            "allowed_skip": "tests/test_warmup_experiments.py::test_r5_benchmark",
            "failed_allowed": 0,
            "xfailed_allowed": 0,
            "deselected_allowed": 0,
            "guard_order_regression_required": True,
            "periodic_and_channel_cuda_execution_finite": True,
            "checkpoint_replay_and_gradient_gates_required": True,
            "scientific_control_result_claimed": False,
        },
        "success": {
            "classification": "PASS_RC6_SINGLE_H100_CUMULATIVE_QUALIFICATION",
            "qualification_complete": True,
            "eligible_for_release_integration_review": True,
            "independent_post_release_verification_required": True,
            "eligible_for_automatic_merge": False,
            "eligible_for_default_promotion": False,
            "production_default_changed": False,
        },
        "submission": {
            "formal_h100_submission_count": 1,
            "automatic_retry": False,
            "second_job_without_new_authorization": False,
            "fail_fast": True,
            "complete_written_only_after_all_gates_and_checksums": True,
        },
    }


def _atomic_json(path: Path, value: object) -> None:
    path = path.expanduser().resolve()
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
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
        if json.loads(path.read_text(encoding="utf-8")) != value:
            raise RuntimeError("plan JSON round-trip differs")
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
