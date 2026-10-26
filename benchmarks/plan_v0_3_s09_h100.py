#!/usr/bin/env python3
"""Build the pre-authorization v0.3 S-09 single-H100 contract.

The plan deliberately remains non-executable until the public backward TF32
ownership decision is confirmed and its Provider/Consumer implementations
have passed installed-wheel CPU gates.  Building this plan never submits a
job and never changes a production default.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable


SCHEMA = "pssolver.v0_3.s09.single_h100_plan.v2"
CLASSIFICATION = "READY_S09_SINGLE_H100_PLAN_FOR_INDEPENDENT_REVIEW"

BASELINE_COMMIT = "2a1d0f7e9e4b1ecf8fc26ef8eda5c2efb1290e70"
BASELINE_WHEEL_SHA256 = (
    "a7e94585e62785c69522d6cc194fd479b5af55577444926d1bb661ed073cc00e"
)
PLANNING_PARENT = "7b796679925020a869fe59cfb56e0a771ef6d47e"
CONSUMER_BASE = "208edd03e6b26a0e788c07d295fddd7a67237189"
CANDIDATE_COMMIT = "85ba3aef75548d549d9db453ca4a5ab604b17a9d"
CANDIDATE_ARCHIVE_SHA256 = (
    "bff21905eb7d4c5e8313b10c871f40d8b96b411975260fdb0edb75b16a99a3b9"
)
CANDIDATE_WHEEL_SHA256 = (
    "4003645025da6d81317eeb7261ccf606aef7f60f20d072518f8baacf56cd4aa8"
)
CONSUMER_RUNTIME_COMMIT = "48dd4d3a9a101f8b24d8311c57ed67fab46ceea0"
CONSUMER_EVIDENCE_COMMIT = "97d3c06450c4a7a6d72e0fe07a7a157ef095c133"
CONSUMER_ARCHIVE_SHA256 = (
    "02f3e8249bc02cbb0484636d9f8bf46cc8a6815d90e13f8286f7aaf8291157fd"
)
CONSUMER_SUPPORT_ARCHIVE_SHA256 = (
    "2243f14cbdad1a58cb8a1ff73e2bb550752d97f5edf2190302005d9d69ff66a1"
)
CONSUMER_WHEEL_SHA256 = (
    "f7ae4c1f5f3c73e0be540b5e4ec5e05a9eef662867a149325462835820ae94ef"
)

H100_CONTROL_ROOT = (
    "/home/fansenwei/pssolver_v0_3_s09_h100_85ba3ae_48dd4d3_20261023_v1"
)
H100_OUTPUT_ROOT = (
    "/scratch1/vincent/PSSolver/"
    "data_pssolver_v0_3_s09_h100_85ba3ae_48dd4d3_20261023_v1"
)
H100_VENV_ROOT = "/scratch1/vincent/PSSolver/venvs"
H100_BASE_ENV = "/home/fansenwei/miniconda3/envs/Nematics3D"
H100_BASELINE_ENV = f"{H100_VENV_ROOT}/pssolver_v03_s09_85ba3ae_baseline"
H100_CANDIDATE_ENV = f"{H100_VENV_ROOT}/pssolver_v03_s09_85ba3ae_candidate"
H100_CONSUMER_ENV = f"{H100_VENV_ROOT}/pssolver_v03_s09_85ba3ae_consumer"

CUDA_ONLY_NODE_IDS = (
    "tests/test_benchmark_bounded_axis_applicability.py::test_cuda_smoke_records_actual_device_and_memory",
    "tests/test_bounded_axis_execution_plan.py::test_cuda_plan_key_uses_the_allocated_device_identity",
    "tests/test_phase4_combined_sbdf2_modal_block.py::test_cuda_canary_binds_concrete_device_and_remains_finite",
    "tests/test_phase8_p845_h100_qualification.py::test_unindexed_cuda_request_binds_lifting_to_allocated_device_identity",
    "tests/test_phase8_p856_finite_q_h100_qualification.py::test_cuda_runtime_matches_cpu_and_binds_allocated_device",
    "tests/test_phase8_p856_finite_q_h100_qualification.py::test_cuda_continuous_split_and_file_restart_are_exact",
    "tests/test_rc4_plane_nyquist_cuda.py::test_plane_nyquist_repair_binds_allocated_cuda_device_and_storage_equivalence",
)

SCHEDULED_REFRESH_FAMILIES = frozenset(("channel_legacy", "channel_compiled_v2"))
SCHEDULED_REFRESH_INTERVAL = 20

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
    "channel_R128x20x20": {
        "shape": [128, 20, 20],
        "lengths": [40.0, 6.0, 6.0],
    },
    "channel_R512x40x40": {
        "shape": [512, 40, 40],
        "lengths": [160.0, 12.0, 12.0],
    },
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


def _profile_order(trial: int) -> tuple[str, str]:
    return ("baseline", "candidate") if trial % 2 else ("candidate", "baseline")


def build_profile_matrix() -> list[dict[str, object]]:
    """Return the frozen RC6-style paired matrix without artifact hashes."""

    entries: list[dict[str, object]] = []
    sequence = 0
    for family in FAMILIES:
        for grid in family["grids"]:
            for trial in range(1, 7):
                for position, role in enumerate(_profile_order(trial), start=1):
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
                            "key": f"{role}_{family['key']}_{grid}_trial{trial}",
                            "family": family["key"],
                            "grid": grid,
                            "trial": trial,
                            "order_position": position,
                            "provider_role": role,
                            "provider_commit": (
                                BASELINE_COMMIT if role == "baseline" else CANDIDATE_COMMIT
                            ),
                            "provider_wheel_sha256": (
                                BASELINE_WHEEL_SHA256
                                if role == "baseline"
                                else CANDIDATE_WHEEL_SHA256
                            ),
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
    families = [str(value["key"]) for value in FAMILIES]
    return {
        "schema": SCHEMA,
        "classification": CLASSIFICATION,
        "plan_step": "S-09",
        "planning_parent": PLANNING_PARENT,
        "issues": ["A3-18", "NEW-14"],
        "diagnosis": {
            "new_14_is_a_confirmed_scope_gap": True,
            "periodic_backward_observed_caller_tf32_on": True,
            "channel_backward_observed_caller_tf32_on": True,
            "current_runtime_owns_forward_calls_only": True,
            "h100_must_not_be_submitted_before_resolution": True,
        },
        "decision_questions": [
            {
                "id": "S09-Q01",
                "question": "How should functional backward own the declared TF32 policy?",
                "recommended": "A",
                "options": {
                    "A": "Add a public runtime execution_scope() covering forward and backward; migrate Control to use it; qualify nested restoration and exceptions.",
                    "B": "Add Provider-owned value_and_grad APIs for every functional differentiation workflow.",
                    "C": "Only force TF32 off globally in the H100 harness; leave ordinary consumers outside the runtime policy contract.",
                },
            },
            {
                "id": "S09-Q02",
                "question": "Which Control source should be the stable v0.3 consumer qualification base?",
                "recommended": "A",
                "options": {
                    "A": f"Use last H100-qualified stable source {CONSUMER_BASE} and a minimal execution_scope/provider-identity migration.",
                    "B": "Use the current research/3d-warmup tip, including unrelated evolving research code.",
                    "C": "Omit the independent Control H100 gate.",
                },
            },
            {
                "id": "S09-Q03",
                "question": "How broad should the paired performance matrix be?",
                "recommended": "A",
                "options": {
                    "A": "Reuse the accepted RC6 eight-family, two-grid, six-trial alternating-order matrix and frozen 1.03/1.05 limits.",
                    "B": "Use only one representative Plane, Periodic, and Channel path.",
                    "C": "Treat performance and memory as informational.",
                },
            },
            {
                "id": "S09-Q04",
                "question": "Which CUDA long-horizon Hermitian cases close A3-18?",
                "recommended": "A",
                "options": {
                    "A": "Run loop3d half/full spectrum and R1 half spectrum, each through t=200.",
                    "B": "Run only loop3d half spectrum through t=200.",
                    "C": "Reuse CPU long-horizon evidence without a CUDA long run.",
                },
            },
        ],
        "selected_answers": {
            "S09-Q01": "A",
            "S09-Q02": "A",
            "S09-Q03": "A",
            "S09-Q04": "A",
        },
        "user_requirements": {
            "control_base_commit": CONSUMER_BASE,
            "control_dedicated_branch_and_worktree": True,
            "control_existing_checkout_must_remain_untouched": (
                "/home/fansenwei/Desktop/Develop/PSSolver-Control"
            ),
            "control_push_requires_separate_authorization": True,
            "control_gate_precedes_profile_matrix": True,
            "s06_cpu_timing_and_memory_preflight_required": True,
            "stop_if_cpu_preflight_predicts_threshold_failure": True,
            "threshold_relaxation_allowed": False,
        },
        "functional_api_impact": {
            "version_before": "1.0",
            "version_after": "1.1",
            "version_changed": True,
            "provider_identity_changed": True,
            "checkpoint_forward_compatibility_identity_changed": False,
            "api_1_0_current_format_checkpoint_reader_required": True,
            "outside_execution_scope_backward_uses_caller_tf32_state": True,
        },
        "provider": {
            "baseline_commit": BASELINE_COMMIT,
            "baseline_version": "0.2.2",
            "baseline_wheel_sha256": BASELINE_WHEEL_SHA256,
            "candidate_planning_parent": PLANNING_PARENT,
            "candidate_implementation_commit": CANDIDATE_COMMIT,
            "candidate_source_archive_sha256": CANDIDATE_ARCHIVE_SHA256,
            "candidate_wheel_sha256": CANDIDATE_WHEEL_SHA256,
            "installed_wheels_only": True,
        },
        "consumer": {
            "stable_qualification_base_commit": CONSUMER_BASE,
            "migration_commit": CONSUMER_RUNTIME_COMMIT,
            "evidence_commit": CONSUMER_EVIDENCE_COMMIT,
            "source_archive_sha256": CONSUMER_ARCHIVE_SHA256,
            "support_archive_sha256": CONSUMER_SUPPORT_ARCHIVE_SHA256,
            "wheel_sha256": CONSUMER_WHEEL_SHA256,
            "functional_api_version": "1.1",
            "installed_cpu": {
                "provider_gate_passed": 7,
                "consumer_passed": 351,
                "consumer_skipped": 6,
                "consumer_failed": 0,
                "source_shadow_import": False,
            },
            "scientific_control_result_claimed": False,
        },
        "scheduler": {
            "partition": "hagan-gpu",
            "account": "hagan-lab",
            "qos": "medium",
            "gres": "gpu:H100:1",
            "cpus_per_task": 8,
            "memory": "64G",
            "time_limit": "03:00:00",
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
            "source_shadow_import": False,
            "pip_check": "PASS",
        },
        "deployment": {
            "upload_root": H100_CONTROL_ROOT,
            "control_root": H100_CONTROL_ROOT,
            "output_root": H100_OUTPUT_ROOT,
            "output_root_must_initially_not_exist": True,
            "python_environment_source": H100_BASE_ENV,
            "python_environment_construction": (
                "create each scratch environment with the pinned base interpreter "
                "using -m venv --system-site-packages, then install the exact "
                "role-specific wheel set with --no-deps --no-cache-dir; import-origin, "
                "sys.prefix, purelib, RECORD and source-shadow gates enforce identity"
            ),
            "python_environment_commands": [
                f"test ! -e {H100_BASELINE_ENV}",
                f"{H100_BASE_ENV}/bin/python -m venv --system-site-packages {H100_BASELINE_ENV}",
                f"{H100_BASELINE_ENV}/bin/python -m pip install --no-deps --no-cache-dir "
                f"{H100_CONTROL_ROOT}/artifacts/pssolver-0.2.2-py3-none-any.whl",
                f"test ! -e {H100_CANDIDATE_ENV}",
                f"{H100_BASE_ENV}/bin/python -m venv --system-site-packages {H100_CANDIDATE_ENV}",
                f"{H100_CANDIDATE_ENV}/bin/python -m pip install --no-deps --no-cache-dir "
                f"{H100_CONTROL_ROOT}/artifacts/pssolver-0.3.0.dev0-py3-none-any.whl",
                f"test ! -e {H100_CONSUMER_ENV}",
                f"{H100_BASE_ENV}/bin/python -m venv --system-site-packages {H100_CONSUMER_ENV}",
                f"{H100_CONSUMER_ENV}/bin/python -m pip install --no-deps --no-cache-dir "
                f"{H100_CONTROL_ROOT}/artifacts/pssolver-0.3.0.dev0-py3-none-any.whl "
                f"{H100_CONTROL_ROOT}/artifacts/pssolver_control-0.0.1.dev0-py3-none-any.whl",
            ],
            "pre_submission_commands": [
                f"test ! -e {H100_OUTPUT_ROOT}",
                f"mkdir -p {H100_CONTROL_ROOT}/logs",
            ],
            "sbatch_command": (
                "sbatch --export=ALL,S09_BASELINE_PYTHON,S09_CANDIDATE_PYTHON,"
                "S09_CONSUMER_PYTHON,S09_BASELINE_WHEEL,S09_CANDIDATE_WHEEL,"
                "S09_CONSUMER_WHEEL,S09_PROVIDER_SUPPORT,S09_CONSUMER_SUPPORT,"
                "S09_PROVIDER_REFERENCE,S09_CONTROL_ROOT,S09_OUTPUT_ROOT "
                f"{H100_CONTROL_ROOT}/jobs/v0_3_s09_single_h100.sbatch"
            ),
            "created_directories": [
                {
                    "path": H100_CONTROL_ROOT,
                    "created": "before submission",
                    "disposition": "retain as the immutable control and provenance archive",
                },
                *[
                    {
                        "path": f"{H100_CONTROL_ROOT}/{name}",
                        "created": "before submission",
                        "disposition": "retain inside the control archive",
                    }
                    for name in (
                        "artifacts",
                        "provider_85ba3ae",
                        "provider_support",
                        "consumer_support",
                        "jobs",
                        "logs",
                    )
                ],
                *[
                    {
                        "path": path,
                        "created": "before submission",
                        "disposition": (
                            "delete only after evidence retrieval and checksum verification; "
                            "record each deletion in control-root PRUNED.txt"
                        ),
                    }
                    for path in (
                        H100_BASELINE_ENV,
                        H100_CANDIDATE_ENV,
                        H100_CONSUMER_ENV,
                    )
                ],
                {
                    "path": H100_OUTPUT_ROOT,
                    "created": "inside the allocated job after all initial-absence checks",
                    "disposition": "retain as the checksummed scratch evidence archive",
                },
                *[
                    {
                        "path": f"{H100_OUTPUT_ROOT}/{name}",
                        "created": "inside the allocated job",
                        "disposition": "retain as part of the checksummed scratch evidence",
                    }
                    for name in (
                        "results",
                        "logs",
                        "compiler_preflight",
                        "long_horizon_cases",
                        "profile_matrix",
                    )
                ],
                {
                    "path": "/tmp/pssolver_v03_s09_${SLURM_JOB_ID}",
                    "created": "inside the allocated job",
                    "disposition": "delete through an EXIT trap on success or failure",
                },
            ],
            "variables": {
                "S09_BASELINE_PYTHON": (
                    f"{H100_BASELINE_ENV}/bin/python"
                ),
                "S09_CANDIDATE_PYTHON": (
                    f"{H100_CANDIDATE_ENV}/bin/python"
                ),
                "S09_CONSUMER_PYTHON": (
                    f"{H100_CONSUMER_ENV}/bin/python"
                ),
                "S09_BASELINE_WHEEL": (
                    f"{H100_CONTROL_ROOT}/artifacts/pssolver-0.2.2-py3-none-any.whl"
                ),
                "S09_CANDIDATE_WHEEL": (
                    f"{H100_CONTROL_ROOT}/artifacts/pssolver-0.3.0.dev0-py3-none-any.whl"
                ),
                "S09_CONSUMER_WHEEL": (
                    f"{H100_CONTROL_ROOT}/artifacts/"
                    "pssolver_control-0.0.1.dev0-py3-none-any.whl"
                ),
                "S09_PROVIDER_SUPPORT": (
                    f"{H100_CONTROL_ROOT}/provider_support"
                ),
                "S09_CONSUMER_SUPPORT": (
                    f"{H100_CONTROL_ROOT}/consumer_support"
                ),
                "S09_PROVIDER_REFERENCE": (
                    f"{H100_CONTROL_ROOT}/provider_85ba3ae"
                ),
                "S09_CONTROL_ROOT": H100_CONTROL_ROOT,
                "S09_OUTPUT_ROOT": H100_OUTPUT_ROOT,
            },
        },
        "pre_h100_gates": [
            "confirmed answers to S09-Q01 through S09-Q04",
            "Provider backward-scope implementation and complete CPU suite",
            "Control minimal migration and installed-wheel CPU suite",
            "clean source archives and reproducible installed wheels",
            "qualification helper compile/import/synthetic negative gates",
            "all artifact and source hashes frozen before sbatch",
        ],
        "cuda_only_tests": list(CUDA_ONLY_NODE_IDS),
        "job_stage_order": [
            "node_local_tmp_environment_and_host_compiler",
            "installed_wheel_cuda_only_tests",
            "cuda_long_horizon_hermitian",
            "tf32_forward_backward_scope",
            "checkpoint_restart_and_released_fixture_matrix",
            "numerical_gradient_taylor_and_replay",
            "independent_control_consumer",
            "paired_performance_memory_profiles",
            "cumulative_analyzer_and_checksums",
        ],
        "s06_cpu_preflight": {
            "runtime_families": [
                "channel_legacy",
                "channel_compiled_v2",
                "channel_complete_stress",
                "channel_functional",
            ],
            "modes": ["tolerance", "fixed_work"],
            "measurements": [
                "single_step_wall_time",
                "peak_process_memory",
                "pressure_iteration_count",
                "termination_reason",
            ],
            "paired_baseline": BASELINE_COMMIT,
            "grid": "channel_R128x20x20",
            "shape": GRIDS["channel_R128x20x20"]["shape"],
            "lengths": GRIDS["channel_R128x20x20"]["lengths"],
            "paired_trials": 6,
            "warmup_steps": 10,
            "measured_steps_per_trial": 60,
            "fixed_block_size": 10,
            "fixed_block_count": 6,
            "center": "median_of_six_fixed_block_means",
            "provider_process_isolation": True,
            "alternating_pair_order": True,
            "runtime_ratio_warning": 1.03,
            "runtime_ratio_stop": 1.05,
            "memory_ratio_stop": 1.03,
            "threshold_relaxation_allowed": False,
            "all_samples_retained": True,
            "single_execution_only": True,
            "telemetry_classification": "NO_PAIRED_LIMIT_EXCEEDANCE",
            "focused_median_ratio": 1.0137810840953951,
            "frozen_function_tolerance_median_ratio": 1.0168360838379518,
            "maximum_paired_ratio": 1.0213293381953248,
            "candidate_optimization_required": False,
            "formal_performance_decision_remains_h100": True,
        },
        "long_horizon": {
            "cases": [
                {"case": "loop3d", "storage": "hermitian_half", "horizon": 200.0},
                {"case": "loop3d", "storage": "full_complex", "horizon": 200.0},
                {"case": "r1", "storage": "hermitian_half", "horizon": 200.0},
            ],
            "finite_required": True,
            "tolerance_factor": 64.0,
            "violation_must_not_exceed_bound": True,
            "fallback_allowed": False,
        },
        "tf32_backward": {
            "caller_states": list(CALLER_STATES),
            "functional_families": ["periodic_functional", "channel_functional"],
            "forward_and_backward_observe_declared_policy": True,
            "success_restores_exact_caller_state": True,
            "exception_restores_exact_caller_state": True,
            "nested_scope_restores_lifo": True,
            "control_consumer_uses_public_scope_only": True,
        },
        "consumer_h100_gate": {
            "expected_passed": 356,
            "expected_skipped": 1,
            "allowed_skip": "tests/test_warmup_experiments.py::test_r5_benchmark",
            "failed_allowed": 0,
            "xfailed_allowed": 0,
            "deselected_allowed": 0,
            "periodic_identity_sha256": "316df489836b7698f5afa9e828ac42348a2ca8c9527fb92c5ac75719b01fee71",
            "channel_fixed_identity_sha256": "f364a1407f731e7f98268a0aeee4d82119c2d4d7c4f11ae1db5b41007b45eb10",
            "channel_tolerance_identity_sha256": "1f8f27e875fa32357d1b312dd559f6dbfaadde63a6982c640f7b56370e3aa418",
            "scientific_control_result_claimed": False,
        },
        "numerical": {
            "standard_families": ["plane", "periodic", "channel"],
            "unchanged_runtime_families": [
                "plane_legacy_production",
                "plane_compiled_v2",
                "periodic_complete_stress",
                "periodic_functional",
            ],
            "changed_runtime_families": [
                "channel_legacy",
                "channel_compiled_v2",
                "channel_complete_stress",
                "channel_functional",
            ],
            "unchanged_paths_byte_identical_to_v0_2_2": True,
            "changed_float64_paths_relative_l2_max": 1.0e-12,
            "changed_float64_paths_linf_max": 1.0e-12,
            "changed_float32_paths_atol_rtol": "8 * finfo(dtype).eps",
            "checkpoint_restart_exact": True,
            "gradient_relative_l2_max": 2.0e-5,
            "finite_difference_relative_error_max": 2.0e-5,
            "threshold_relaxation_allowed": False,
        },
        "grid_contracts": GRIDS,
        "profile_matrix": profiles,
        "profile_contract": {
            "runtime_families": families,
            "grid_count": 16,
            "trials_per_cell": 6,
            "provider_roles": 2,
            "profile_count": len(profiles),
            "samples_per_profile": 60,
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
            "transform_counts_exact": True,
            "graph_breaks": 0,
            "compile_fallback": False,
            "runtime_fallback": False,
            "finite_required": True,
            "paired_initial_sha_equal": True,
        },
        "performance_adjudication": {
            "per_cell_minimum_trials_at_or_below_1_05": "at least 4 of 6",
            "baseline_then_candidate_order_stratum_median_ratio_max": 1.05,
            "candidate_then_baseline_order_stratum_median_ratio_max": 1.05,
            "between_order_strata_imbalance_ratio_max": 1.05,
            "aggregate_all_sample_mean_ratio_max": 1.03,
            "median_of_profile_means_ratio_max": 1.03,
            "median_of_six_trial_ratios_max": 1.03,
            "peak_allocated_group_ratio_max": 1.03,
            "peak_reserved_group_ratio_max": 1.03,
            "every_individual_trial_must_be_at_or_below_1_05": False,
            "all_samples_retained": True,
        },
        "submission": {
            "formal_h100_submission_count": 1,
            "automatic_retry": False,
            "fail_fast": True,
            "complete_only_after_all_gates_and_checksums": True,
        },
        "authorization": {
            "planning_only": True,
            "provider_runtime_change_authorized": False,
            "consumer_change_authorized": False,
            "push_authorized": False,
            "h100_submission_authorized": False,
            "automatic_retry": False,
            "tag_authorized": False,
            "release_authorized": False,
            "pypi_authorized": False,
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
            raise RuntimeError("S-09 plan JSON round-trip differs")
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
