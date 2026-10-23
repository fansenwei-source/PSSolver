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


SCHEMA = "pssolver.v0_3.s09.single_h100_plan.v1"
CLASSIFICATION = "S09_PLAN_READY_DECISIONS_AND_IMPLEMENTATION_PENDING"

BASELINE_COMMIT = "2a1d0f7e9e4b1ecf8fc26ef8eda5c2efb1290e70"
BASELINE_WHEEL_SHA256 = (
    "a7e94585e62785c69522d6cc194fd479b5af55577444926d1bb661ed073cc00e"
)
PLANNING_PARENT = "7b796679925020a869fe59cfb56e0a771ef6d47e"
CONSUMER_BASE = "208edd03e6b26a0e788c07d295fddd7a67237189"

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
                    entries.append(
                        {
                            "sequence": sequence,
                            "key": f"{role}_{family['key']}_{grid}_trial{trial}",
                            "family": family["key"],
                            "grid": grid,
                            "trial": trial,
                            "order_position": position,
                            "provider_role": role,
                            "device": "cuda",
                            "dtype": family["dtype"],
                            "warmup_steps": 10,
                            "timed_steps": 60,
                            "samples_per_profile": 60,
                            "fixed_block_size": 10,
                            "fixed_block_count": 6,
                            "steady_forward_transforms_per_step": family["forward"],
                            "steady_inverse_transforms_per_step": family["inverse"],
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
        "selected_answers": {},
        "provider": {
            "baseline_commit": BASELINE_COMMIT,
            "baseline_version": "0.2.2",
            "baseline_wheel_sha256": BASELINE_WHEEL_SHA256,
            "candidate_planning_parent": PLANNING_PARENT,
            "candidate_implementation_commit": None,
            "candidate_wheel_sha256": None,
            "installed_wheels_only": True,
        },
        "consumer": {
            "stable_qualification_base_commit": CONSUMER_BASE,
            "migration_commit": None,
            "wheel_sha256": None,
            "functional_api_version": "1.0",
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
        "pre_h100_gates": [
            "confirmed answers to S09-Q01 through S09-Q04",
            "Provider backward-scope implementation and complete CPU suite",
            "Control minimal migration and installed-wheel CPU suite",
            "clean source archives and reproducible installed wheels",
            "qualification helper compile/import/synthetic negative gates",
            "all artifact and source hashes frozen before sbatch",
        ],
        "job_stage_order": [
            "node_local_tmp_environment_and_host_compiler",
            "installed_wheel_cuda_only_tests",
            "cuda_long_horizon_hermitian",
            "tf32_forward_backward_scope",
            "checkpoint_restart_and_released_fixture_matrix",
            "numerical_gradient_taylor_and_replay",
            "paired_performance_memory_profiles",
            "independent_control_consumer",
            "cumulative_analyzer_and_checksums",
        ],
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
        "numerical": {
            "standard_families": ["plane", "periodic", "channel"],
            "unchanged_paths_byte_identical_to_v0_2_2": True,
            "changed_float64_paths_relative_l2_max": 1.0e-12,
            "changed_float64_paths_linf_max": 1.0e-12,
            "changed_float32_paths_atol_rtol": "8 * finfo(dtype).eps",
            "checkpoint_restart_exact": True,
            "gradient_relative_l2_max": 2.0e-5,
            "finite_difference_relative_error_max": 2.0e-5,
            "threshold_relaxation_allowed": False,
        },
        "profile_matrix": profiles,
        "profile_contract": {
            "runtime_families": families,
            "grid_count": 16,
            "trials_per_cell": 6,
            "provider_roles": 2,
            "profile_count": len(profiles),
            "samples_per_profile": 60,
            "profile_center": "median_of_six_fixed_block_means",
            "all_samples_retained": True,
            "outlier_removal_allowed": False,
            "trial_replacement_allowed": False,
        },
        "thresholds": {
            "aggregate_and_median_runtime_ratio_max": 1.03,
            "order_stratum_and_individual_reference_ratio_max": 1.05,
            "minimum_trials_at_or_below_individual_reference": 4,
            "peak_allocated_group_ratio_max": 1.03,
            "peak_reserved_group_ratio_max": 1.03,
            "transform_counts_exact": True,
            "graph_breaks": 0,
            "compile_fallback": False,
            "runtime_fallback": False,
            "finite_required": True,
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
