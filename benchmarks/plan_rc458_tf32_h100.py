"""Build the frozen RC4.5.8 single-H100 qualification matrix."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Iterable


SCHEMA = "pssolver.rc4_5_8.tf32_h100_plan.v1"
PROVIDER_BASELINE = "51f47e9f06bd3e019d785305bff47c80f5519b10"
PROVIDER_CANDIDATE = "a195bd05992bc6553c72c400587d2b07afe722d0"
CONSUMER_SOURCE = "b8539e0ec4c137a6c13746f6832e26e74be24515"


@dataclass(frozen=True)
class RuntimeFamily:
    key: str
    application: str
    runtime_path: str
    grids: tuple[str, str]
    forward_transforms_per_step: int
    inverse_transforms_per_step: int
    dtype: str = "float64"


FAMILIES = (
    RuntimeFamily(
        "plane_legacy_production",
        "plane",
        "legacy_production",
        ("plane_R128", "plane_R320"),
        7,
        32,
    ),
    RuntimeFamily(
        "plane_compiled_v2",
        "plane",
        "compiled_v2",
        ("plane_R128", "plane_R320"),
        7,
        32,
    ),
    RuntimeFamily(
        "periodic_complete_stress",
        "periodic",
        "periodic_spectral",
        ("periodic_R128", "periodic_R320"),
        7,
        46,
    ),
    RuntimeFamily(
        "channel_legacy",
        "channel",
        "legacy_channel",
        ("channel_R128x20x20", "channel_R512x40x40"),
        11,
        36,
        "float32",
    ),
    RuntimeFamily(
        "channel_compiled_v2",
        "channel",
        "compiled_channel_v2",
        ("channel_R128x20x20", "channel_R512x40x40"),
        11,
        36,
        "float32",
    ),
    RuntimeFamily(
        "channel_complete_stress",
        "channel",
        "channel_complete_stress",
        ("channel_R128x20x20", "channel_R512x40x40"),
        15,
        42,
    ),
    RuntimeFamily(
        "periodic_functional",
        "periodic_functional",
        "periodic_activity_batch_one",
        ("periodic_R128", "periodic_R320"),
        7,
        46,
    ),
    RuntimeFamily(
        "channel_functional",
        "channel_functional",
        "channel_activity_batch_one",
        ("channel_R128x20x20", "channel_R512x40x40"),
        15,
        57,
    ),
)


GRID_CONTRACTS = {
    "plane_R128": {"shape": [128, 128, 32], "lengths": [40.0, 40.0, 10.0]},
    "plane_R320": {"shape": [320, 320, 80], "lengths": [100.0, 100.0, 20.0]},
    "periodic_R128": {"shape": [128, 128, 32], "lengths": [40.0, 40.0, 10.0]},
    "periodic_R320": {"shape": [320, 320, 80], "lengths": [100.0, 100.0, 20.0]},
    "channel_R128x20x20": {"shape": [128, 20, 20], "lengths": [40.0, 6.0, 6.0]},
    "channel_R512x40x40": {"shape": [512, 40, 40], "lengths": [160.0, 12.0, 12.0]},
}


CALLER_STATES = (
    {
        "key": "caller_off",
        "float32_matmul_precision": "highest",
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
    },
    {
        "key": "caller_on",
        "float32_matmul_precision": "high",
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": True,
    },
    {
        "key": "caller_mixed",
        "float32_matmul_precision": "medium",
        "cuda_matmul_allow_tf32": True,
        "cudnn_allow_tf32": False,
    },
)


CUDA_ONLY_NODE_IDS = (
    "tests/test_bounded_axis_execution_plan.py::test_cuda_plan_key_uses_the_allocated_device_identity",
    "tests/test_benchmark_bounded_axis_applicability.py::test_cuda_smoke_records_actual_device_and_memory",
    "tests/test_phase4_combined_sbdf2_modal_block.py::test_cuda_canary_binds_concrete_device_and_remains_finite",
    "tests/test_phase8_p845_h100_qualification.py::test_unindexed_cuda_request_binds_lifting_to_allocated_device_identity",
    "tests/test_phase8_p856_finite_q_h100_qualification.py::test_cuda_runtime_matches_cpu_and_binds_allocated_device",
    "tests/test_phase8_p856_finite_q_h100_qualification.py::test_cuda_continuous_split_and_file_restart_are_exact",
    "tests/test_rc4_plane_nyquist_cuda.py::test_plane_nyquist_repair_binds_allocated_cuda_device_and_storage_equivalence",
)


def build_profile_matrix() -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    sequence = 0
    for family in FAMILIES:
        for grid in family.grids:
            for trial in range(1, 4):
                for provider_role, commit in (
                    ("baseline", PROVIDER_BASELINE),
                    ("candidate", PROVIDER_CANDIDATE),
                ):
                    sequence += 1
                    entries.append(
                        {
                            "sequence": sequence,
                            "key": (
                                f"{provider_role}_{family.key}_{grid}_trial{trial}"
                            ),
                            "provider_role": provider_role,
                            "provider_commit": commit,
                            "family": family.key,
                            "application": family.application,
                            "runtime_path": family.runtime_path,
                            "grid": grid,
                            "trial": trial,
                            "warmup_steps": 10,
                            "timed_steps": 60,
                            "device": "cuda",
                            "dtype": family.dtype,
                            "tf32": "off",
                            "forward_transforms_per_step": (
                                family.forward_transforms_per_step
                            ),
                            "inverse_transforms_per_step": (
                                family.inverse_transforms_per_step
                            ),
                        }
                    )
    return entries


def build_plan() -> dict[str, object]:
    profiles = build_profile_matrix()
    return {
        "schema": SCHEMA,
        "provider": {
            "baseline_commit": PROVIDER_BASELINE,
            "candidate_runtime_commit": PROVIDER_CANDIDATE,
        },
        "consumer": {"qualified_source_commit": CONSUMER_SOURCE},
        "environment": {
            "gpu": "NVIDIA H100 PCIe",
            "visible_cuda_device_count": 1,
            "device_argument": "cuda",
            "allocated_device": "cuda:0",
            "tf32_policy": "off",
            "source_shadow_import": False,
            "installed_wheels_only": True,
            "isolated_compiler_caches_per_profile": True,
        },
        "tf32_scope_gate": {
            "caller_states": list(CALLER_STATES),
            "runtime_families": [family.key for family in FAMILIES],
            "construction_must_not_mutate_caller_state": True,
            "successful_execution_must_restore_exact_caller_state": True,
            "exceptional_execution_must_restore_exact_caller_state": True,
            "nested_scope_must_restore_lifo": True,
            "invalid_policy_must_not_mutate_caller_state": True,
            "metadata_must_report_requested_and_effective_policy": True,
        },
        "cuda_only_tests": list(CUDA_ONLY_NODE_IDS),
        "grid_contracts": GRID_CONTRACTS,
        "profile_matrix": profiles,
        "profile_count": len(profiles),
        "thresholds": {
            "completed_steps": 60,
            "finite": True,
            "runtime_fallback": False,
            "compile_fallback": False,
            "graph_breaks": 0,
            "paired_initial_sha_equal": True,
            "paired_final_state_sha_equal": True,
            "mean_timestep_ratio_max": 1.03,
            "median_timestep_ratio_max": 1.03,
            "paired_timestep_ratio_max": 1.05,
            "peak_allocated_ratio_max": 1.03,
            "peak_reserved_ratio_max": 1.03,
            "transform_counts_exact": True,
        },
        "consumer_h100_gate": {
            "source_commit": CONSUMER_SOURCE,
            "installed_wheel": True,
            "full_suite_required": True,
            "allowed_skip": "tests/test_warmup_experiments.py::test_r5_benchmark",
            "r5_benchmark_authorized": False,
        },
        "submission": {
            "formal_h100_submission_count": 1,
            "automatic_retry": False,
            "second_job_without_new_authorization": False,
            "fail_fast": True,
        },
    }


def _atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        json.loads(path.read_text(encoding="utf-8"))
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args(list(argv) if argv is not None else None)
    _atomic_json(arguments.output, build_plan())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
