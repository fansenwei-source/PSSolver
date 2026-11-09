"""Synthetic contracts for the RC6 single-H100 qualification support."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path
from benchmarks.plan_rc6_single_h100_qualification import (
    CONSUMER_SOURCE,
    CONSUMER_WHEEL_SHA256,
    PROVIDER_BASELINE,
    PROVIDER_BASELINE_WHEEL_SHA256,
    PROVIDER_CANDIDATE,
    PROVIDER_CANDIDATE_WHEEL_SHA256,
    build_plan,
)
from benchmarks.run_rc6_profile_matrix import build_command_plan


OFF = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


def _hash(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _component_reports() -> dict[str, dict[str, object]]:
    plan = build_plan()
    return {
        "environment": {
            "schema": "pssolver.rc6.environment.v1",
            "passed": True,
            "cuda_available": True,
            "visible_device_count": 1,
            "allocated_device": "cuda:0",
            "device_name": "NVIDIA H100 PCIe",
            "tf32_matmul": False,
            "tf32_cudnn": False,
            "source_shadow_import": False,
            "pip_check": "PASS",
            "imports_from_site_packages": True,
            "installed_wheels": {
                "baseline": PROVIDER_BASELINE_WHEEL_SHA256,
                "candidate": PROVIDER_CANDIDATE_WHEEL_SHA256,
                "consumer": CONSUMER_WHEEL_SHA256,
            },
            "preflight": {
                "node_local_tmp_rename_noreplace": True,
                "pytest_tmpdir_is_unique_node_local_tmp": True,
                "host_compiler_stdatomic_and_cuda_compile": True,
                "empty_triton_cache_compile": True,
                "empty_torchinductor_cache_compile": True,
            },
        },
        "cuda_only": {
            "schema": "pssolver.rc6.cuda_only.v1",
            "provider_commit": PROVIDER_CANDIDATE,
            "provider_wheel_sha256": PROVIDER_CANDIDATE_WHEEL_SHA256,
            "passed": True,
            "passed_count": 7,
            "failed_count": 0,
            "skipped_count": 0,
            "deselected_count": 0,
            "node_ids": plan["cuda_only_tests"],
        },
        "contract_smokes": {
            "schema": "pssolver.rc6.contract_smokes.v1",
            "provider_commit": PROVIDER_CANDIDATE,
            "provider_wheel_sha256": PROVIDER_CANDIDATE_WHEEL_SHA256,
            "passed": True,
            "checks": plan["rc6_contract_smokes"],
        },
        "tf32_scope": {
            "schema": "pssolver.rc4_5_8.tf32_scope_h100.v1",
            "provider_commit": PROVIDER_CANDIDATE,
            "provider_wheel_sha256": PROVIDER_CANDIDATE_WHEEL_SHA256,
            "caller_state_count": 6,
            "runtime_family_case_count": 48,
            "original_state_restored": True,
            "passed": True,
        },
        "current_checkpoints": {
            "schema": "pssolver.rc6.current_checkpoint_h100.v1",
            "provider_commit": PROVIDER_CANDIDATE,
            "consumer_source_commit": CONSUMER_SOURCE,
            "passed": True,
            "family_count": 4,
            "direction_count": 2,
            "continuous_split_restart_passed": True,
            "transactional_restore_passed": True,
            "negative_gates_passed": True,
            "rejected_restart_output_remains_absent": True,
            "source_checkpoints_unchanged": True,
        },
        "released_checkpoints": {
            "schema": "pssolver.rc6.released_checkpoint_h100.v1",
            "passed": True,
            "primary_case_count": 13,
            "dispositions_match_cpu_record": True,
            "source_checkpoints_unchanged": True,
        },
        "numerical": {
            "schema": "pssolver.rc6.numerical_h100.v1",
            "provider_baseline_commit": PROVIDER_BASELINE,
            "provider_candidate_commit": PROVIDER_CANDIDATE,
            "passed": True,
            "families": [
                {
                    "family": family,
                    "completed_steps": 2,
                    "finite": True,
                    "fallback_used": False,
                    "byte_identical": True,
                    "relative_l2": 0.0,
                    "linf": 0.0,
                    "candidate_hermitian_violation": 0.0,
                }
                for family in plan["numerical_gate"]["runtime_families"]
            ],
        },
        "consumer": {
            "schema": "pssolver.rc6.consumer_h100.v1",
            "passed": True,
            "source_commit": CONSUMER_SOURCE,
            "wheel_sha256": CONSUMER_WHEEL_SHA256,
            "passed_count": 344,
            "skipped_count": 1,
            "failed_count": 0,
            "xfailed_count": 0,
            "deselected_count": 0,
            "skip_node_id": "tests/test_warmup_experiments.py::test_r5_benchmark",
            "identities": {
                "periodic": plan["consumer"]["periodic_identity_sha256"],
                "channel_fixed": plan["consumer"]["channel_fixed_identity_sha256"],
                "channel_tolerance": plan["consumer"]["channel_tolerance_identity_sha256"],
            },
            "guard_order_regression_passed": True,
            "periodic_and_channel_cuda_execution_finite": True,
            "checkpoint_replay_and_gradient_gates_passed": True,
            "scientific_control_result": False,
        },
    }


def _write_profiles(root: Path) -> None:
    plan = build_plan()
    root.mkdir()
    for entry in plan["profile_matrix"]:
        role = entry["provider_role"]
        group = f"{entry['family']}:{entry['grid']}:{entry['trial']}"
        mean = 1.01 if role == "candidate" else 1.0
        samples = [mean] * 60
        profile = {
            "schema": "pssolver.rc4_5_8.runtime_family_profile.v1",
            "config": {
                "family": entry["family"],
                "trial": entry["trial"],
                "shape": plan["grid_contracts"][entry["grid"]]["shape"],
                "lengths": plan["grid_contracts"][entry["grid"]]["lengths"],
                "device": "cuda",
                "dtype": entry["dtype"],
            },
            "provider_identity": {
                "role": role,
                "commit": entry["provider_commit"],
                "wheel_sha256": entry["provider_wheel_sha256"],
            },
            "runtime_identity": {
                "requested": entry["family"],
                "effective": entry["family"],
                "fallback_used": False,
            },
            "environment": {
                "allocated_device": "cuda:0",
                "device_name": "NVIDIA H100 PCIe",
                "tf32_caller_before": OFF,
                "tf32_caller_after": OFF,
            },
            "timing": {
                "samples_seconds": samples,
                "mean_timestep_seconds": mean,
                "median_timestep_seconds": mean,
            },
            "memory": {
                "peak_allocated_bytes": 101 if role == "candidate" else 100,
                "peak_reserved_bytes": 101 if role == "candidate" else 100,
            },
            "transform_calls": {
                "forward": entry["expected_forward_raw_calls"],
                "inverse": entry["expected_inverse_raw_calls"],
                "forward_per_step": entry["expected_forward_raw_calls"] / 60,
                "inverse_per_step": entry["expected_inverse_raw_calls"] / 60,
            },
            "compiler": {"graph_breaks": 0, "compile_fallback": False},
            "initial_state_sha256": _hash("initial:" + group),
            "final_state_sha256": _hash("final:" + group),
            "completed_profile_steps": 60,
            "runtime_completed_steps": 70,
            "finite": True,
        }
        (root / f"{entry['key']}.json").write_text(
            json.dumps(profile, allow_nan=False), encoding="utf-8"
        )


def test_rc6_profile_command_plan_is_complete_order_balanced_and_shell_free(tmp_path):
    command = build_command_plan(
        baseline_python=tmp_path / "baseline/bin/python",
        candidate_python=tmp_path / "candidate/bin/python",
        support_root=tmp_path / "support",
        output_root=tmp_path / "output",
    )
    assert command["profile_count"] == 192
    assert [row["sequence"] for row in command["rows"]] == list(range(1, 193))
    assert len({row["key"] for row in command["rows"]}) == 192
    assert len({row["compiler_cache"] for row in command["rows"]}) == 192
    assert len({row["tf32_bootstrap_output"] for row in command["rows"]}) == 192
    assert all(isinstance(row["command"], list) and "-c" not in row["command"] for row in command["rows"])
    plan = build_plan()
    for entry, row in zip(plan["profile_matrix"], command["rows"]):
        assert row["provider_role"] == entry["provider_role"]
        assert row["command"][row["command"].index("--device") + 1] == "cuda"
        assert row["command"][1].endswith("run_rc6_profile_child.py")
        assert row["command"][row["command"].index("--profile-script") + 1].endswith(
            "profile_rc458_runtime_family.py"
        )
