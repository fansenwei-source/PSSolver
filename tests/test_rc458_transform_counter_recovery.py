"""RC4.5.8 raw/refresh/base transform-counter recovery tests."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.plan_rc458_tf32_h100 import build_plan


ROOT = Path(__file__).resolve().parents[1]


RECOVERY_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc458_transform_counter_recovery.json"
)


RC5_R55 = ROOT / "notes/PSSolver_v0_2_0rc5_r5_5_tf32_state_restore.json"


def _resolve_r55(relative: str, historical: str) -> str:
    successor = json.loads(RC5_R55.read_text(encoding="utf-8"))[
        "historical_successor_sources"
    ].get(relative)
    if successor is not None and successor["historical_sha256"] == historical:
        return successor["current_sha256"]
    return historical


def _auxiliary() -> dict[str, object]:
    return {
        "schema": "pssolver.rc4_5_8.tf32_h100_auxiliary.v1",
        "installed_wheel_identity_passed": True,
        "environment_passed": True,
        "cuda_only_tests_passed": True,
        "tf32_scope_gate_passed": True,
        "consumer_installed_wheel_h100_suite_passed": True,
        "source_shadow_import": False,
        "automatic_retry": False,
        "formal_h100_submission_count": 1,
        "cuda_only_passed": 7,
        "cuda_only_failed": 0,
        "cuda_only_skipped": 0,
        "consumer_failed": 0,
        "consumer_xfailed": 0,
        "consumer_skipped": 1,
        "consumer_skip_node_id": "tests/test_warmup_experiments.py::test_r5_benchmark",
    }


def _profile(entry: dict[str, object]) -> dict[str, object]:
    plan = build_plan()
    role = entry["provider_role"]
    group = f"{entry['family']}:{entry['grid']}:{entry['trial']}"
    mean = (1.0 + 0.01 * int(entry["trial"])) * (
        1.01 if role == "candidate" else 1.0
    )
    timed = int(entry["timed_steps"])
    refresh_calls = 3 if entry["family"] in {"channel_legacy", "channel_compiled_v2"} else 0
    raw_forward = int(entry["forward_transforms_per_step"]) * timed + refresh_calls
    raw_inverse = int(entry["inverse_transforms_per_step"]) * timed
    return {
        "schema": "pssolver.rc4_5_8.runtime_family_profile.v1",
        "config": {
            "family": entry["family"],
            "provider_role": role,
            "trial": entry["trial"],
            "shape": plan["grid_contracts"][entry["grid"]]["shape"],
            "lengths": plan["grid_contracts"][entry["grid"]]["lengths"],
            "device": "cuda",
            "dtype": entry["dtype"],
            "warmup_steps": entry["warmup_steps"],
            "profile_steps": entry["timed_steps"],
        },
        "provider_identity": {
            "role": role,
            "commit": entry["provider_commit"],
            "wheel_sha256": ("a" if role == "baseline" else "b") * 64,
        },
        "runtime_identity": {
            "requested": entry["runtime_path"],
            "effective": entry["runtime_path"],
            "fallback_used": False,
        },
        "environment": {
            "allocated_device": "cuda:0",
            "device_name": "NVIDIA H100 PCIe",
        },
        "timing": {
            "samples_seconds": [mean] * timed,
            "mean_timestep_seconds": mean,
            "median_timestep_seconds": mean,
        },
        "memory": {
            "peak_allocated_bytes": 101 if role == "candidate" else 100,
            "peak_reserved_bytes": 101 if role == "candidate" else 100,
        },
        "transform_calls": {
            "forward": raw_forward,
            "inverse": raw_inverse,
            "forward_per_step": raw_forward / timed,
            "inverse_per_step": raw_inverse / timed,
        },
        "compiler": {"graph_breaks": 0, "compile_fallback": False},
        "initial_state_sha256": ("1" + group).encode().hex()[:64].ljust(64, "0"),
        "final_state_sha256": ("2" + group).encode().hex()[:64].ljust(64, "0"),
        "completed_profile_steps": timed,
        "runtime_completed_steps": int(entry["warmup_steps"]) + timed,
        "finite": True,
    }


def _write_profiles(root: Path) -> None:
    root.mkdir()
    for entry in build_plan()["profile_matrix"]:
        (root / f"{entry['key']}.json").write_text(
            json.dumps(_profile(entry), allow_nan=False), encoding="utf-8"
        )
