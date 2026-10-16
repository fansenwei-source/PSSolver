"""Synthetic fail-closed tests for the rc5 G-1 cumulative analyzer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.analyze_rc5_g1_h100 import PASS_CLASSIFICATION, analyze
from benchmarks.assemble_rc5_g1_auxiliary import assemble
from benchmarks.plan_rc5_g1_h100 import (
    CONSUMER_SOURCE,
    CONSUMER_WHEEL_SHA256,
    PROVIDER_BASELINE,
    PROVIDER_BASELINE_WHEEL_SHA256,
    PROVIDER_CANDIDATE,
    PROVIDER_CANDIDATE_WHEEL_SHA256,
    build_plan,
)


OFF = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


def _auxiliary() -> dict[str, object]:
    plan = build_plan()
    numerical = []
    for family in plan["tf32_scope_gate"]["runtime_families"]:
        entry: dict[str, object] = {
            "family": family,
            "finite": True,
            "fallback_used": False,
        }
        if family.startswith("periodic_"):
            entry.update(
                {
                    "comparison": "physical_tolerance",
                    "components": {
                        name: {"relative_l2": 1.0e-14, "linf": 1.0e-14}
                        for name in ("Q", "velocity", "pressure")
                    },
                    "candidate_hermitian_violation": 0.0,
                }
            )
        else:
            entry.update({"comparison": "byte_identity", "byte_identical": True})
        numerical.append(entry)
    return {
        "schema": "pssolver.rc5.g1.h100_auxiliary.v1",
        "formal_h100_submission_count": 1,
        "automatic_retry": False,
        "second_job_submitted": False,
        "environment": {
            "schema": "pssolver.rc5.g1.environment.v1",
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
        },
        "cuda_only": {
            "schema": "pssolver.rc5.g1.cuda_only.v1",
            "provider_commit": PROVIDER_CANDIDATE,
            "provider_wheel_sha256": PROVIDER_CANDIDATE_WHEEL_SHA256,
            "passed": True,
            "passed_count": 7,
            "failed_count": 0,
            "skipped_count": 0,
            "deselected_count": 0,
            "node_ids": plan["cuda_only_tests"],
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
        "checkpoints": {
            "schema": "pssolver.rc5.g1.checkpoint_h100.v1",
            "provider_commit": PROVIDER_CANDIDATE,
            "consumer_source_commit": CONSUMER_SOURCE,
            "passed": True,
            "current_family_count": 4,
            "direction_count": 2,
            "continuous_split_restart_passed": True,
            "transactional_restore_passed": True,
            "negative_gates_passed": True,
            "released_primary_case_count": 13,
            "released_dispositions_match_cpu_record": True,
            "source_checkpoints_unchanged": True,
        },
        "numerical": {
            "schema": "pssolver.rc5.g1.numerical_h100.v1",
            "provider_baseline_commit": PROVIDER_BASELINE,
            "provider_candidate_commit": PROVIDER_CANDIDATE,
            "passed": True,
            "families": numerical,
        },
        "consumer": {
            "schema": "pssolver.rc5.g1.consumer_h100.v1",
            "passed": True,
            "source_commit": CONSUMER_SOURCE,
            "wheel_sha256": CONSUMER_WHEEL_SHA256,
            "passed_count": plan["consumer_h100_gate"]["expected_passed"],
            "skipped_count": 1,
            "failed_count": 0,
            "xfailed_count": 0,
            "deselected_count": 0,
            "skip_node_id": "tests/test_warmup_experiments.py::test_r5_benchmark",
            "identities": {
                "periodic": plan["consumer_h100_gate"]["periodic_identity_sha256"],
                "channel_fixed": plan["consumer_h100_gate"]["channel_fixed_identity_sha256"],
                "channel_tolerance": plan["consumer_h100_gate"]["channel_tolerance_identity_sha256"],
            },
            "old_periodic_pin_rejected_before_mutation": True,
        },
    }


def _hash(label: str) -> str:
    return label.encode().hex()[:64].ljust(64, "0")


def _write_profiles(root: Path) -> None:
    plan = build_plan()
    root.mkdir()
    for entry in plan["profile_matrix"]:
        role = entry["provider_role"]
        group = f"{entry['family']}:{entry['grid']}:{entry['trial']}"
        base = 1.0 + 0.01 * int(entry["trial"])
        mean = base * (1.01 if role == "candidate" else 1.0)
        final = _hash("final:" + group)
        if role == "candidate" and entry["family"].startswith("periodic_"):
            final = _hash("repaired:" + group)
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
                "requested": entry["runtime_path"],
                "effective": entry["runtime_path"],
                "fallback_used": False,
            },
            "environment": {
                "allocated_device": "cuda:0",
                "device_name": "NVIDIA H100 PCIe",
                "tf32_caller_before": OFF,
                "tf32_caller_after": OFF,
            },
            "timing": {
                "samples_seconds": [mean] * 60,
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
                "forward_per_step": (
                    entry["expected_forward_raw_calls"] / entry["timed_steps"]
                ),
                "inverse_per_step": (
                    entry["expected_inverse_raw_calls"] / entry["timed_steps"]
                ),
            },
            "compiler": {"graph_breaks": 0, "compile_fallback": False},
            "initial_state_sha256": _hash("initial:" + group),
            "final_state_sha256": final,
            "completed_profile_steps": 60,
            "runtime_completed_steps": 70,
            "finite": True,
        }
        (root / f"{entry['key']}.json").write_text(
            json.dumps(profile, allow_nan=False), encoding="utf-8"
        )


def test_rc5_analyzer_accepts_periodic_repair_with_physical_gate(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    result = analyze(root, _auxiliary())
    assert result["classification"] == PASS_CLASSIFICATION
    assert result["qualification_complete"] is True
    assert result["profile_count"] == 96
    assert result["group_count"] == 16
    assert result["g1_complete"] is False
    window = result["transform_measurement_window"]
    assert window["passed"] is True
    assert window["profile_count"] == 96
    assert window["scheduled_refresh_profile_count"] == 24
    assert len(window["records"]) == 96
    channel = next(
        record
        for record in window["records"]
        if record["family"] == "channel_legacy"
    )
    assert channel["raw_forward_calls"] == 663
    assert channel["scheduled_forward_calls"] == 3
    assert channel["steady_forward_calls"] == 660
    assert channel["steady_forward_per_step"] == 11.0
    repaired = [group for group in result["groups"] if group["family"].startswith("periodic_")]
    assert repaired
    assert {group["final_state_policy"] for group in repaired} == {"physical_tolerance"}


def test_rc5_auxiliary_assembler_revalidates_every_subgate():
    auxiliary = _auxiliary()
    result = assemble(
        environment=auxiliary["environment"],
        cuda_only=auxiliary["cuda_only"],
        tf32_scope=auxiliary["tf32_scope"],
        checkpoints=auxiliary["checkpoints"],
        numerical=auxiliary["numerical"],
        consumer=auxiliary["consumer"],
    )
    assert result == auxiliary

    broken = _auxiliary()
    broken["checkpoints"]["source_checkpoints_unchanged"] = False
    with pytest.raises(ValueError, match="source checkpoint mutated"):
        assemble(
            environment=broken["environment"],
            cuda_only=broken["cuda_only"],
            tf32_scope=broken["tf32_scope"],
            checkpoints=broken["checkpoints"],
            numerical=broken["numerical"],
            consumer=broken["consumer"],
        )


def test_rc5_analyzer_rejects_nonperiodic_state_mismatch(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    target = root / "candidate_plane_legacy_production_plane_R128_trial1.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["final_state_sha256"] = "f" * 64
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="final state differs"):
        analyze(root, _auxiliary())


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("relative_l2", 2.0e-12), "relative L2 failed"),
        (("linf", 2.0e-12), "Linf failed"),
        (("candidate_hermitian_violation", 1.0e-30), "not exactly zero"),
    ],
)
def test_rc5_analyzer_rejects_periodic_numerical_gate_failures(
    tmp_path, mutation, message
):
    root = tmp_path / "profiles"
    _write_profiles(root)
    auxiliary = _auxiliary()
    periodic = auxiliary["numerical"]["families"][2]
    name, value = mutation
    if name in {"relative_l2", "linf"}:
        periodic["components"]["Q"][name] = value
    else:
        periodic[name] = value
    with pytest.raises(ValueError, match=message):
        analyze(root, auxiliary)


def test_rc5_analyzer_rejects_tf32_and_consumer_identity_failures(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    auxiliary = _auxiliary()
    auxiliary["tf32_scope"]["original_state_restored"] = False
    with pytest.raises(ValueError, match="TF32 state not restored"):
        analyze(root, auxiliary)

    auxiliary = _auxiliary()
    auxiliary["consumer"]["identities"]["periodic"] = "0" * 64
    with pytest.raises(ValueError, match="Periodic identity differs"):
        analyze(root, auxiliary)


def test_rc5_analyzer_rejects_performance_and_reserved_memory_regression(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    target = root / "candidate_channel_functional_channel_R128x20x20_trial1.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["timing"]["mean_timestep_seconds"] *= 2.0
    payload["timing"]["median_timestep_seconds"] *= 2.0
    payload["timing"]["samples_seconds"] = [
        value * 2.0 for value in payload["timing"]["samples_seconds"]
    ]
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="regression"):
        analyze(root, _auxiliary())

    _write_profiles(tmp_path / "profiles2")
    target = tmp_path / "profiles2/candidate_channel_functional_channel_R128x20x20_trial1.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["memory"]["peak_reserved_bytes"] = 200
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reserved regression"):
        analyze(tmp_path / "profiles2", _auxiliary())


@pytest.mark.parametrize(
    ("raw_forward", "message"),
    (
        (662, "raw forward transform count differs"),
        (664, "raw forward transform count differs"),
    ),
)
def test_rc5_analyzer_rejects_wrong_scheduled_refresh_count(
    tmp_path, raw_forward, message
):
    root = tmp_path / "profiles"
    _write_profiles(root)
    target = root / "candidate_channel_legacy_channel_R128x20x20_trial1.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["transform_calls"]["forward"] = raw_forward
    payload["transform_calls"]["forward_per_step"] = raw_forward / 60
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        analyze(root, _auxiliary())


def test_rc5_analyzer_rejects_raw_rate_or_inverse_tamper(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    target = root / "baseline_channel_compiled_v2_channel_R128x20x20_trial1.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["transform_calls"]["forward_per_step"] = 11.0
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="reported forward rate differs"):
        analyze(root, _auxiliary())

    second = tmp_path / "profiles2"
    _write_profiles(second)
    target = second / "baseline_channel_compiled_v2_channel_R128x20x20_trial1.json"
    payload = json.loads(target.read_text(encoding="utf-8"))
    payload["transform_calls"]["inverse"] = 2161
    payload["transform_calls"]["inverse_per_step"] = 2161 / 60
    target.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="raw inverse transform count differs"):
        analyze(second, _auxiliary())
