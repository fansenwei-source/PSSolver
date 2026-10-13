"""RC4.5.8 raw/refresh/base transform-counter recovery tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.analyze_rc458_transform_counter_recovery import (
    PASS_CLASSIFICATION,
    adjudicate_transform_counters,
    analyze,
)
from benchmarks.plan_rc458_tf32_h100 import build_plan


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc458_transform_counter_recovery.json"
)


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


def test_counter_recovery_distinguishes_raw_refresh_and_base_layers():
    entry = next(
        item
        for item in build_plan()["profile_matrix"]
        if item["key"] == "baseline_channel_legacy_channel_R128x20x20_trial1"
    )
    original = _profile(entry)
    normalized, record = adjudicate_transform_counters(original, entry)
    assert original["transform_calls"] == {
        "forward": 663,
        "inverse": 2160,
        "forward_per_step": 11.05,
        "inverse_per_step": 36.0,
    }
    assert normalized["transform_calls"]["forward_per_step"] == 11.0
    assert normalized["transform_calls"]["forward"] == 663
    assert record["scheduled_refreshes"] == 3
    assert record["scheduled_refresh_forward_calls"] == 3
    assert record["base_forward_calls"] == 660
    assert record["base_forward_per_step"] == 11.0
    assert original is not normalized


@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("forward", 662, "raw forward rate differs"),
        ("inverse", 2159, "raw inverse rate differs"),
    ),
)
def test_counter_recovery_rejects_raw_count_or_rate_tamper(field, value, message):
    entry = next(
        item for item in build_plan()["profile_matrix"]
        if item["family"] == "channel_legacy"
    )
    profile = _profile(entry)
    profile["transform_calls"][field] = value
    with pytest.raises(ValueError, match=message):
        adjudicate_transform_counters(profile, entry)


def test_counter_recovery_rejects_consistent_but_wrong_raw_forward_count():
    entry = next(
        item for item in build_plan()["profile_matrix"]
        if item["family"] == "channel_compiled_v2"
    )
    profile = _profile(entry)
    profile["transform_calls"]["forward"] = 664
    profile["transform_calls"]["forward_per_step"] = 664 / 60
    with pytest.raises(ValueError, match="base forward transform count differs"):
        adjudicate_transform_counters(profile, entry)


def test_counter_recovery_does_not_subtract_refreshes_from_other_families():
    entry = next(
        item for item in build_plan()["profile_matrix"]
        if item["family"] == "channel_complete_stress"
    )
    profile = _profile(entry)
    normalized, record = adjudicate_transform_counters(profile, entry)
    assert record["scheduled_refreshes"] == 0
    assert record["raw_forward_calls"] == record["base_forward_calls"] == 900
    assert normalized["transform_calls"] == profile["transform_calls"]


def test_counter_recovery_replays_the_complete_matrix_without_mutating_evidence(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    result = analyze(root, _auxiliary())
    after = {path.name: path.read_bytes() for path in root.iterdir()}
    assert after == before
    assert result["classification"] == PASS_CLASSIFICATION
    assert result["qualification_complete"] is True
    assert result["profile_count"] == 96
    assert result["group_count"] == 16
    contract = result["transform_counter_contract"]
    assert contract["raw_evidence_modified"] is False
    assert contract["original_analyzer_modified"] is False
    assert contract["refresh_profile_count"] == 24
    assert len(contract["records"]) == 96


def test_counter_recovery_retains_original_performance_and_auxiliary_gates(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    candidate = root / "candidate_plane_legacy_production_plane_R128_trial1.json"
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["timing"]["mean_timestep_seconds"] = 2.0
    payload["timing"]["median_timestep_seconds"] = 2.0
    payload["timing"]["samples_seconds"] = [2.0] * 60
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="paired time regression"):
        analyze(root, _auxiliary())

    auxiliary = _auxiliary()
    auxiliary["tf32_scope_gate_passed"] = False
    with pytest.raises(ValueError, match="tf32_scope_gate_passed failed"):
        analyze(root, auxiliary)


def test_counter_recovery_rejects_profile_symlinks(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    target = root / "baseline_plane_legacy_production_plane_R128_trial1.json"
    payload = target.read_bytes()
    target.unlink()
    real = tmp_path / "outside.json"
    real.write_bytes(payload)
    target.symlink_to(real)
    with pytest.raises(ValueError, match="profile missing"):
        analyze(root, _auxiliary())


def test_counter_recovery_rejects_extra_profile_json(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    (root / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source profile file set differs"):
        analyze(root, _auxiliary())


def test_counter_recovery_record_binds_sources_and_authorization_boundary():
    record = json.loads(RECOVERY_RECORD.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "READY_RC4_5_8_TRANSFORM_COUNTER_LAYER_CPU_REPLAY"
    )
    assert record["qualification_complete"] is False
    assert record["failed_h100_evidence"] == {
        "job_id": 10879606,
        "control_directory": (
            "/home/fansenwei/pssolver_rc458_tf32_h100_5c286f8_b8539e0_"
            "20261013_recovery_v2"
        ),
        "checksums_sha256": (
            "73fd2ce700bf6ec9b367074bb240952698b2e3948c616557b997c712ae084880"
        ),
        "checksums_verified": 3371,
        "profiles_completed": 96,
        "consumer_passed": 310,
        "consumer_skipped": 1,
        "consumer_failed": 0,
        "automatic_retry": False,
    }
    for relative, expected in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert record["authorization"] == {
        "cpu_only_replay_authorized": True,
        "new_h100_submission_authorized": False,
        "automatic_retry": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }
