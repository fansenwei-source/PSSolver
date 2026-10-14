"""RC4.5.8 CUDA reserved-memory reducer recovery tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks.analyze_rc458_reserved_memory_recovery import (
    PASS_CLASSIFICATION,
    adjudicate_reserved_group,
    analyze,
)
from benchmarks.analyze_rc458_transform_counter_recovery import (
    analyze as analyze_counter_recovery,
)
from benchmarks.plan_rc458_tf32_h100 import build_plan


ROOT = Path(__file__).resolve().parents[1]
RECOVERY_RECORD = ROOT / (
    "notes/PSSolver_v0_2_0rc4_rc458_reserved_memory_recovery.json"
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
        "consumer_skip_node_id": (
            "tests/test_warmup_experiments.py::test_r5_benchmark"
        ),
    }


def _profile(entry: dict[str, object]) -> dict[str, object]:
    plan = build_plan()
    role = entry["provider_role"]
    group = f"{entry['family']}:{entry['grid']}:{entry['trial']}"
    mean = (1.0 + 0.01 * int(entry["trial"])) * (
        1.01 if role == "candidate" else 1.0
    )
    timed = int(entry["timed_steps"])
    refresh_calls = (
        3
        if entry["family"] in {"channel_legacy", "channel_compiled_v2"}
        else 0
    )
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


def test_reserved_memory_recovery_uses_group_peaks_not_trial_labels():
    record = adjudicate_reserved_group(
        [1_151_336_448, 1_151_336_448, 1_203_765_248],
        [1_151_336_448, 1_203_765_248, 1_151_336_448],
        ratio_limit=1.03,
    )
    assert record["raw_paired_candidate_over_baseline"][1] > 1.03
    assert record["candidate_over_baseline_group_peak_ratio"] == 1.0
    assert record["candidate_over_baseline_group_mean_ratio"] == 1.0
    assert record["trial_multiset_equal"] is True


def test_reserved_memory_recovery_rejects_real_group_peak_regression():
    with pytest.raises(ValueError, match="group-peak reserved-memory regression"):
        adjudicate_reserved_group([100, 100, 100], [100, 100, 104], ratio_limit=1.03)


def test_reserved_memory_recovery_replays_allocator_permutation_without_mutation(
    tmp_path,
):
    root = tmp_path / "profiles"
    _write_profiles(root)
    family = "channel_complete_stress"
    grid = "channel_R512x40x40"
    baseline = [1_151_336_448, 1_151_336_448, 1_203_765_248]
    candidate = [1_151_336_448, 1_203_765_248, 1_151_336_448]
    for role, values in (("baseline", baseline), ("candidate", candidate)):
        for trial, value in enumerate(values, start=1):
            path = root / f"{role}_{family}_{grid}_trial{trial}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["memory"]["peak_reserved_bytes"] = value
            path.write_text(json.dumps(payload), encoding="utf-8")
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    with pytest.raises(ValueError, match="reserved regression"):
        analyze_counter_recovery(root, _auxiliary())
    result = analyze(root, _auxiliary())
    assert {path.name: path.read_bytes() for path in root.iterdir()} == before
    assert result["classification"] == PASS_CLASSIFICATION
    assert result["qualification_complete"] is True
    contract = result["reserved_memory_contract"]
    assert contract["threshold_changed"] is False
    assert contract["peak_reserved_ratio_max"] == 1.03
    assert contract["peak_allocated_contract"] == (
        "original_strict_per_trial_gate_unchanged"
    )
    group = next(
        item
        for item in contract["groups"]
        if item["family"] == family and item["grid"] == grid
    )
    assert group["candidate_over_baseline_group_peak_ratio"] == 1.0
    assert group["trial_multiset_equal"] is True


def test_reserved_memory_recovery_retains_allocated_memory_hard_gate(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    path = root / "candidate_channel_functional_channel_R512x40x40_trial2.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["memory"]["peak_allocated_bytes"] = 104
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="allocated regression"):
        analyze(root, _auxiliary())


def test_reserved_memory_recovery_rejects_symlink_and_extra_json(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    target = root / "baseline_channel_functional_channel_R128x20x20_trial1.json"
    payload = target.read_bytes()
    target.unlink()
    real = tmp_path / "outside.json"
    real.write_bytes(payload)
    target.symlink_to(real)
    with pytest.raises(ValueError, match="profile missing"):
        analyze(root, _auxiliary())

    target.unlink()
    target.write_bytes(payload)
    (root / "unexpected.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source profile file set differs"):
        analyze(root, _auxiliary())


def test_reserved_memory_recovery_record_binds_contract_and_sources():
    record = json.loads(RECOVERY_RECORD.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "READY_RC4_5_8_RESERVED_MEMORY_REDUCER_CPU_REPLAY"
    )
    assert record["qualification_complete"] is False
    contract = record["contract_correction"]
    assert contract["threshold_changed"] is False
    assert contract["peak_reserved_ratio_max"] == 1.03
    assert contract["peak_allocated_gate_changed"] is False
    assert contract["raw_profiles_modified"] is False
    for relative, expected in record["source_sha256"].items():
        resolved = _resolve_r55(relative, expected)
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == resolved
    assert record["authorization"] == {
        "cpu_only_replay_authorized": True,
        "new_h100_submission_authorized": False,
        "automatic_retry": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
        "production_default_changed": False,
    }
