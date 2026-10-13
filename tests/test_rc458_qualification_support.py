"""RC4.5.8 qualification-support contracts and synthetic analyzer tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from benchmarks.analyze_rc458_tf32_h100 import (
    PASS_CLASSIFICATION,
    analyze,
)
from benchmarks.plan_rc458_tf32_h100 import FAMILIES, build_plan
from benchmarks.profile_rc458_runtime_family import (
    RC458ProfileConfig,
    _validate,
)
from benchmarks.run_rc458_profile_matrix import build_command_plan


ROOT = Path(__file__).resolve().parents[1]
SUPPORT_RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc458_qualification_support.json"


def test_rc458_per_family_dtype_and_transform_contract_is_explicit():
    expected = {
        "plane_legacy_production": ("float64", 7, 32),
        "plane_compiled_v2": ("float64", 7, 32),
        "periodic_complete_stress": ("float64", 7, 46),
        "channel_legacy": ("float32", 11, 36),
        "channel_compiled_v2": ("float32", 11, 36),
        "channel_complete_stress": ("float64", 15, 42),
        "periodic_functional": ("float64", 7, 46),
        "channel_functional": ("float64", 15, 57),
    }
    assert {
        family.key: (
            family.dtype,
            family.forward_transforms_per_step,
            family.inverse_transforms_per_step,
        )
        for family in FAMILIES
    } == expected
    for entry in build_plan()["profile_matrix"]:
        assert (
            entry["dtype"],
            entry["forward_transforms_per_step"],
            entry["inverse_transforms_per_step"],
        ) == expected[entry["family"]]


def test_rc458_profile_config_rejects_cross_family_dtype_substitution():
    common = dict(
        provider_role="candidate",
        provider_commit="1" * 40,
        provider_wheel_sha256="a" * 64,
        trial=1,
        shape=(8, 6, 4),
        lengths=(8.0, 6.0, 4.0),
        device="cuda",
    )
    _validate(RC458ProfileConfig(family="channel_legacy", dtype="float32", **common))
    _validate(
        RC458ProfileConfig(
            family="channel_complete_stress", dtype="float64", **common
        )
    )
    with pytest.raises(ValueError, match="qualified float32"):
        _validate(
            RC458ProfileConfig(
                family="channel_legacy", dtype="float64", **common
            )
        )
    with pytest.raises(ValueError, match="qualified float64"):
        _validate(
            RC458ProfileConfig(
                family="channel_complete_stress", dtype="float32", **common
            )
        )


def test_rc458_command_plan_is_exact_and_shell_free(tmp_path):
    command = build_command_plan(
        baseline_python=tmp_path / "a" / "bin" / "python",
        candidate_python=tmp_path / "b" / "bin" / "python",
        baseline_wheel_sha256="a" * 64,
        candidate_wheel_sha256="b" * 64,
        support_root=tmp_path / "support",
        output_root=tmp_path / "output",
    )
    rows = command["rows"]
    assert command["profile_count"] == 96
    assert len(rows) == 96
    assert [row["sequence"] for row in rows] == list(range(1, 97))
    assert len({row["key"] for row in rows}) == 96
    assert len({row["compiler_cache"] for row in rows}) == 96
    assert all(isinstance(row["command"], list) for row in rows)
    assert all("--device" in row["command"] for row in rows)
    assert all(row["command"][row["command"].index("--device") + 1] == "cuda" for row in rows)
    dtype_by_family = {family.key: family.dtype for family in FAMILIES}
    for row in rows:
        arguments = row["command"]
        family = arguments[arguments.index("--family") + 1]
        assert arguments[arguments.index("--dtype") + 1] == dtype_by_family[family]


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


def _write_profiles(root: Path) -> None:
    plan = build_plan()
    root.mkdir()
    for entry in plan["profile_matrix"]:
        role = entry["provider_role"]
        group = f"{entry['family']}:{entry['grid']}:{entry['trial']}"
        base = 1.0 + 0.01 * int(entry["trial"])
        mean = base * (1.01 if role == "candidate" else 1.0)
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
                "samples_seconds": [mean] * int(entry["timed_steps"]),
                "mean_timestep_seconds": mean,
                "median_timestep_seconds": mean,
            },
            "memory": {
                "peak_allocated_bytes": 101 if role == "candidate" else 100,
                "peak_reserved_bytes": 101 if role == "candidate" else 100,
            },
            "transform_calls": {
                "forward_per_step": entry["forward_transforms_per_step"],
                "inverse_per_step": entry["inverse_transforms_per_step"],
            },
            "compiler": {"graph_breaks": 0, "compile_fallback": False},
            "initial_state_sha256": ("1" + group).encode().hex()[:64].ljust(64, "0"),
            "final_state_sha256": ("2" + group).encode().hex()[:64].ljust(64, "0"),
            "completed_profile_steps": 60,
            "runtime_completed_steps": 70,
            "finite": True,
        }
        (root / f"{entry['key']}.json").write_text(
            json.dumps(profile, allow_nan=False), encoding="utf-8"
        )


def test_rc458_analyzer_accepts_only_the_complete_frozen_matrix(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    result = analyze(root, _auxiliary())
    assert result["classification"] == PASS_CLASSIFICATION
    assert result["qualification_complete"] is True
    assert result["profile_count"] == 96
    assert result["group_count"] == 16


def test_rc458_analyzer_rejects_state_mismatch_and_auxiliary_failure(tmp_path):
    root = tmp_path / "profiles"
    _write_profiles(root)
    candidate = root / "candidate_plane_legacy_production_plane_R128_trial1.json"
    payload = json.loads(candidate.read_text(encoding="utf-8"))
    payload["final_state_sha256"] = "f" * 64
    candidate.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="final state differs"):
        analyze(root, _auxiliary())

    payload = _auxiliary()
    payload["tf32_scope_gate_passed"] = False
    with pytest.raises(ValueError, match="tf32_scope_gate_passed failed"):
        analyze(root, payload)


def test_rc458_support_record_binds_sources_and_preserves_authorization_boundary():
    import hashlib

    record = json.loads(SUPPORT_RECORD.read_text(encoding="utf-8"))
    assert record["classification"] == "READY_RC4_5_8_H100_QUALIFICATION_SUPPORT"
    assert record["qualification_complete"] is False
    assert record["consumer"]["qualified_source_commit"] == (
        "b8539e0ec4c137a6c13746f6832e26e74be24515"
    )
    assert record["consumer"]["evidence_only_tip"] == (
        "0e3895d664d1b81925127848b2dfde638f39763d"
    )
    for relative, expected in record["source_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    assert record["authorization"] == {
        "push_authorized": False,
        "h100_submission_authorized": False,
        "automatic_retry": False,
        "production_default_changed": False,
    }
