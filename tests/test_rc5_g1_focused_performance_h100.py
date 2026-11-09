"""Contracts for the rc5 G-1 focused performance/memory recovery."""

from __future__ import annotations

import json
from pathlib import Path
import statistics

import pytest
from benchmarks.plan_rc5_g1_focused_performance_h100 import (
    FOCUSED_CELLS,
    SCHEMA,
    TIMING_CV_MAX,
    V5_MANIFEST_SHA256,
    V6_MANIFEST_SHA256,
    build_plan,
    main as plan_main,
)


OFF = {
    "float32_matmul_precision": "highest",
    "cuda_matmul_allow_tf32": False,
    "cudnn_allow_tf32": False,
}


def _hash(label: str) -> str:
    import hashlib

    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _write_profiles(root: Path) -> None:
    plan = build_plan()
    root.mkdir()
    for entry in plan["profile_matrix"]:
        role = entry["provider_role"]
        group = f"{entry['family']}:{entry['grid']}:trial{entry['trial']}"
        mean = (1.0 + 0.001 * int(entry["trial"])) * (
            1.01 if role == "candidate" else 1.0
        )
        samples = [mean] * 60
        final = _hash("final:" + group)
        if role == "candidate" and entry["family"] == "periodic_functional":
            final = _hash("repaired:" + group)
        profile = {
            "schema": "pssolver.rc4_5_8.runtime_family_profile.v1",
            "config": {
                "family": entry["family"],
                "provider_role": role,
                "provider_commit": entry["provider_commit"],
                "provider_wheel_sha256": entry["provider_wheel_sha256"],
                "trial": entry["trial"],
                "shape": plan["grid_contracts"][entry["grid"]]["shape"],
                "lengths": plan["grid_contracts"][entry["grid"]]["lengths"],
                "device": "cuda",
                "dtype": entry["dtype"],
                "warmup_steps": 10,
                "profile_steps": 60,
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
                "samples_seconds": samples,
                "mean_timestep_seconds": mean,
                "median_timestep_seconds": mean,
                "sample_standard_deviation_seconds": 0.0,
                "coefficient_of_variation": 0.0,
            },
            "memory": {
                "peak_allocated_bytes": 101 if role == "candidate" else 100,
                "peak_active_bytes": 101 if role == "candidate" else 100,
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
            "final_state_sha256": final,
            "completed_profile_steps": 60,
            "runtime_completed_steps": 70,
            "finite": True,
        }
        (root / f"{entry['key']}.json").write_text(
            json.dumps(profile, allow_nan=False), encoding="utf-8"
        )


def _load(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _save(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, allow_nan=False), encoding="utf-8")


def _replace_samples(profile: dict[str, object], samples: list[float]) -> None:
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples)
    profile["timing"] = {
        "samples_seconds": samples,
        "mean_timestep_seconds": mean,
        "median_timestep_seconds": statistics.median(samples),
        "sample_standard_deviation_seconds": stdev,
        "coefficient_of_variation": stdev / mean,
    }


def test_focused_plan_is_exact_order_balanced_and_keeps_original_thresholds():
    plan = build_plan()
    assert plan["schema"] == SCHEMA
    assert plan["profile_count"] == 48
    assert len(FOCUSED_CELLS) == 4
    assert [entry["sequence"] for entry in plan["profile_matrix"]] == list(
        range(1, 49)
    )
    assert len({entry["key"] for entry in plan["profile_matrix"]}) == 48
    for cell in FOCUSED_CELLS:
        rows = [
            row
            for row in plan["profile_matrix"]
            if row["family"] == cell["family"] and row["grid"] == cell["grid"]
        ]
        assert len(rows) == 12
        orders = [
            [row["provider_role"] for row in rows if row["trial"] == trial]
            for trial in range(1, 7)
        ]
        assert orders == [
            ["baseline", "candidate"],
            ["candidate", "baseline"],
            ["baseline", "candidate"],
            ["candidate", "baseline"],
            ["baseline", "candidate"],
            ["candidate", "baseline"],
        ]
    assert plan["timing_validity"] == {
        "raw_sample_count": 60,
        "sample_standard_deviation": "sample",
        "coefficient_of_variation_max": TIMING_CV_MAX,
        "reported_summary_must_match_raw_samples": True,
        "invalid_profile_disposition": "fail_closed_stop",
        "outlier_removal_allowed": False,
        "trial_replacement_allowed": False,
    }
    assert plan["thresholds"]["mean_timestep_ratio_max"] == 1.03
    assert plan["thresholds"]["median_of_profile_means_ratio_max"] == 1.03
    assert plan["thresholds"]["every_paired_mean_timestep_ratio_max"] == 1.05
    assert plan["thresholds"]["peak_reserved_group_ratio_max"] == 1.03
    assert plan["immutable_inputs"]["v5_manifest_sha256"] == V5_MANIFEST_SHA256
    assert plan["immutable_inputs"]["v6_manifest_sha256"] == V6_MANIFEST_SHA256
    assert not any(plan["authorization"].values())


def test_focused_plan_cli_writes_strict_non_overwriting_json(tmp_path):
    output = tmp_path / "plan.json"
    assert plan_main(["--output", str(output)]) == 0
    assert _load(output) == build_plan()
    with pytest.raises(FileExistsError):
        plan_main(["--output", str(output)])
    assert not list(tmp_path.glob("*.tmp"))
    assert not list(tmp_path.glob(".*.tmp"))


def test_focused_slurm_script_is_single_h100_fail_closed_and_not_g1_complete():
    root = Path(__file__).resolve().parents[1]
    source = (root / "jobs/rc5_g1_focused_performance_h100.sbatch").read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=hagan-gpu" in source
    assert "#SBATCH --account=hagan-lab" in source
    assert "#SBATCH --qos=medium" in source
    assert "#SBATCH --gres=gpu:H100:1" in source
    assert "#SBATCH --no-requeue" in source
    assert "run_rc5_g1_checkpoint_tmp_preflight.py" in source
    assert "run_rc5_g1_environment.py" in source
    assert "run_rc5_g1_focused_profile_matrix.py" in source
    assert 'touch "${RC5_G1_OUTPUT_ROOT}/FOCUSED_COMPLETE"' in source
    assert 'touch "${RC5_G1_OUTPUT_ROOT}/COMPLETE"' not in source
    assert "sbatch" not in "\n".join(
        line for line in source.splitlines() if not line.startswith("#SBATCH")
    )
