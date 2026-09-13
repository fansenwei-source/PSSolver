"""CPU qualification tests for the bounded Stage M H100 gate."""

from __future__ import annotations

import copy
import json

import pytest

from pssolver.experimental.h100_shadow_qualification import (
    analyze_stage_m_h100_qualification,
)
from pssolver.experimental.plane_shadow_driver import (
    _require_stage_m_h100_production_metadata,
    build_h100_plane_shadow_runtime_from_production_metadata,
)


def _h100_metadata():
    return {
        "script": "Plane_beris_edwards_stokes.py",
        "status": "complete",
        "runtime_environment": {
            "device_type": "cuda",
            "cuda_device_name": "NVIDIA H100 PCIe",
        },
        "solver": {"real_dtype": "float64"},
        "model": {
            "name": "active_nematics",
            "variant": "beris_edwards_complete_nematic_stress_stokes",
        },
        "save_start_step": 0,
        "save_hydrodynamics": True,
        "numerics": {
            "q_gradient_reuse": {"enabled": True},
            "precision": {"tf32_effective": False},
        },
    }


def test_stage_m_h100_metadata_contract_is_explicit():
    metadata = _h100_metadata()
    _require_stage_m_h100_production_metadata(
        metadata,
        expected_gpu_name="H100",
    )

    for path, value, message in (
        (("runtime_environment", "device_type"), "cpu", "CUDA"),
        (("runtime_environment", "cuda_device_name"), "NVIDIA L40S", "GPU"),
        (("solver", "real_dtype"), "float32", "float64"),
        (("numerics", "precision", "tf32_effective"), True, "TF32"),
    ):
        changed = copy.deepcopy(metadata)
        target = changed
        for key in path[:-1]:
            target = target[key]
        target[path[-1]] = value
        with pytest.raises(ValueError, match=message):
            _require_stage_m_h100_production_metadata(
                changed,
                expected_gpu_name="H100",
            )


def test_stage_m_builder_cannot_fall_back_to_cpu():
    with pytest.raises(ValueError, match="requires a CUDA device"):
        build_h100_plane_shadow_runtime_from_production_metadata(
            _h100_metadata(),
            device="cpu",
        )


def _profile_config():
    return {
        "shape": [128, 128, 32],
        "lengths": [100.0, 100.0, 20.0],
        "device": "cuda",
        "dtype": "float64",
        "dt": 0.005,
        "dealias_rule": "cubic_half",
        "projected_transform_execution": "truncated",
        "warmup_steps": 3,
        "profile_steps": 10,
        "spectral_refresh_interval": 2,
        "reuse_q_gradients": True,
        "molecular_field_linear_space": "spectral",
        "stress_divergence_sum_space": "spectral",
        "pointwise_execution": "compile",
        "transform_execution_order": "real_first",
        "spectral_storage": "hermitian_half",
        "initial_q_path": "/scratch/run/Q_0.npy",
    }


def _profile(*, shadow: bool, mean: float, allocated: int, reserved: int):
    config_key = "configuration" if shadow else "config"
    result = {
        config_key: _profile_config(),
        "environment": {
            "cuda_available": True,
            "device_name": "NVIDIA H100 PCIe",
            "cuda_matmul_allow_tf32": False,
        },
        "throughput": {"mean_timestep_seconds": mean},
        "memory": {
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": reserved,
        },
    }
    if shadow:
        result["production_initial_q_sha256"] = "b" * 64
    else:
        result["profile_input"] = {"initial_q_sha256": "b" * 64}
    return result


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_stage_m_analysis_separates_science_time_and_memory(tmp_path):
    trajectory = _write_json(
        tmp_path / "trajectory.json",
        {
            "classification": "PASS",
            "configuration": {"compatible": True},
            "maximum_gate_relative_l2": 2.0e-15,
            "relative_l2_tolerance": 1.0e-10,
            "array_count": 21,
            "arrays": [
                {
                    "field": "Q",
                    "step": 0,
                    "production_sha256": "b" * 64,
                }
            ],
        },
    )
    production = [
        _write_json(
            tmp_path / f"production_{index}.json",
            _profile(
                shadow=False,
                mean=value,
                allocated=100,
                reserved=200,
            ),
        )
        for index, value in enumerate((0.05, 0.052, 0.051), start=1)
    ]
    shadow = [
        _write_json(
            tmp_path / f"shadow_{index}.json",
            _profile(
                shadow=True,
                mean=value,
                allocated=120,
                reserved=240,
            ),
        )
        for index, value in enumerate((0.06, 0.061, 0.059), start=1)
    ]

    report = analyze_stage_m_h100_qualification(
        trajectory,
        production,
        shadow,
    )

    assert report["classification"] == "PASS"
    assert report["numerical_equivalence_passed"] is True
    assert report["eligible_for_stage_n_architecture_decision"] is True
    assert report["eligible_for_production_promotion"] is False
    assert report["trajectory"]["maximum_gate_relative_l2"] == 2.0e-15
    assert report["comparison"][
        "peak_allocated_ratio_shadow_over_production"
    ] == pytest.approx(1.2)
    assert report["comparison"][
        "performance_is_observational_not_a_promotion_gate"
    ] is True


def test_stage_m_analysis_rejects_failed_science_or_mismatched_profiles(tmp_path):
    failed = _write_json(
        tmp_path / "failed.json",
        {
            "classification": "FAIL",
            "configuration": {"compatible": True},
            "maximum_gate_relative_l2": 1.0,
            "relative_l2_tolerance": 1.0e-10,
            "array_count": 1,
            "arrays": [],
        },
    )
    production = _write_json(
        tmp_path / "production.json",
        _profile(shadow=False, mean=0.05, allocated=100, reserved=200),
    )
    shadow_value = _profile(
        shadow=True,
        mean=0.06,
        allocated=120,
        reserved=240,
    )
    shadow = _write_json(tmp_path / "shadow.json", shadow_value)
    with pytest.raises(ValueError, match="did not pass"):
        analyze_stage_m_h100_qualification(failed, [production], [shadow])

    passed = _write_json(
        tmp_path / "passed.json",
        {
            "classification": "PASS",
            "configuration": {"compatible": True},
            "maximum_gate_relative_l2": 1.0e-15,
            "relative_l2_tolerance": 1.0e-10,
            "array_count": 1,
            "arrays": [
                {
                    "field": "Q",
                    "step": 0,
                    "production_sha256": "b" * 64,
                }
            ],
        },
    )
    shadow_value["configuration"]["dt"] = 0.01
    shadow.write_text(json.dumps(shadow_value), encoding="utf-8")
    with pytest.raises(ValueError, match="configurations differ"):
        analyze_stage_m_h100_qualification(passed, [production], [shadow])
