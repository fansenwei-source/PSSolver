"""Analysis and planning tests for bounded Stage N.2 qualification."""

from __future__ import annotations

import json

import pytest

from pssolver.experimental import (
    analyze_stage_n2_qualification,
    build_stage_n2_h100_plan,
)


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _comparison():
    return {
        "classification": "PASS",
        "maximum_gate_relative_l2": 4.0e-15,
        "relative_l2_tolerance": 1.0e-10,
        "array_count": 21,
        "saved_steps": list(range(7)),
        "arrays": [
            {
                "field": field,
                "step": step,
                "production_sha256": (
                    "b" * 64 if field == "Q" and step == 0 else "c" * 64
                ),
            }
            for step in range(7)
            for field in ("Q", "u", "p")
        ],
    }


def _profile(
    mode: str,
    *,
    mean: float,
    forward: float,
    inverse: float,
    allocated: int,
):
    candidate = mode == "candidate"
    return {
        "qualification_stage": "N.2",
        "classification": "PROFILE_COMPLETE",
        "mode": mode,
        "finite": True,
        "algebraic_representation_reuse": True,
        "lazy_algebraic_materialization": candidate,
        "production_metadata_sha256": "a" * 64,
        "production_initial_q_sha256": "b" * 64,
        "configuration": {
            "shape": [128, 128, 32],
            "lengths": [100.0, 100.0, 20.0],
            "dtype": "float64",
            "dt": 0.005,
            "dealias_rule": "cubic_half",
            "projected_transform_execution": "truncated",
            "spectral_storage": "hermitian_half",
            "spectral_refresh_interval": 2,
            "warmup_steps": 10,
            "profile_steps": 20,
            "transform_audit_steps": 2,
        },
        "throughput": {"mean_timestep_seconds": mean},
        "memory": {
            "peak_allocated_bytes": allocated,
            "peak_reserved_bytes": allocated * 2,
        },
        "transform_call_audit": {
            "forward_calls_per_step": forward,
            "inverse_calls_per_step": inverse,
            "representation_reuse": {
                "retained_pairs_after_generation": 0,
            },
            "physical_materialization": (
                {
                    "physical_materializations": 29,
                    "unmaterialized_published_components": 25,
                }
                if candidate
                else None
            ),
        },
        "environment": {
            "cuda_available": True,
            "device_name": "NVIDIA H100 PCIe",
            "cuda_matmul_allow_tf32": False,
        },
    }


def test_stage_n2_analysis_accepts_lazy_inverse_reduction(tmp_path):
    comparison = _write_json(tmp_path / "comparison.json", _comparison())
    controls = [
        _write_json(
            tmp_path / f"control_{index}.json",
            _profile(
                "control",
                mean=value,
                forward=34.5,
                inverse=60.5,
                allocated=100,
            ),
        )
        for index, value in enumerate((0.030, 0.031, 0.029), start=1)
    ]
    candidates = [
        _write_json(
            tmp_path / f"candidate_{index}.json",
            _profile(
                "candidate",
                mean=value,
                forward=31.5,
                inverse=36.5,
                allocated=102,
            ),
        )
        for index, value in enumerate((0.024, 0.025, 0.023), start=1)
    ]

    report = analyze_stage_n2_qualification(
        comparison,
        controls,
        candidates,
    )

    assert report["classification"] == "A_recommended"
    assert report["numerical_equivalence_passed"] is True
    assert report["eligible_for_stage_n3_optimization_design"] is True
    assert report["eligible_for_production_promotion"] is False
    assert report["comparison"]["inverse_transform_reduction_gate"] is True
    assert report["comparison"]["candidate_lifecycle_gate"] is True
    assert report["comparison"]["mean_speedup_control_over_candidate"] == (
        pytest.approx(1.25)
    )


def test_stage_n2_analysis_keeps_slow_candidate_experimental(tmp_path):
    comparison = _write_json(tmp_path / "comparison.json", _comparison())
    control = _write_json(
        tmp_path / "control.json",
        _profile(
            "control",
            mean=0.03,
            forward=34.5,
            inverse=60.5,
            allocated=100,
        ),
    )
    candidate = _write_json(
        tmp_path / "candidate.json",
        _profile(
            "candidate",
            mean=0.031,
            forward=31.5,
            inverse=36.5,
            allocated=100,
        ),
    )

    report = analyze_stage_n2_qualification(
        comparison,
        [control],
        [candidate],
    )

    assert report["classification"] == "B_neutral"
    assert report["eligible_for_stage_n3_optimization_design"] is False
    assert report["eligible_for_production_promotion"] is False


def test_stage_n2_analysis_rejects_failed_trajectory_before_profiles(tmp_path):
    value = _comparison()
    value["classification"] = "FAIL"
    comparison = _write_json(tmp_path / "comparison.json", value)
    with pytest.raises(ValueError, match="trajectory comparison did not pass"):
        analyze_stage_n2_qualification(comparison, [], [])


def test_stage_n2_plan_is_bounded_balanced_and_non_promoting(tmp_path):
    project = tmp_path / "project"
    production = tmp_path / "production"
    project.mkdir()
    production.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")

    plan = build_stage_n2_h100_plan(
        project_root=project,
        production_reference=production,
        control_root=tmp_path / "control",
        scratch_root=tmp_path / "scratch",
        python=python,
        expected_commit="c" * 40,
    )

    assert plan["qualification_stage"] == "N.2"
    assert plan["planning_only"] is True
    assert plan["control_definition"] == "stage_n1_representation_reuse"
    assert plan["production_path_changed"] is False
    assert plan["production_promotion_authorized"] is False
    assert plan["fixed_gates"]["candidate_inverse_calls_must_decrease"] is True
    profiles = plan["commands"]["profiles_balanced"]
    assert [(item["trial"], item["mode"]) for item in profiles] == [
        (1, "control"),
        (1, "candidate"),
        (2, "candidate"),
        (2, "control"),
        (3, "control"),
        (3, "candidate"),
    ]
    assert not (tmp_path / "control").exists()
    assert not (tmp_path / "scratch").exists()
