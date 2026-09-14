"""Analysis and planning tests for bounded Stage N.4 qualification."""

from __future__ import annotations

import json

import pytest

from pssolver.experimental import (
    analyze_stage_n4_qualification,
    build_stage_n4_h100_plan,
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


def _materialization():
    return {
        "physical_materializations": 29,
        "on_demand_physical_materializations": 0,
        "physical_materialization_batches": 4,
        "batched_physical_components": 29,
        "singleton_materialization_batches": 0,
        "maximum_materialization_batch_size": 15,
        "physical_island_prefetches": 7,
        "physical_island_requested_components": 35,
        "unmaterialized_published_components": 25,
    }


def _profile(mode: str, *, mean: float, allocated: int = 100):
    value = {
        "qualification_stage": "N.3" if mode == "control" else "N.4",
        "classification": "PROFILE_COMPLETE",
        "mode": "candidate",
        "finite": True,
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
            "forward_calls_per_step": 6.5,
            "inverse_calls_per_step": 11.5,
            "execution_policy": {"mode": "batched_physical_islands"},
            "representation_reuse": {
                "retained_pairs_after_generation": 0,
            },
            "physical_materialization": _materialization(),
        },
        "environment": {
            "cuda_available": True,
            "device_name": "NVIDIA H100 PCIe",
            "cuda_matmul_allow_tf32": False,
        },
        "final_q_sha256": "d" * 64,
    }
    if mode == "control":
        value.update(
            algebraic_representation_reuse=True,
            lazy_algebraic_materialization=True,
            batched_physical_islands=True,
        )
        value["transform_call_audit"].pop("execution_policy")
    else:
        value.update(
            configuration_authority="unified_execution_policy",
            algebraic_execution_policy={
                "mode": "batched_physical_islands",
            },
        )
    return value


def test_stage_n4_analysis_accepts_equivalent_non_regressing_consolidation(
    tmp_path,
):
    comparison = _write_json(tmp_path / "comparison.json", _comparison())
    controls = [
        _write_json(
            tmp_path / f"control_{index}.json",
            _profile("control", mean=value),
        )
        for index, value in enumerate((0.00730, 0.00740, 0.00735), start=1)
    ]
    candidates = [
        _write_json(
            tmp_path / f"candidate_{index}.json",
            _profile("candidate", mean=value, allocated=101),
        )
        for index, value in enumerate((0.00731, 0.00739, 0.00736), start=1)
    ]

    report = analyze_stage_n4_qualification(
        comparison,
        controls,
        candidates,
    )

    assert report["classification"] == "A_recommended"
    assert report["numerical_equivalence_passed"] is True
    assert report["eligible_for_stage_o_production_migration_design"] is True
    assert report["eligible_for_production_promotion"] is False
    assert report["comparison"]["transform_count_identity_gate"] is True
    assert report["comparison"]["lifecycle_identity_gate"] is True
    assert report["comparison"]["performance_non_regression_gate"] is True


def test_stage_n4_analysis_rejects_a_structural_count_change(tmp_path):
    comparison = _write_json(tmp_path / "comparison.json", _comparison())
    control = _profile("control", mean=0.0073)
    candidate = _profile("candidate", mean=0.0073)
    candidate["transform_call_audit"]["forward_calls_per_step"] = 7.0

    report = analyze_stage_n4_qualification(
        comparison,
        [_write_json(tmp_path / "control.json", control)],
        [_write_json(tmp_path / "candidate.json", candidate)],
    )

    assert report["classification"] == "B_neutral"
    assert report["comparison"]["transform_count_identity_gate"] is False
    assert report["eligible_for_stage_o_production_migration_design"] is False


def test_stage_n4_analysis_rejects_a_performance_regression(tmp_path):
    comparison = _write_json(tmp_path / "comparison.json", _comparison())
    report = analyze_stage_n4_qualification(
        comparison,
        [_write_json(tmp_path / "control.json", _profile("control", mean=0.01))],
        [
            _write_json(
                tmp_path / "candidate.json",
                _profile("candidate", mean=0.011),
            )
        ],
    )
    assert report["classification"] == "B_neutral"
    assert report["comparison"]["performance_non_regression_gate"] is False


def test_stage_n4_analysis_rejects_mixed_configuration_authority(tmp_path):
    comparison = _write_json(tmp_path / "comparison.json", _comparison())
    candidate = _profile("candidate", mean=0.0073)
    candidate["configuration_authority"] = "legacy_stage_flags"
    with pytest.raises(ValueError, match="candidate profile is incomplete"):
        analyze_stage_n4_qualification(
            comparison,
            [
                _write_json(
                    tmp_path / "control.json",
                    _profile("control", mean=0.0073),
                )
            ],
            [_write_json(tmp_path / "candidate.json", candidate)],
        )


def test_stage_n4_plan_is_bounded_balanced_and_non_promoting(tmp_path):
    project = tmp_path / "project"
    stage_n3_project = tmp_path / "stage_n3_project"
    production = tmp_path / "production"
    project.mkdir()
    stage_n3_project.mkdir()
    production.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")

    plan = build_stage_n4_h100_plan(
        project_root=project,
        stage_n3_project_root=stage_n3_project,
        production_reference=production,
        control_root=tmp_path / "control",
        scratch_root=tmp_path / "scratch",
        python=python,
        expected_commit="c" * 40,
        expected_stage_n3_commit="b" * 40,
    )

    assert plan["qualification_stage"] == "N.4"
    assert plan["planning_only"] is True
    assert plan["control_definition"] == "frozen_stage_n3_parent_commit"
    assert plan["candidate_definition"] == (
        "unified_execution_policy_and_scheduler"
    )
    assert plan["production_path_changed"] is False
    assert plan["production_promotion_authorized"] is False
    assert plan["fixed_gates"]["transform_call_counts_must_match"] is True
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
