"""P8.4.5 H100-closure helper, analyzer, and frozen-contract tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from benchmarks.analyze_plane_static_lifting_qualification import analyze
from benchmarks.qualify_plane_static_lifting import (
    LiftingProfileConfig,
    run_manufactured,
    run_profile,
    run_restart,
)


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_8_p845_h100_qualification_plan.json"
)
RECOVERY_PATH = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_8_p845_cuda_device_identity_recovery.json"
)
MEMORY_RECOVERY_PATH = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_8_p845_memory_recovery.json"
)
FUSED_RECOVERY_PATH = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_8_p845_fused_reconstruction_recovery.json"
)
CORRECTED_PLAN_PATH = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_8_p845_corrected_closure_plan.json"
)
QUALIFICATION_PATH = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_8_p845_h100_qualification.json"
)


def _plan() -> dict[str, object]:
    return json.loads(PLAN_PATH.read_text(encoding="utf-8"))


def _corrected_plan() -> dict[str, object]:
    return json.loads(CORRECTED_PLAN_PATH.read_text(encoding="utf-8"))


def test_p845_plan_freezes_narrow_scope_and_keeps_neumann_deferred():
    plan = _plan()
    assert plan["phase"] == "P8.4.5"
    assert plan["classification"] == (
        "READY_P8_4_5_H100_QUALIFICATION_NOT_EXECUTED"
    )
    assert plan["authorization"] == {
        "compiled_runtime_promotion": False,
        "nonhomogeneous_neumann": False,
        "p8_4_5_h100_submission": True,
        "p8_5": False,
        "phase_9": False,
        "production_default_change": False,
    }
    contract = plan["qualification_contract"]
    assert contract["profile_report_count"] == 12
    assert contract["profile_shapes"] == [[128, 128, 32], [320, 320, 80]]
    assert contract["forward_transforms_per_step"] == 7.0
    assert contract["inverse_transforms_per_step"] == 32.0


def test_p845_recovery_record_binds_the_cuda_device_fix_without_overclaim():
    record = json.loads(RECOVERY_PATH.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "READY_P8_4_5_CUDA_DEVICE_IDENTITY_RECOVERY"
    )
    assert record["failed_job"]["job_id"] == 10843514
    assert record["failed_job"]["timestep_started"] is False
    assert record["fix"]["requested_device"] == "cuda"
    assert record["fix"]["allocated_device_example"] == "cuda:0"
    assert record["authorization"]["nonhomogeneous_neumann"] is False
    assert record["source_sha256"]["pssolver/operators/lifting.py"] == (
        "66cba570635caf2b471eb6582a4820281bf69a59d3771db917f03b79e4688bfe"
    )


def test_p845_memory_recovery_records_real_failure_and_compact_scope():
    record = json.loads(MEMORY_RECOVERY_PATH.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "READY_P8_4_5_STATIC_LIFTING_MEMORY_RECOVERY"
    )
    assert record["failed_job"]["job_id"] == 10843551
    assert record["failed_job"]["numerical_gates_passed"] is True
    assert record["failed_job"]["memory_gate_passed"] is False
    assert record["cpu_equivalence"]["restart_metadata_equal"] is True
    assert record["recovery"]["lift_storage"] == (
        "wall_normal_profile_broadcast"
    )
    assert record["recovery"]["zero_linear_correction"] == (
        "broadcast_zero"
    )
    assert record["authorization"]["nonhomogeneous_neumann"] is False
    superseded_by_fused_recovery = {
        "benchmarks/qualify_plane_static_lifting.py",
        "pssolver/runtime/static_lifting.py",
    }
    for relative, expected in record["source_sha256"].items():
        if relative in superseded_by_fused_recovery:
            continue
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == (
            expected
        )


def test_p845_fused_reconstruction_recovery_records_measured_failure_and_scope():
    record = json.loads(FUSED_RECOVERY_PATH.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "READY_P8_4_5_FUSED_RECONSTRUCTION_MEMORY_RECOVERY"
    )
    assert record["failed_job"]["job_id"] == 10843753
    assert record["failed_job"]["scientific_gates_passed"] is True
    assert record["failed_job"]["memory_gate_passed"] is False
    assert record["recovery"]["persistent_physical_workspace_bytes"] == 0
    assert record["recovery"]["physical_q_hot_path"] == (
        "fused_pointwise_reconstruction"
    )
    assert record["frozen_contract"]["thresholds_relaxed"] is False
    assert record["authorization"]["nonhomogeneous_neumann"] is False
    superseded_by_corrected_contract = {
        "pssolver/models/active_nematics/beris_edwards.py",
        "pssolver/models/active_nematics/stokes.py",
        "tests/test_phase8_p844_lifting_workflow_restart.py",
        "tests/test_phase8_p845_h100_qualification.py",
    }
    for relative, expected in record["source_sha256"].items():
        if relative in superseded_by_corrected_contract:
            continue
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == (
            expected
        )


def test_p845_corrected_plan_separates_live_memory_and_roundoff_contracts():
    plan = _corrected_plan()
    assert plan["classification"] == (
        "READY_P8_4_5_CORRECTED_CONTRACT_CLOSURE"
    )
    contract = plan["qualification_contract"]
    assert contract["lifting_control_peak_allocated_ratio_max"] == 1.1
    assert contract["lifting_control_peak_active_ratio_max"] == 1.1
    assert contract["reserved_memory_role"] == "diagnostic_only"
    assert "lifting_control_peak_reserved_ratio_max" not in contract
    assert contract["cross_version_field_relative_l2_max"] == 1e-12
    assert contract["cross_version_field_linf_max"] == 1e-12
    assert contract["cross_version_minimum_steps"] == 20
    assert plan["authorization"]["nonhomogeneous_neumann"] is False
    assert plan["authorization"]["p8_5"] is False


def test_p845_final_h100_record_closes_only_the_qualified_scope():
    record = json.loads(QUALIFICATION_PATH.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "PASS_P8_4_5_PLANE_STATIC_LIFTING_H100_CLOSURE"
    )
    assert record["qualification_complete"] is True
    assert record["h100_job"] == {
        "archive_checksum_entries_passed": 2739,
        "archive_manifest_sha256": (
            "11212eddf626a9adf07cba743cfd49429df5dac43adb787b655679cb2a8dafdf"
        ),
        "elapsed": "00:06:03",
        "exit_code": "0:0",
        "job_id": 10843953,
        "node": "gpu-h100-4-0",
        "state": "COMPLETED",
    }
    assert record["qualification"]["restart_byte_identity_passed"] is True
    assert record["cross_version_equivalence"][
        "all_fields_within_tolerance"
    ] is True
    assert record["memory_and_performance"]["reserved_memory_role"] == (
        "diagnostic_only"
    )
    assert record["authorization"] == {
        "compiled_runtime_promotion": False,
        "nonhomogeneous_neumann": False,
        "p8_5_implementation": False,
        "p8_5_planning": True,
        "phase_9": False,
        "production_default_change": False,
    }


@pytest.mark.parametrize(
    ("variant", "has_lifting"),
    (("homogeneous_control", False), ("strong_planar_lifting", True)),
)
def test_profile_helper_uses_real_runtime_and_counts_only_timesteps(
    variant,
    has_lifting,
):
    report = run_profile(
        LiftingProfileConfig(
            variant=variant,
            shape=(8, 8, 6),
            pointwise_execution="eager",
            warmup_steps=1,
            profile_steps=2,
        )
    )
    assert report["finite"] is True
    assert report["completed_steps"] == 3
    assert report["runtime_identity"]["effective"] == "legacy_production"
    assert report["transform_calls"]["forward_per_step"] == 7.0
    assert report["transform_calls"]["inverse_per_step"] == (
        33.0 if has_lifting else 32.0
    )
    assert (report["lifting"] is not None) is has_lifting
    assert (report["lifting_storage"] is not None) is has_lifting
    assert (report["wall_residual"] is not None) is has_lifting
    if has_lifting:
        assert report["wall_residual"]["max_linf"] <= 2e-16
        assert report["lifting_storage"]["operator"]["layout"] == (
            "wall_normal_profile_broadcast"
        )
        assert report["lifting_storage"]["physical_workspace_bytes"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_unindexed_cuda_request_binds_lifting_to_allocated_device_identity():
    report = run_profile(
        LiftingProfileConfig(
            variant="strong_planar_lifting",
            shape=(8, 8, 6),
            device="cuda",
            pointwise_execution="eager",
            warmup_steps=0,
            profile_steps=1,
        )
    )
    assert report["finite"] is True
    assert report["lifting"]["lifting"]["device"] == (
        f"cuda:{torch.cuda.current_device()}"
    )


def test_manufactured_helper_recovers_second_order_rate():
    report = run_manufactured(device="cpu")
    assert report["finite"] is True
    assert report["minimum_rate"] > 1.99


def test_restart_helper_is_byte_exact_for_physical_and_evolved_state():
    report = run_restart(
        shape=(8, 8, 6),
        device="cpu",
        segment_steps=1,
        final_steps=2,
        seed=24,
        pointwise_execution="eager",
    )
    assert report["finite"] is True
    assert report["all_byte_identical"] is True
    assert report["continuous_physical_sha256"] == (
        report["resumed_physical_sha256"]
    )
    assert report["lifting_restart"]["representation"] == (
        "homogeneous_remainder"
    )


def _synthetic_profile(
    *,
    shape: tuple[int, int, int],
    trial: int,
    variant: str,
    timestep: float,
) -> dict[str, object]:
    lifting = variant == "strong_planar_lifting"
    return {
        "phase": "P8.4.5",
        "kind": "plane_static_lifting_profile",
        "config": {"shape": list(shape), "trial": trial, "variant": variant},
        "runtime_identity": {
            "requested": "legacy_production",
            "effective": "legacy_production",
            "fallback_used": False,
        },
        "finite": True,
        "git": {"head": "candidate"},
        "pssolver_import": "/installed/pssolver/__init__.py",
        "transform_calls": {"forward_per_step": 7.0, "inverse_per_step": 32.0},
        "pointwise_compile": {
            "execution": {
                "requested": "compile",
                "effective": "compile",
                "fallback_allowed": False,
            },
            "during_profile": {},
        },
        "lifting": {"representation": "homogeneous_remainder"} if lifting else None,
        "wall_residual": {"max_linf": 1e-16} if lifting else None,
        "timing": {
            "mean_timestep_seconds": timestep,
            "median_timestep_seconds": timestep,
        },
        "memory": {
            "peak_allocated_bytes": 105 if lifting else 100,
            "peak_active_bytes": 105 if lifting else 100,
            "peak_reserved_bytes": 105 if lifting else 100,
        },
        "initial_q_sha256": f"initial-{shape}-{trial}",
    }


def _synthetic_reports() -> list[dict[str, object]]:
    reports = []
    for shape in ((128, 128, 32), (320, 320, 80)):
        for trial in range(1, 4):
            reports.append(
                _synthetic_profile(
                    shape=shape,
                    trial=trial,
                    variant="homogeneous_control",
                    timestep=1.0,
                )
            )
            reports.append(
                _synthetic_profile(
                    shape=shape,
                    trial=trial,
                    variant="strong_planar_lifting",
                    timestep=1.05,
                )
            )
    reports.extend(
        (
            {
                "phase": "P8.4.5",
                "kind": "plane_static_lifting_manufactured_convergence",
                "device": "cuda",
                "finite": True,
                "minimum_rate": 1.99,
            },
            {
                "phase": "P8.4.5",
                "kind": "plane_static_lifting_restart",
                "device": "cuda",
                "pointwise_execution": "compile",
                "finite": True,
                "all_byte_identical": True,
                "continuous_physical_sha256": "same",
                "resumed_physical_sha256": "same",
                "lifting_restart": {"representation": "homogeneous_remainder"},
            },
        )
    )
    return reports


def _synthetic_cross_version_report() -> dict[str, object]:
    contract = _corrected_plan()["qualification_contract"]
    return {
        "phase": "P8.4.5",
        "kind": "plane_static_lifting_cross_version_equivalence",
        "reference_commit": contract["cross_version_reference_commit"],
        "actual_commit": "candidate",
        "shape": [128, 128, 32],
        "steps": 20,
        "pointwise_execution": "compile",
        "initial_q_identical": True,
        "finite": True,
        "maximum_field_relative_l2": 5e-16,
        "maximum_field_linf": 5e-19,
        "all_fields_within_tolerance": True,
        "relative_l2_tolerance": 1e-12,
        "linf_tolerance": 1e-12,
        "same_runtime_restart_byte_identity_required_separately": True,
    }


def _corrected_reports() -> list[dict[str, object]]:
    return [*_synthetic_reports(), _synthetic_cross_version_report()]


def test_analyzer_accepts_complete_frozen_evidence():
    result = analyze(
        _plan(),
        _synthetic_reports(),
        expected_commit="candidate",
        expected_package_root=Path("/installed"),
    )
    assert result["classification"] == (
        "PASS_P8_4_5_PLANE_STATIC_LIFTING_H100_CLOSURE"
    )
    assert result["qualification_complete"] is True
    assert result["nonhomogeneous_neumann_supported"] is False


def test_analyzer_fails_closed_on_transform_or_restart_regression():
    reports = _synthetic_reports()
    reports[0]["transform_calls"]["forward_per_step"] = 8.0
    with pytest.raises(RuntimeError, match="forward transform"):
        analyze(
            _plan(),
            reports,
            expected_commit="candidate",
            expected_package_root=Path("/installed"),
        )

    reports = _synthetic_reports()
    reports[-1]["all_byte_identical"] = False
    with pytest.raises(RuntimeError, match="restart is not exact"):
        analyze(
            _plan(),
            reports,
            expected_commit="candidate",
            expected_package_root=Path("/installed"),
        )


def test_corrected_analyzer_gates_live_memory_but_not_allocator_reservation():
    reports = _corrected_reports()
    for report in reports:
        if report.get("kind") != "plane_static_lifting_profile":
            continue
        if report["config"]["variant"] == "strong_planar_lifting":
            report["memory"]["peak_reserved_bytes"] = 200

    result = analyze(
        _corrected_plan(),
        reports,
        expected_commit="candidate",
        expected_package_root=Path("/installed"),
    )

    assert result["qualification_complete"] is True
    assert result["reserved_memory_role"] == "diagnostic_only"
    assert all(
        value["max_peak_reserved_ratio"] == 2.0
        for value in result["performance"].values()
    )
    assert result["cross_version_numerical_equivalence"][
        "all_fields_within_tolerance"
    ] is True

    reports = _corrected_reports()
    reports[1]["memory"]["peak_active_bytes"] = 111
    with pytest.raises(RuntimeError, match="active-memory"):
        analyze(
            _corrected_plan(),
            reports,
            expected_commit="candidate",
            expected_package_root=Path("/installed"),
        )


def test_corrected_analyzer_fails_closed_on_cross_version_regression():
    reports = _corrected_reports()
    reports[-1]["maximum_field_relative_l2"] = 2e-12
    reports[-1]["all_fields_within_tolerance"] = False

    with pytest.raises(RuntimeError, match="numerical-equivalence"):
        analyze(
            _corrected_plan(),
            reports,
            expected_commit="candidate",
            expected_package_root=Path("/installed"),
        )
