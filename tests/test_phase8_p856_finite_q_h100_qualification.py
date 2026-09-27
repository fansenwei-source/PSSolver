"""P8.5.6 GPU execution, profiler, and analyzer qualification tests."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from benchmarks.analyze_finite_q_anchoring_qualification import analyze
from benchmarks.profile_finite_q_anchoring import (
    FiniteQAnchoringProfileConfig,
    build_runtime,
    profile,
)
from pssolver.workflows.finite_q_anchoring import (
    PlaneFiniteQAnchoringWorkflow,
    load_finite_q_anchoring_checkpoint,
)


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def _config(device: str, *, mode: str = "aggregate_runtime"):
    return FiniteQAnchoringProfileConfig(
        mode=mode,
        shape=(8, 8, 6),
        lengths=(12.0, 10.0, 20.0),
        device=device,
        warmup_steps=1,
        profile_steps=2,
    )


def test_cpu_profiler_records_frozen_operation_counts_and_finite_state():
    report = profile(_config("cpu"))
    assert report["finite"] is True
    assert report["completed_steps"] == 3
    assert report["operation_counts"]["per_step"] == {
        "forward_transform": 10.0,
        "inverse_transform": 5.0,
        "helmholtz_apply": 0.0,
        "helmholtz_solve": 5.0,
    }
    assert report["runtime"]["fallback_used"] is False
    assert report["runtime"]["construction_in_timed_loop"] is False
    assert report["conditioning"]["maximum_basis_condition_number"] < 2.0
    assert report["torch"]["tf32_matmul"] is False
    assert report["torch"]["tf32_cudnn"] is False


def _synthetic_report(shape, trial, mode, package_root):
    ratio = 1.02 if mode == "aggregate_runtime" else 1.0
    return {
        "phase": "P8.5.6",
        "kind": "finite_q_anchoring_profile",
        "config": {"shape": list(shape), "trial": trial, "mode": mode},
        "timing": {
            "mean_timestep_seconds": ratio,
            "median_timestep_seconds": ratio,
        },
        "memory": {
            "peak_allocated_bytes": int(1000 * ratio),
            "peak_active_bytes": int(1000 * ratio),
            "peak_reserved_bytes": int(1000 * ratio),
        },
        "operation_counts": {
            "per_step": {
                "forward_transform": 10.0,
                "inverse_transform": 5.0,
                "helmholtz_apply": 0.0,
                "helmholtz_solve": 5.0,
            }
        },
        "conditioning": {"maximum_basis_condition_number": 1.01},
        "runtime": {
            "fallback_allowed": False,
            "fallback_used": False,
            "graph_breaks": 0,
            "construction_in_timed_loop": False,
            "root_solve_in_timed_loop": False,
            "matrix_factorization_in_timed_loop": False,
        },
        "finite": True,
        "cuda": {"available": True, "name": "NVIDIA H100 PCIe"},
        "torch": {"tf32_matmul": False, "tf32_cudnn": False},
        "git": {"head": "a" * 40, "dirty": False},
        "pssolver_import": str(package_root / "pssolver" / "__init__.py"),
        "initial_q_sha256": f"initial-{shape}",
        "final_q_sha256": f"final-{shape}-{trial}",
    }


def _synthetic_matrix(tmp_path):
    return [
        _synthetic_report(shape, trial, mode, tmp_path)
        for shape in ((128, 128, 32), (320, 320, 80))
        for trial in (1, 2, 3)
        for mode in ("scalar_reference", "aggregate_runtime")
    ]


def test_analyzer_accepts_only_complete_paired_non_regression_matrix(tmp_path):
    reports = _synthetic_matrix(tmp_path)
    result = analyze(
        reports,
        expected_commit="a" * 40,
        expected_package_root=tmp_path,
    )
    assert result["classification"] == "PASS_P8_5_6_PROFILE_MATRIX"
    assert result["profile_count"] == 12
    assert result["eligible_for_final_scientific_gates"] is True

    incomplete = reports[:-1]
    with pytest.raises(RuntimeError, match="12 profile"):
        analyze(
            incomplete,
            expected_commit="a" * 40,
            expected_package_root=tmp_path,
        )
    tampered = copy.deepcopy(reports)
    tampered[0]["operation_counts"]["per_step"]["forward_transform"] = 9.0
    with pytest.raises(RuntimeError, match="forward count"):
        analyze(
            tampered,
            expected_commit="a" * 40,
            expected_package_root=tmp_path,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_runtime_matches_cpu_and_binds_allocated_device():
    cpu = build_runtime(_config("cpu"))
    gpu = build_runtime(_config("cuda"))
    assert gpu.device == torch.device("cuda", torch.cuda.current_device())
    assert all(
        value.cache_key.device == str(gpu.device)
        for value in gpu.components.values()
    )
    cpu.advance(3)
    gpu.advance(3)
    torch.testing.assert_close(
        gpu.physical_q().cpu(),
        cpu.physical_q(),
        rtol=2.0e-13,
        atol=2.0e-14,
    )
    assert gpu.operation_counts() == {
        "forward_transform": 35,
        "inverse_transform": 15,
        "helmholtz_apply": 0,
        "helmholtz_solve": 15,
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_continuous_split_and_file_restart_are_exact(tmp_path):
    continuous = build_runtime(_config("cuda"))
    segment = build_runtime(_config("cuda"))
    resumed = build_runtime(_config("cuda"))
    continuous.advance(4)
    PlaneFiniteQAnchoringWorkflow(
        segment,
        tmp_path / "segment",
        checkpoint_interval=2,
    ).run(2)
    checkpoint_path = tmp_path / "segment" / "checkpoint_2"
    checkpoint = load_finite_q_anchoring_checkpoint(
        checkpoint_path,
        device=resumed.device,
    )
    assert all(
        value.remainder.device == resumed.device
        for _name, value in checkpoint.components
    )
    resumed.restore_checkpoint(checkpoint)
    resumed.advance(2)
    assert torch.equal(continuous.physical_q(), resumed.physical_q())
    for component in continuous.components:
        assert torch.equal(
            continuous.components[component].remainder,
            resumed.components[component].remainder,
        )
        assert torch.equal(
            continuous.components[component].bounded_modal,
            resumed.components[component].bounded_modal,
        )


def test_p856_plan_and_archive_entries_are_present():
    record = json.loads(
        (NOTES / "phase_8_p856_h100_qualification_plan.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["phase"] == "P8.5.6"
    assert record["status"] == "READY_P8_5_6_H100_QUALIFICATION_NOT_EXECUTED"
    assert record["qualification_contract"]["profile_report_count"] == 12
    assert record["authorization"]["single_h100_job"] is True
    assert record["authorization"]["p8_6_authorized"] is False
    archive = (NOTES / "build_verbatim_archive_pdf.py").read_text()
    for name in (
        "phase_8_p856_h100_qualification_plan.md",
        "phase_8_p856_h100_qualification_plan.json",
    ):
        assert archive.count(f'"{name}"') == 1
