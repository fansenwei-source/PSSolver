"""Architecture and planning contracts for Plane Stage P."""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

import pssolver
from pssolver.experimental.stage_o_closure import (
    build_stage_o_closure_decision,
    stage_o_closure_identity_sha256,
)
from pssolver.experimental.stage_p_plan import build_stage_p_h100_plan


PROJECT_ROOT = Path(__file__).parents[1]
COMMIT = "a" * 40


def test_stage_o_closure_identity_is_stable_and_portable():
    decision = build_stage_o_closure_decision(PROJECT_ROOT)
    first = stage_o_closure_identity_sha256(decision)
    second = stage_o_closure_identity_sha256(decision)
    assert first == second
    assert len(first) == 64
    assert all(character in "0123456789abcdef" for character in first)


def test_stage_p_plan_is_balanced_bounded_and_diagnostic_only(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    control = tmp_path / "control"
    plan = build_stage_p_h100_plan(
        project_root=PROJECT_ROOT,
        control_root=control,
        python=sys.executable,
        r320_production_reference_dir=reference,
        expected_commit=COMMIT,
    )
    assert plan["planning_only"] is True
    assert plan["classification"] == "PLANNING_COMPLETE"
    assert plan["diagnostic_contract"]["shape"] == [320, 320, 80]
    assert plan["diagnostic_contract"]["measurement_windows_are_separate"] is True
    assert plan["diagnostic_contract"]["raw_profiler_trace_retained"] is False
    assert plan["diagnostic_contract"]["large_simulation_arrays_written"] is False
    assert plan["authorizations"] == {
        "single_h100_diagnostic_job": True,
        "stage_q_target_review": True,
        "stage_q_candidate_implementation": False,
        "production_promotion": False,
        "default_change": False,
        "benchmark_or_long_run": False,
    }
    commands = plan["commands"]["profiles_balanced"]
    assert len(commands) == 6
    assert [item["runtime_role"] for item in commands] == [
        "legacy_production",
        "separated_canary",
        "separated_canary",
        "legacy_production",
        "legacy_production",
        "separated_canary",
    ]
    outputs = [
        item["argv"][item["argv"].index("--output") + 1]
        for item in commands
    ]
    assert len(set(outputs)) == 6
    assert not control.exists()


def test_stage_p_plan_rejects_existing_output_and_bad_commit(tmp_path):
    reference = tmp_path / "reference"
    reference.mkdir()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        build_stage_p_h100_plan(
            project_root=PROJECT_ROOT,
            control_root=existing,
            python=sys.executable,
            r320_production_reference_dir=reference,
            expected_commit=COMMIT,
        )
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        build_stage_p_h100_plan(
            project_root=PROJECT_ROOT,
            control_root=tmp_path / "new",
            python=sys.executable,
            r320_production_reference_dir=reference,
            expected_commit="short",
        )


def test_stage_p_remains_outside_production_imports_and_retains_no_trace_api():
    assert not hasattr(pssolver, "build_stage_p_h100_plan")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "stage_p" not in source
        assert "Stage P" not in source
    profiler = (PROJECT_ROOT / "benchmarks/profile_plane_stage_p.py").read_text(
        encoding="utf-8"
    )
    assert "export_chrome_trace" not in profiler
    assert "_throughput_window" in profiler
    assert "_operator_kernel_window" in profiler
    assert "_semantic_window" in profiler
