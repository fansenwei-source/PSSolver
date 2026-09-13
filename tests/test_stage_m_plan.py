"""Safety tests for the read-only Stage M H100 command plan."""

from __future__ import annotations

from pathlib import Path

import pytest

from pssolver.experimental.stage_m_plan import build_stage_m_h100_plan


PROJECT_ROOT = Path(__file__).parents[1]
PYTHON = Path(__import__("sys").executable)
COMMIT = "a" * 40


def test_stage_m_plan_is_bounded_explicit_and_read_only(tmp_path):
    control = tmp_path / "control"
    scratch = tmp_path / "scratch"

    plan = build_stage_m_h100_plan(
        project_root=PROJECT_ROOT,
        control_root=control,
        scratch_root=scratch,
        python=PYTHON,
        expected_commit=COMMIT,
    )

    assert plan["planning_only"] is True
    assert plan["long_simulation_authorized"] is False
    assert plan["default_promotion_authorized"] is False
    assert plan["fixed_numerical_gate"]["steps"] == 6
    assert plan["fixed_numerical_gate"]["relative_l2_tolerance"] == 1e-10
    assert len(plan["commands"]["profiles_balanced"]) == 6
    assert [
        value["implementation"]
        for value in plan["commands"]["profiles_balanced"]
    ] == [
        "production",
        "shadow",
        "shadow",
        "production",
        "production",
        "shadow",
    ]
    assert not control.exists()
    assert not scratch.exists()


def test_stage_m_plan_rejects_existing_or_overlapping_roots(tmp_path):
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        build_stage_m_h100_plan(
            project_root=PROJECT_ROOT,
            control_root=existing,
            scratch_root=tmp_path / "scratch",
            python=PYTHON,
            expected_commit=COMMIT,
        )

    with pytest.raises(ValueError, match="independent"):
        build_stage_m_h100_plan(
            project_root=PROJECT_ROOT,
            control_root=tmp_path / "new",
            scratch_root=tmp_path / "new" / "scratch",
            python=PYTHON,
            expected_commit=COMMIT,
        )
