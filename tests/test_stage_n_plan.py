"""Safety tests for read-only Stage N H100 diagnostic planning."""

from __future__ import annotations

import json

import pytest

from pssolver.experimental.stage_n_plan import build_stage_n_h100_plan


def test_stage_n_plan_is_bounded_read_only_and_non_promoting(tmp_path):
    project = tmp_path / "project"
    production = tmp_path / "production"
    project.mkdir()
    production.mkdir()
    python = tmp_path / "python"
    python.write_text("", encoding="utf-8")
    control = tmp_path / "control"

    plan = build_stage_n_h100_plan(
        project_root=project,
        production_reference=production,
        control_root=control,
        python=python,
        expected_commit="a" * 40,
    )

    assert plan["planning_only"] is True
    assert plan["architecture_decision"] == (
        "retain_shadow_without_promotion"
    )
    assert plan["production_path_changed"] is False
    assert plan["production_promotion_authorized"] is False
    assert plan["optimization_authorized"] is False
    assert plan["fixed_diagnostic_gate"] == {
        "diagnostic_trials": 3,
        "warmup_steps": 10,
        "profile_steps": 20,
        "operator_audit_steps": 2,
        "semantic_timing_and_operator_audit_are_separate": True,
    }
    commands = plan["commands"]["diagnostic_profiles"]
    assert len(commands) == 3
    assert [command["trial"] for command in commands] == [1, 2, 3]
    assert all("--operator-audit-steps" in command["argv"] for command in commands)
    assert plan["commands"]["analysis"].count("--profile") == 3
    assert not control.exists()
    json.dumps(plan, allow_nan=False, sort_keys=True)


def test_stage_n_plan_rejects_reuse_and_invalid_commit(tmp_path):
    project = tmp_path / "project"
    production = tmp_path / "production"
    control = tmp_path / "control"
    python = tmp_path / "python"
    for directory in (project, production, control):
        directory.mkdir()
    python.write_text("", encoding="utf-8")

    with pytest.raises(FileExistsError, match="control root"):
        build_stage_n_h100_plan(
            project_root=project,
            production_reference=production,
            control_root=control,
            python=python,
            expected_commit="a" * 40,
        )
    control.rmdir()
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        build_stage_n_h100_plan(
            project_root=project,
            production_reference=production,
            control_root=control,
            python=python,
            expected_commit="short",
        )
