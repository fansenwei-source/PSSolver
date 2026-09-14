"""Characterization tests for the Stage O production-migration design."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

import pssolver
from pssolver.configuration import (
    PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES,
)
from pssolver.experimental import (
    MigrationDisposition,
    PlaneRuntimePath,
    StageOMigrationDesign,
    build_stage_o_migration_design,
)
from pssolver.experimental.stage_o_migration import main


PROJECT_ROOT = Path(__file__).parents[1]
SOURCE_COMMIT = "a" * 40
BASELINE_COMMIT = "b" * 40


def _qualified_report():
    return {
        "schema_version": 1,
        "qualification_stage": "N.4",
        "classification": "A_recommended",
        "numerical_equivalence_passed": True,
        "eligible_for_stage_o_production_migration_design": True,
        "eligible_for_production_promotion": False,
        "production_path_changed": False,
        "trajectory": {
            "maximum_gate_relative_l2": 2.6e-15,
            "relative_l2_tolerance": 1.0e-10,
        },
        "comparison": {
            "mean_timestep_ratio_candidate_over_control": 0.986,
            "peak_allocated_ratio_candidate_over_control": 1.0,
            "peak_reserved_ratio_candidate_over_control": 1.0,
            "maximum_mean_timestep_ratio": 1.03,
            "maximum_memory_ratio": 1.03,
            "final_q_identity_gate": True,
            "transform_count_identity_gate": True,
            "lifecycle_identity_gate": True,
            "performance_non_regression_gate": True,
            "memory_gate": True,
        },
    }


def _write_report(tmp_path, report=None):
    path = tmp_path / "stage_n41_qualification.json"
    path.write_text(
        json.dumps(_qualified_report() if report is None else report),
        encoding="utf-8",
    )
    return path


def _design(tmp_path):
    return build_stage_o_migration_design(
        project_root=PROJECT_ROOT,
        stage_n41_qualification=_write_report(tmp_path),
        architecture_source_commit=SOURCE_COMMIT,
        production_baseline_commit=BASELINE_COMMIT,
    )


def test_stage_o_design_is_bounded_dual_path_and_non_promoting(tmp_path):
    design = _design(tmp_path)
    assert isinstance(design, StageOMigrationDesign)
    metadata = design.to_metadata()

    assert metadata["qualification_stage"] == "O"
    assert metadata["planning_only"] is True
    assert metadata["scope"]["geometry"] == "plane_slab"
    assert metadata["scope"]["excluded_geometries"] == [
        "periodic_box",
        "rectangular_channel",
    ]
    assert metadata["runtime_paths"] == {
        "default": PlaneRuntimePath.LEGACY_PRODUCTION.value,
        "opt_in_candidate": PlaneRuntimePath.SEPARATED_CANARY.value,
        "simultaneous_selection_forbidden": True,
        "selection_authority": "resolved_plane_run_spec",
        "legacy_rollback_retained": True,
    }
    assert metadata["authorizations"] == {
        "design_complete": True,
        "implementation_started": False,
        "production_path_changed": False,
        "production_default_change": False,
        "channel_migration": False,
        "generic_solver_migration": False,
        "long_simulation": False,
    }
    assert metadata["next_authorized_action"] == (
        "implement_O.1_pure_configuration_extraction"
    )


def test_stage_o_responsibilities_follow_the_spectral_first_dependency_order(
    tmp_path,
):
    metadata = _design(tmp_path).to_metadata()
    assert metadata["dependency_direction"] == [
        "model",
        "physical_boundary_conditions",
        "geometry",
        "spectral_plan",
        "backend",
        "runtime",
        "workflow",
    ]
    responsibilities = {
        value["name"]: value for value in metadata["responsibilities"]
    }
    assert responsibilities["model_equations"]["disposition"] == (
        MigrationDisposition.PRESERVE.value
    )
    assert responsibilities["runtime_assembly"]["disposition"] == (
        MigrationDisposition.DUAL_PATH.value
    )
    assert responsibilities["other_geometries"]["disposition"] == (
        MigrationDisposition.DEFER.value
    )
    assert responsibilities["checkpoint_and_restart"]["first_phase"] == "O.3"


def test_stage_o_phases_are_incremental_and_never_authorize_a_default(tmp_path):
    metadata = _design(tmp_path).to_metadata()
    phases = metadata["phases"]
    assert [phase["name"] for phase in phases] == [
        "O.1",
        "O.2",
        "O.3",
        "O.4",
        "O.5",
    ]
    assert all(
        phase["production_default_may_change"] is False for phase in phases
    )
    gates = metadata["fixed_migration_gates"]
    assert gates["legacy_characterization"][
        "short_trajectory_byte_identity"
    ] is True
    assert gates["candidate_numerics"][
        "one_hundred_step_relative_l2_tolerance"
    ] == 1.0e-10
    assert gates["h100_non_regression"][
        "maximum_mean_timestep_ratio"
    ] == 1.03
    assert gates["promotion"]["explicit_authorization_required"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        lambda report: report.update(classification="B_neutral"),
        lambda report: report.update(
            eligible_for_stage_o_production_migration_design=False
        ),
        lambda report: report["comparison"].update(
            lifecycle_identity_gate=False
        ),
        lambda report: report["comparison"].update(
            mean_timestep_ratio_candidate_over_control=1.031
        ),
        lambda report: report["trajectory"].update(
            maximum_gate_relative_l2=1.1e-10
        ),
    ),
)
def test_stage_o_rejects_unqualified_n41_evidence(tmp_path, mutation):
    report = _qualified_report()
    mutation(report)
    with pytest.raises(ValueError, match="does not authorize|not qualified"):
        build_stage_o_migration_design(
            project_root=PROJECT_ROOT,
            stage_n41_qualification=_write_report(tmp_path, report),
            architecture_source_commit=SOURCE_COMMIT,
            production_baseline_commit=BASELINE_COMMIT,
        )


def test_stage_o_design_is_immutable_json_and_records_input_hashes(tmp_path):
    design = _design(tmp_path)
    metadata = design.to_metadata()
    json.dumps(metadata, allow_nan=False, sort_keys=True)
    assert set(metadata["implementation_file_sha256"]) == {
        *PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES,
        "pssolver/channel.py",
        "pssolver/execution/policy.py",
        "pssolver/experimental/model_execution.py",
        "pssolver/experimental/projected_scheduler.py",
    }
    assert all(
        len(value) == 64
        for value in metadata["implementation_file_sha256"].values()
    )
    with pytest.raises(FrozenInstanceError):
        design.project_root = "/tmp/changed"


def test_stage_o_cli_prints_the_same_read_only_design(tmp_path, capsys):
    report = _write_report(tmp_path)
    result = main(
        [
            "--project-root",
            str(PROJECT_ROOT),
            "--stage-n41-qualification",
            str(report),
            "--architecture-source-commit",
            SOURCE_COMMIT,
            "--production-baseline-commit",
            BASELINE_COMMIT,
        ]
    )
    assert result == 0
    output = json.loads(capsys.readouterr().out)
    assert output["planning_only"] is True
    assert output["architecture_source_commit"] == SOURCE_COMMIT
    assert output["qualification_evidence"]["path"] == str(report.resolve())


def test_stage_o_remains_outside_all_current_production_paths():
    assert not hasattr(pssolver, "StageOMigrationDesign")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "stage_o_migration" not in text
        assert "StageOMigrationDesign" not in text
        assert "pssolver.experimental" not in text
