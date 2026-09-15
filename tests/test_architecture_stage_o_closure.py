"""Contracts for closing the Plane Stage O migration study."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
from pathlib import Path

import pytest

import pssolver
from pssolver.experimental import (
    StageOClosureDecision,
    StageOPathDisposition,
    build_stage_o_closure_decision,
)
from pssolver.experimental.stage_o_closure import main


PROJECT_ROOT = Path(__file__).parents[1]


def _metadata() -> dict[str, object]:
    return build_stage_o_closure_decision(PROJECT_ROOT).to_metadata()


def test_stage_o_closure_retains_production_and_the_experimental_oracle():
    metadata = _metadata()
    assert metadata["classification"] == "CLOSED_RETAIN_LEGACY_PRODUCTION"
    assert metadata["path_dispositions"]["legacy_production"] == (
        StageOPathDisposition.RETAIN_PRODUCTION.value
    )
    assert metadata["path_dispositions"]["separated_canary"] == (
        StageOPathDisposition.RETAIN_EXPERIMENTAL_ORACLE.value
    )
    assert metadata["production_conclusion"] == {
        "default_runtime": "legacy_production",
        "separated_canary_is_production_replacement": False,
        "stage_o5_entered": False,
        "stage_o44_authorized": False,
        "production_default_changed": False,
        "channel_changed": False,
        "generic_solver_changed": False,
    }


def test_stage_o_closure_preserves_complete_authoritative_evidence_lineage():
    evidence = _metadata()["evidence"]
    assert [item["stage"] for item in evidence] == [
        "O.4",
        "O.4.1",
        "O.4.2.7",
        "O.4.3",
        "O.4.3.1",
        "O.4.3.3",
        "O.4.3.4",
    ]
    assert [item["classification"] for item in evidence] == [
        "B_neutral",
        "B_neutral",
        "DIAGNOSTIC_COMPLETE",
        "C_rejected",
        "B_neutral",
        "B_neutral",
        "DIAGNOSTIC_COMPLETE",
    ]
    assert all(len(item["commit"]) == 40 for item in evidence)
    assert all(len(item["report_sha256"]) == 64 for item in evidence)


def test_stage_o_closure_does_not_turn_science_into_a_promotion_claim():
    metadata = _metadata()
    assert metadata["scientific_conclusion"] == {
        "separated_equations_and_timestep_are_qualified": True,
        "six_and_one_hundred_step_equivalence_passed": True,
        "same_backend_restart_passed": True,
        "initial_q_identity_passed": True,
        "scientific_failure_observed": False,
    }
    authorizations = metadata["authorizations"]
    assert authorizations["stage_o_closed"] is True
    assert authorizations["stage_p_design"] is True
    assert all(
        authorizations[name] is False
        for name in (
            "stage_p_implementation",
            "stage_p_h100_execution",
            "stage_o44",
            "stage_o5",
            "production_promotion",
            "default_change",
            "benchmark_run",
        )
    )


def test_materialization_line_is_closed_below_the_frozen_screening_signal():
    closure = _metadata()["materialization_line_closure"]
    assert closure["closed"] is True
    assert closure["dominant_remaining_source"] == (
        "algebraic.nematic_stress.dependencies"
    )
    assert closure["dominant_fraction_of_timestep"] == pytest.approx(
        0.01997209883
    )
    assert closure["dominant_fraction_of_timestep"] < closure[
        "minimum_screening_fraction"
    ]
    assert closure["new_layout_candidate_authorized"] is False


def test_stage_p_boundary_is_diagnostic_only_and_excludes_more_layout_work():
    boundary = _metadata()["next_stage_boundary"]
    assert boundary["stage"] == "P"
    assert boundary["optimization_authorized"] is False
    assert boundary["h100_execution_authorized"] is False
    assert boundary["materialization_layout_work_reopened"] is False
    assert boundary["focus"] == [
        "transform_scheduling",
        "kernel_launch_structure",
        "compiled_graph_boundaries",
        "repeated_spectral_operations",
    ]


def test_closure_is_immutable_json_and_hashes_its_architecture_boundary():
    decision = build_stage_o_closure_decision(PROJECT_ROOT)
    assert isinstance(decision, StageOClosureDecision)
    metadata = decision.to_metadata()
    json.dumps(metadata, allow_nan=False, sort_keys=True)
    assert set(metadata["implementation_file_sha256"]) == {
        "pssolver/configuration/plane_beris_edwards.py",
        "pssolver/runtime/plane_beris_edwards.py",
        "pssolver/workflows/plane_beris_edwards.py",
        "pssolver/experimental/model_execution.py",
        "pssolver/experimental/projected_scheduler.py",
    }
    assert all(
        len(digest) == 64
        for digest in metadata["implementation_file_sha256"].values()
    )
    with pytest.raises(FrozenInstanceError):
        decision.project_root = "/tmp/changed"


def test_closure_cli_prints_the_same_read_only_decision(capsys):
    assert main(["--project-root", str(PROJECT_ROOT)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output == _metadata()


def test_stage_o_closure_remains_outside_production_imports():
    assert not hasattr(pssolver, "StageOClosureDecision")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "stage_o_closure" not in source
        assert "StageOClosureDecision" not in source
