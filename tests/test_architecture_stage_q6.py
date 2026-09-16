"""CPU contracts for the Stage Q.6 read-only data-movement map."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pssolver.experimental._shadow_support import file_sha256
from pssolver.experimental.stage_q6_diagnostics import (
    STAGE_Q6_ARCHITECTURE_DECISION,
    STAGE_Q6_CLASSIFICATION,
    STAGE_Q6_REQUIRED_DISTINCTIONS,
    analysis_main,
    analyze_stage_q6_data_movement_map,
)


_SOURCES = (
    "algebraic.nematic_stress.dependencies",
    "explicit_rhs.dependencies",
    "explicit_rhs.outputs",
)


def _write(path: Path, report: dict[str, object]) -> tuple[Path, str]:
    path.write_text(json.dumps(report), encoding="utf-8")
    return path, file_sha256(path)


def _reports(tmp_path: Path) -> dict[str, tuple[Path, str]]:
    rows = [
        {
            "source": source,
            "mean_milliseconds_per_step": milliseconds,
            "mean_fraction_of_timestep": fraction,
            "copy_cat_batches_per_step": batches,
            "copy_cat_components_per_step": components,
            "copy_cat_materialized_output_bytes_per_step": bytes_per_step,
            "contiguous_view_batches_per_step": 0.0,
            "fallback_reasons": {"policy_copy_cat": int(batches)},
        }
        for source, milliseconds, fraction, batches, components, bytes_per_step in (
            (_SOURCES[0], 1.46, 0.0199, 4.0, 23.0, 1_300_000.0),
            (_SOURCES[1], 0.70, 0.0096, 3.0, 15.0, 900_000.0),
            (_SOURCES[2], 0.36, 0.0049, 2.0, 5.0, 500_000.0),
        )
    ]
    o434_path, o434_sha = _write(
        tmp_path / "o434.json",
        {
            "qualification_stage": "O.4.3.4",
            "classification": "DIAGNOSTIC_COMPLETE",
            "architecture_decision": "remaining_materialization_attribution",
            "remaining_copy_sources": rows,
            "dominant_source_meets_screening_signal": False,
            "eligible_for_target_selection_review": True,
            "eligible_for_new_optimization_candidate": False,
            "eligible_for_production_promotion": False,
            "production_default_changed": False,
        },
    )
    q5_path, q5_sha = _write(
        tmp_path / "q5.json",
        {
            "qualification_stage": "Q.5",
            "classification": "POST_Q4_RETARGETING_COMPLETE",
            "primary_target": (
                "end_to_end_algebraic_data_movement_and_consumer_layout"
            ),
            "stage_q6_scope": {
                "action": "define_one_read_only_end_to_end_data_movement_map",
                "must_cover_sources": list(_SOURCES),
                "must_distinguish": list(STAGE_Q6_REQUIRED_DISTINCTIONS),
                "may_reuse_existing_profiles": True,
                "may_execute_solver": False,
                "may_implement_candidate": False,
                "may_change_production_default": False,
            },
            "eligible_for_stage_q6_diagnostic_design": True,
            "eligible_for_stage_q6_candidate_implementation": False,
            "eligible_for_production_promotion": False,
            "production_default_changed": False,
            "inputs": [
                {
                    "kind": "stage_o434",
                    "path": str(o434_path),
                    "sha256": o434_sha,
                }
            ],
        },
    )
    return {"q5": (q5_path, q5_sha), "o434": (o434_path, o434_sha)}


def _analyze(inputs: dict[str, tuple[Path, str]]) -> dict[str, object]:
    return analyze_stage_q6_data_movement_map(
        stage_q5_report=inputs["q5"][0],
        expected_stage_q5_sha256=inputs["q5"][1],
        stage_o434_report=inputs["o434"][0],
        expected_stage_o434_sha256=inputs["o434"][1],
    )


def test_stage_q6_maps_every_frozen_source_and_authorizes_design_only(tmp_path):
    report = _analyze(_reports(tmp_path))

    assert report["qualification_stage"] == "Q.6"
    assert report["classification"] == STAGE_Q6_CLASSIFICATION
    assert report["architecture_decision"] == STAGE_Q6_ARCHITECTURE_DECISION
    assert [row["source"] for row in report["data_movement_map"]] == list(
        _SOURCES
    )
    assert report["aggregate_measured_evidence"]["source_count"] == 3
    assert report["aggregate_measured_evidence"][
        "mean_fraction_of_timestep"
    ] == pytest.approx(0.0344)
    assert report["cross_source_findings"][
        "copy_cat_is_present_at_every_remaining_source"
    ] is True
    assert report["eligible_for_stage_q61_design"] is True
    assert report["eligible_for_stage_q61_candidate_implementation"] is False
    assert report["eligible_for_production_promotion"] is False
    assert report["production_default_changed"] is False
    assert report["changes_runtime_implementation"] is False


def test_stage_q6_distinguishes_every_required_mechanism(tmp_path):
    report = _analyze(_reports(tmp_path))

    assert set(report["required_distinctions"]) == set(
        STAGE_Q6_REQUIRED_DISTINCTIONS
    )
    for row in report["data_movement_map"]:
        assert set(STAGE_Q6_REQUIRED_DISTINCTIONS).issubset(row)
        assert row["tensor_copy"]["present"] is True
        assert row["view_or_alias"]["zero_copy_currently_exercised"] is False
        assert row["operator_launch_fragmentation"][
            "timing_is_nonadditive"
        ] is True


def test_stage_q6_rejects_changed_input_identity(tmp_path):
    inputs = _reports(tmp_path)
    inputs["q5"][0].write_text(
        inputs["q5"][0].read_text(encoding="utf-8") + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="SHA-256 differs"):
        _analyze(inputs)


def test_stage_q6_rejects_unmapped_or_missing_source(tmp_path):
    inputs = _reports(tmp_path)
    q5_path, _ = inputs["q5"]
    q5 = json.loads(q5_path.read_text(encoding="utf-8"))
    q5["stage_q6_scope"]["must_cover_sources"] = list(_SOURCES[:-1])
    inputs["q5"] = _write(q5_path, q5)

    with pytest.raises(ValueError, match="not fully mapped"):
        _analyze(inputs)


def test_stage_q6_rejects_candidate_or_solver_authorization(tmp_path):
    inputs = _reports(tmp_path)
    q5_path, _ = inputs["q5"]
    q5 = json.loads(q5_path.read_text(encoding="utf-8"))
    q5["stage_q6_scope"]["may_execute_solver"] = True
    inputs["q5"] = _write(q5_path, q5)

    with pytest.raises(ValueError, match="Q.5 evidence contract differs"):
        _analyze(inputs)


def test_stage_q6_cli_writes_once(tmp_path):
    inputs = _reports(tmp_path)
    output = tmp_path / "q6.json"
    argv = [
        "--stage-q5-report",
        str(inputs["q5"][0]),
        "--expected-stage-q5-sha256",
        inputs["q5"][1],
        "--stage-o434-report",
        str(inputs["o434"][0]),
        "--expected-stage-o434-sha256",
        inputs["o434"][1],
        "--output",
        str(output),
    ]

    assert analysis_main(argv) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["classification"] == (
        STAGE_Q6_CLASSIFICATION
    )
    with pytest.raises(FileExistsError):
        analysis_main(argv)
