"""Synthetic tests for Stage O.4.3.4 profile aggregation and planning."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from pssolver.experimental.stage_o434_diagnostics import (
    analyze_stage_o434_diagnostics,
    summarize_projected_batch_source_attribution,
)
from pssolver.experimental.stage_o434_plan import build_stage_o434_h100_plan


PROJECT_ROOT = Path(__file__).parents[1]


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evidence(path: Path) -> str:
    _write_json(
        path,
        {
            "qualification_stage": "O.4.3.3",
            "classification": "B_neutral",
            "architecture_decision": (
                "producer_owned_boundary_packed_h_and_stress"
            ),
            "eligible_for_stage_o44_decision": False,
            "eligible_for_production_promotion": False,
            "production_default_changed": False,
        },
    )
    return _sha256(path)


def _source(
    *,
    milliseconds: float,
    copy_batches: float,
    copy_components: float,
    copy_bytes: float,
    view_batches: float = 0.0,
    view_components: float = 0.0,
) -> dict[str, object]:
    return {
        "timing_region": "projected_batch_assembly.source.synthetic",
        "timing_calls": 20,
        "milliseconds_per_profile_step": milliseconds,
        "peak_allocated_bytes_observed": 100,
        "peak_reserved_bytes_observed": 200,
        "singleton_batches_per_profile_step": 0.0,
        "singleton_components_per_profile_step": 0.0,
        "singleton_logical_input_bytes_per_profile_step": 0.0,
        "copy_cat_batches_per_profile_step": copy_batches,
        "copy_cat_components_per_profile_step": copy_components,
        "copy_cat_logical_input_bytes_per_profile_step": copy_bytes,
        "copy_cat_materialized_output_bytes_per_profile_step": copy_bytes,
        "contiguous_view_batches_per_profile_step": view_batches,
        "contiguous_view_components_per_profile_step": view_components,
        "contiguous_view_logical_input_bytes_per_profile_step": 0.0,
        "contiguous_view_materialized_output_bytes_per_profile_step": 0.0,
        "fallback_reasons": (
            {"distinct_storage": int(copy_batches * 20)}
            if copy_batches
            else {}
        ),
        "retained_tensor_references": 0,
        "copy_materialized_byte_fraction": 0.0,
        "assembly_time_fraction": 0.0,
    }


def _profile(path: Path, *, trial: int) -> None:
    _write_json(
        path,
        {
            "qualification_stage": "O.4.3.4",
            "classification": "ATTRIBUTION_PROFILE_COMPLETE",
            "measurement_role": "candidate_attribution",
            "production_default_changed": False,
            "eligible_for_production_promotion": False,
            "production_metadata_sha256": "a" * 64,
            "production_initial_q_sha256": "b" * 64,
            "configuration": {
                "shape": [320, 320, 80],
                "lengths": [100.0, 100.0, 20.0],
                "dtype": "torch.float64",
                "dt": 0.005,
                "dealias_rule": "cubic_half",
                "projected_transform_execution": "truncated",
                "spectral_storage": "hermitian_half",
                "spectral_refresh_interval": None,
                "warmup_steps": 10,
                "profile_steps": 20,
            },
            "throughput": {
                "mean_timestep_seconds": 0.05 + trial * 0.0001,
            },
            "finite": True,
            "source_attribution": {
                "schema_version": 1,
                "profile_steps": 20,
                "sources": {
                    "algebraic.nematic_stress.dependencies": _source(
                        milliseconds=2.0 + trial * 0.1,
                        copy_batches=2.0,
                        copy_components=20.0,
                        copy_bytes=1000.0,
                    ),
                    "explicit_rhs.outputs": _source(
                        milliseconds=0.5 + trial * 0.01,
                        copy_batches=1.0,
                        copy_components=5.0,
                        copy_bytes=400.0,
                    ),
                    "algebraic.nematic_stress.outputs": _source(
                        milliseconds=0.2,
                        copy_batches=0.0,
                        copy_components=0.0,
                        copy_bytes=0.0,
                        view_batches=2.0,
                        view_components=18.0,
                    ),
                },
                "total_copy_cat_materialized_output_bytes_per_step": 1400.0,
                "total_measured_assembly_milliseconds_per_step": 2.7,
                "unattributed_batches_per_step": 0.0,
                "all_sources_attributed": True,
                "retained_tensor_references": 0,
            },
        },
    )


def test_raw_profile_source_counters_are_normalized_per_step():
    source = {
        "singleton_batches": 0,
        "singleton_components": 0,
        "singleton_logical_input_bytes": 0,
        "copy_cat_batches": 4,
        "copy_cat_components": 20,
        "copy_cat_logical_input_bytes": 1600,
        "copy_cat_materialized_output_bytes": 1600,
        "contiguous_view_batches": 2,
        "contiguous_view_components": 10,
        "contiguous_view_logical_input_bytes": 800,
        "contiguous_view_materialized_output_bytes": 0,
        "fallback_reasons": {"distinct_storage": 4},
        "retained_tensor_references": 0,
        "timing_region": "projected_batch_assembly.source.explicit_rhs.outputs",
    }
    report = summarize_projected_batch_source_attribution(
        {
            "configuration": {"profile_steps": 2},
            "batch_assembly_diagnostics": {
                "schema_version": 2,
                "source_attribution": {"explicit_rhs.outputs": source},
            },
            "semantic_regions": {
                source["timing_region"]: {
                    "calls": 6,
                    "total_seconds": 0.004,
                    "peak_allocated_bytes_observed": 1024,
                    "peak_reserved_bytes_observed": 2048,
                }
            },
        }
    )
    normalized = report["sources"]["explicit_rhs.outputs"]
    assert normalized["copy_cat_batches_per_profile_step"] == 2.0
    assert normalized["copy_cat_components_per_profile_step"] == 10.0
    assert normalized[
        "copy_cat_materialized_output_bytes_per_profile_step"
    ] == 800.0
    assert normalized["contiguous_view_batches_per_profile_step"] == 1.0
    assert normalized["milliseconds_per_profile_step"] == 2.0
    assert report["all_sources_attributed"] is True


def test_analysis_ranks_remaining_copy_sources_without_authorizing_candidate(
    tmp_path: Path,
):
    evidence = tmp_path / "o433.json"
    evidence_sha = _evidence(evidence)
    profiles = []
    for trial in range(1, 4):
        path = tmp_path / f"profile_{trial}.json"
        _profile(path, trial=trial)
        profiles.append(path)

    report = analyze_stage_o434_diagnostics(
        profiles,
        stage_o433_report=evidence,
        expected_stage_o433_sha256=evidence_sha,
    )
    assert report["classification"] == "DIAGNOSTIC_COMPLETE"
    assert report["stage_o433_evidence"]["classification_preserved"] == (
        "B_neutral"
    )
    assert report["dominant_remaining_copy_source"]["source"] == (
        "algebraic.nematic_stress.dependencies"
    )
    assert report["dominant_source_meets_screening_signal"] is True
    assert report["eligible_for_target_selection_review"] is True
    assert report["eligible_for_new_optimization_candidate"] is False
    assert report["eligible_for_stage_o44_decision"] is False
    assert report["eligible_for_production_promotion"] is False
    assert json.dumps(report, allow_nan=False, sort_keys=True)


def test_analysis_rejects_unattributed_or_unstable_profiles(tmp_path: Path):
    evidence = tmp_path / "o433.json"
    evidence_sha = _evidence(evidence)
    profiles = []
    for trial in range(1, 4):
        path = tmp_path / f"profile_{trial}.json"
        _profile(path, trial=trial)
        profiles.append(path)
    value = json.loads(profiles[0].read_text(encoding="utf-8"))
    value["source_attribution"]["all_sources_attributed"] = False
    _write_json(profiles[0], value)
    with pytest.raises(ValueError, match="profile is incomplete"):
        analyze_stage_o434_diagnostics(
            profiles,
            stage_o433_report=evidence,
            expected_stage_o433_sha256=evidence_sha,
        )

    _profile(profiles[0], trial=1)
    value = json.loads(profiles[2].read_text(encoding="utf-8"))
    value["source_attribution"]["sources"]["explicit_rhs.outputs"][
        "copy_cat_batches_per_profile_step"
    ] = 2.0
    _write_json(profiles[2], value)
    with pytest.raises(ValueError, match="unstable"):
        analyze_stage_o434_diagnostics(
            profiles,
            stage_o433_report=evidence,
            expected_stage_o433_sha256=evidence_sha,
        )


def test_planner_is_r320_diagnostic_only(tmp_path: Path):
    control = tmp_path / "control"
    reference = tmp_path / "reference"
    reference.mkdir()
    evidence = tmp_path / "o433.json"
    evidence_sha = _evidence(evidence)
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        text=True,
    ).strip()
    plan = build_stage_o434_h100_plan(
        project_root=PROJECT_ROOT,
        control_root=control,
        python=sys.executable,
        r320_production_reference_dir=reference,
        expected_commit=commit,
        stage_o433_report=evidence,
        expected_stage_o433_sha256=evidence_sha,
    )
    assert plan["planning_only"] is True
    assert plan["qualification_stage"] == "O.4.3.4"
    assert len(plan["commands"]["profiles"]) == 3
    assert plan["diagnostic_contract"]["shape"] == [320, 320, 80]
    assert plan["diagnostic_contract"]["changes_tensor_layout"] is False
    assert plan["diagnostic_contract"]["production_default_may_change"] is False
    assert "trajectories" not in plan["commands"]
    assert not control.exists()
