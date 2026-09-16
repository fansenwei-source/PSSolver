"""CPU contracts for Stage Q.3 algebraic target refinement."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pssolver
from pssolver.experimental._shadow_support import file_sha256
from pssolver.experimental.stage_q3_diagnostics import (
    analysis_main,
    analyze_stage_q3_algebraic_target,
)


PROJECT_ROOT = Path(__file__).parents[1]


def _timing(total_ms: float, calls: int = 3) -> dict[str, object]:
    return {
        "calls": calls,
        "mean_seconds": total_ms / 1000.0 / calls,
        "total_seconds": total_ms / 1000.0,
        "timed": True,
    }


def _candidate_profile(trial: int) -> dict[str, object]:
    regions = {
        "timestep.algebraic_update": _timing(126.0),
        "timestep.explicit_rhs": _timing(23.0),
        "timestep.spectral_update": _timing(7.5),
        "timestep.dynamic_inverse": _timing(6.9),
        "timestep.spectral_refresh": _timing(4.5),
        "timestep.total": _timing(168.0),
        "transform.forward": _timing(18.0, 21),
        "transform.inverse": _timing(45.0, 96),
        "algebraic.molecular_field.solve": _timing(24.0),
        "algebraic.q_gradient.solve": _timing(18.0),
        "algebraic.velocity_gradient.solve": _timing(12.0),
        "algebraic.nematic_stress.solve": _timing(33.0),
        "algebraic.nematic_force.solve": _timing(21.0),
        "algebraic.flow.solve": _timing(9.0),
        "algebraic.publish_spectral": _timing(3.0),
        (
            "projected_batch_assembly.source."
            "algebraic.molecular_field.outputs"
        ): _timing(6.0),
        (
            "projected_batch_assembly.source."
            "algebraic.nematic_stress.dependencies"
        ): _timing(12.0),
        (
            "projected_batch_assembly.source."
            "algebraic.nematic_stress.outputs"
        ): _timing(9.0),
    }
    return {
        "schema_version": 1,
        "qualification_stage": "Q.1",
        "classification": "OPERATOR_KERNEL_PROFILE_COMPLETE",
        "runtime_role": "separated_canary",
        "runtime_variant": f"compiled_rhs_{trial}",
        "finite": True,
        "production_default_changed": False,
        "eligible_for_production_promotion": False,
        "production_metadata_sha256": "1" * 64,
        "production_initial_q_sha256": "2" * 64,
        "stage_o_closure_sha256": "3" * 64,
        "configuration": {
            "shape": [320, 320, 80],
            "lengths": [100.0, 100.0, 20.0],
            "dtype": "float64",
            "dt": 0.005,
            "dealias_rule": "cubic_half",
            "spectral_storage": "hermitian_half",
            "throughput_steps": 20,
            "operator_audit_steps": 3,
            "semantic_steps": 3,
        },
        "throughput": {
            "instrumentation": "outer_cuda_events_only",
            "mean_timestep_seconds": 0.055 + trial * 1.0e-5,
        },
        "matched_semantic_regions": {
            "semantic_steps": 3,
            "nested_regions_overlap": True,
            "separate_from_authoritative_throughput": True,
            "total_milliseconds_per_step": 56.0,
            "algebraic_milliseconds_per_step": 42.0,
            "explicit_rhs_milliseconds_per_step": 7.7,
            "spectral_update_milliseconds_per_step": 2.5,
            "dynamic_inverse_milliseconds_per_step": 2.3,
            "spectral_refresh_milliseconds_per_step": 1.5,
            "transform_milliseconds_per_step": 21.0,
        },
        "raw_semantic_regions": {
            "schema_version": 1,
            "enabled": True,
            "device": "cuda:0",
            "timing_backend": "cuda_events",
            "nested_regions_overlap": True,
            "regions": regions,
        },
        "operator_kernel_audit": {
            "bounded_aggregate_only": True,
            "raw_trace_retained": False,
            "operator_entries_complete": True,
            "kernel_entries_complete": True,
            "operators": {"entries": []},
            "kernels": {"entries": []},
            "category_totals": {},
        },
        "measurement_contract": {
            "throughput_operator_and_semantic_windows_are_separate": True,
            "throughput_uses_internal_semantic_instrumentation": False,
            "raw_trace_retained": False,
            "changes_equations": False,
            "changes_production_default": False,
        },
        "dynamo_delta": {"graph_breaks": 0},
    }


def _write_inputs(tmp_path: Path, *, include_cat: bool = True):
    paths = []
    for trial in range(1, 4):
        path = tmp_path / f"candidate_{trial}.json"
        path.write_text(json.dumps(_candidate_profile(trial)), encoding="utf-8")
        paths.append(path)
    positive_operators = []
    positive_kernels = []
    if include_cat:
        positive_operators.append(
            {
                "name": "aten::cat",
                "device_microseconds_delta_per_step": 3700.0,
            }
        )
        positive_kernels.append(
            {
                "name": "CatArrayBatchedCopy_contig",
                "device_microseconds_delta_per_step": 3800.0,
            }
        )
    q2 = {
        "schema_version": 1,
        "qualification_stage": "Q.2",
        "classification": "TARGET_REVIEW_COMPLETE",
        "target_review": {
            "primary_target": "algebraic_runtime_orchestration",
            "primary_source_region": "algebraic",
            "explicit_rhs_line_closed": True,
            "transform_forward_line_closed": True,
            "projected_materialization_line_remains_closed": True,
        },
        "eligible_for_stage_q2_candidate_design": True,
        "eligible_for_stage_q2_candidate_implementation": False,
        "eligible_for_production_promotion": False,
        "production_default_changed": False,
        "nested_nonadditive_evidence": {
            "regions": [
                {
                    "region": "transform_inverse",
                    "delta_milliseconds_per_step": 0.94,
                }
            ]
        },
        "operator_kernel_evidence": {
            "largest_positive_operator_deltas": positive_operators,
            "largest_positive_kernel_deltas": positive_kernels,
        },
        "inputs": [
            {
                "kind": "profile",
                "path": str(path.resolve()),
                "sha256": file_sha256(path),
            }
            for path in paths
        ],
    }
    q2_path = tmp_path / "stage_q2.json"
    q2_path.write_text(json.dumps(q2), encoding="utf-8")
    return paths, q2_path


def test_stage_q3_selects_one_preplanned_workspace_candidate(tmp_path):
    profiles, q2 = _write_inputs(tmp_path)
    report = analyze_stage_q3_algebraic_target(
        profiles,
        stage_q2_report_path=q2,
    )

    assert report["classification"] == (
        "ALGEBRAIC_TARGET_REFINEMENT_COMPLETE"
    )
    assert report["candidate_design"]["selected_candidate"] == (
        "preplanned_algebraic_batch_workspace"
    )
    assert len(report["algebraic_solver_regions"]) == 6
    assert len(report["batch_assembly_source_regions"]) == 3
    assert report["batch_assembly_source_sum_milliseconds_per_step"] == (
        pytest.approx(9.0)
    )
    assert report["supporting_excess_signals"][
        "aten_cat_delta_milliseconds_per_step"
    ] == pytest.approx(3.7)
    assert report["eligible_for_stage_q4_candidate_implementation"] is True
    assert report["eligible_for_production_promotion"] is False
    assert report["production_default_changed"] is False


def test_stage_q3_uses_inverse_fallback_only_without_copy_signal(tmp_path):
    profiles, q2 = _write_inputs(tmp_path, include_cat=False)
    report = analyze_stage_q3_algebraic_target(
        profiles,
        stage_q2_report_path=q2,
    )
    assert report["candidate_design"]["selected_candidate"] == (
        "inverse_transform_batch_scheduling"
    )


def test_stage_q3_rejects_unfrozen_profile_and_bad_q2_gate(tmp_path):
    profiles, q2 = _write_inputs(tmp_path)
    changed = json.loads(profiles[0].read_text(encoding="utf-8"))
    changed["runtime_variant"] = "changed"
    profiles[0].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="not frozen"):
        analyze_stage_q3_algebraic_target(
            profiles,
            stage_q2_report_path=q2,
        )

    profiles, q2 = _write_inputs(tmp_path)
    q2_data = json.loads(q2.read_text(encoding="utf-8"))
    q2_data["target_review"]["primary_target"] = "explicit_rhs_execution"
    q2.write_text(json.dumps(q2_data), encoding="utf-8")
    with pytest.raises(ValueError, match="does not authorize"):
        analyze_stage_q3_algebraic_target(
            profiles,
            stage_q2_report_path=q2,
        )


def test_stage_q3_cli_writes_once_and_stays_outside_production(tmp_path):
    profiles, q2 = _write_inputs(tmp_path)
    output = tmp_path / "stage_q3.json"
    argv = []
    for path in profiles:
        argv.extend(("--candidate-profile", str(path)))
    argv.extend(("--stage-q2-report", str(q2), "--output", str(output)))
    assert analysis_main(argv) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["classification"] == (
        "ALGEBRAIC_TARGET_REFINEMENT_COMPLETE"
    )
    with pytest.raises(FileExistsError):
        analysis_main(argv)

    assert not hasattr(pssolver, "analyze_stage_q3_algebraic_target")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "stage_q3" not in source
        assert "Stage Q.3" not in source
