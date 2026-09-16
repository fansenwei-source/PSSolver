"""CPU contracts for Stage Q.2 residual-gap attribution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import pssolver
from pssolver.experimental.stage_q2_diagnostics import (
    analysis_main,
    analyze_stage_q2_residual_gap,
)


PROJECT_ROOT = Path(__file__).parents[1]


def _region(total_seconds: float, calls: int = 3) -> dict[str, object]:
    return {
        "calls": calls,
        "mean_seconds": total_seconds / calls,
        "timed": True,
        "total_seconds": total_seconds,
    }


def _profile(
    *,
    role: str,
    mean_ms: float,
    algebraic_ms: float,
    explicit_rhs_ms: float,
    inverse_ms: float,
    trial: int,
) -> dict[str, object]:
    forward_ms = 6.0
    spectral_update_ms = 2.0
    dynamic_inverse_ms = 1.5
    spectral_refresh_ms = 0.5
    total_ms = (
        algebraic_ms
        + explicit_rhs_ms
        + spectral_update_ms
        + dynamic_inverse_ms
        + spectral_refresh_ms
    )
    direct = {
        "transform_forward": _region(3.0 * forward_ms / 1000.0, 21),
        "transform_inverse": _region(3.0 * inverse_ms / 1000.0, 96),
        "static_fields": _region(3.0 * algebraic_ms / 1000.0),
        "q_nonlinear": _region(3.0 * explicit_rhs_ms / 1000.0),
    }
    wrapped = {
        "schema_version": 1,
        "enabled": True,
        "device": "cuda:0",
        "timing_backend": "cuda_events",
        "nested_regions_overlap": True,
        "regions": {
            "transform.forward": _region(3.0 * forward_ms / 1000.0, 21),
            "transform.inverse": _region(3.0 * inverse_ms / 1000.0, 96),
            "timestep.algebraic_update": _region(
                3.0 * algebraic_ms / 1000.0
            ),
            "timestep.explicit_rhs": _region(
                3.0 * explicit_rhs_ms / 1000.0
            ),
        },
    }
    kernel_calls = 500.0 if role == "legacy_production" else 300.0
    category_time = 8_000.0 if role == "legacy_production" else 9_000.0
    return {
        "schema_version": 1,
        "qualification_stage": "Q.1",
        "classification": "OPERATOR_KERNEL_PROFILE_COMPLETE",
        "runtime_role": role,
        "runtime_variant": f"{role}_{trial}",
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
            "mean_timestep_seconds": mean_ms / 1000.0,
            "median_timestep_seconds": mean_ms / 1000.0,
            "sample_count": 20,
            "sample_std_seconds": 0.0001,
            "timesteps_per_second": 1000.0 / mean_ms,
            "total_seconds": 20.0 * mean_ms / 1000.0,
            "peak_allocated_bytes": 100,
            "peak_reserved_bytes": 200,
        },
        "matched_semantic_regions": {
            "semantic_steps": 3,
            "nested_regions_overlap": True,
            "separate_from_authoritative_throughput": True,
            "total_milliseconds_per_step": total_ms,
            "algebraic_milliseconds_per_step": algebraic_ms,
            "explicit_rhs_milliseconds_per_step": explicit_rhs_ms,
            "spectral_update_milliseconds_per_step": spectral_update_ms,
            "dynamic_inverse_milliseconds_per_step": dynamic_inverse_ms,
            "spectral_refresh_milliseconds_per_step": spectral_refresh_ms,
            "transform_milliseconds_per_step": forward_ms + inverse_ms,
        },
        "raw_semantic_regions": direct if role == "legacy_production" else wrapped,
        "operator_kernel_audit": {
            "bounded_aggregate_only": True,
            "raw_trace_retained": False,
            "operator_entries_complete": True,
            "kernel_entries_complete": True,
            "operators": {
                "entries": [
                    {
                        "name": "aten::mul",
                        "category": "other",
                        "calls_per_step": kernel_calls,
                        "device_microseconds_per_step": category_time,
                    }
                ]
            },
            "kernels": {
                "entries": [
                    {
                        "name": "elementwise",
                        "category": "other",
                        "calls_per_step": kernel_calls,
                        "device_microseconds_per_step": category_time,
                    }
                ]
            },
            "category_totals": {
                "other": {
                    "operator_calls_per_step": kernel_calls,
                    "kernel_calls_per_step": kernel_calls,
                    "kernel_device_microseconds_per_step": category_time,
                }
            },
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


def _write_profiles(tmp_path: Path):
    production = []
    candidate = []
    for trial in range(1, 4):
        p_path = tmp_path / f"production_{trial}.json"
        c_path = tmp_path / f"candidate_{trial}.json"
        p_path.write_text(
            json.dumps(
                _profile(
                    role="legacy_production",
                    mean_ms=51.0 + 0.1 * trial,
                    algebraic_ms=35.0,
                    explicit_rhs_ms=7.0,
                    inverse_ms=14.0,
                    trial=trial,
                )
            ),
            encoding="utf-8",
        )
        c_path.write_text(
            json.dumps(
                _profile(
                    role="separated_canary",
                    mean_ms=55.0 + 0.1 * trial,
                    algebraic_ms=43.0,
                    explicit_rhs_ms=1.0,
                    inverse_ms=15.0,
                    trial=trial,
                )
            ),
            encoding="utf-8",
        )
        production.append(p_path)
        candidate.append(c_path)
    return production, candidate


def test_stage_q2_attributes_only_disjoint_regions_and_ranks_target(tmp_path):
    production, candidate = _write_profiles(tmp_path)
    support = tmp_path / "stage_q1_final_diagnostic.json"
    support.write_text('{"classification":"A_recommended"}\n', encoding="utf-8")
    report = analyze_stage_q2_residual_gap(
        production,
        candidate,
        supporting_artifact_paths=[support],
    )

    assert report["classification"] == "TARGET_REVIEW_COMPLETE"
    assert report["target_review"]["primary_target"] == (
        "algebraic_runtime_orchestration"
    )
    assert report["target_review"]["explicit_rhs_line_closed"] is True
    assert report["target_review"]["transform_forward_line_closed"] is True
    assert report["target_review"][
        "projected_materialization_line_remains_closed"
    ] is True
    additive = report["additive_semantic_attribution"]
    assert additive["top_level_delta_sum_milliseconds_per_step"] == pytest.approx(2.0)
    nested = report["nested_nonadditive_evidence"]
    assert nested["must_not_be_added_to_top_level_attribution"] is True
    assert nested["transform_inverse_materially_changed"] is True
    assert report["eligible_for_stage_q2_candidate_design"] is True
    assert report["eligible_for_stage_q2_candidate_implementation"] is False
    assert report["eligible_for_production_promotion"] is False
    assert report["production_default_changed"] is False
    assert report["inputs"][-1]["kind"] == "supporting_artifact"


def test_stage_q2_rejects_identity_and_schema_mismatches(tmp_path):
    production, candidate = _write_profiles(tmp_path)
    changed = json.loads(candidate[0].read_text(encoding="utf-8"))
    changed["production_initial_q_sha256"] = "f" * 64
    candidate[0].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="identities differ"):
        analyze_stage_q2_residual_gap(production, candidate)

    production, candidate = _write_profiles(tmp_path)
    changed = json.loads(candidate[0].read_text(encoding="utf-8"))
    changed["raw_semantic_regions"]["transform_forward"] = _region(0.01)
    candidate[0].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="mixes semantic schemas"):
        analyze_stage_q2_residual_gap(production, candidate)


def test_stage_q2_rejects_bad_counts_nonfinite_and_graph_breaks(tmp_path):
    production, candidate = _write_profiles(tmp_path)
    with pytest.raises(ValueError, match="exactly three"):
        analyze_stage_q2_residual_gap(production[:2], candidate)

    changed = json.loads(candidate[0].read_text(encoding="utf-8"))
    changed["matched_semantic_regions"][
        "algebraic_milliseconds_per_step"
    ] = float("nan")
    candidate[0].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="must be finite"):
        analyze_stage_q2_residual_gap(production, candidate)

    production, candidate = _write_profiles(tmp_path)
    changed = json.loads(candidate[0].read_text(encoding="utf-8"))
    changed["dynamo_delta"]["graph_breaks"] = 1
    candidate[0].write_text(json.dumps(changed), encoding="utf-8")
    with pytest.raises(ValueError, match="graph break"):
        analyze_stage_q2_residual_gap(production, candidate)


def test_stage_q2_cli_writes_once_and_remains_analysis_only(tmp_path):
    production, candidate = _write_profiles(tmp_path)
    output = tmp_path / "stage_q2.json"
    argv = []
    for path in production:
        argv.extend(("--production-profile", str(path)))
    for path in candidate:
        argv.extend(("--candidate-profile", str(path)))
    argv.extend(("--output", str(output)))
    assert analysis_main(argv) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["classification"] == (
        "TARGET_REVIEW_COMPLETE"
    )
    with pytest.raises(FileExistsError):
        analysis_main(argv)

    assert not hasattr(pssolver, "analyze_stage_q2_residual_gap")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "stage_q2" not in source
        assert "Stage Q.2" not in source
