from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.diagnose_plane_nyquist_storage import (
    BASELINE_COMMIT,
    SCHEMA,
    diagnose,
    summary,
    write_json_atomic,
)


RECORD_PATH = (
    Path(__file__).resolve().parents[1]
    / "notes"
    / "PSSolver_v0_2_0rc4_rc410_plane_nyquist_diagnostic.json"
)


@pytest.fixture(scope="module")
def report():
    return diagnose()


def _cases(report):
    return {case["name"]: case for case in report["cases"]}


def test_diagnostic_is_analysis_only_and_binds_rc3(report):
    assert report["schema"] == SCHEMA
    assert report["analysis_only"] is True
    assert report["solver_modified"] is False
    assert report["baseline"] == {
        "release": "v0.2.0rc3",
        "commit": BASELINE_COMMIT,
    }
    assert report["configuration"]["dealias_rule"] == "none"
    assert report["classification"] == (
        "REPRODUCED_PLANE_NON_REDUCED_PERIODIC_NYQUIST_STORAGE_MISMATCH"
    )


def test_even_default_plane_storage_mismatch_is_reproduced(report):
    case = _cases(report)["even_xy_default_axis1"]
    assert case["non_reduced_periodic_axis"] == 0
    assert case["non_reduced_axis_is_even"] is True
    assert case["summary"]["raw_max_velocity_relative_l2"] > 1.0e-2
    assert case["summary"]["raw_pressure_relative_l2"] > 5.0e-2
    assert all(
        record["finite"]
        for record in (
            case["raw"]["full_complex"],
            case["raw"]["hermitian_half"],
        )
    )


def test_odd_grid_is_the_storage_equivalence_control(report):
    case = _cases(report)["odd_xy_default_axis1"]
    assert case["non_reduced_axis_is_even"] is False
    assert case["summary"]["raw_max_velocity_relative_l2"] < 1.0e-12
    assert case["summary"]["raw_pressure_relative_l2"] < 1.0e-12


def test_only_the_non_reduced_periodic_axis_triggers_the_default_layout(report):
    cases = _cases(report)
    even_x = cases["even_x_only_default_axis1"]
    even_y = cases["even_y_only_default_axis1"]
    assert even_x["non_reduced_periodic_axis"] == 0
    assert even_x["summary"]["raw_max_velocity_relative_l2"] > 1.0e-2
    assert even_y["non_reduced_periodic_axis"] == 0
    assert even_y["summary"]["raw_max_velocity_relative_l2"] < 1.0e-12


def test_rotating_the_hermitian_axis_rotates_the_trigger(report):
    case = _cases(report)["even_xy_rotated_axis0"]
    assert case["hermitian_axis"] == 0
    assert case["non_reduced_periodic_axis"] == 1
    assert case["summary"]["raw_max_velocity_relative_l2"] > 1.0e-2


def test_removing_only_the_causal_nyquist_plane_restores_equivalence(report):
    for case in _cases(report).values():
        filtered = case["summary"]["filtered_max_velocity_relative_l2"]
        filtered_pressure = case["summary"]["filtered_pressure_relative_l2"]
        assert filtered < 1.0e-12
        assert filtered_pressure < 1.0e-12


def test_native_saddle_residuals_remain_small_while_storage_results_differ(report):
    case = _cases(report)["even_xy_default_axis1"]
    for storage in ("full_complex", "hermitian_half"):
        result = case["raw"][storage]
        assert result["native_divergence_relative_l2"] < 1.0e-12
        assert result["pressure_relative_residual"] < 1.0e-12
    assert case["summary"]["raw_max_velocity_relative_l2"] > 1.0e-2


def test_summary_writer_is_atomic_json_and_rejects_no_fields(tmp_path, report):
    path = tmp_path / "diagnostic.json"
    payload = summary(report)
    write_json_atomic(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.iterdir()) == [path]


def test_frozen_record_binds_the_live_diagnostic_and_next_stage(report):
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    assert record["classification"] == report["classification"]
    assert record["baseline"] == {
        "release": "v0.2.0rc3",
        "commit": BASELINE_COMMIT,
        "audit_parent": "2e7b2198c4629f614d52acf26c5c9837124e725d",
        "known_issue": "B5",
    }
    assert record["scope"] == {
        "analysis_only": True,
        "solver_modified": False,
        "h100_required": False,
        "production_default_changed": False,
        "pssolver_control_modified": False,
        "nematics3d_modified": False,
    }
    observed = record["observations"]
    for case in report["cases"]:
        frozen = observed[case["name"]]
        assert frozen["shape"] == case["shape"]
        assert frozen["hermitian_axis"] == case["hermitian_axis"]
        assert frozen["non_reduced_periodic_axis"] == case[
            "non_reduced_periodic_axis"
        ]
        for metric, value in case["summary"].items():
            if metric in frozen:
                assert frozen[metric] == pytest.approx(value, rel=1.0e-13)
    assert record["next_stage"]["b5_closed"] is False
    assert record["next_stage"]["h100_qualification_still_required"] is True
