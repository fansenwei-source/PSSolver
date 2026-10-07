from __future__ import annotations

import hashlib
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
REPAIR_RECORD_PATH = (
    Path(__file__).resolve().parents[1]
    / "notes"
    / "PSSolver_v0_2_0rc4_rc411_plane_nyquist_repair.json"
)


@pytest.fixture(scope="module")
def report():
    return diagnose()


def _cases(report):
    return {case["name"]: case for case in report["cases"]}


def test_diagnostic_is_analysis_only_and_binds_the_rc3_baseline(report):
    assert report["schema"] == SCHEMA
    assert report["analysis_only"] is True
    assert report["diagnostic_modifies_solver"] is False
    assert report["baseline"] == {
        "release": "v0.2.0rc3",
        "commit": BASELINE_COMMIT,
    }
    assert report["configuration"]["dealias_rule"] == "none"
    assert report["classification"] == (
        "PASS_PLANE_PERIODIC_NYQUIST_STORAGE_EQUIVALENCE"
    )
    assert report["causal_finding"]["current_storage_equivalence_passed"] is True


def test_even_default_plane_storage_equivalence_is_repaired(report):
    case = _cases(report)["even_xy_default_axis1"]
    assert case["non_reduced_periodic_axis"] == 0
    assert case["non_reduced_axis_is_even"] is True
    assert case["summary"]["raw_max_velocity_relative_l2"] < 1.0e-12
    assert case["summary"]["raw_pressure_relative_l2"] < 1.0e-12
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


def test_even_x_and_even_y_are_storage_equivalent_in_the_default_layout(report):
    cases = _cases(report)
    even_x = cases["even_x_only_default_axis1"]
    even_y = cases["even_y_only_default_axis1"]
    assert even_x["non_reduced_periodic_axis"] == 0
    assert even_x["summary"]["raw_max_velocity_relative_l2"] < 1.0e-12
    assert even_y["non_reduced_periodic_axis"] == 0
    assert even_y["summary"]["raw_max_velocity_relative_l2"] < 1.0e-12


def test_rotated_hermitian_axis_is_storage_equivalent(report):
    case = _cases(report)["even_xy_rotated_axis0"]
    assert case["hermitian_axis"] == 0
    assert case["non_reduced_periodic_axis"] == 1
    assert case["summary"]["raw_max_velocity_relative_l2"] < 1.0e-12


def test_removing_only_the_causal_nyquist_plane_restores_equivalence(report):
    for case in _cases(report).values():
        filtered = case["summary"]["filtered_max_velocity_relative_l2"]
        filtered_pressure = case["summary"]["filtered_pressure_relative_l2"]
        assert filtered < 1.0e-12
        assert filtered_pressure < 1.0e-12


def test_native_saddle_residuals_and_storage_equivalence_both_pass(report):
    case = _cases(report)["even_xy_default_axis1"]
    for storage in ("full_complex", "hermitian_half"):
        result = case["raw"][storage]
        assert result["native_divergence_relative_l2"] < 1.0e-12
        assert result["pressure_relative_residual"] < 1.0e-12
    assert case["summary"]["raw_max_velocity_relative_l2"] < 1.0e-12


def test_summary_writer_is_atomic_json_and_rejects_no_fields(tmp_path, report):
    path = tmp_path / "diagnostic.json"
    payload = summary(report)
    write_json_atomic(path, payload)
    assert json.loads(path.read_text(encoding="utf-8")) == payload
    assert list(tmp_path.iterdir()) == [path]


def test_frozen_record_preserves_the_rc3_failure_and_authorizes_repair(report):
    record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
    assert record["classification"] == (
        "REPRODUCED_PLANE_NON_REDUCED_PERIODIC_NYQUIST_STORAGE_MISMATCH"
    )
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
    assert (
        observed["even_xy_default_axis1"]["raw_max_velocity_relative_l2"]
        > record["frozen_oracle"][
            "mismatch_case_raw_velocity_relative_l2_min"
        ]
    )
    assert (
        observed["even_xy_default_axis1"]["raw_pressure_relative_l2"]
        > record["frozen_oracle"][
            "mismatch_case_raw_pressure_relative_l2_min"
        ]
    )
    live = _cases(report)
    for name, case in live.items():
        frozen = observed[name]
        assert frozen["shape"] == case["shape"]
        assert frozen["hermitian_axis"] == case["hermitian_axis"]
        assert frozen["non_reduced_periodic_axis"] == case[
            "non_reduced_periodic_axis"
        ]
        assert case["summary"]["raw_max_velocity_relative_l2"] < 1.0e-12
        assert case["summary"]["raw_pressure_relative_l2"] < 1.0e-12
    assert record["next_stage"]["b5_closed"] is False
    assert record["next_stage"]["h100_qualification_still_required"] is True


def test_repair_record_binds_the_implementation_and_cpu_scope(report):
    record = json.loads(REPAIR_RECORD_PATH.read_text(encoding="utf-8"))
    root = Path(__file__).resolve().parents[1]
    source = root / record["implementation"]["source"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record[
        "implementation"
    ]["source_sha256"]
    assert record["classification"] == (
        "PASS_RC4_1_1_PLANE_NYQUIST_CPU_NON_REGRESSION"
    )
    assert record["storage_equivalence"]["classification"] == report[
        "classification"
    ]
    assert record["implementation"]["state_dict_key_schema_changed"] is False
    assert record["implementation"]["production_default_changed"] is False
    assert record["scope"]["pssolver_control_modified"] is False
    assert record["scope"]["nematics3d_modified"] is False
    assert record["qualification"] == {
        "rc4_1_1_cpu_complete": True,
        "b5_closed": False,
        "eligible_for_single_h100_non_regression": True,
        "single_h100_non_regression_complete": False,
        "eligible_for_default_promotion": False,
        "eligible_for_release": False,
    }
