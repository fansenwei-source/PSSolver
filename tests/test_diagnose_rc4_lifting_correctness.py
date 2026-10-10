from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from benchmarks import diagnose_rc4_lifting_correctness as diagnostic


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc431_lifting_correctness_analysis.json"


@pytest.fixture(scope="module")
def report():
    return diagnostic.run_analysis()


def test_rc431_runs_all_five_frozen_cases_and_selects_one_decision(report):
    assert report["phase"] == "RC4.3.1"
    assert list(report["cases"]) == ["L01", "L02", "L03", "L04", "L05"]
    assert report["decision"]["classification"] == (
        "PUBLIC_CAPABILITY_RESTRICTION_REQUIRED"
    )
    assert report["decision"]["all_arrays_finite"] is True


def test_checked_in_record_is_the_reproducible_analysis_output(report):
    assert json.loads(RECORD.read_text(encoding="utf-8")) == report


def test_independent_reference_does_not_call_production_constitutive_helpers():
    source = (
        ROOT / "benchmarks/diagnose_rc4_lifting_correctness.py"
    ).read_text(encoding="utf-8")
    for helper in (
        "beris_edwards_molecular_field_components",
        "beris_edwards_algebraic_stress_components",
        "beris_edwards_distortion_stress_components",
    ):
        assert helper not in source


def test_l01_control_resolves_second_order_convergence(report):
    case = report["cases"]["L01"]
    assert case["wall_points"] == [16, 32, 64, 128, 256]
    assert case["claimed_second_order_passed"] is True
    assert case["regression_rate"] >= 1.8
    assert min(case["adjacent_rates"][-2:]) >= 1.8
    assert all(
        left > right for left, right in zip(case["errors"], case["errors"][1:])
    )


def test_l02_real_solver_wall_molecular_field_plateaus(report):
    case = report["cases"]["L02"]
    assert case["reference"] == (
        "direct_constitutive_evaluation_with_analytic_q_laplacian"
    )
    assert case["representation_repair_supported"] is False
    assert case["wall_error_regression_rate"] < 0.2
    assert case["bulk_error_regression_rate"] < 1.0
    assert case["last_two_wall_errors_strictly_decrease"] is True
    assert all(row["q_wall_linf"] == 0.0 for row in case["rows"])
    assert all(row["q_first_interior_linf"] == 0.0 for row in case["rows"])
    assert all(row["runtime_fallback"] is False for row in case["rows"])


def test_l03_real_solver_twisted_velocity_is_only_first_order(report):
    case = report["cases"]["L03"]
    assert "independent_dense_physical_quadrature" in case["reference"]
    assert case["claimed_second_order_velocity_passed"] is False
    assert 0.9 < case["velocity_relative_l2_regression_rate"] < 1.2
    assert 0.8 < case["pressure_relative_l2_regression_rate"] < 1.2
    assert all(row["finite"] for row in case["rows"])
    assert all(row["runtime_fallback"] is False for row in case["rows"])


def test_l04_fresh_conditioning_is_global_and_underspecified(report):
    case = report["cases"]["L04"]
    assert case["policy_kind"] == "sine_wall_taper"
    assert case["whole_slab_sine_taper_confirmed"] is True
    assert case["localized_unchanged_interior_supported"] is False
    assert case["formula_present"] is False
    assert case["transition_width_present"] is False
    assert case["transition_width_units_present"] is False
    for row in (*case["uniform"], *case["defect_gas"]):
        assert row["exactly_unchanged_cell_count"] == 0
        assert row["quarter_height_deviation_linf"] > 0.0


def test_l05_two_requests_execute_one_physical_path(report):
    case = report["cases"]["L05"]
    assert case["lifting_forces_physical_branch_guard_present"] is True
    assert case["observed_outputs_and_transform_counts_identical"] is True
    assert case["requested_space_is_silently_ignored"] is True
    assert case["explicit_effective_space_metadata_present"] is False
    spectral = case["spectral_request"]
    physical = case["physical_request"]
    assert spectral["requested"] == "spectral"
    assert physical["requested"] == "physical"
    assert spectral["observed_physical_projection_calls"] == 1
    assert physical["observed_physical_projection_calls"] == 1
    assert spectral["runtime_fallback"] is False
    assert physical["runtime_fallback"] is False


def test_rc431_authorization_does_not_claim_implementation_or_h100(report):
    assert report["authorization"] == {
        "analysis_authorized_by_user": True,
        "runtime_source_modified": False,
        "production_default_changed": False,
        "h100_submitted": False,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
        "rc4_3_2_authorized": False,
        "eligible_for_automatic_merge": False,
        "eligible_for_default_promotion": False,
    }


def test_atomic_json_writer_round_trips_and_refuses_overwrite(tmp_path):
    output = tmp_path / "analysis.json"
    value = {"finite": True, "rate": 2.0}
    diagnostic._atomic_write_json(output, value)
    assert json.loads(output.read_text(encoding="utf-8")) == value
    assert not list(tmp_path.glob(".*.tmp"))
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnostic._atomic_write_json(output, value)


def test_report_contains_no_nonfinite_json_values(report):
    payload = json.dumps(report, allow_nan=False, sort_keys=True)
    assert "NaN" not in payload
    assert "Infinity" not in payload
    assert all(math.isfinite(value) for value in report["cases"]["L01"]["errors"])
