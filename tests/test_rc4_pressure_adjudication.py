"""RC4.4.1 analysis-only pressure-contract adjudication gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from benchmarks import diagnose_channel_pressure_contract as diagnostic


ROOT = Path(__file__).resolve().parents[1]
RECORD = ROOT / "notes/PSSolver_v0_2_0rc4_rc441_pressure_adjudication.json"
PLAN = ROOT / "notes/PSSolver_v0_2_0rc4_rc440_pressure_contract_plan.json"
DIAGNOSTIC = ROOT / "benchmarks/diagnose_channel_pressure_contract.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_rc441_record_binds_the_plan_and_reproducible_diagnostic():
    record = _record()

    assert record["classification"] == (
        "PASS_RC4_4_1_ACTIVE_MODE_RMS_AND_SCOPED_CAPABILITY_CONTRACT"
    )
    assert record["identity"]["rc4_4_0_plan_sha256"] == _sha256(PLAN)
    assert record["identity"]["diagnostic_sha256"] == _sha256(DIAGNOSTIC)
    assert record["completion"] == {
        "analysis_complete": True,
        "B25_formula_selected": True,
        "B28_truth_table_complete": True,
        "B28_source_repair_complete": False,
        "B29_policy_reconfirmed": True,
        "eligible_for_rc4_4_2": True,
        "solver_source_modified": False,
        "PSSolver_Control_modified": False,
        "nematics3d_modified": False,
        "production_default_changed": False,
        "installed_wheel_qualified": False,
        "H100_qualified": False,
        "rc4_4_complete": False,
    }


def test_active_mode_rms_survives_the_registered_float64_stress_matrix():
    result = diagnostic._norm_stress()

    assert result["active_mode_rms_passed"] is True
    assert result["active_mode_rms_nonfinite_or_zero_count"] == 0
    assert result["scaled_global_l2_public_float_passed"] is False
    assert result["scaled_global_l2_nonfinite_or_zero_count"] == 3
    failures = [
        item for item in result["records"] if not item["global_l2_finite"]
    ]
    assert [item["shape"] for item in failures] == [
        [3, 2, 2],
        [4, 3, 2],
        [5, 4, 3],
    ]
    assert all(item["scale_exponent"] == 1023 for item in failures)
    assert all(item["active_rms_finite"] for item in failures)


def test_scale_relative_breakdown_rejects_the_absolute_magic_threshold():
    result = diagnostic._breakdown_cases()

    assert result["absolute_magic_threshold_rejected"] is True
    assert result["scale_relative_test_distinguishes_tiny_from_zero"] is True
    tiny = result["records"]["operator_tiny_but_well_scaled"]
    assert tiny["absolute_1e_30_declares_breakdown"] is True
    assert tiny["relative_test_declares_breakdown"] is False
    assert result["records"]["true_orthogonal_breakdown"][
        "relative_test_declares_breakdown"
    ] is True


def test_registered_scale_grid_and_operator_matrix_is_reproducible():
    result = diagnostic._scale_sweep()

    assert result["passed"] is True
    assert result["all_solve_outcomes_acceptable"] is True
    assert result["maximum_scale_covariance_relative_l2"] <= result[
        "scale_covariance_bound"
    ]
    assert [item["active_pressure_degrees_of_freedom"] for item in result["grids"]] == [
        11,
        17,
        59,
    ]
    for grid in result["grids"]:
        for action in grid["actions"].values():
            fixed = action["fixed_work"]["scale_trials"]
            tolerance = action["relative_tolerance"]["scale_trials"]
            assert {item["iterations"] for item in fixed} == {5}
            assert {item["termination_reason"] for item in fixed} == {
                "fixed_work_completed"
            }
            assert len({item["iterations"] for item in tolerance}) == 1
            assert {item["termination_reason"] for item in tolerance} == {
                "relative_tolerance_met"
            }


def test_dense_gradient_and_capability_truth_gates_are_live():
    dense = diagnostic._dense_oracle()
    gradient = diagnostic._gradient_oracle()
    capability = diagnostic._capability_truth_table()

    assert dense["passed"] is True
    assert gradient["passed"] is True
    assert capability["declaration_runtime_capabilities_equal"] is True
    assert capability["explicit_vjp_false_is_truthful"] is True
    assert capability["custom_autograd_adjoint_exists"] is True
    assert capability["finding"].startswith("METADATA_REPAIR_REQUIRED")
    components = {
        row["subject"]: row for row in capability["truth_table"]
    }
    assert components["functional_runtime"]["is_runtime"] is True
    assert components["implicit_adjoint_component"]["is_runtime"] is False
    assert components["implicit_adjoint_component"][
        "connected_to_executable_runtime"
    ] is True


def test_complete_analysis_matches_the_frozen_record_aggregates():
    live = diagnostic.build_report()
    record = _record()

    assert live["classification"] == record["classification"]
    assert live["selected_scale_contract"]["decision_class"] == (
        record["selected_scale_contract"]["decision_class"]
    )
    assert live["completion"]["analysis_complete"] is True
    assert live["evidence"]["P02_P03_P04_P05_scale_grid_primal_transpose"][
        "maximum_scale_covariance_relative_l2"
    ] == pytest.approx(
        record["results"]["P03_scale_sweep"][
            "maximum_pressure_scale_covariance_relative_l2"
        ],
        rel=0.0,
        abs=0.0,
    )
    assert live["evidence"]["P08_dense_and_gradient_oracles"]["dense"][
        "primal_transpose_inner_product_relative_error"
    ] == pytest.approx(
        record["results"]["P05_primal_transpose"][
            "inner_product_relative_error"
        ],
        rel=1.0e-15,
        abs=0.0,
    )
