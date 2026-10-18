"""Tests for the component-resolved RC6 pressure H100 diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks import diagnose_rc6_pressure_floor_h100 as diagnostic


def _tolerance(**changes):
    value = {
        "schema": diagnostic.COMPONENT_SCHEMA,
        "dtype": "float32",
        "mode": "tolerance",
        "floor": 1.0e-5,
        "termination_reason": "relative_tolerance_met",
        "recorded_relative_residual": 9.0e-6,
        "recomputed_relative_residual": 8.0e-6,
        "exception": None,
    }
    value.update(changes)
    return value


def _fixed(**changes):
    value = {
        "schema": diagnostic.COMPONENT_SCHEMA,
        "dtype": "float32",
        "mode": "fixed",
        "termination_reason": "fixed_iterations_completed",
        "pressure_finite": True,
        "exception": None,
    }
    value.update(changes)
    return value


def test_tolerance_adjudication_persists_every_original_subpredicate():
    result = diagnostic.adjudicate_tolerance(_tolerance())
    assert result["passed"] is True
    assert result["predicates"] == {
        "termination_reason_is_relative_tolerance_met": True,
        "recorded_relative_residual_is_finite": True,
        "recorded_relative_residual_at_or_below_floor": True,
        "recomputed_relative_residual_is_finite": True,
        "recomputed_relative_residual_at_or_below_floor": True,
    }


@pytest.mark.parametrize(
    "changes,failed",
    [
        ({"termination_reason": "iteration_cap_reached"}, "termination_reason_is_relative_tolerance_met"),
        ({"recorded_relative_residual": None}, "recorded_relative_residual_is_finite"),
        ({"recorded_relative_residual": 1.1e-5}, "recorded_relative_residual_at_or_below_floor"),
        ({"recomputed_relative_residual": float("inf")}, "recomputed_relative_residual_is_finite"),
        ({"recomputed_relative_residual": 1.1e-5}, "recomputed_relative_residual_at_or_below_floor"),
    ],
)
def test_tolerance_adjudication_identifies_the_exact_failed_predicate(changes, failed):
    result = diagnostic.adjudicate_tolerance(_tolerance(**changes))
    assert result["passed"] is False
    assert result["predicates"][failed] is False


@pytest.mark.parametrize(
    "changes,failed",
    [
        ({"termination_reason": "iteration_cap_reached"}, "termination_reason_is_fixed_iterations_completed"),
        ({"pressure_finite": False}, "pressure_is_finite"),
    ],
)
def test_fixed_adjudication_identifies_the_exact_failed_predicate(changes, failed):
    result = diagnostic.adjudicate_fixed(_fixed(**changes))
    assert result["passed"] is False
    assert result["predicates"][failed] is False


def test_execute_persists_all_components_and_returns_zero_for_completed_gate_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    monkeypatch.setattr(diagnostic, "_capture_environment", lambda device: {
        "schema": diagnostic.ENVIRONMENT_SCHEMA, "passed": True,
    })
    monkeypatch.setattr(
        diagnostic, "_build_case", lambda dtype, device: {"dtype_name": dtype},
    )
    monkeypatch.setattr(
        diagnostic,
        "_observe_tolerance",
        lambda case: diagnostic.adjudicate_tolerance(_tolerance(dtype=case["dtype_name"])),
    )
    monkeypatch.setattr(
        diagnostic,
        "_observe_fixed",
        lambda case: diagnostic.adjudicate_fixed(
            _fixed(
                dtype=case["dtype_name"],
                pressure_finite=case["dtype_name"] != "float64",
            )
        ),
    )
    root = tmp_path / "diagnostic"
    result = diagnostic.execute(root)
    assert result["complete"] is True
    assert result["passed"] is False
    assert result["classification"] == "FAIL_RC6_PRESSURE_FLOOR_COMPONENT_DIAGNOSTIC"
    assert diagnostic.main(["--output-root", str(tmp_path / "second")]) == 0
    for dtype in diagnostic.DTYPES:
        for mode in diagnostic.MODES:
            value = json.loads((root / f"{dtype}_{mode}.json").read_text())
            assert value["dtype"] == dtype
            assert value["mode"] == mode
    assert not list(root.glob(".*.tmp-*"))


def test_execute_persists_exceptions_and_returns_nonzero_when_incomplete(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
):
    monkeypatch.setattr(diagnostic, "_capture_environment", lambda device: {
        "schema": diagnostic.ENVIRONMENT_SCHEMA, "passed": True,
    })
    monkeypatch.setattr(
        diagnostic, "_build_case", lambda dtype, device: (_ for _ in ()).throw(RuntimeError("setup")),
    )
    root = tmp_path / "diagnostic"
    result = diagnostic.execute(root)
    assert result["complete"] is False
    assert result["component_count"] == 4
    assert all(item["exception"]["type"] == "RuntimeError" for item in result["components"])
    assert diagnostic.main(["--output-root", str(tmp_path / "second")]) == 1


def test_atomic_json_refuses_overwrite_and_rejects_nan(tmp_path: Path):
    path = tmp_path / "value.json"
    diagnostic._atomic_json(path, {"passed": True})
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnostic._atomic_json(path, {"passed": False})
    with pytest.raises(ValueError, match="Out of range float values"):
        diagnostic._atomic_json(tmp_path / "nan.json", {"value": float("nan")})


def test_versioned_job_runs_preflights_before_the_analysis_only_diagnostic():
    root = Path(__file__).resolve().parents[1]
    source = (root / "jobs/rc6_pressure_component_diagnostic_h100.sbatch").read_text(
        encoding="utf-8",
    )
    checkpoint = source.index("run_rc5_g1_checkpoint_tmp_preflight.py\" \\")
    toolkit = source.index("run_rc6_cuda_toolkit_preflight.py\" \\")
    diagnostic_position = source.index("diagnose_rc6_pressure_floor_h100.py\" \\")
    assert checkpoint < toolkit < diagnostic_position
    assert "module load cuda/12.1" in source
    assert "set +u\n  source /opt/ohpc/admin/lmod/lmod/init/bash\n  set -u" in source
    assert 'export TMPDIR="${pytest_tmp}"' in source
    assert 'test ! -e "${RC6_DIAGNOSTIC_OUTPUT}"' in source
    assert 'test ! -e "${RC6_SUPPORT_ROOT}/pssolver"' in source
    assert "run_rc6_single_h100_stage_plan.py" not in source
    assert "COMPLETE" not in source
