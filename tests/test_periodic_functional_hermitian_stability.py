"""Long-horizon regression for packed-spectrum periodic functional state."""

from __future__ import annotations

import json

import pytest

from benchmarks.check_periodic_hermitian_stability import (
    HermitianStabilityConfig,
    _atomic_json,
    run_stability_check,
)


def test_periodic_functional_state_remains_hermitian_through_time_200():
    """Prevent the pre-fix failure near ``t=86.48`` from recurring."""

    report = run_stability_check(
        HermitianStabilityConfig(
            case="loop3d",
            device="cpu",
            horizon=200.0,
        )
    )

    assert report["passed"] is True
    assert report["failure"] is None
    assert report["completed_time"] == 200.0
    assert report["hermitian_state_projection"] == (
        "self_conjugate_planes_each_step"
    )
    assert report["samples"]
    assert all(sample["finite"] for sample in report["samples"])
    assert all(
        sample["hermitian_violation"] <= sample["violation_bound"]
        for sample in report["samples"]
    )


def test_hermitian_stability_report_is_strict_json_and_atomic(tmp_path):
    report = run_stability_check(
        HermitianStabilityConfig(
            case="loop3d",
            device="cpu",
            horizon=0.04,
            sample_interval=0.02,
        )
    )
    output = tmp_path / "report.json"

    _atomic_json(output, report)

    assert json.loads(output.read_text(encoding="utf-8")) == report
    with pytest.raises(FileExistsError, match="overwrite"):
        _atomic_json(output, report)
