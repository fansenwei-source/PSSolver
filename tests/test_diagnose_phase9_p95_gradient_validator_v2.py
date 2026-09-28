"""CPU contracts for the P9.5 validator-v2 diagnostic."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from benchmarks.diagnose_phase9_p95_gradient_validator_v2 import (
    ACTIVITY_EPSILONS,
    DIRECTION_KINDS,
    METHODS,
    STATE_EPSILONS,
    GradientValidatorV2DiagnosticConfig,
    _atomic_json,
    _low_mode_direction,
    _paired_quadratic_derivative,
    run_diagnostic,
)


def _initial_q(tmp_path):
    path = tmp_path / "Q_0.npy"
    values = np.zeros((6, 6, 4, 5), dtype=np.float64)
    values[..., 0] = 0.18
    values[..., 3] = -0.09
    np.save(path, values, allow_pickle=False)
    return path


def test_paired_quadratic_derivative_matches_analytic_complex_result():
    epsilon = 1.0e-4
    base = torch.tensor([1.0 + 2.0j, -0.5 + 0.25j], dtype=torch.complex128)
    tangent = torch.tensor([0.2 - 0.1j, 0.4 + 0.3j], dtype=torch.complex128)
    plus = (base + epsilon * tangent,)
    minus = (base - epsilon * tangent,)

    actual = _paired_quadratic_derivative(plus, minus, epsilon)
    expected = 2.0 * (base.conj() * tangent).real.mean()

    assert torch.allclose(actual, expected, atol=2.0e-12, rtol=2.0e-12)


def test_low_mode_direction_is_finite_normalized_and_complex_compatible():
    direction = _low_mode_direction(
        torch.zeros((2, 4, 5), dtype=torch.complex128), 0.31
    )

    assert direction.dtype is torch.complex128
    assert torch.isfinite(direction).all()
    assert float(direction.abs().amax()) == pytest.approx(1.0)


def test_validator_v2_diagnostic_records_complete_analysis_only_matrix(tmp_path):
    report = run_diagnostic(
        GradientValidatorV2DiagnosticConfig(
            grid_id="R128",
            shape=(6, 6, 4),
            lengths=(6.0, 6.0, 4.0),
            initial_q_path=str(_initial_q(tmp_path)),
            device="cpu",
        )
    )

    assert report["kind"] == "p95_gradient_validator_v2_diagnostic"
    assert len(report["frozen_validation"]["gradient_paths"]) == 6
    assert report["objective_replay_bitwise"] is True
    assert report["all_finite"] is True
    assert len(report["sweeps"]) == 6
    assert {
        (sweep["target"], sweep["direction"]) for sweep in report["sweeps"]
    } == {
        (target, direction)
        for target in ("state", "activity")
        for direction in DIRECTION_KINDS
    }
    for sweep in report["sweeps"]:
        expected = STATE_EPSILONS if sweep["target"] == "state" else ACTIVITY_EPSILONS
        assert [row["epsilon"] for row in sweep["rows"]] == list(expected)
        assert all(set(row["methods"]) == set(METHODS) for row in sweep["rows"])
    assert report["qualification_changed"] is False
    assert report["p9_5_pass_claimed"] is False
    assert report["p9_6_authorized"] is False
    json.dumps(report, allow_nan=False)


def test_validator_v2_atomic_output_refuses_overwrite(tmp_path):
    output = tmp_path / "diagnostic.json"
    _atomic_json(output, {"complete": True})
    with pytest.raises(FileExistsError, match="overwrite"):
        _atomic_json(output, {"complete": False})
