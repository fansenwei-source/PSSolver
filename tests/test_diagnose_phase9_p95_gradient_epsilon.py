"""CPU contract tests for the P9.5 gradient epsilon diagnostic."""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.diagnose_phase9_p95_gradient_epsilon import (
    ACTIVITY_EPSILONS,
    GradientEpsilonDiagnosticConfig,
    _atomic_json,
    run_diagnostic,
)


def _initial_q(tmp_path):
    path = tmp_path / "Q_0.npy"
    values = np.zeros((6, 6, 4, 5), dtype=np.float64)
    values[..., 0] = 0.18
    values[..., 3] = -0.09
    np.save(path, values, allow_pickle=False)
    return path


def _config(path):
    return GradientEpsilonDiagnosticConfig(
        grid_id="R128",
        trial=1,
        shape=(6, 6, 4),
        lengths=(6.0, 6.0, 4.0),
        initial_q_path=str(path),
        device="cpu",
    )


def test_diagnostic_persists_raw_validator_and_complete_epsilon_sweep(tmp_path):
    report = run_diagnostic(_config(_initial_q(tmp_path)))

    assert report["kind"] == "p95_activity_gradient_epsilon_diagnostic"
    assert len(report["frozen_validation"]["gradient_paths"]) == 6
    assert len(report["frozen_validation"]["directional_derivatives"]) == 2
    assert [row["epsilon"] for row in report["activity_sweep"]] == list(
        ACTIVITY_EPSILONS
    )
    assert all(row["finite"] for row in report["activity_sweep"])
    assert all(row["objective_replay_bitwise"] for row in report["activity_sweep"])
    assert report["qualification_changed"] is False
    assert report["p9_5_pass_claimed"] is False
    json.dumps(report, allow_nan=False)


def test_diagnostic_rejects_bad_input_and_atomic_overwrite(tmp_path):
    path = tmp_path / "Q_0.npy"
    np.save(path, np.zeros((6, 6, 4, 4), dtype=np.float64))
    with pytest.raises(ValueError, match="shape or dtype"):
        run_diagnostic(_config(path))

    output = tmp_path / "diagnostic.json"
    _atomic_json(output, {"complete": True})
    with pytest.raises(FileExistsError, match="overwrite"):
        _atomic_json(output, {"complete": False})
