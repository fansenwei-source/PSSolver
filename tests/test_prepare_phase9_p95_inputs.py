"""Tests for immutable P9.5 analytic input preparation."""

from __future__ import annotations

import json

import numpy as np
import pytest

from benchmarks.prepare_phase9_p95_inputs import write_analytic_q


def test_analytic_q_is_reproducible_finite_and_has_production_layout(tmp_path):
    first = tmp_path / "first.npy"
    second = tmp_path / "second.npy"

    write_analytic_q(first, (6, 5, 4))
    write_analytic_q(second, (6, 5, 4))

    left = np.load(first, allow_pickle=False)
    right = np.load(second, allow_pickle=False)
    assert left.shape == (6, 5, 4, 5)
    assert left.dtype == np.float64
    assert np.isfinite(left).all()
    assert first.read_bytes() == second.read_bytes()
    assert np.array_equal(left, right)
    json.dumps(left[0, 0, 0].tolist())


def test_analytic_q_refuses_invalid_shape_and_overwrite(tmp_path):
    path = tmp_path / "Q_0.npy"
    with pytest.raises(ValueError, match="shape"):
        write_analytic_q(path, (6, 1, 4))

    write_analytic_q(path, (6, 5, 4))
    with pytest.raises(FileExistsError, match="overwrite"):
        write_analytic_q(path, (6, 5, 4))
