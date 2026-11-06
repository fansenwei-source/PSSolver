"""Draft harness regressions; these do not declare the full S-23 matrix complete."""

from __future__ import annotations

import numpy as np
import pytest

from pssolver.functional.api import FunctionalValueError
from qualification.identity.runtime_case import RuntimeCase, initial_values
from qualification.identity.scenarios import scenarios


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_snapshot_matches_requested_dtype(dtype):
    scene = {"shape": [16, 12, 12], "dtype": dtype}
    values = initial_values(scene)
    assert values.dtype == np.dtype(dtype)
    assert values.shape == (16, 12, 12, 5)
    assert np.isfinite(values).all()
    assert np.array_equal(values, initial_values(scene))


@pytest.mark.parametrize("family", ["functional_periodic", "functional_channel"])
@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_functional_scaffold_uses_stable_state_without_private_solver(tmp_path, family, dtype):
    scene = next(
        row
        for row in scenarios()
        if row["family"] == family and row["dtype"] == dtype and not row["options"]
    )
    if family == "functional_channel" and dtype == "float32":
        with pytest.raises(FunctionalValueError, match="Channel declarations require float64"):
            RuntimeCase(scene, tmp_path)
        return
    case = RuntimeCase(scene, tmp_path)
    assert case.solver is None
    assert case.record()["step"] == 0
    assert all(item["finite"] for item in case.record()["tensors"].values())
    assert case.identity()["runtime"]
