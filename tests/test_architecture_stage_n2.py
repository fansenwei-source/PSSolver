"""CPU qualification for Stage N.2 lazy algebraic materialization."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import pssolver
from pssolver.core import HomogeneousNeumannBC, PeriodicBC
from pssolver.experimental.representations import AlgebraicGenerationState
from tests.v0_3_s03_successor import read_registered_source


P = PeriodicBC()


N = HomogeneousNeumannBC()


PROJECT_ROOT = Path(__file__).parents[1]


def test_generation_state_materializes_once_and_releases_every_tensor():
    physical = torch.arange(6.0, dtype=torch.float64).reshape(2, 3)
    spectral = torch.complex(physical, torch.zeros_like(physical))
    calls = []

    def materialize(name, value):
        calls.append((name, value))
        return value.real.clone()

    generation = AlgebraicGenerationState(
        1,
        {"phi": physical},
        {"phi": spectral},
        materialize=materialize,
    )
    output_hat = spectral + 1.0
    generation.publish_spectral("psi", output_hat)
    view = generation.view(("phi", "psi"))

    assert view.spectral("psi") is output_hat
    assert calls == []
    first = view["psi"]
    assert view["psi"] is first
    assert len(calls) == 1
    snapshot = generation.snapshot()
    assert snapshot["physical_materializations"] == 1
    assert snapshot["spectral_dependency_hits"] == 1
    assert snapshot["unmaterialized_published_components"] == 0

    final = generation.invalidate()
    assert final["active"] is False
    assert final["retained_physical_components"] == 0
    assert final["retained_spectral_components"] == 0
    with pytest.raises(RuntimeError, match="no longer current"):
        view["phi"]


def test_stage_n2_remains_outside_production_and_generic_solver_paths():
    assert not hasattr(pssolver, "AlgebraicGenerationState")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
    ):
        text = read_registered_source(relative).decode("utf-8")
        assert "enable_lazy_algebraic_materialization" not in text
        assert "AlgebraicGenerationState" not in text
