"""CPU qualification for Stage N.1 generation-local representation reuse."""

from __future__ import annotations

from pathlib import Path

import pssolver
from pssolver.core import (
    BoundarySet,
    HomogeneousNeumannBC,
    PeriodicBC,
)


P = PeriodicBC()


N = HomogeneousNeumannBC()


BOUNDARIES = BoundarySet((P, N))


PROJECT_ROOT = Path(__file__).parents[1]


def test_stage_n1_remains_outside_production_and_generic_solver_paths():
    assert not hasattr(pssolver, "AlgebraicRepresentationCache")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "enable_algebraic_representation_reuse" not in text
        assert "AlgebraicRepresentationCache" not in text
