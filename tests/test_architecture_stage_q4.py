"""CPU contracts for the bounded Stage Q.4 batch-workspace candidate."""

from __future__ import annotations

from pathlib import Path

from pssolver.core import (
    HomogeneousNeumannBC,
    PeriodicBC,
)


PROJECT_ROOT = Path(__file__).parents[1]


P = PeriodicBC()


N = HomogeneousNeumannBC()
