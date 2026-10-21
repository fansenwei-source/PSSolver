"""Compatibility facade for stable public simulation execution errors."""

from __future__ import annotations

from pssolver.core.errors import (
    SimulationExecutionError,
    SimulationInputError,
    SimulationRestartError,
    SimulationStateError,
)


__all__ = [
    "SimulationExecutionError",
    "SimulationInputError",
    "SimulationRestartError",
    "SimulationStateError",
]
