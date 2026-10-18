"""Compatibility facade for stable public simulation execution errors."""

from __future__ import annotations

from pssolver.core.errors import (
    SimulationExecutionError,
    SimulationRestartError,
    SimulationStateError,
)


__all__ = [
    "SimulationExecutionError",
    "SimulationRestartError",
    "SimulationStateError",
]
