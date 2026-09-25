"""Stokes solver implementations selected by exact geometry capability."""

from .channel_no_slip import (
    CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    CHANNEL_VELOCITY_BOUNDARY_CONDITIONS,
    ChannelNoSlipModalStokesSolver,
)
from .plane_free_slip import FreeSlipModalStokesSolver
from .periodic import PeriodicModalStokesSolver

__all__ = [
    "CHANNEL_PRESSURE_BOUNDARY_CONDITIONS",
    "CHANNEL_VELOCITY_BOUNDARY_CONDITIONS",
    "ChannelNoSlipModalStokesSolver",
    "FreeSlipModalStokesSolver",
    "PeriodicModalStokesSolver",
]
