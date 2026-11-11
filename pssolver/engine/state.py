"""Caller-owned tensor state and progress of one execution."""

from collections.abc import Mapping
from dataclasses import dataclass

import torch


@dataclass(slots=True)
class State:
    """Execution state; replace solver entries together rather than mutating aliases.

    Changing a Channel guess requires clearing its cached action and exponent
    unless both replacements are known to match the new guess.
    """

    spectral: torch.Tensor
    spatial: torch.Tensor
    history: tuple[torch.Tensor, ...]
    solver: Mapping[str, torch.Tensor | int | tuple[int | None, ...] | None]
    completed_steps: int
    refresh_phase: tuple[int, int]
    constrained_current: bool
    # Pure calls retain the original evolved tensors, rather than routing
    # their independent gradient consumers through padded-storage slices.
    pure_evolved: tuple[torch.Tensor, torch.Tensor] | None = None
    constraint_physical: torch.Tensor | None = None
