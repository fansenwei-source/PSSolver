"""Diagonal tier: the same two resident tensors as the legacy integrator."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True, slots=True, init=False)
class DiagonalOperator:
    L_hat: torch.Tensor
    denominator: torch.Tensor

    def __init__(self, L_hat, dt):
        object.__setattr__(self, "L_hat", L_hat)
        object.__setattr__(self, "denominator", 1 - L_hat * dt)
