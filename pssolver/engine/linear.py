"""A common implicit-solve interface and its diagonal and modal-block tiers."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import torch


class LinearSolve(Protocol):
    def solve(
        self, rhs: torch.Tensor, *, shift: float, dt: float, out: torch.Tensor | None,
        diagnostics: dict | None,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class SolveReport:
    converged: bool
    iterations: int
    residuals: tuple[float, ...]


class MatrixFreeSolve(Protocol):
    def apply(self, value: torch.Tensor) -> torch.Tensor: ...

    def precondition(self, residual: torch.Tensor) -> torch.Tensor: ...

    def iterate(
        self, rhs: torch.Tensor, *, shift: float, dt: float,
        precondition: Callable | None, record: bool,
    ) -> tuple[torch.Tensor, SolveReport]: ...


@dataclass(frozen=True, slots=True, init=False)
class DiagonalOperator:
    L_hat: torch.Tensor
    denominator: torch.Tensor
    _dt: float

    def __init__(self, L_hat, dt):
        object.__setattr__(self, "L_hat", L_hat)
        object.__setattr__(self, "denominator", 1 - L_hat * dt)
        object.__setattr__(self, "_dt", dt)

    def solve(self, rhs, *, shift, dt, out, diagnostics):
        """Reuse the original Euler denominator without a new tensor operation."""
        denominator = self.denominator if shift == 1.0 and dt == self._dt else (
            shift - self.L_hat * dt
        )
        if out is None:
            return rhs / denominator
        if out is not rhs:
            raise ValueError("implicit solve output must alias its RHS")
        return out.div_(denominator)


@dataclass(frozen=True, slots=True)
class ModalBlockOperator:
    """Private reference for n-component solves with trailing per-mode blocks."""

    coefficients: torch.Tensor

    def solve(self, rhs, *, shift, dt, out, diagnostics):
        if out is not None and out is not rhs:
            raise ValueError("implicit solve output must alias its RHS")
        vector = rhs.movedim(0, -1)
        n = rhs.shape[0]
        if self.coefficients.shape != (*vector.shape[:-1], n, n):
            raise ValueError("modal block shape differs from the RHS")
        identity = torch.eye(n, device=rhs.device, dtype=rhs.dtype)
        matrix = shift * identity - dt * self.coefficients
        solution = torch.linalg.solve(matrix, vector.unsqueeze(-1))
        if diagnostics is not None:
            diagnostics["linear_residual"] = (matrix @ solution).squeeze(-1) - vector
        result = solution.squeeze(-1).movedim(-1, 0)
        return result if out is None else out.copy_(result)
