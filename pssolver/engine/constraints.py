"""Stateless Periodic constraint adapter with caller-owned diagnostics."""

from collections.abc import Mapping
from typing import Literal, Protocol

import torch


SolverState = Mapping[str, torch.Tensor | None]


class ConstraintSolver(Protocol):
    key: tuple[str, str]
    adjoint_mode: Literal["autograd", "custom_transpose"]

    def seed(self, state) -> SolverState: ...
    def solve(self, rhs, solver_state, *, out=None): ...
    def solve_transpose(self, rhs_bar, solver_state): ...


class PeriodicStokesConstraint:
    key = ("stokes", "periodic^3")
    adjoint_mode = "autograd"

    def __init__(self, term, *, pressure_diagnostics):
        self.term = term
        self.pressure_diagnostics = pressure_diagnostics

    def seed(self, state):
        return {}

    def constrained_hats(self, view, alpha, *, diagnostics):
        def solve(*forces):
            values, report = self.term.solve_force_hats_with_diagnostics(
                *forces,
                pressure_diagnostics=self.pressure_diagnostics,
            )
            if diagnostics is not None:
                diagnostics.update(report)
            return values

        return self.term.constrained_hats(view, alpha, solve=solve, record=None)

    def solve(self, rhs, solver_state, *, out=None):
        values, _ = self.term.solve_force_hats_with_diagnostics(
            *rhs,
            pressure_diagnostics=False,
        )
        result = torch.stack(values)
        if out is not None:
            out.copy_(result)
            result = out
        return result, {}

    def solve_transpose(self, rhs_bar, solver_state):
        with torch.enable_grad():
            rhs = torch.zeros_like(rhs_bar[:3], requires_grad=True)
            result, _ = self.solve(rhs, solver_state)
            return torch.autograd.grad(result, rhs, rhs_bar, create_graph=True)[0]
