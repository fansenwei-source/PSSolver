"""Temporal scheme contract; only semi-implicit Euler is used in S-25."""

from typing import Protocol


class ImexScheme(Protocol):
    history_length: int

    def implicit_denominator(self, linear, dt): ...
    def explicit_update(self, u, nonlinear, history, dt, *, out=None): ...
    def next_history(self, u, nonlinear, history): ...


class SemiImplicitEuler:
    history_length = 0

    def implicit_denominator(self, linear, dt):
        return 1 - linear * dt

    def explicit_update(self, u, nonlinear, history, dt, *, out=None):
        if history:
            raise ValueError("Euler history must be empty")
        if out is None:
            return u + dt * nonlinear
        if out is not u:
            out.copy_(u)
        return out.add_(dt * nonlinear)

    def next_history(self, u, nonlinear, history):
        return ()
