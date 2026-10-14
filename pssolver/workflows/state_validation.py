"""Fail-closed finite-state checks at production workflow boundaries."""

from __future__ import annotations

from collections.abc import Iterable

import torch

from pssolver.api.errors import SimulationStateError


def require_finite_tensors(
    values: Iterable[tuple[str, torch.Tensor]],
    *,
    application_family: str,
    stage: str,
    step: int,
) -> None:
    """Validate a boundary snapshot with one device-to-host synchronization.

    This helper is deliberately used only when an observation, diagnostic,
    checkpoint, or final result already crosses a synchronization boundary. It
    is not part of the per-timestep hot path.
    """

    named = tuple(values)
    if not named:
        raise ValueError("finite-state validation requires at least one tensor")
    for name, value in named:
        if not isinstance(name, str) or not name:
            raise ValueError("finite-state tensor names must be non-empty")
        if not isinstance(value, torch.Tensor):
            raise TypeError("finite-state values must be tensors")
    devices = {value.device for _name, value in named}
    if len(devices) != 1:
        raise ValueError("finite-state tensors must share one device")
    flags = torch.stack(tuple(torch.isfinite(value).all() for _name, value in named))
    observed = tuple(bool(value) for value in flags.detach().cpu().tolist())
    failures = tuple(name for (name, _value), finite in zip(named, observed, strict=True) if not finite)
    if failures:
        raise SimulationStateError(
            f"{application_family} state is non-finite at {stage} step {step}",
            application_family=application_family,
            stage=stage,
            step=step,
            details={"nonfinite_fields": list(failures)},
        )


__all__ = ["require_finite_tensors"]
