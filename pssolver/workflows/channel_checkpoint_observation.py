"""Side-effect-free Channel checkpoint observation helpers."""

from __future__ import annotations

import torch


def capture_synchronized_pressure_guess(adapter: object) -> torch.Tensor:
    """Synchronize checkpoint fields without changing the live warm start.

    The synchronized pressure guess belongs in the checkpoint so a resumed
    runtime follows the same trajectory as an uninterrupted runtime.  The
    running adapter, however, must retain the warm start it had before the
    observational solve; otherwise merely requesting a checkpoint changes its
    subsequent trajectory.
    """

    capture = getattr(adapter, "capture_pressure_guess", None)
    restore = getattr(adapter, "restore_pressure_guess", None)
    synchronize = getattr(adapter, "synchronize_for_observation", None)
    if not all(callable(value) for value in (capture, restore, synchronize)):
        raise TypeError(
            "Channel checkpoint adapter lacks pressure observation methods"
        )
    live_pressure_guess = capture()
    try:
        synchronize()
        checkpoint_pressure_guess = capture()
    finally:
        restore(live_pressure_guess)
    return checkpoint_pressure_guess


__all__ = ["capture_synchronized_pressure_guess"]
