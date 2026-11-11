from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from pssolver.core.errors import SimulationStateError
from pssolver.models.active_nematics import Q_COMPONENTS, VELOCITY_COMPONENTS


@dataclass(frozen=True, slots=True)
class ChannelBerisEdwardsObservation:
    step: int
    q: np.ndarray
    velocity: np.ndarray
    pressure: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError("observation step must be non-negative")
        expected = (
            ("Q", self.q, 4, len(Q_COMPONENTS)),
            ("velocity", self.velocity, 4, len(VELOCITY_COMPONENTS)),
            ("pressure", self.pressure, 3, None),
        )
        for description, value, ndim, components in expected:
            if not isinstance(value, np.ndarray) or value.ndim != ndim:
                raise ValueError(f"{description} observation shape is invalid")
            if components is not None and value.shape[-1] != components:
                raise ValueError(f"{description} observation component count is invalid")
            if not np.issubdtype(value.dtype, np.floating):
                raise TypeError(f"{description} observation must be floating")
            if not np.isfinite(value).all():
                raise SimulationStateError(
                    f"Channel {description} observation is non-finite at step {self.step}",
                    application_family="channel_complete_stress",
                    stage="observation",
                    step=self.step,
                    details={"field": description},
                )
        if self.velocity.shape[:-1] != self.q.shape[:-1]:
            raise ValueError("velocity grid does not match Q")
        if self.pressure.shape != self.q.shape[:-1]:
            raise ValueError("pressure grid does not match Q")


@dataclass(frozen=True, slots=True)
class ChannelBerisEdwardsDiagnostic:
    step: int
    divergence_max: float
    divergence_rms: float
    pressure_mean: float
    pressure_iterations: int
    pressure_relative_residual: float

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError("diagnostic step must be non-negative")
        if not all(math.isfinite(float(value)) for value in self.as_tuple()[1:]):
            raise SimulationStateError(
                f"Channel diagnostic is non-finite at step {self.step}",
                application_family="channel_complete_stress",
                stage="diagnostic",
                step=self.step,
            )

    def as_tuple(self) -> tuple[int | float, ...]:
        return (
            self.step,
            self.divergence_max,
            self.divergence_rms,
            self.pressure_mean,
            self.pressure_iterations,
            self.pressure_relative_residual,
        )


CHANNEL_BERIS_EDWARDS_DIAGNOSTIC_DTYPE = np.dtype(
    [
        ("step", np.int64),
        ("divergence_max", np.float64),
        ("divergence_rms", np.float64),
        ("pressure_mean", np.float64),
        ("pressure_iterations", np.int64),
        ("pressure_relative_residual", np.float64),
    ]
)

CHANNEL_BERIS_EDWARDS_DIAGNOSTIC_HEADER = (
    "step,divergence_max,divergence_rms,pressure_mean,pressure_iterations,"
    "pressure_relative_residual"
)


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


def capture_channel_observation(fields, *, step):
    """Read the already synchronized Channel fields without another solve."""

    def stack(names):
        return np.stack([fields[name][0].detach().cpu().numpy() for name in names], axis=-1)

    return ChannelBerisEdwardsObservation(
        step, stack(Q_COMPONENTS), stack(VELOCITY_COMPONENTS), fields["p"][0].detach().cpu().numpy()
    )


def capture_channel_diagnostic(fields, *, step, projector, constraint_diagnostics):
    """Evaluate the historical diagnostic at the Runner observation boundary."""
    divergence = sum(
        projector.inverse_transform_full(*fields.gradient_hat(name, axis=axis))
        for axis, name in enumerate(VELOCITY_COMPONENTS)
    )
    return ChannelBerisEdwardsDiagnostic(
        step=step,
        divergence_max=float(divergence.abs().max().item()),
        divergence_rms=float(torch.sqrt(divergence.square().mean()).item()),
        pressure_mean=float(fields["p"].mean().item()),
        pressure_iterations=constraint_diagnostics["achieved_iteration_count"],
        pressure_relative_residual=constraint_diagnostics["achieved_normalized_residual"],
    )


class ChannelCapture:
    application_family = "channel_complete_stress"
    runtime_path = "channel_complete_stress"
    diagnostic_dtype = CHANNEL_BERIS_EDWARDS_DIAGNOSTIC_DTYPE
    diagnostic_header = CHANNEL_BERIS_EDWARDS_DIAGNOSTIC_HEADER
    progress_order_error = "progress must yield consecutive local steps from zero"
    wants_constraint_diagnostics = True

    def __init__(self, projector, *, hydrodynamics, checkpoint_diagnostics=None):
        self.projector = projector
        self.hydrodynamics = hydrodynamics
        self._checkpoint_diagnostics = checkpoint_diagnostics

    def checkpoint_provenance(self, metadata):
        return {
            "legacy_configuration_sha256": metadata["configuration_sha256"],
            "legacy_runtime_identity_sha256": metadata["runtime_identity_sha256"],
            "channel_constraint_diagnostics": (
                {} if self._checkpoint_diagnostics is None else self._checkpoint_diagnostics()
            ),
        }

    def observation(self, fields, *, step):
        return capture_channel_observation(fields, step=step)

    def diagnostic(self, fields, *, step, constraint_diagnostics):
        return capture_channel_diagnostic(
            fields,
            step=step,
            projector=self.projector,
            constraint_diagnostics=constraint_diagnostics,
        )
