"""Caller-owned Periodic observation capture, independent of execution scheduling."""

import math
from dataclasses import dataclass

import numpy as np
import torch

from pssolver.errors import SimulationStateError
from pssolver.models.active_nematics.fields import Q_COMPONENTS


VELOCITY_COMPONENTS = ("ux", "uy", "uz")


@dataclass(frozen=True, slots=True)
class PeriodicObservation:
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
                    f"periodic {description} observation is non-finite at step {self.step}",
                    application_family="periodic_complete_stress",
                    stage="observation",
                    step=self.step,
                    details={"field": description},
                )
        if self.velocity.shape[:-1] != self.q.shape[:-1]:
            raise ValueError("velocity grid does not match Q")
        if self.pressure.shape != self.q.shape[:-1]:
            raise ValueError("pressure grid does not match Q")


@dataclass(frozen=True, slots=True)
class PeriodicDiagnostic:
    step: int
    divergence_max: float
    divergence_rms: float
    pressure_mean: float
    velocity_mean_norm: float

    def __post_init__(self) -> None:
        if not isinstance(self.step, int) or isinstance(self.step, bool) or self.step < 0:
            raise ValueError("diagnostic step must be non-negative")
        if not all(math.isfinite(float(value)) for value in self.as_tuple()[1:]):
            raise SimulationStateError(
                f"periodic diagnostic is non-finite at step {self.step}",
                application_family="periodic_complete_stress",
                stage="diagnostic",
                step=self.step,
            )

    def as_tuple(self) -> tuple[int | float, ...]:
        return (
            self.step,
            self.divergence_max,
            self.divergence_rms,
            self.pressure_mean,
            self.velocity_mean_norm,
        )


PERIODIC_DIAGNOSTIC_DTYPE = np.dtype(
    [
        ("step", np.int64),
        ("divergence_max", np.float64),
        ("divergence_rms", np.float64),
        ("pressure_mean", np.float64),
        ("velocity_mean_norm", np.float64),
    ]
)


PERIODIC_DIAGNOSTIC_HEADER = "step,divergence_max,divergence_rms,pressure_mean,velocity_mean_norm"


def capture_periodic_observation(fields, *, step):

    def array(name):
        return fields[name][0].detach().cpu().contiguous().numpy()

    return PeriodicObservation(
        step,
        np.stack(tuple(array(name) for name in Q_COMPONENTS), axis=-1),
        np.stack(tuple(array(name) for name in VELOCITY_COMPONENTS), axis=-1),
        array("p"),
    )


def capture_periodic_diagnostic(fields, *, step):
    divergence = sum(
        fields.gradient(name, axis=axis) for axis, name in enumerate(VELOCITY_COMPONENTS)
    )
    velocity_means = torch.stack(tuple(fields[name].mean() for name in VELOCITY_COMPONENTS))
    return PeriodicDiagnostic(
        step=step,
        divergence_max=float(divergence.abs().max().item()),
        divergence_rms=float(torch.sqrt(divergence.square().mean()).item()),
        pressure_mean=float(fields["p"].mean().item()),
        velocity_mean_norm=float(torch.linalg.vector_norm(velocity_means).item()),
    )


@dataclass(frozen=True, slots=True)
class PeriodicCapture:
    """Periodic output recipe supplied to the generic step scheduler."""

    hydrodynamics: bool
    runtime_path: str = "periodic_spectral"
    application_family: str = "periodic_complete_stress"
    observation = staticmethod(capture_periodic_observation)
    diagnostic = staticmethod(capture_periodic_diagnostic)
    diagnostic_dtype = PERIODIC_DIAGNOSTIC_DTYPE
    diagnostic_header = PERIODIC_DIAGNOSTIC_HEADER
