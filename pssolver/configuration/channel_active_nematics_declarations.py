"""Tensor-free declarations for the provisional Channel run specification.

These value objects describe the existing Channel oracle.  They do not build
the solver, allocate tensors, import Torch, or select a production runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math
from numbers import Integral, Real
from pathlib import Path

from pssolver.core import NumericsConfig
from pssolver.geometries import RectangularChannel
from pssolver.systems.stokes import (
    IncompressibleStokesSystemSpec,
    PressureGauge,
    TangentialZeroModePolicy,
)


CHANNEL_Q_BOUNDARIES = ("periodic", "neumann", "neumann")
CHANNEL_VELOCITY_BOUNDARIES = ("periodic", "dirichlet", "dirichlet")
CHANNEL_PRESSURE_BOUNDARIES = ("periodic", "neumann", "neumann")


def _finite(value: object, description: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{description} must be finite")
    return float(value)


def _positive(value: object, description: str) -> float:
    normalized = _finite(value, description)
    if normalized <= 0.0:
        raise ValueError(f"{description} must be positive")
    return normalized


def _non_negative(value: object, description: str) -> float:
    normalized = _finite(value, description)
    if normalized < 0.0:
        raise ValueError(f"{description} must be non-negative")
    return normalized


def _integer(value: object, description: str, *, positive: bool) -> int:
    if not isinstance(value, Integral) or isinstance(value, bool):
        raise ValueError(f"{description} must be an integer")
    normalized = int(value)
    if positive and normalized <= 0:
        raise ValueError(f"{description} must be positive")
    if not positive and normalized < 0:
        raise ValueError(f"{description} must be non-negative")
    return normalized


def _path(value: object, description: str) -> Path:
    try:
        return Path(value)
    except TypeError as exc:
        raise TypeError(f"{description} must be path-like") from exc


class ChannelRuntimePath(str, Enum):
    """Runtime selectors available before the compiled Channel path exists."""

    LEGACY_CHANNEL = "legacy_channel"


@dataclass(frozen=True, slots=True)
class ChannelBoundaryConditions:
    """Frozen homogeneous boundary spaces of the current Channel oracle."""

    q: tuple[str, str, str] = CHANNEL_Q_BOUNDARIES
    velocity: tuple[str, str, str] = CHANNEL_VELOCITY_BOUNDARIES
    pressure: tuple[str, str, str] = CHANNEL_PRESSURE_BOUNDARIES

    def __post_init__(self) -> None:
        values = (
            ("q", self.q, CHANNEL_Q_BOUNDARIES),
            ("velocity", self.velocity, CHANNEL_VELOCITY_BOUNDARIES),
            ("pressure", self.pressure, CHANNEL_PRESSURE_BOUNDARIES),
        )
        for name, value, expected in values:
            try:
                normalized = tuple(value)
            except TypeError as exc:
                raise TypeError(f"{name} boundaries must be iterable") from exc
            if normalized != expected:
                raise ValueError(
                    f"P7.1 only declares the qualified Channel {name} "
                    f"boundaries {expected!r}"
                )
            object.__setattr__(self, name, normalized)

    def to_metadata(self) -> dict[str, list[str]]:
        return {
            "q": list(self.q),
            "velocity": list(self.velocity),
            "pressure": list(self.pressure),
        }


CHANNEL_BOUNDARIES = ChannelBoundaryConditions()


@dataclass(frozen=True, slots=True)
class ChannelActiveNematicMaterialSpec:
    """Physical parameters in the rho-parameterized legacy Channel model."""

    rho: float
    elastic_constant: float
    activity: float
    beta: float
    friction: float
    viscosity: float
    flow_alignment: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "rho", _positive(self.rho, "rho"))
        object.__setattr__(
            self,
            "elastic_constant",
            _positive(self.elastic_constant, "elastic_constant"),
        )
        object.__setattr__(self, "activity", _finite(self.activity, "activity"))
        object.__setattr__(self, "beta", _finite(self.beta, "beta"))
        object.__setattr__(
            self,
            "friction",
            _non_negative(self.friction, "friction"),
        )
        object.__setattr__(
            self,
            "viscosity",
            _positive(self.viscosity, "viscosity"),
        )
        object.__setattr__(
            self,
            "flow_alignment",
            _finite(self.flow_alignment, "flow_alignment"),
        )

    @property
    def ldg_a(self) -> float:
        return 1.0 - self.rho / 3.0

    @property
    def ldg_b(self) -> float:
        return -self.rho

    @property
    def ldg_c(self) -> float:
        return self.rho

    def to_metadata(self) -> dict[str, float]:
        return {
            "rho": self.rho,
            "ldg_a": self.ldg_a,
            "ldg_b": self.ldg_b,
            "ldg_c": self.ldg_c,
            "elastic_constant": self.elastic_constant,
            "activity": self.activity,
            "beta": self.beta,
            "friction": self.friction,
            "viscosity": self.viscosity,
            "flow_alignment": self.flow_alignment,
        }


@dataclass(frozen=True, slots=True)
class ChannelPressureSolverSpec:
    """Convergence and deterministic-iteration controls for Channel PCG."""

    relative_tolerance: float
    max_iterations: int
    fixed_iterations: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relative_tolerance",
            _positive(self.relative_tolerance, "relative_tolerance"),
        )
        object.__setattr__(
            self,
            "max_iterations",
            _integer(self.max_iterations, "max_iterations", positive=True),
        )
        if self.fixed_iterations is not None:
            object.__setattr__(
                self,
                "fixed_iterations",
                _integer(
                    self.fixed_iterations,
                    "fixed_iterations",
                    positive=True,
                ),
            )

    def to_metadata(self) -> dict[str, object]:
        return {
            "algorithm": "preconditioned_conjugate_gradient",
            "relative_tolerance": self.relative_tolerance,
            "max_iterations": self.max_iterations,
            "fixed_iterations": self.fixed_iterations,
            "warm_start": True,
        }


def build_channel_stokes_spec(
    material: ChannelActiveNematicMaterialSpec,
) -> IncompressibleStokesSystemSpec:
    """Build the tensor-free Channel Stokes request from material values."""

    if not isinstance(material, ChannelActiveNematicMaterialSpec):
        raise TypeError("material must be a ChannelActiveNematicMaterialSpec")
    return IncompressibleStokesSystemSpec(
        name="channel_stokes",
        force_components=("fx", "fy", "fz"),
        velocity_components=("ux", "uy", "uz"),
        pressure_component="p",
        viscosity=material.viscosity,
        friction=material.friction,
        pressure_gauge=PressureGauge.ZERO_MEAN,
        tangential_zero_mode_policy=TangentialZeroModePolicy.NOT_APPLICABLE,
    )


@dataclass(frozen=True, slots=True)
class ChannelInitialConditionSpec:
    """Generated or snapshot-backed initial-condition request."""

    mode: str
    seed: int
    initial_s: float
    noise_theta: float
    noise_phi: float
    smoothing_sigma: tuple[float, float, float]
    snapshot_mode: str
    snapshot_directory: Path
    snapshot_step: int

    def __post_init__(self) -> None:
        if self.mode not in {"generated", "snapshot"}:
            raise ValueError("initialization mode must be generated or snapshot")
        if self.snapshot_mode not in {"resume", "branch"}:
            raise ValueError("snapshot_mode must be resume or branch")
        object.__setattr__(self, "seed", _integer(self.seed, "seed", positive=False))
        object.__setattr__(
            self,
            "initial_s",
            _positive(self.initial_s, "initial_s"),
        )
        object.__setattr__(
            self,
            "noise_theta",
            _non_negative(self.noise_theta, "noise_theta"),
        )
        object.__setattr__(
            self,
            "noise_phi",
            _non_negative(self.noise_phi, "noise_phi"),
        )
        try:
            sigma = tuple(self.smoothing_sigma)
        except TypeError as exc:
            raise TypeError("smoothing_sigma must be iterable") from exc
        if len(sigma) != 3:
            raise ValueError("smoothing_sigma must contain three values")
        object.__setattr__(
            self,
            "smoothing_sigma",
            tuple(_positive(value, "smoothing sigma") for value in sigma),
        )
        object.__setattr__(
            self,
            "snapshot_directory",
            _path(self.snapshot_directory, "snapshot_directory"),
        )
        object.__setattr__(
            self,
            "snapshot_step",
            _integer(self.snapshot_step, "snapshot_step", positive=False),
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "generated": {
                "name": "aligned_x_smooth_noise",
                "seed": self.seed,
                "initial_s": self.initial_s,
                "noise_theta": self.noise_theta,
                "noise_phi": self.noise_phi,
                "smoothing_sigma": list(self.smoothing_sigma),
            },
            "snapshot": {
                "mode": self.snapshot_mode,
                "directory": str(self.snapshot_directory),
                "step": self.snapshot_step,
            },
        }


@dataclass(frozen=True, slots=True)
class ChannelExecutionSpec:
    """Requested device, precision, and runtime identity."""

    device: str
    dtype: str
    batch_size: int
    runtime_path: ChannelRuntimePath

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        if self.dtype not in {"float32", "float64"}:
            raise ValueError("dtype must be float32 or float64")
        object.__setattr__(
            self,
            "batch_size",
            _integer(self.batch_size, "batch_size", positive=True),
        )
        if not isinstance(self.runtime_path, ChannelRuntimePath):
            raise TypeError("runtime_path must be a ChannelRuntimePath")

    def to_metadata(self) -> dict[str, object]:
        return {
            "device": self.device,
            "dtype": self.dtype,
            "batch_size": self.batch_size,
            "runtime_path": self.runtime_path.value,
            "fallback_allowed": False,
        }


@dataclass(frozen=True, slots=True)
class ChannelWorkflowSpec:
    """Finite run and output schedule without execution behavior."""

    steps: int
    save_interval: int
    diagnostics_enabled: bool
    diagnostic_interval: int
    generated_output_directory: Path
    snapshot_output_directory: Path

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "steps",
            _integer(self.steps, "steps", positive=True),
        )
        object.__setattr__(
            self,
            "save_interval",
            _integer(self.save_interval, "save_interval", positive=True),
        )
        object.__setattr__(
            self,
            "diagnostic_interval",
            _integer(
                self.diagnostic_interval,
                "diagnostic_interval",
                positive=True,
            ),
        )
        if not isinstance(self.diagnostics_enabled, bool):
            raise TypeError("diagnostics_enabled must be a bool")
        object.__setattr__(
            self,
            "generated_output_directory",
            _path(
                self.generated_output_directory,
                "generated_output_directory",
            ),
        )
        object.__setattr__(
            self,
            "snapshot_output_directory",
            _path(self.snapshot_output_directory, "snapshot_output_directory"),
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "steps": self.steps,
            "save_interval": self.save_interval,
            "diagnostics_enabled": self.diagnostics_enabled,
            "diagnostic_interval": self.diagnostic_interval,
            "generated_output_directory": str(self.generated_output_directory),
            "snapshot_output_directory": str(self.snapshot_output_directory),
        }


@dataclass(frozen=True, slots=True)
class ChannelRunComponents:
    """Ownership-separated view of one legacy-compatible Channel request."""

    geometry: RectangularChannel
    boundaries: ChannelBoundaryConditions
    numerics: NumericsConfig
    material: ChannelActiveNematicMaterialSpec
    stokes: IncompressibleStokesSystemSpec
    pressure_solver: ChannelPressureSolverSpec
    initial_condition: ChannelInitialConditionSpec
    execution: ChannelExecutionSpec
    workflow: ChannelWorkflowSpec
    dt: float

    def __post_init__(self) -> None:
        declarations = (
            ("geometry", self.geometry, RectangularChannel),
            ("boundaries", self.boundaries, ChannelBoundaryConditions),
            ("numerics", self.numerics, NumericsConfig),
            ("material", self.material, ChannelActiveNematicMaterialSpec),
            ("stokes", self.stokes, IncompressibleStokesSystemSpec),
            ("pressure_solver", self.pressure_solver, ChannelPressureSolverSpec),
            ("initial_condition", self.initial_condition, ChannelInitialConditionSpec),
            ("execution", self.execution, ChannelExecutionSpec),
            ("workflow", self.workflow, ChannelWorkflowSpec),
        )
        for name, value, value_type in declarations:
            if not isinstance(value, value_type):
                raise TypeError(f"{name} must be a {value_type.__name__}")
        object.__setattr__(self, "dt", _positive(self.dt, "dt"))
        if self.geometry.periodic_axes != (0,):
            raise ValueError("P7.1 Channel requires streamwise axis 0")
        if self.stokes.friction != self.material.friction:
            raise ValueError("material and Stokes friction must agree")
        if self.stokes.viscosity != self.material.viscosity:
            raise ValueError("material and Stokes viscosity must agree")

    def to_metadata(self) -> dict[str, object]:
        return {
            "geometry": self.geometry.to_metadata(),
            "boundaries": self.boundaries.to_metadata(),
            "numerics": self.numerics.to_metadata(),
            "material": self.material.to_metadata(),
            "stokes": self.stokes.to_metadata(),
            "pressure_solver": self.pressure_solver.to_metadata(),
            "initial_condition": self.initial_condition.to_metadata(),
            "execution": self.execution.to_metadata(),
            "workflow": self.workflow.to_metadata(),
            "dt": self.dt,
        }


__all__ = [
    "CHANNEL_BOUNDARIES",
    "CHANNEL_PRESSURE_BOUNDARIES",
    "CHANNEL_Q_BOUNDARIES",
    "CHANNEL_VELOCITY_BOUNDARIES",
    "ChannelActiveNematicMaterialSpec",
    "ChannelBoundaryConditions",
    "ChannelExecutionSpec",
    "ChannelInitialConditionSpec",
    "ChannelPressureSolverSpec",
    "ChannelRunComponents",
    "ChannelRuntimePath",
    "ChannelWorkflowSpec",
    "build_channel_stokes_spec",
]
