"""Immutable tensor-free inputs to the public simulation compiler."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from types import MappingProxyType

from pssolver.core._canonical import canonical_json
from pssolver.core.device_tokens import validate_execution_device
from pssolver.core.numerics import (
    DEFAULT_DEALIAS_RULE,
    DEFAULT_PROJECTED_TRANSFORM_EXECUTION,
    DEFAULT_TRANSFORM_EXECUTION_ORDER,
    DealiasRule,
    NumericsConfig,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)


def _enum(value, enum_type, name):
    try:
        return enum_type(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"unsupported {name}: {value!r}") from exc


def _integer(value: object, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        qualifier = "positive" if minimum else "non-negative"
        raise ValueError(f"{name} must be a {qualifier} integer")
    return value


def _positive_real(value: object, name: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or value <= 0
    ):
        raise ValueError(f"{name} must be positive and finite")
    return float(value)


def _freeze_json(value):
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _mapping(values: Mapping[str, object] | None, name: str) -> Mapping[str, object]:
    if values is None:
        return MappingProxyType({})
    if not isinstance(values, Mapping) or not all(isinstance(key, str) for key in values):
        raise TypeError(f"{name} must be a string-keyed mapping")
    return _freeze_json(json.loads(canonical_json(dict(values), description=name)))


def _plain_json(value):
    if isinstance(value, Mapping):
        return {key: _plain_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain_json(item) for item in value]
    return value


class SpectralNumerics(NumericsConfig):
    """Precision, storage and nonlinear projection choices for one simulation.

    Attributes:
        dtype: Canonical precision token.
        pressure_solver: Optional immutable Channel pressure-solver configuration.
    """

    __slots__ = ("pressure_solver",)
    pressure_solver: Mapping[str, object] | None

    def __init__(
        self,
        *,
        dtype: str | Precision,
        spectral_storage: str | SpectralStorage,
        dealias_rule: str | DealiasRule = DEFAULT_DEALIAS_RULE,
        projected_transform_execution: str | ProjectedTransformExecution = (
            DEFAULT_PROJECTED_TRANSFORM_EXECUTION
        ),
        hermitian_axis: int | None = None,
        transform_execution_order: str | TransformExecutionOrder = (
            DEFAULT_TRANSFORM_EXECUTION_ORDER
        ),
        pressure_solver: Mapping[str, object] | None = None,
    ) -> None:
        """Declare spectral policies without allocating tensors.

        Args:
            dtype: Real precision token or precision enum.
            spectral_storage: Full or Hermitian-half spectral storage policy.
            dealias_rule: Nonlinear projection rule.
            projected_transform_execution: Implementation of projected transforms.
            hermitian_axis: Optional axis that stores the Hermitian half spectrum.
            transform_execution_order: Execution order of multidimensional transforms.
            pressure_solver: Optional string-keyed pressure-solver policy.

        Raises:
            TypeError: Pressure-solver settings are not a string-keyed mapping.
            ValueError: A numerical policy token or combination is not supported.
        """
        super().__init__(
            precision=_enum(dtype, Precision, "dtype"),
            dealias_rule=_enum(dealias_rule, DealiasRule, "dealias rule"),
            transform_execution_order=_enum(
                transform_execution_order, TransformExecutionOrder, "transform execution order"
            ),
            projected_transform_execution=_enum(
                projected_transform_execution,
                ProjectedTransformExecution,
                "projected transform execution",
            ),
            spectral_storage=_enum(spectral_storage, SpectralStorage, "spectral storage"),
            hermitian_axis=hermitian_axis,
        )
        object.__setattr__(
            self,
            "pressure_solver",
            None if pressure_solver is None else _mapping(pressure_solver, "pressure_solver"),
        )

    def __setattr__(self, name, value):
        raise AttributeError(f"cannot assign to field {name!r}")

    def __delattr__(self, name):
        raise AttributeError(f"cannot delete field {name!r}")

    @property
    def dtype(self) -> str:
        """Return the canonical precision token, not a torch dtype.

        Returns:
            The configured real-precision token.
        """
        return self.precision.value

    def to_metadata(self) -> dict[str, object]:
        """Serialize spectral and pressure policies.

        Returns:
            A fresh dictionary of tensor-free numerical options.
        """
        return {**super().to_metadata(), "pressure_solver": _plain_json(self.pressure_solver)}


@dataclass(frozen=True, slots=True)
class SemiImplicitEuler:
    """Declare the fixed timestep and optional dynamic-spectrum refresh cadence.

    Attributes:
        dt: Positive finite timestep in simulation time units.
        refresh: Optional cadence specified by steps or by physical time.
    """

    dt: float
    refresh: Mapping[str, object] | None = None

    def __post_init__(self):
        object.__setattr__(self, "dt", _positive_real(self.dt, "dt"))
        if self.refresh is None:
            return
        refresh = _mapping(self.refresh, "refresh")
        if set(refresh) == {"steps"}:
            _integer(refresh["steps"], "refresh steps", minimum=1)
        elif set(refresh) == {"physical_time"}:
            _positive_real(refresh["physical_time"], "refresh physical_time")
        else:
            raise ValueError("refresh must contain exactly 'steps' or 'physical_time'")
        object.__setattr__(self, "refresh", refresh)

    def to_metadata(self) -> dict[str, object]:
        """Serialize the timestep and refresh declaration.

        Returns:
            A fresh dictionary of time-stepping options.
        """
        return {"dt": self.dt, "refresh": _plain_json(self.refresh)}


@dataclass(frozen=True, slots=True, kw_only=True)
class TorchSpectralExecution:
    """Device and implementation options; construction does not resolve devices.

    Attributes:
        device: Explicit execution-device token.
        options: Immutable implementation options; do not repeat the device here.
    """

    device: str = "cpu"
    options: Mapping[str, object] | None = None

    def __post_init__(self):
        validate_execution_device(self.device)
        options = _mapping(self.options, "execution options")
        if "device" in options:
            raise ValueError("execution options must not repeat device")
        object.__setattr__(self, "options", options)

    def to_metadata(self) -> dict[str, object]:
        """Serialize the requested execution policy.

        Returns:
            A dictionary with the unresolved device token and plain options.
        """
        return {"device": self.device, "options": _plain_json(self.options)}


@dataclass(frozen=True, slots=True)
class GeneratedInitialCondition:
    """Request a generated field from a named initialization family.

    Attributes:
        family: Identifier of the initial-field generator.
        parameters: Immutable generator inputs; available keys depend on the family.
    """

    family: str
    parameters: Mapping[str, object] | None = None

    def __post_init__(self):
        if not isinstance(self.family, str) or not self.family.isidentifier():
            raise ValueError("initial condition family must be a Python identifier")
        object.__setattr__(
            self, "parameters", _mapping(self.parameters, "initial condition parameters")
        )

    def to_metadata(self) -> dict[str, object]:
        """Serialize the generated initial-field request.

        Returns:
            A fresh dictionary identifying the generator and its options.
        """
        return {
            "source": "generated",
            "family": self.family,
            "parameters": _plain_json(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class SnapshotInitialCondition:
    """Declare a compact-Q file in an existing snapshot directory.

    Attributes:
        directory: Directory containing the compact-Q snapshot series.
        step: Non-negative step at which to load the field.
        mode: Whether to branch from or resume the snapshot clock.
    """

    directory: str | Path
    step: int
    mode: str = "branch"

    def __post_init__(self):
        if self.mode not in {"branch", "resume"}:
            raise ValueError("snapshot mode must be 'branch' or 'resume'")
        _integer(self.step, "snapshot step", minimum=0)
        object.__setattr__(self, "directory", Path(self.directory))

    def to_metadata(self) -> dict[str, object]:
        """Serialize the initial snapshot declaration, without reading its arrays.

        Returns:
            A dictionary containing the directory, step, and clock mode.
        """
        return {
            "source": "snapshot",
            "directory": str(self.directory),
            "step": self.step,
            "mode": self.mode,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class Output:
    """Declare output scheduling and checkpoint/restart options.

    Attributes:
        directory: Directory in which the run publishes its outputs.
        steps: Positive number of timesteps to execute.
        save_interval: Number of timesteps between saved fields.
        diagnostic_interval: Number of timesteps between diagnostic records.
        save_start_step: First eligible field-save step.
        diagnostics: Whether diagnostics are enabled.
        save_hydrodynamics: Whether to save velocity and pressure with Q.
        checkpoint_interval: Optional number of timesteps between checkpoints.
        restart_from: Optional path to the checkpoint used for continuation.
        dry_run: Whether to validate without executing the simulation.
        validation_config_sha256: Optional lowercase SHA-256 of validation input.
    """

    directory: str | Path
    steps: int
    save_interval: int
    diagnostic_interval: int
    save_start_step: int
    diagnostics: bool
    save_hydrodynamics: bool
    checkpoint_interval: int | None = None
    restart_from: str | Path | None = None
    dry_run: bool = False
    validation_config_sha256: str | None = None

    def __post_init__(self):
        _integer(self.steps, "steps", minimum=1)
        _integer(self.save_start_step, "save_start_step", minimum=0)
        if self.restart_from is None and self.save_start_step > self.steps:
            raise ValueError("save_start_step must not exceed steps")
        for name in ("diagnostics", "save_hydrodynamics", "dry_run"):
            if type(getattr(self, name)) is not bool:
                raise TypeError(f"{name} must be a bool")
        _integer(self.save_interval, "save_interval", minimum=1)
        _integer(self.diagnostic_interval, "diagnostic_interval", minimum=1)
        if self.checkpoint_interval is not None:
            _integer(self.checkpoint_interval, "checkpoint_interval", minimum=1)
        digest = self.validation_config_sha256
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "validation_config_sha256 must be null or exactly 64 lowercase hexadecimal characters"
            )
        object.__setattr__(self, "directory", Path(self.directory))
        if self.restart_from is not None:
            object.__setattr__(self, "restart_from", Path(self.restart_from))

    def to_metadata(self) -> dict[str, object]:
        """Serialize the output declaration without creating its directory.

        Returns:
            A fresh dictionary with the output and continuation settings.
        """
        return {
            "directory": str(self.directory),
            "steps": self.steps,
            "save_interval": self.save_interval,
            "diagnostic_interval": self.diagnostic_interval,
            "save_start_step": self.save_start_step,
            "diagnostics": self.diagnostics,
            "save_hydrodynamics": self.save_hydrodynamics,
            "checkpoint_interval": self.checkpoint_interval,
            "restart_from": None if self.restart_from is None else str(self.restart_from),
            "dry_run": self.dry_run,
            "validation_config_sha256": self.validation_config_sha256,
        }


__all__ = [
    "GeneratedInitialCondition",
    "Output",
    "SemiImplicitEuler",
    "SnapshotInitialCondition",
    "SpectralNumerics",
    "TorchSpectralExecution",
]
