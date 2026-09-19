"""Provisional Plane run-configuration value objects.

The types in this module are disconnected Phase 2 leaves.  They do not
construct a runtime, decompose the supported flat facade, serialize schema-v1
metadata, or select a numerical implementation.  Their small validation
boundaries make ownership explicit before any production consumer migrates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path

from .plane_beris_edwards import PlaneRuntimePath, SpectralRefreshSpec


def _positive_finite(value: object, description: str) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or float(value) <= 0.0
    ):
        raise ValueError(f"{description} must be positive and finite")
    return float(value)


def _positive_integer(value: object, description: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value <= 0
    ):
        raise ValueError(f"{description} must be a positive integer")
    return value


def _non_negative_integer(value: object, description: str) -> int:
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
        or value < 0
    ):
        raise ValueError(f"{description} must be a non-negative integer")
    return value


def _require_bool(value: object, description: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{description} must be a bool")
    return value


def _path(value: object, description: str) -> Path:
    try:
        return Path(value)
    except TypeError as exc:
        raise TypeError(f"{description} must be path-like") from exc


def _optional_path(value: object, description: str) -> Path | None:
    if value is None:
        return None
    return _path(value, description)


def _require_choice(
    value: object,
    choices: frozenset[str],
    description: str,
) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ValueError(
            f"{description} must be one of {tuple(sorted(choices))!r}"
        )
    return value


@dataclass(frozen=True, slots=True)
class PlaneTimeSteppingSpec:
    """Time increment and resolved dynamic-spectrum refresh schedule."""

    dt: float
    spectral_refresh: SpectralRefreshSpec

    def __post_init__(self) -> None:
        dt = _positive_finite(self.dt, "dt")
        if not isinstance(self.spectral_refresh, SpectralRefreshSpec):
            raise TypeError(
                "spectral_refresh must be a SpectralRefreshSpec"
            )
        # Consistency between dt and the resolved refresh interval is a
        # composition-root invariant.  Keeping it out of this disconnected
        # leaf avoids moving legacy direct-constructor validation in P2.1.
        object.__setattr__(self, "dt", dt)

    def to_metadata(self) -> dict[str, object]:
        return {
            "dt": self.dt,
            "spectral_refresh": self.spectral_refresh.to_metadata(),
        }


@dataclass(frozen=True, slots=True)
class PlaneBerisEdwardsExecutionSpec:
    """Requested implementation and device policies for one Plane run."""

    device: str
    tf32: str
    molecular_field_linear_space: str
    stress_divergence_sum_space: str
    pointwise_execution: str
    disable_q_gradient_reuse: bool
    runtime_path: PlaneRuntimePath

    def __post_init__(self) -> None:
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")
        _require_choice(self.tf32, frozenset({"off", "on"}), "tf32")
        _require_choice(
            self.molecular_field_linear_space,
            frozenset({"physical", "spectral"}),
            "molecular_field_linear_space",
        )
        _require_choice(
            self.stress_divergence_sum_space,
            frozenset({"physical", "spectral"}),
            "stress_divergence_sum_space",
        )
        _require_choice(
            self.pointwise_execution,
            frozenset({"eager", "compile"}),
            "pointwise_execution",
        )
        _require_bool(
            self.disable_q_gradient_reuse,
            "disable_q_gradient_reuse",
        )
        if not isinstance(self.runtime_path, PlaneRuntimePath):
            raise TypeError("runtime_path must be a PlaneRuntimePath")
        # The separated-canary/cache compatibility rule also spans facade
        # ownership groups and remains a later composition-root check.

    def to_metadata(self) -> dict[str, object]:
        return {
            "device": self.device,
            "tf32": self.tf32,
            "molecular_field_linear_space": (
                self.molecular_field_linear_space
            ),
            "stress_divergence_sum_space": (
                self.stress_divergence_sum_space
            ),
            "pointwise_execution": self.pointwise_execution,
            "disable_q_gradient_reuse": self.disable_q_gradient_reuse,
            "runtime_path": self.runtime_path.value,
        }


@dataclass(frozen=True, slots=True)
class PlaneWorkflowSpec:
    """Output, observation, checkpoint, and restart choices."""

    output_dir: Path
    steps: int
    save_start_step: int
    save_interval: int
    diagnostic_interval: int
    diagnostics: bool
    save_hydrodynamics: bool
    checkpoint_interval: int | None
    restart_from: Path | None

    def __post_init__(self) -> None:
        output_dir = _path(self.output_dir, "output_dir")
        steps = _positive_integer(self.steps, "steps")
        save_start = _non_negative_integer(
            self.save_start_step,
            "save_start_step",
        )
        if save_start > steps:
            raise ValueError("save_start_step must not exceed steps")
        _positive_integer(self.save_interval, "save_interval")
        _positive_integer(
            self.diagnostic_interval,
            "diagnostic_interval",
        )
        _require_bool(self.diagnostics, "diagnostics")
        _require_bool(self.save_hydrodynamics, "save_hydrodynamics")
        checkpoint_interval = self.checkpoint_interval
        if checkpoint_interval is not None:
            _positive_integer(checkpoint_interval, "checkpoint_interval")
        restart_from = _optional_path(self.restart_from, "restart_from")
        object.__setattr__(self, "output_dir", output_dir)
        object.__setattr__(self, "steps", steps)
        object.__setattr__(self, "save_start_step", save_start)
        object.__setattr__(self, "restart_from", restart_from)

    def to_metadata(self) -> dict[str, object]:
        return {
            "output_dir": str(self.output_dir),
            "steps": self.steps,
            "save_start_step": self.save_start_step,
            "save_interval": self.save_interval,
            "diagnostic_interval": self.diagnostic_interval,
            "diagnostics": self.diagnostics,
            "save_hydrodynamics": self.save_hydrodynamics,
            "checkpoint_interval": self.checkpoint_interval,
            "restart_from": (
                str(self.restart_from)
                if self.restart_from is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class PlaneInvocationSpec:
    """Non-scientific invocation and validation provenance controls."""

    validation_config_sha256: str | None
    dry_run: bool

    def __post_init__(self) -> None:
        digest = self.validation_config_sha256
        if digest is not None and (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError(
                "validation_config_sha256 must be 64 lowercase hexadecimal "
                "characters or None"
            )
        _require_bool(self.dry_run, "dry_run")

    def to_metadata(self) -> dict[str, object]:
        return {
            "validation_config_sha256": self.validation_config_sha256,
            "dry_run": self.dry_run,
        }


__all__ = [
    "PlaneBerisEdwardsExecutionSpec",
    "PlaneInvocationSpec",
    "PlaneTimeSteppingSpec",
    "PlaneWorkflowSpec",
]
