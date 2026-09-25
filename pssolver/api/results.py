"""Stable public result contracts shared by qualified applications."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable


class SimulationRunStatus(str, Enum):
    """Terminal state returned by the synchronous public runner."""

    DRY_RUN = "dry_run"
    COMPLETE = "complete"


@runtime_checkable
class SimulationObservationProtocol(Protocol):
    """Structural Q/velocity/pressure observation shared by applications."""

    step: int
    q: object
    velocity: object
    pressure: object


@runtime_checkable
class SimulationDiagnosticProtocol(Protocol):
    """Minimum structural identity of one application diagnostic."""

    step: int


def _frozen_json(value: Mapping[str, object]) -> Mapping[str, object]:
    payload = json.dumps(
        dict(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return MappingProxyType(json.loads(payload))


def _sha256(value: str, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256")
    return value


def _step_tuple(value: tuple[int, ...], description: str) -> tuple[int, ...]:
    try:
        normalized = tuple(value)
    except TypeError as exc:
        raise TypeError(f"{description} must be iterable") from exc
    if any(
        not isinstance(step, int) or isinstance(step, bool) or step < 0
        for step in normalized
    ):
        raise ValueError(f"{description} must contain non-negative integers")
    if tuple(sorted(set(normalized))) != normalized:
        raise ValueError(f"{description} must be strictly increasing")
    return normalized


@dataclass(frozen=True, slots=True)
class SimulationResult:
    """Application-neutral completion evidence returned by ``run_simulation``.

    Array-bearing observations and diagnostic records remain structural
    protocols, so the public API does not expose Plane- or Channel-specific
    workflow result classes.  No array is copied again while constructing this
    lightweight wrapper.
    """

    status: SimulationRunStatus
    application: str
    runtime_path: str
    source_simulation_sha256: str
    application_request_sha256: str
    output_directory: Path | None
    start_step: int | None
    final_step: int | None
    elapsed_seconds: float | None
    saved_steps: tuple[int, ...] = ()
    checkpoint_steps: tuple[int, ...] = ()
    final_observation: SimulationObservationProtocol | None = None
    diagnostics: tuple[SimulationDiagnosticProtocol, ...] = ()
    provenance: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.status, SimulationRunStatus):
            raise TypeError("status must be a SimulationRunStatus")
        for value, description in (
            (self.application, "application"),
            (self.runtime_path, "runtime_path"),
        ):
            if not isinstance(value, str) or not value:
                raise ValueError(f"{description} must be a non-empty string")
        _sha256(self.source_simulation_sha256, "source simulation identity")
        _sha256(self.application_request_sha256, "application request identity")
        object.__setattr__(
            self,
            "saved_steps",
            _step_tuple(self.saved_steps, "saved_steps"),
        )
        object.__setattr__(
            self,
            "checkpoint_steps",
            _step_tuple(self.checkpoint_steps, "checkpoint_steps"),
        )
        diagnostics = tuple(self.diagnostics)
        if any(
            not isinstance(value, SimulationDiagnosticProtocol)
            for value in diagnostics
        ):
            raise TypeError("diagnostics must satisfy SimulationDiagnosticProtocol")
        if any(
            not isinstance(value.step, int)
            or isinstance(value.step, bool)
            or value.step < 0
            for value in diagnostics
        ):
            raise ValueError("diagnostic steps must be non-negative integers")
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "provenance", _frozen_json(self.provenance))

        if self.status is SimulationRunStatus.DRY_RUN:
            if any(
                value is not None
                for value in (
                    self.output_directory,
                    self.start_step,
                    self.final_step,
                    self.elapsed_seconds,
                    self.final_observation,
                )
            ):
                raise ValueError("dry-run result cannot contain execution outputs")
            if self.saved_steps or self.checkpoint_steps or self.diagnostics:
                raise ValueError("dry-run result cannot contain workflow records")
            return

        if self.output_directory is None:
            raise ValueError("complete result requires an output directory")
        object.__setattr__(
            self,
            "output_directory",
            Path(self.output_directory).expanduser().resolve(),
        )
        for value, description in (
            (self.start_step, "start_step"),
            (self.final_step, "final_step"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(
                    f"complete result {description} must be non-negative"
                )
        if self.final_step < self.start_step:
            raise ValueError("final_step must not precede start_step")
        if (
            self.elapsed_seconds is None
            or not math.isfinite(float(self.elapsed_seconds))
            or self.elapsed_seconds < 0.0
        ):
            raise ValueError("complete result elapsed_seconds must be non-negative")
        if not isinstance(
            self.final_observation,
            SimulationObservationProtocol,
        ):
            raise TypeError(
                "complete result requires SimulationObservationProtocol"
            )
        if self.final_observation.step != self.final_step:
            raise ValueError("final observation step differs from final_step")

    @property
    def completed(self) -> bool:
        return self.status is SimulationRunStatus.COMPLETE

    def to_metadata(self) -> dict[str, object]:
        observation = self.final_observation
        observation_metadata = None
        if observation is not None:
            observation_metadata = {
                "step": observation.step,
                "q_shape": list(observation.q.shape),
                "q_dtype": str(observation.q.dtype),
                "velocity_shape": list(observation.velocity.shape),
                "velocity_dtype": str(observation.velocity.dtype),
                "pressure_shape": list(observation.pressure.shape),
                "pressure_dtype": str(observation.pressure.dtype),
            }
        return {
            "schema_version": 1,
            "status": self.status.value,
            "completed": self.completed,
            "application": self.application,
            "runtime_path": self.runtime_path,
            "source_simulation_sha256": self.source_simulation_sha256,
            "application_request_sha256": self.application_request_sha256,
            "output_directory": (
                None
                if self.output_directory is None
                else str(self.output_directory)
            ),
            "start_step": self.start_step,
            "final_step": self.final_step,
            "elapsed_seconds": self.elapsed_seconds,
            "saved_steps": list(self.saved_steps),
            "checkpoint_steps": list(self.checkpoint_steps),
            "final_observation": observation_metadata,
            "diagnostic_steps": [value.step for value in self.diagnostics],
            "provenance": dict(self.provenance),
        }


__all__ = [
    "SimulationDiagnosticProtocol",
    "SimulationObservationProtocol",
    "SimulationResult",
    "SimulationRunStatus",
]
