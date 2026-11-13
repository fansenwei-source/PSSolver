"""Stable public result contracts shared by qualified applications."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Protocol, runtime_checkable

from pssolver.core._canonical import canonical_json


class SimulationRunStatus(str, Enum):
    """Terminal state returned by the synchronous public runner.

    Attributes:
        value: Stable string indicating dry-run validation or completion.
    """

    DRY_RUN = "dry_run"
    COMPLETE = "complete"


@runtime_checkable
class SimulationObservationProtocol(Protocol):
    """Structural Q/velocity/pressure observation shared by applications.

    Attributes:
        step: Step at which the constraints and fields were observed.
        q: Compact Q tensor in the field's physical layout.
        velocity: Vector velocity in the field's physical layout.
        pressure: Scalar pressure in the field's physical layout.
    """

    step: int
    q: object
    velocity: object
    pressure: object


@runtime_checkable
class SimulationDiagnosticProtocol(Protocol):
    """Minimum structural identity of one application diagnostic.

    Attributes:
        step: Step at which the diagnostic was captured.
    """

    step: int


def _frozen_json(value: Mapping[str, object]) -> Mapping[str, object]:
    payload = canonical_json(dict(value))
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

    Attributes:
        status: Whether execution completed or only validation was requested.
        application: Name of the qualified application.
        plan_identity_sha256: Digest of the executable Plan.
        source_simulation_sha256: Digest of the source simulation declaration.
        output_directory: Output location, absent for a dry run.
        start_step: Starting clock, absent for a dry run.
        final_step: Final clock, absent for a dry run.
        elapsed_seconds: Elapsed execution time, absent for a dry run.
        saved_steps: Steps at which fields were saved.
        checkpoint_steps: Steps at which checkpoints were written.
        final_observation: Final constrained fields, absent for a dry run.
        diagnostics: Diagnostic records captured during execution.
        provenance: Immutable JSON-compatible run provenance.
    """

    status: SimulationRunStatus
    application: str
    plan_identity_sha256: str
    source_simulation_sha256: str
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
        if not isinstance(self.application, str) or not self.application:
            raise ValueError("application must be a non-empty string")
        _sha256(self.plan_identity_sha256, "Plan identity")
        _sha256(self.source_simulation_sha256, "source simulation identity")
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
        """Report whether the simulation actually executed.

        Returns:
            True for a completed run, false for a dry run.
        """
        return self.status is SimulationRunStatus.COMPLETE

    def to_metadata(self) -> dict[str, object]:
        """Serialize run evidence without copying field arrays.

        Returns:
            A fresh dictionary with identities, timing, output steps and array descriptors.
        """
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
            "plan_identity_sha256": self.plan_identity_sha256,
            "source_simulation_sha256": self.source_simulation_sha256,
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
