"""Tensor-free execution errors shared by solver and public API layers."""

from __future__ import annotations

from collections.abc import Mapping
import json
from types import MappingProxyType


def _frozen_details(value: Mapping[str, object]) -> Mapping[str, object]:
    try:
        payload = json.dumps(
            dict(value),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise TypeError("simulation error details must be JSON-compatible") from exc
    return MappingProxyType(json.loads(payload))


class SimulationExecutionError(RuntimeError):
    """Base class for failures after public simulation validation succeeds."""

    code = "simulation_execution_error"

    def __init__(
        self,
        message: str,
        *,
        application_family: str,
        stage: str,
        step: int,
        details: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        if not isinstance(application_family, str) or not application_family:
            raise ValueError("application_family must be a non-empty string")
        if not isinstance(stage, str) or not stage:
            raise ValueError("stage must be a non-empty string")
        if not isinstance(step, int) or isinstance(step, bool) or step < 0:
            raise ValueError("step must be a non-negative integer")
        self.application_family = application_family
        self.stage = stage
        self.step = step
        self.details = _frozen_details(details or {})

    def to_metadata(self) -> dict[str, object]:
        return {
            "code": self.code,
            "message": str(self),
            "application_family": self.application_family,
            "stage": self.stage,
            "step": self.step,
            "details": dict(self.details),
        }


class SimulationStateError(SimulationExecutionError, ValueError):
    """A production observation, diagnostic, or checkpoint is non-finite."""

    code = "simulation_state_error"


class SimulationRestartError(SimulationExecutionError, ValueError):
    """A requested restart is incompatible before timestep execution."""

    code = "simulation_restart_error"


__all__ = [
    "SimulationExecutionError",
    "SimulationRestartError",
    "SimulationStateError",
]
