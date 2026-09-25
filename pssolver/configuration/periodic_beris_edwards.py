"""Immutable P8.2 request for complete-stress periodic-box execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .simulation import SimulationSpec


PERIODIC_RUNTIME_PATH = "periodic_spectral"


@dataclass(frozen=True, slots=True)
class PeriodicBerisEdwardsRunSpec:
    """Exact public declaration consumed by the periodic application."""

    simulation: SimulationSpec

    def __post_init__(self) -> None:
        if not isinstance(self.simulation, SimulationSpec):
            raise TypeError("simulation must be a SimulationSpec")
        if self.simulation.equation_system.variant != (
            "complete_stress_beris_edwards"
        ):
            raise ValueError("periodic request requires complete-stress equations")
        if self.simulation.geometry.name != "periodic_box":
            raise ValueError("periodic request requires PeriodicBox geometry")
        if self.simulation.execution.runtime_path != PERIODIC_RUNTIME_PATH:
            raise ValueError("periodic request has the wrong runtime path")

    @property
    def runtime_path(self) -> str:
        return PERIODIC_RUNTIME_PATH

    @property
    def output_dir(self) -> Path:
        return Path(self.simulation.workflow.options["output_dir"])

    @property
    def dry_run(self) -> bool:
        return bool(self.simulation.invocation.options.get("dry_run", False))

    def canonical_sha256(self) -> str:
        return self.simulation.canonical_sha256()

    def runtime_identity_sha256(self) -> str:
        identities = self.simulation.identity_metadata()
        payload = {
            name: identities[name]
            for name in ("scientific", "discretization", "execution")
        }
        encoded = json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


__all__ = ["PERIODIC_RUNTIME_PATH", "PeriodicBerisEdwardsRunSpec"]
