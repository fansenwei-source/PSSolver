"""Immutable P8.3 request for complete-stress Channel execution."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from .simulation import SimulationSpec


CHANNEL_COMPLETE_STRESS_RUNTIME_PATH = "channel_complete_stress"


@dataclass(frozen=True, slots=True)
class ChannelBerisEdwardsRunSpec:
    """Exact public declaration consumed by the P8.3 application."""

    simulation: SimulationSpec

    def __post_init__(self) -> None:
        if not isinstance(self.simulation, SimulationSpec):
            raise TypeError("simulation must be a SimulationSpec")
        if self.simulation.equation_system.variant != (
            "complete_stress_beris_edwards"
        ):
            raise ValueError("Channel request requires complete-stress equations")
        if self.simulation.geometry.name != "rectangular_channel":
            raise ValueError("Channel request requires RectangularChannel")
        if self.simulation.execution.runtime_path != (
            CHANNEL_COMPLETE_STRESS_RUNTIME_PATH
        ):
            raise ValueError("Channel request has the wrong runtime path")

    @property
    def runtime_path(self) -> str:
        return CHANNEL_COMPLETE_STRESS_RUNTIME_PATH

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


__all__ = [
    "CHANNEL_COMPLETE_STRESS_RUNTIME_PATH",
    "ChannelBerisEdwardsRunSpec",
]
