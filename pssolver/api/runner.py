"""Public compile-once and execute-once simulation connection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from .simulation import Simulation

if TYPE_CHECKING:
    from pssolver.configuration.simulation import SimulationSpec
    from pssolver.planning.package_construction import (
        PackageRuntimeConstructionPlan,
    )
    from pssolver.planning.simulation import SimulationLoweringPlan


def _frozen_json(value: Mapping[str, object]) -> Mapping[str, object]:
    payload = json.dumps(
        dict(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return MappingProxyType(json.loads(payload))


@dataclass(frozen=True, slots=True)
class CompiledSimulation:
    """Immutable public evidence for one qualified application connection."""

    source: Simulation
    application: str
    application_specification: SimulationSpec
    lowering_plan: SimulationLoweringPlan
    construction_plan: PackageRuntimeConstructionPlan
    application_request: object
    normalization: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "normalization", _frozen_json(self.normalization))

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "source_simulation_sha256": self.source.canonical_sha256(),
            "application": self.application,
            "application_simulation_sha256": (
                self.application_specification.canonical_sha256()
            ),
            "lowering_plan": self.lowering_plan.to_metadata(),
            "construction_plan": self.construction_plan.to_metadata(),
            "application_request_type": (
                f"{type(self.application_request).__module__}."
                f"{type(self.application_request).__qualname__}"
            ),
            "application_request_sha256": (
                self.application_request.canonical_sha256()
            ),
            "normalization": dict(self.normalization),
            "fallback_allowed": False,
        }


def compile_simulation(simulation: Simulation) -> CompiledSimulation:
    """Compile a public declaration without allocating or running tensors."""

    from pssolver.configuration.public_simulation_runner import (
        compile_public_simulation,
    )

    if not isinstance(simulation, Simulation):
        raise TypeError("simulation must be a public Simulation")
    value = compile_public_simulation(simulation.specification)
    return CompiledSimulation(
        source=simulation,
        application=value.application,
        application_specification=value.application_simulation,
        lowering_plan=value.lowering_plan,
        construction_plan=value.construction_plan,
        application_request=value.run_spec,
        normalization=value.normalization,
    )


def run_simulation(
    simulation: Simulation | CompiledSimulation,
    *,
    progress: Iterable[int] | None = None,
    emit_metadata: bool = False,
) -> Any:
    """Run one qualified public simulation through its existing application.

    Compilation and application selection happen once before runtime
    construction.  There is no runtime fallback and no public dispatch in the
    timestep.
    """

    compiled = (
        simulation
        if isinstance(simulation, CompiledSimulation)
        else compile_simulation(simulation)
    )
    from pssolver.configuration.public_simulation_runner import (
        PUBLIC_PLANE_APPLICATION,
    )

    if compiled.application != PUBLIC_PLANE_APPLICATION:
        raise RuntimeError(
            "the compiled application is not connected to run_simulation yet; "
            "P7.7.9 compiles both qualified combinations, while P7.7.10 "
            "owns the common application runner and result protocol"
        )
    from pssolver.applications.plane_beris_edwards import (
        run_plane_beris_edwards,
    )

    return run_plane_beris_edwards(
        compiled.application_request,
        progress=progress,
        emit_metadata=emit_metadata,
    )


__all__ = ["CompiledSimulation", "compile_simulation", "run_simulation"]
