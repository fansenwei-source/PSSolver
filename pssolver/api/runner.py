"""Public compile-once and execute-once simulation connection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import json
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Callable

from .results import SimulationResult, SimulationRunStatus
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


@dataclass(frozen=True, slots=True)
class _PublicApplicationRunnerRegistration:
    """One application-level dispatch resolved before the timestep."""

    application: str
    runner: str
    invoke: Callable[
        [CompiledSimulation, Iterable[int] | None, bool],
        tuple[object | None, str | Path | None],
    ]


def _run_plane_application(
    compiled: CompiledSimulation,
    progress: Iterable[int] | None,
    emit_metadata: bool,
) -> tuple[object | None, str | Path | None]:
    from pssolver.applications.plane_beris_edwards import (
        run_plane_beris_edwards,
    )

    result = run_plane_beris_edwards(
        compiled.application_request,
        progress=progress,
        emit_metadata=emit_metadata,
    )
    return result, compiled.application_request.output_dir


def _run_channel_application(
    compiled: CompiledSimulation,
    progress: Iterable[int] | None,
    emit_metadata: bool,
) -> tuple[object, str | Path]:
    from pssolver.applications.channel_active_nematics import (
        run_channel_active_nematics,
    )

    if emit_metadata:
        print(json.dumps(compiled.to_metadata(), indent=2, sort_keys=True))
    result = run_channel_active_nematics(
        compiled.application_request,
        progress=progress,
    )
    request = compiled.application_request
    output = (
        request.generated_output_directory
        if request.initialization_mode == "generated"
        else request.snapshot_output_directory
    )
    return result, output


def _run_periodic_application(
    compiled: CompiledSimulation,
    progress: Iterable[int] | None,
    emit_metadata: bool,
) -> tuple[object | None, str | Path | None]:
    from pssolver.applications.periodic_beris_edwards import (
        run_periodic_beris_edwards,
    )

    result = run_periodic_beris_edwards(
        compiled.application_request,
        progress=progress,
        emit_metadata=emit_metadata,
    )
    output = None if result is None else compiled.application_request.output_dir
    return result, output


_PUBLIC_APPLICATION_RUNNERS = MappingProxyType(
    {
        "plane_complete_stress_beris_edwards": (
            _PublicApplicationRunnerRegistration(
                "plane_complete_stress_beris_edwards",
                (
                    "pssolver.applications.plane_beris_edwards."
                    "run_plane_beris_edwards"
                ),
                _run_plane_application,
            )
        ),
        "channel_legacy_active_force_active_nematics": (
            _PublicApplicationRunnerRegistration(
                "channel_legacy_active_force_active_nematics",
                (
                    "pssolver.applications.channel_active_nematics."
                    "run_channel_active_nematics"
                ),
                _run_channel_application,
            )
        ),
        "periodic_complete_stress_beris_edwards": (
            _PublicApplicationRunnerRegistration(
                "periodic_complete_stress_beris_edwards",
                (
                    "pssolver.applications.periodic_beris_edwards."
                    "run_periodic_beris_edwards"
                ),
                _run_periodic_application,
            )
        ),
    }
)


def _runtime_path(compiled: CompiledSimulation) -> str:
    value = compiled.application_request.runtime_path
    return value.value if hasattr(value, "value") else str(value)


def _result_provenance(compiled: CompiledSimulation) -> dict[str, object]:
    return {
        "application_simulation_sha256": (
            compiled.application_specification.canonical_sha256()
        ),
        "lowering_plan_sha256": compiled.lowering_plan.canonical_sha256(),
        "construction_plan_sha256": (
            compiled.construction_plan.canonical_sha256()
        ),
        "fallback_allowed": False,
    }


def _public_result(
    compiled: CompiledSimulation,
    application_result: object | None,
    *,
    output_directory: str | Path | None,
) -> SimulationResult:
    common = {
        "application": compiled.application,
        "runtime_path": _runtime_path(compiled),
        "source_simulation_sha256": compiled.source.canonical_sha256(),
        "application_request_sha256": (
            compiled.application_request.canonical_sha256()
        ),
        "provenance": _result_provenance(compiled),
    }
    if application_result is None:
        if (
            compiled.application
            not in {
                "plane_complete_stress_beris_edwards",
                "periodic_complete_stress_beris_edwards",
            }
            or compiled.application_request.dry_run is not True
        ):
            raise RuntimeError(
                "application returned no workflow result outside a qualified "
                "Plane dry-run"
            )
        return SimulationResult(
            status=SimulationRunStatus.DRY_RUN,
            output_directory=None,
            start_step=None,
            final_step=None,
            elapsed_seconds=None,
            **common,
        )
    required = (
        "start_step",
        "final_step",
        "elapsed_seconds",
        "saved_steps",
        "checkpoint_steps",
        "final_observation",
        "diagnostics",
    )
    missing = tuple(
        name for name in required if not hasattr(application_result, name)
    )
    if missing:
        raise TypeError(
            "application result does not satisfy the public workflow "
            f"contract; missing={missing!r}"
        )
    return SimulationResult(
        status=SimulationRunStatus.COMPLETE,
        output_directory=output_directory,
        start_step=application_result.start_step,
        final_step=application_result.final_step,
        elapsed_seconds=application_result.elapsed_seconds,
        saved_steps=tuple(application_result.saved_steps),
        checkpoint_steps=tuple(application_result.checkpoint_steps),
        final_observation=application_result.final_observation,
        diagnostics=tuple(application_result.diagnostics),
        **common,
    )


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
) -> SimulationResult:
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
    registration = _PUBLIC_APPLICATION_RUNNERS.get(compiled.application)
    if registration is None:
        raise RuntimeError(
            "compiled simulation has no qualified public application runner"
        )
    application_result, output_directory = registration.invoke(
        compiled,
        progress,
        emit_metadata,
    )
    return _public_result(
        compiled,
        application_result,
        output_directory=output_directory,
    )


__all__ = [
    "CompiledSimulation",
    "compile_simulation",
    "run_simulation",
]
