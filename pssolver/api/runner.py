"""Compile public declarations and run their prepared immutable inputs."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from pssolver.core._canonical import canonical_sha256
from pssolver.planning.assemble import AssembledProblem, AssemblyRejection, assemble
from pssolver.planning.engine_plan import Plan

from .declarations import GeneratedInitialCondition, Output, SnapshotInitialCondition
from .errors import PublicSimulationCompilationError
from .lowering import EngineConstruction, lower_simulation
from .results import SimulationResult, SimulationRunStatus
from .simulation import Simulation


class _ResultIdentity(TypedDict):
    application: str
    plan_identity_sha256: str
    source_simulation_sha256: str
    provenance: dict[str, object]


@dataclass(frozen=True, slots=True)
class _RunCarrier:
    """Tensor-free owner of everything required to execute a compiled run."""

    assembled: AssembledProblem
    construction: EngineConstruction
    initial_condition: GeneratedInitialCondition | SnapshotInitialCondition
    output: Output


@dataclass(frozen=True, slots=True)
class CompiledSimulation:
    """Public Plan, source declaration and their independent identities.

    Attributes:
        source: Original immutable simulation declaration.
        plan: Assembled tensor-free execution plan.
        plan_identity_sha256: Canonical digest of the execution plan.
        declaration_sha256: Canonical digest of the source declaration.
    """

    source: Simulation
    plan: Plan
    plan_identity_sha256: str
    declaration_sha256: str
    _run: _RunCarrier | None = field(default=None, init=False, compare=False, hash=False, repr=False)

    def validate_integrity(self) -> None:
        """Check the declaration, Plan and private preparation agree.

        Raises:
            ValueError: The declaration, Plan, or prepared inputs disagree.
        """
        if self.declaration_sha256 != self.source.canonical_sha256():
            raise ValueError("compiled simulation declaration identity does not match source")
        plan_metadata = self.plan.to_metadata()
        stored = plan_metadata.pop("identity_sha256")
        if stored != self.plan_identity_sha256 or canonical_sha256(plan_metadata) != stored:
            raise ValueError("compiled simulation Plan identity does not match Plan")
        if self._run is not None and (
            self._run.assembled.plan is not self.plan
            or self._run.initial_condition is not self.source.initial_condition
            or self._run.output is not self.source.output
        ):
            raise ValueError("compiled simulation runtime inputs do not match source and Plan")


def compile_simulation(simulation: Simulation) -> CompiledSimulation:
    """Assemble once, without resolving devices or importing tensor runtimes.

    Args:
        simulation: A complete public simulation declaration.

    Returns:
        The source declaration and its prepared immutable Plan.

    Raises:
        TypeError: The input is not a public 0.5 simulation declaration.
        PublicSimulationCompilationError: Lowering or assembly rejects the declaration.
    """
    if not isinstance(simulation, Simulation):
        raise TypeError("expected a 0.5 Simulation; see docs/migration_0_4_to_0_5.md")
    try:
        problem, construction = lower_simulation(simulation)
        assembled = assemble(problem)
    except AssemblyRejection as exc:
        raise PublicSimulationCompilationError(
            exc.code, exc.reason, kind=exc.kind, source=exc.source, opens_in=exc.opens_in,
        ) from exc
    result = CompiledSimulation(
        simulation, assembled.plan, assembled.plan.identity_sha256, simulation.canonical_sha256(),
    )
    carrier = _RunCarrier(assembled, construction, simulation.initial_condition, simulation.output)
    object.__setattr__(result, "_run", carrier)
    return result


def run_simulation(
    simulation: Simulation | CompiledSimulation,
    *,
    progress: Iterable[int] | None = None,
    emit_metadata: bool = False,
) -> SimulationResult:
    """Run a public declaration or the exact product of compile_simulation.

    Args:
        simulation: A declaration or the unchanged result of compilation.
        progress: Optional iterable of progress indices for the run.
        emit_metadata: Whether to emit metadata through the runtime's output channel.

    Returns:
        Dry-run or completion evidence with identities and scheduled outputs.

    Raises:
        TypeError: The input is invalid or lacks the original prepared runtime data.
        ValueError: A compiled declaration or Plan fails its integrity check.
        PublicSimulationCompilationError: Compilation rejects a supplied declaration.
    """
    compiled = simulation if isinstance(simulation, CompiledSimulation) else compile_simulation(simulation)
    if compiled._run is None:
        raise TypeError("run_simulation requires a product of compile_simulation")
    compiled.validate_integrity()
    from pssolver.runtime.simulation_run import run_assembled_simulation

    preflight = None
    if compiled.source.output.restart_from is not None:
        from .legacy_checkpoints import continue_from_v04_checkpoint

        def preflight(dtype):
            return continue_from_v04_checkpoint(compiled, dtype=dtype)

    runtime_result = run_assembled_simulation(
        compiled, progress=progress, emit_metadata=emit_metadata,
        legacy_preflight=preflight,
    )
    application = {
        "periodic_box": "periodic_complete_stress_beris_edwards",
        "plane_slab": "plane_complete_stress_beris_edwards",
        "rectangular_channel": "channel_complete_stress_beris_edwards",
    }[compiled.plan.geometry.kind]
    provenance: dict[str, object] = {
        "plan_identity_sha256": compiled.plan_identity_sha256,
        "declaration_sha256": compiled.declaration_sha256,
    }
    common: _ResultIdentity = {
        "application": application,
        "plan_identity_sha256": compiled.plan_identity_sha256,
        "source_simulation_sha256": compiled.declaration_sha256,
        "provenance": provenance,
    }
    if runtime_result is None:
        return SimulationResult(
            SimulationRunStatus.DRY_RUN, output_directory=None, start_step=None,
            final_step=None, elapsed_seconds=None, **common,
        )
    return SimulationResult(
        SimulationRunStatus.COMPLETE,
        output_directory=Path(compiled.source.output.directory),
        start_step=runtime_result.start_step, final_step=runtime_result.final_step,
        elapsed_seconds=runtime_result.elapsed_seconds,
        saved_steps=runtime_result.saved_steps,
        checkpoint_steps=runtime_result.checkpoint_steps,
        final_observation=runtime_result.final_observation,
        diagnostics=runtime_result.diagnostics,
        **common,
    )


__all__ = ["CompiledSimulation", "compile_simulation", "run_simulation"]
