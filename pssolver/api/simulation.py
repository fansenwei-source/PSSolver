"""Public tensor-free composition contract for one simulation request.

P7.7.7 freezes ownership and naming only. This module deliberately does not
select a capability implementation, allocate tensors, construct a runtime, or
execute a workflow. Later public convenience objects may construct the
canonical declarations accepted here without changing this top-level shape.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from pssolver.configuration.simulation import (
    ExecutionSpec,
    InitialConditionSpec,
    InvocationSpec,
    SimulationSpec,
    TimeIntegrationSpec,
    WorkflowSpec,
)
from pssolver.core.boundary import BoundaryAssignment
from pssolver.core.geometry import GeometrySpec
from pssolver.core.numerics import NumericsConfig
from pssolver.systems.equations import EquationSystemSpec


@dataclass(frozen=True, slots=True)
class Simulation:
    """Immutable public declaration of one complete finite simulation.

    Field ownership is intentionally explicit:

    - ``model`` owns equations, physical parameters, algebraic subsystem
      requests, diagnostics, and admitted initial-condition families;
    - ``geometry`` owns domain shape, lengths, axes, and topology;
    - ``boundaries`` attaches physical or compatibility laws to components
      and oriented faces;
    - ``numerics`` owns precision, dealiasing, transform intent, and storage;
    - ``time`` owns the mathematical integrator and timestep;
    - ``initial_condition`` owns the realization family, source, and seed;
    - ``execution`` owns backend, runtime, device, and implementation policy;
    - ``output`` owns finite duration, observation/output schedule, restart,
      checkpoint, and output-location policy;
    - ``discretization`` carries solver controls not already represented by
      the generic numerical contracts;
    - ``invocation`` carries non-scientific invocation provenance.

    The validated canonical :class:`SimulationSpec` is available through
    :attr:`specification`. It remains tensor-free and is not an executable
    runtime.
    """

    model: EquationSystemSpec
    geometry: GeometrySpec
    boundaries: BoundaryAssignment
    numerics: NumericsConfig
    time: TimeIntegrationSpec
    initial_condition: InitialConditionSpec
    execution: ExecutionSpec
    output: WorkflowSpec
    discretization: Mapping[str, object] = field(default_factory=dict)
    invocation: InvocationSpec = field(default_factory=InvocationSpec)
    _specification: SimulationSpec = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        specification = SimulationSpec(
            equation_system=self.model,
            geometry=self.geometry,
            boundaries=self.boundaries,
            numerics=self.numerics,
            time_integration=self.time,
            discretization_parameters=self.discretization,
            initial_condition=self.initial_condition,
            execution=self.execution,
            workflow=self.output,
            invocation=self.invocation,
        )
        object.__setattr__(
            self,
            "discretization",
            specification.discretization_parameters,
        )
        object.__setattr__(self, "_specification", specification)

    @property
    def specification(self) -> SimulationSpec:
        """Return the canonical tensor-free declaration with stable identity."""

        return self._specification

    def identity_metadata(self) -> dict[str, object]:
        """Return scientific, discretization, execution, and run identities."""

        return self.specification.identity_metadata()

    def to_metadata(self) -> dict[str, object]:
        """Return the complete canonical declaration metadata."""

        return self.specification.to_metadata()

    def canonical_sha256(self) -> str:
        """Hash the complete normalized declaration deterministically."""

        return self.specification.canonical_sha256()


__all__ = ["Simulation"]
