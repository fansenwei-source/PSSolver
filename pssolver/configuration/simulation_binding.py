"""Fail-closed P7.7.4 binding of lowered simulations to runtime factories.

This module is tensor-free.  It converts a validated ``SimulationSpec`` and
its exact ``SimulationLoweringPlan`` into an immutable construction identity.
It imports no runtime factory and performs no allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import NoReturn

from pssolver.planning.construction import (
    BuilderProvision,
    RuntimeConstructionBinding,
    RuntimeConstructionKind,
)
from pssolver.planning.simulation import SimulationLoweringPlan

from .simulation import SimulationSpec
from .simulation_lowering import lower_simulation_spec


_PLANE_VARIANT = "complete_stress_beris_edwards"
_PLANE_GEOMETRY = "plane_slab"
_CHANNEL_VARIANT = "legacy_active_force_active_nematics"
_CHANNEL_GEOMETRY = "rectangular_channel"
_PERIODIC_GEOMETRY = "periodic_box"

_PLANE_REQUEST = (
    "pssolver.runtime.plane_beris_edwards.PlaneRuntimeBuildRequest"
)
_PLANE_FACTORY = (
    "pssolver.runtime.plane_beris_edwards."
    "build_plane_beris_edwards_runtime"
)
_PLANE_PROTOCOL = (
    "pssolver.runtime.plane_beris_edwards.PlaneRuntimeAdapterProtocol"
)
_CHANNEL_REQUEST = (
    "pssolver.runtime.channel_active_nematics.ChannelRuntimeBuildRequest"
)
_CHANNEL_FACTORY = (
    "pssolver.runtime.channel_active_nematics."
    "build_channel_active_nematic_runtime"
)
_CHANNEL_PROTOCOL = (
    "pssolver.runtime.channel_active_nematics.ChannelRuntimeAdapterProtocol"
)
_PERIODIC_REQUEST = (
    "pssolver.runtime.periodic_beris_edwards.PeriodicRuntimeBuildRequest"
)
_PERIODIC_FACTORY = (
    "pssolver.runtime.periodic_beris_edwards."
    "build_periodic_beris_edwards_runtime"
)
_PERIODIC_PROTOCOL = (
    "pssolver.runtime.periodic_beris_edwards.PeriodicRuntimeAdapterProtocol"
)
_CHANNEL_COMPLETE_REQUEST = (
    "pssolver.runtime.channel_beris_edwards."
    "ChannelBerisEdwardsRuntimeBuildRequest"
)
_CHANNEL_COMPLETE_FACTORY = (
    "pssolver.runtime.channel_beris_edwards."
    "build_channel_beris_edwards_runtime"
)
_CHANNEL_COMPLETE_PROTOCOL = (
    "pssolver.runtime.channel_beris_edwards."
    "ChannelBerisEdwardsRuntimeAdapterProtocol"
)


class BindingRejectionCode(str, Enum):
    """Stable reason that a lowered declaration cannot be connected."""

    INVALID_LOWERING_PLAN = "invalid_lowering_plan"
    UNSUPPORTED_RUNTIME_PATH = "unsupported_runtime_path"


@dataclass(frozen=True, slots=True)
class BindingRejection:
    """JSON-compatible construction-binding rejection."""

    code: BindingRejectionCode
    message: str
    context_json: str = "{}"

    def __post_init__(self) -> None:
        if not isinstance(self.code, BindingRejectionCode):
            raise TypeError("code must be a BindingRejectionCode")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("message must be a non-empty string")
        try:
            context = json.loads(self.context_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("context_json must be valid JSON") from exc
        canonical = json.dumps(
            context,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if canonical != self.context_json:
            raise ValueError("context_json must use canonical JSON")

    def to_metadata(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": json.loads(self.context_json),
        }


class SimulationBindingError(ValueError):
    """Raised before allocation when no qualified runtime binding exists."""

    def __init__(self, rejection: BindingRejection) -> None:
        if not isinstance(rejection, BindingRejection):
            raise TypeError("rejection must be a BindingRejection")
        self.rejection = rejection
        super().__init__(f"{rejection.code.value}: {rejection.message}")


def _reject(
    code: BindingRejectionCode,
    message: str,
    **context: object,
) -> NoReturn:
    raise SimulationBindingError(
        BindingRejection(
            code,
            message,
            json.dumps(
                context,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ),
        )
    )


def _require_exact_plan(
    simulation: SimulationSpec,
    lowering_plan: SimulationLoweringPlan | None,
) -> SimulationLoweringPlan:
    expected = lower_simulation_spec(simulation)
    if lowering_plan is None:
        return expected
    if not isinstance(lowering_plan, SimulationLoweringPlan):
        raise TypeError("lowering_plan must be a SimulationLoweringPlan")
    if lowering_plan.canonical_sha256() != expected.canonical_sha256():
        _reject(
            BindingRejectionCode.INVALID_LOWERING_PLAN,
            "the supplied lowering plan is not the exact plan for the simulation",
            expected_sha256=expected.canonical_sha256(),
            observed_sha256=lowering_plan.canonical_sha256(),
            source_simulation_sha256=simulation.canonical_sha256(),
        )
    return lowering_plan


def bind_simulation_runtime(
    simulation: SimulationSpec,
    lowering_plan: SimulationLoweringPlan | None = None,
) -> RuntimeConstructionBinding:
    """Bind one qualified declaration to an immutable runtime identity.

    Resolution is a construction-time operation.  The returned product names
    implementations but contains no imported callables and permits no runtime
    fallback.
    """

    if not isinstance(simulation, SimulationSpec):
        raise TypeError("simulation must be a SimulationSpec")
    plan = _require_exact_plan(simulation, lowering_plan)
    key = (
        plan.equation_variant,
        plan.geometry_name,
        simulation.execution.runtime_path,
    )
    table = {
        (_PLANE_VARIANT, _PERIODIC_GEOMETRY, "periodic_spectral"): (
            RuntimeConstructionKind.PERIODIC_COMPLETE_STRESS,
            _PERIODIC_REQUEST,
            _PERIODIC_FACTORY,
            _PERIODIC_PROTOCOL,
            BuilderProvision.PACKAGE,
            None,
        ),
        (_PLANE_VARIANT, _CHANNEL_GEOMETRY, "channel_complete_stress"): (
            RuntimeConstructionKind.CHANNEL_COMPLETE_STRESS,
            _CHANNEL_COMPLETE_REQUEST,
            _CHANNEL_COMPLETE_FACTORY,
            _CHANNEL_COMPLETE_PROTOCOL,
            BuilderProvision.PACKAGE,
            None,
        ),
        (_PLANE_VARIANT, _PLANE_GEOMETRY, "legacy_production"): (
            RuntimeConstructionKind.PLANE_LEGACY_PRODUCTION,
            _PLANE_REQUEST,
            _PLANE_FACTORY,
            _PLANE_PROTOCOL,
            BuilderProvision.PACKAGE,
            None,
        ),
        (_PLANE_VARIANT, _PLANE_GEOMETRY, "compiled_v2"): (
            RuntimeConstructionKind.PLANE_COMPILED_V2,
            _PLANE_REQUEST,
            _PLANE_FACTORY,
            _PLANE_PROTOCOL,
            BuilderProvision.CALLER,
            "compiled_builder",
        ),
        (_CHANNEL_VARIANT, _CHANNEL_GEOMETRY, "legacy_channel"): (
            RuntimeConstructionKind.CHANNEL_LEGACY,
            _CHANNEL_REQUEST,
            _CHANNEL_FACTORY,
            _CHANNEL_PROTOCOL,
            BuilderProvision.CALLER,
            "legacy_builder",
        ),
        (_CHANNEL_VARIANT, _CHANNEL_GEOMETRY, "compiled_channel_v2"): (
            RuntimeConstructionKind.CHANNEL_COMPILED_V2,
            _CHANNEL_REQUEST,
            _CHANNEL_FACTORY,
            _CHANNEL_PROTOCOL,
            BuilderProvision.CALLER,
            "compiled_builder",
        ),
    }
    try:
        (
            kind,
            request_type,
            factory,
            protocol,
            builder_provision,
            builder_parameter,
        ) = table[key]
    except KeyError:
        _reject(
            BindingRejectionCode.UNSUPPORTED_RUNTIME_PATH,
            "the lowered model/geometry pair has no qualified runtime binding",
            equation_variant=plan.equation_variant,
            geometry_name=plan.geometry_name,
            runtime_path=simulation.execution.runtime_path,
            supported_runtime_paths=sorted(
                candidate[2]
                for candidate in table
                if candidate[:2] == key[:2]
            ),
        )
    return RuntimeConstructionBinding(
        source_simulation_sha256=simulation.canonical_sha256(),
        lowering_plan_sha256=plan.canonical_sha256(),
        equation_variant=plan.equation_variant,
        geometry_name=plan.geometry_name,
        runtime_path=simulation.execution.runtime_path,
        kind=kind,
        request_type=request_type,
        runtime_factory=factory,
        adapter_protocol=protocol,
        solver_implementation=plan.solver.implementation,
        builder_provision=builder_provision,
        builder_parameter=builder_parameter,
    )


__all__ = [
    "BindingRejection",
    "BindingRejectionCode",
    "SimulationBindingError",
    "bind_simulation_runtime",
]
