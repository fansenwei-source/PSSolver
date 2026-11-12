"""Fail-closed P7.7.4 binding of lowered simulations to runtime factories.

This module is tensor-free.  It converts a validated ``SimulationSpec`` and
its exact ``SimulationLoweringPlan`` into an immutable construction identity.
It imports no runtime factory and performs no allocation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import NoReturn

from pssolver.core._canonical import canonical_json
from pssolver.functional.legacy_declarations.application_registry import RUNTIME_BY_KEY
from pssolver.functional.legacy_declarations.planning_construction import (
    BuilderProvision,
    RuntimeConstructionBinding,
    RuntimeConstructionKind,
)
from pssolver.functional.legacy_declarations.planning_simulation import SimulationLoweringPlan
from pssolver.functional.legacy_declarations.simulation import SimulationSpec
from pssolver.functional.legacy_declarations.simulation_lowering import lower_simulation_spec


_PLANE_VARIANT = "complete_stress_beris_edwards"
_PLANE_GEOMETRY = "plane_slab"


class BindingRejectionCode(str, Enum):
    """Stable reason that a lowered declaration cannot be connected."""

    INVALID_LOWERING_PLAN = "invalid_lowering_plan"
    UNSUPPORTED_LIFTING_RUNTIME = "unsupported_lifting_runtime"
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
        canonical = canonical_json(context)
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
            canonical_json(context),
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
    lifting_supported = (
        plan.equation_variant == _PLANE_VARIANT
        and plan.geometry_name == _PLANE_GEOMETRY
        and simulation.execution.runtime_path == "legacy_production"
    )
    if plan.lifting_plan is not None and not lifting_supported:
        _reject(
            BindingRejectionCode.UNSUPPORTED_LIFTING_RUNTIME,
            "static lifting is connected only to the qualified Plane "
            "legacy_production runtime",
            lifting_plan_sha256=plan.lifting_plan.canonical_sha256(),
            requested_runtime_path=simulation.execution.runtime_path,
            supported_runtime_path="legacy_production",
        )
    key = (
        plan.equation_variant,
        plan.geometry_name,
        simulation.execution.runtime_path,
    )
    try:
        registration = RUNTIME_BY_KEY[key]
    except KeyError:
        _reject(
            BindingRejectionCode.UNSUPPORTED_RUNTIME_PATH,
            "the lowered model/geometry pair has no qualified runtime binding",
            equation_variant=plan.equation_variant,
            geometry_name=plan.geometry_name,
            runtime_path=simulation.execution.runtime_path,
            supported_runtime_paths=sorted(
                candidate[2]
                for candidate in RUNTIME_BY_KEY
                if candidate[:2] == key[:2]
            ),
        )
    return RuntimeConstructionBinding(
        source_simulation_sha256=simulation.canonical_sha256(),
        lowering_plan_sha256=plan.canonical_sha256(),
        equation_variant=plan.equation_variant,
        geometry_name=plan.geometry_name,
        runtime_path=simulation.execution.runtime_path,
        kind=RuntimeConstructionKind(registration.kind),
        request_type=registration.application.request_type,
        runtime_factory=registration.application.runtime_factory,
        adapter_protocol=registration.application.adapter_protocol,
        solver_implementation=plan.solver.implementation,
        builder_provision=BuilderProvision(registration.builder_provision),
        builder_parameter=registration.builder_parameter,
    )


__all__ = [
    "BindingRejection",
    "BindingRejectionCode",
    "SimulationBindingError",
    "bind_simulation_runtime",
]
