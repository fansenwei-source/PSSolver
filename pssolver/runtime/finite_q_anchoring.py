"""CPU reference runtime for the P8.5 finite-Q anchoring pilot.

The runtime composes five generic scalar Robin runtimes and advances only the
wall-normal elastic relaxation equation.  It is deliberately not a complete
Beris--Edwards or Stokes timestep and is not selected by the public runner.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math

import torch

from pssolver.configuration.finite_q_anchoring import (
    PlaneFiniteQAnchoringLoweringPlan,
)
from pssolver.models.active_nematics.fields import Q_COMPONENTS
from pssolver.runtime.robin_scalar import (
    PlaneRobinOperatorCache,
    PlaneRobinScalarCheckpoint,
    PlaneRobinScalarRuntime,
)


FINITE_Q_ANCHORING_RUNTIME_SCHEMA_VERSION = 1
FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION = 1


def _canonical_json(value: object, description: str) -> str:
    try:
        return json.dumps(
            value,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{description} must be finite and JSON-compatible"
        ) from exc


def _sha256_json(value: object) -> str:
    return hashlib.sha256(
        _canonical_json(value, "identity").encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class PlaneFiniteQAnchoringCheckpoint:
    """All five component snapshots under one immutable runtime identity."""

    identity_json: str
    components: tuple[tuple[str, PlaneRobinScalarCheckpoint], ...]

    def __post_init__(self) -> None:
        try:
            identity = json.loads(self.identity_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("identity_json must be valid JSON") from exc
        if self.identity_json != _canonical_json(identity, "identity"):
            raise ValueError("identity_json must use canonical JSON")
        components = tuple(self.components)
        if tuple(name for name, _value in components) != Q_COMPONENTS:
            raise ValueError("finite-Q checkpoint component order is invalid")
        if not all(
            isinstance(value, PlaneRobinScalarCheckpoint)
            for _name, value in components
        ):
            raise TypeError("finite-Q checkpoint values must be scalar snapshots")
        object.__setattr__(self, "components", components)

    @property
    def completed_steps(self) -> int:
        values = {
            int(json.loads(value.progress_json)["completed_steps"])
            for _name, value in self.components
        }
        if len(values) != 1:
            raise ValueError("finite-Q checkpoint component clocks disagree")
        return values.pop()

    def for_component(self, component: str) -> PlaneRobinScalarCheckpoint:
        for name, value in self.components:
            if name == component:
                return value
        raise KeyError(component)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION,
            "identity": json.loads(self.identity_json),
            "identity_sha256": hashlib.sha256(
                self.identity_json.encode("utf-8")
            ).hexdigest(),
            "component_order": list(Q_COMPONENTS),
            "completed_steps": self.completed_steps,
            "components": {
                name: value.to_metadata() for name, value in self.components
            },
        }


class PlaneFiniteQAnchoringRuntime:
    """Five-component CPU oracle for finite anchoring workflow semantics."""

    def __init__(
        self,
        plan: PlaneFiniteQAnchoringLoweringPlan,
        initial_physical: Mapping[str, torch.Tensor],
        *,
        dt: float,
        cache: PlaneRobinOperatorCache | None = None,
    ) -> None:
        if not isinstance(plan, PlaneFiniteQAnchoringLoweringPlan):
            raise TypeError(
                "plan must be a PlaneFiniteQAnchoringLoweringPlan"
            )
        if not isinstance(initial_physical, Mapping):
            raise TypeError("initial_physical must be a component mapping")
        if tuple(initial_physical.keys()) != Q_COMPONENTS:
            raise ValueError(
                "initial_physical must use canonical Q component order"
            )
        if (
            not isinstance(dt, (int, float))
            or isinstance(dt, bool)
            or not math.isfinite(float(dt))
            or float(dt) <= 0.0
        ):
            raise ValueError("dt must be positive and finite")
        if cache is None:
            cache = PlaneRobinOperatorCache()
        if not isinstance(cache, PlaneRobinOperatorCache):
            raise TypeError("cache must be a PlaneRobinOperatorCache")
        self.plan = plan
        self.dt = float(dt)
        self.cache = cache
        self.components = {
            component: PlaneRobinScalarRuntime(
                plan.for_component(component),
                initial_physical[component],
                dt=self.dt,
                cache=cache,
            )
            for component in Q_COMPONENTS
        }
        self._require_synchronized_clocks()

    @property
    def completed_steps(self) -> int:
        return self._require_synchronized_clocks()

    def _require_synchronized_clocks(self) -> int:
        values = {
            runtime.state.progress.completed_steps
            for runtime in self.components.values()
        }
        if len(values) != 1:
            raise RuntimeError("finite-Q component clocks are not synchronized")
        return values.pop()

    @property
    def k_q(self) -> float:
        value = float(self.plan.anchoring_metadata["k_q"])
        if not math.isfinite(value) or value <= 0.0:
            raise RuntimeError("finite-Q plan has an invalid K_Q")
        return value

    def physical_components(self) -> dict[str, torch.Tensor]:
        return {
            component: runtime.physical_observation()
            for component, runtime in self.components.items()
        }

    def physical_q(self) -> torch.Tensor:
        """Return physical Q in canonical trailing-component layout."""

        return torch.stack(
            tuple(
                self.components[component].physical_observation()
                for component in Q_COMPONENTS
            ),
            dim=-1,
        )

    def boundary_residuals(
        self,
    ) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {
            component: runtime.boundary_residual()
            for component, runtime in self.components.items()
        }

    def advance(self, steps: int = 1) -> None:
        """Advance the implicit wall-normal elastic-relaxation CPU oracle."""

        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or steps < 0
        ):
            raise ValueError("steps must be a non-negative integer")
        mass = 1.0 / (self.dt * self.k_q)
        for _ in range(steps):
            candidates = {}
            for component, runtime in self.components.items():
                physical = runtime.physical_observation()
                candidate = runtime.solve_bounded_helmholtz(
                    physical * mass,
                    mass=mass,
                ).contiguous()
                runtime._validate_domain_tensor(
                    candidate,
                    f"candidate {component}",
                )
                candidates[component] = candidate
            for component, runtime in self.components.items():
                runtime.replace_physical_observation(candidates[component])
            for runtime in self.components.values():
                runtime.state.progress.commit_step(refreshed=False)
            self._require_synchronized_clocks()

    def checkpoint_identity_metadata(self) -> dict[str, object]:
        component_identities = {
            component: runtime.checkpoint_identity_metadata()
            for component, runtime in self.components.items()
        }
        anchoring = self.plan.anchoring_metadata
        return {
            "schema_version": FINITE_Q_ANCHORING_RUNTIME_SCHEMA_VERSION,
            "runtime_kind": "plane_finite_q_anchoring_relaxation_pilot",
            "runtime_path": "legacy_production",
            "backend": "torch_spectral",
            "complete_q_timestep_connected": False,
            "evolution_law": "implicit_wall_normal_elastic_relaxation",
            "dt": self.dt,
            "k_q": self.k_q,
            "lowering_plan_sha256": self.plan.canonical_sha256(),
            "source_simulation_sha256": self.plan.source_simulation_sha256,
            "anchoring_sha256": self.plan.anchoring_sha256,
            "surface_law_id": anchoring["surface_law_id"],
            "q_convention": anchoring["q_convention"],
            "faces": anchoring["faces"],
            "normal_derivative_convention": "outward_unit_normal",
            "component_order": list(Q_COMPONENTS),
            "component_identities": component_identities,
            "component_identities_sha256": _sha256_json(
                component_identities
            ),
            "dtype": "torch.float64",
            "device": "cpu",
        }

    def capture_checkpoint(self) -> PlaneFiniteQAnchoringCheckpoint:
        identity = self.checkpoint_identity_metadata()
        return PlaneFiniteQAnchoringCheckpoint(
            identity_json=_canonical_json(identity, "runtime identity"),
            components=tuple(
                (
                    component,
                    self.components[component].capture_checkpoint(),
                )
                for component in Q_COMPONENTS
            ),
        )

    def restore_checkpoint(
        self,
        checkpoint: PlaneFiniteQAnchoringCheckpoint,
    ) -> None:
        """Restore only after all aggregate and component gates pass."""

        if not isinstance(checkpoint, PlaneFiniteQAnchoringCheckpoint):
            raise TypeError(
                "checkpoint must be a PlaneFiniteQAnchoringCheckpoint"
            )
        expected = _canonical_json(
            self.checkpoint_identity_metadata(),
            "runtime identity",
        )
        if checkpoint.identity_json != expected:
            raise ValueError(
                "checkpoint finite-Q runtime identity does not match target"
            )
        checkpoint.completed_steps
        for component in Q_COMPONENTS:
            self.components[component].validate_checkpoint(
                checkpoint.for_component(component)
            )
        for component in Q_COMPONENTS:
            self.components[component].restore_checkpoint(
                checkpoint.for_component(component)
            )
        self._require_synchronized_clocks()

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": FINITE_Q_ANCHORING_RUNTIME_SCHEMA_VERSION,
            "runtime_kind": "plane_finite_q_anchoring_relaxation_pilot",
            "identity": self.checkpoint_identity_metadata(),
            "identity_sha256": _sha256_json(
                self.checkpoint_identity_metadata()
            ),
            "completed_steps": self.completed_steps,
            "operator_cache_entry_count": self.cache.entry_count,
            "components": {
                component: runtime.to_metadata()
                for component, runtime in self.components.items()
            },
            "complete_q_timestep_connected": False,
            "public_runner_connected": False,
            "production_runtime_selector_connected": False,
        }


__all__ = [
    "FINITE_Q_ANCHORING_CHECKPOINT_FORMAT_VERSION",
    "FINITE_Q_ANCHORING_RUNTIME_SCHEMA_VERSION",
    "PlaneFiniteQAnchoringCheckpoint",
    "PlaneFiniteQAnchoringRuntime",
]
