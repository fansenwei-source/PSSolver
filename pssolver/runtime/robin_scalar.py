"""P8.5.3 runtime state for a generic scalar Plane Robin pilot.

This module binds the P8.5.2 CPU operator once.  It is intentionally not a
complete PDE timestep and is not reachable through the public simulation
runner or any production runtime selector.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math

import torch

from pssolver.execution.state import (
    IntegratorProgress,
    RepresentationLedger,
    RuntimeState,
)
from pssolver.operators.robin import (
    CellCenteredRobinEigenbasisOperator,
    materialize_cell_centered_robin_eigenbasis,
)
from pssolver.planning.robin import PlaneRobinScalarLoweringPlan


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


def _tensor_sha256(value: torch.Tensor) -> str:
    materialized = value.detach().cpu().contiguous()
    header = _canonical_json(
        {
            "dtype": str(materialized.dtype),
            "shape": list(materialized.shape),
        },
        "tensor header",
    ).encode("utf-8")
    digest = hashlib.sha256()
    digest.update(header)
    digest.update(b"\0")
    digest.update(materialized.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_digest(value: object, description: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{description} must be a lowercase SHA-256 digest")
    return value


@dataclass(frozen=True, slots=True)
class PlaneRobinOperatorCacheKey:
    """Exact construction identity for one materialized Robin operator."""

    lowering_plan_sha256: str
    bounded_axis_plan_sha256: str
    geometry_name: str
    domain_shape: tuple[int, ...]
    domain_lengths: tuple[float, ...]
    dtype: str
    device: str

    def __post_init__(self) -> None:
        _require_digest(self.lowering_plan_sha256, "lowering plan identity")
        _require_digest(
            self.bounded_axis_plan_sha256,
            "bounded-axis plan identity",
        )
        if self.geometry_name != "plane_slab":
            raise ValueError("Robin cache key requires plane_slab geometry")
        shape = tuple(self.domain_shape)
        lengths = tuple(float(value) for value in self.domain_lengths)
        if len(shape) != 3 or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            for value in shape
        ):
            raise ValueError("Robin cache domain shape is invalid")
        if len(lengths) != 3 or any(
            not math.isfinite(value) or value <= 0.0 for value in lengths
        ):
            raise ValueError("Robin cache domain lengths are invalid")
        if self.dtype != "torch.float64" or self.device != "cpu":
            raise ValueError("P8.5.3 cache supports torch.float64 CPU only")
        object.__setattr__(self, "domain_shape", shape)
        object.__setattr__(self, "domain_lengths", lengths)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "lowering_plan_sha256": self.lowering_plan_sha256,
            "bounded_axis_plan_sha256": self.bounded_axis_plan_sha256,
            "geometry_name": self.geometry_name,
            "domain_shape": list(self.domain_shape),
            "domain_lengths": list(self.domain_lengths),
            "dtype": self.dtype,
            "device": self.device,
        }

    def canonical_sha256(self) -> str:
        return _sha256_json(self.to_metadata())


class PlaneRobinOperatorCache:
    """Explicit construction-owned cache; never consulted by a hot loop."""

    def __init__(self) -> None:
        self._entries: dict[
            PlaneRobinOperatorCacheKey,
            CellCenteredRobinEigenbasisOperator,
        ] = {}

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def bind(
        self,
        plan: PlaneRobinScalarLoweringPlan,
    ) -> tuple[
        PlaneRobinOperatorCacheKey,
        CellCenteredRobinEigenbasisOperator,
    ]:
        if not isinstance(plan, PlaneRobinScalarLoweringPlan):
            raise TypeError("plan must be a PlaneRobinScalarLoweringPlan")
        key = PlaneRobinOperatorCacheKey(
            lowering_plan_sha256=plan.canonical_sha256(),
            bounded_axis_plan_sha256=plan.robin_plan.canonical_sha256(),
            geometry_name=plan.geometry_name,
            domain_shape=plan.domain_shape,
            domain_lengths=plan.domain_lengths,
            dtype="torch.float64",
            device="cpu",
        )
        operator = self._entries.get(key)
        if operator is None:
            operator = materialize_cell_centered_robin_eigenbasis(
                plan.robin_plan
            )
            self._entries[key] = operator
        return key, operator


@dataclass(frozen=True, slots=True)
class PlaneRobinScalarCheckpoint:
    """In-memory state snapshot with pre-mutation validation metadata."""

    identity_json: str
    remainder: torch.Tensor
    bounded_modal: torch.Tensor
    progress_json: str
    representations_json: str
    remainder_sha256: str
    bounded_modal_sha256: str

    def __post_init__(self) -> None:
        for name in ("identity_json", "progress_json", "representations_json"):
            value = getattr(self, name)
            try:
                decoded = json.loads(value)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{name} must be valid JSON") from exc
            if value != _canonical_json(decoded, name):
                raise ValueError(f"{name} must use canonical JSON")
        for name in ("remainder", "bounded_modal"):
            value = getattr(self, name)
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"checkpoint {name} must be a tensor")
        _require_digest(self.remainder_sha256, "remainder identity")
        _require_digest(self.bounded_modal_sha256, "modal identity")

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "identity": json.loads(self.identity_json),
            "progress": json.loads(self.progress_json),
            "representations": json.loads(self.representations_json),
            "tensors": {
                "remainder": {
                    "shape": list(self.remainder.shape),
                    "dtype": str(self.remainder.dtype),
                    "device": str(self.remainder.device),
                    "sha256": self.remainder_sha256,
                },
                "bounded_modal": {
                    "shape": list(self.bounded_modal.shape),
                    "dtype": str(self.bounded_modal.dtype),
                    "device": str(self.bounded_modal.device),
                    "sha256": self.bounded_modal_sha256,
                },
            },
        }


class PlaneRobinScalarRuntime:
    """Bound P8.5.3 state, observation, and checkpoint semantics."""

    def __init__(
        self,
        plan: PlaneRobinScalarLoweringPlan,
        initial_physical: torch.Tensor,
        *,
        dt: float,
        cache: PlaneRobinOperatorCache | None = None,
    ) -> None:
        if not isinstance(plan, PlaneRobinScalarLoweringPlan):
            raise TypeError("plan must be a PlaneRobinScalarLoweringPlan")
        if cache is None:
            cache = PlaneRobinOperatorCache()
        if not isinstance(cache, PlaneRobinOperatorCache):
            raise TypeError("cache must be a PlaneRobinOperatorCache")
        self.plan = plan
        self.cache = cache
        self.cache_key, self.operator = cache.bind(plan)
        self._validate_domain_tensor(initial_physical, "initial physical field")
        remainder = self.operator.homogeneous_remainder(
            initial_physical
        ).contiguous()
        bounded_modal = self.operator.to_modal(remainder).contiguous()
        self.state = RuntimeState(
            component_names=(plan.component,),
            physical=remainder.unsqueeze(0),
            spectral=bounded_modal.unsqueeze(0),
            progress=IntegratorProgress(dt=dt, refresh_interval=None),
            representations=RepresentationLedger(),
        )

    def _validate_domain_tensor(
        self,
        value: torch.Tensor,
        description: str,
    ) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{description} must be a tensor")
        if tuple(value.shape) != self.plan.domain_shape:
            raise ValueError(f"{description} shape does not match the plan")
        if value.dtype is not torch.float64:
            raise ValueError(f"{description} must use torch.float64")
        if value.device.type != "cpu":
            raise ValueError(f"{description} must be on CPU for P8.5.3")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"{description} must be finite")

    @property
    def remainder(self) -> torch.Tensor:
        return self.state.physical_component(self.plan.component)

    @property
    def bounded_modal(self) -> torch.Tensor:
        return self.state.spectral_component(self.plan.component)

    def physical_observation(self) -> torch.Tensor:
        """Reconstruct the registered physical field at cell centers."""

        self.state.representations.require_physical_current()
        return self.operator.reconstruct_physical(self.remainder)

    def boundary_residual(self) -> tuple[torch.Tensor, torch.Tensor]:
        return self.operator.boundary_residual(self.physical_observation())

    def replace_physical_observation(self, value: torch.Tensor) -> None:
        """Atomically replace the pilot state and synchronize bounded modes."""

        self._validate_domain_tensor(value, "replacement physical field")
        remainder = self.operator.homogeneous_remainder(value).contiguous()
        bounded_modal = self.operator.to_modal(remainder).contiguous()
        if not bool(torch.isfinite(bounded_modal).all().item()):
            raise ValueError("replacement bounded modes must be finite")
        self.remainder.copy_(remainder)
        self.state.representations.mark_physical_updated()
        self.bounded_modal.copy_(bounded_modal)
        self.state.representations.mark_spectral_synchronized()

    def apply_bounded_helmholtz(self, *, mass: float) -> torch.Tensor:
        return self.operator.apply_helmholtz(
            self.physical_observation(),
            mass=mass,
        )

    def solve_bounded_helmholtz(
        self,
        forcing: torch.Tensor,
        *,
        mass: float,
    ) -> torch.Tensor:
        self._validate_domain_tensor(forcing, "Helmholtz forcing")
        return self.operator.solve_helmholtz(forcing, mass=mass)

    def checkpoint_identity_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "runtime_kind": "plane_robin_scalar_bounded_axis_pilot",
            "lowering_plan_sha256": self.plan.canonical_sha256(),
            "source_simulation_sha256": self.plan.source_simulation_sha256,
            "source_boundary_sha256": self.plan.source_boundary_sha256,
            "bounded_axis_operator_kind": self.plan.robin_plan.operator_kind,
            "bounded_axis_plan_sha256": (
                self.plan.robin_plan.canonical_sha256()
            ),
            "raw_lower_coefficients_sha256": (
                self.plan.robin_plan.lower.canonical_sha256()
            ),
            "raw_upper_coefficients_sha256": (
                self.plan.robin_plan.upper.canonical_sha256()
            ),
            "normal_derivative_convention": "outward_unit_normal",
            "operator_materialized_sha256": self.operator.materialized_sha256,
            "operator_cache_key": self.cache_key.to_metadata(),
            "operator_cache_key_sha256": self.cache_key.canonical_sha256(),
            "field_name": self.plan.field_name,
            "component": self.plan.component,
            "geometry_name": self.plan.geometry_name,
            "domain_shape": list(self.plan.domain_shape),
            "domain_lengths": list(self.plan.domain_lengths),
            "wall_normal_axis": self.plan.wall_normal_axis,
            "dtype": "torch.float64",
            "device": "cpu",
            "evolved_representation": "homogeneous_remainder",
            "physical_observation": "homogeneous_remainder_plus_affine_lift",
            "model_specialization": None,
        }

    def capture_checkpoint(self) -> PlaneRobinScalarCheckpoint:
        self.state.representations.require_physical_current()
        self.state.representations.require_spectral_current()
        remainder = self.remainder.detach().clone().contiguous()
        bounded_modal = self.bounded_modal.detach().clone().contiguous()
        return PlaneRobinScalarCheckpoint(
            identity_json=_canonical_json(
                self.checkpoint_identity_metadata(),
                "checkpoint identity",
            ),
            remainder=remainder,
            bounded_modal=bounded_modal,
            progress_json=_canonical_json(
                self.state.progress.to_metadata(),
                "checkpoint progress",
            ),
            representations_json=_canonical_json(
                self.state.representations.to_metadata(),
                "checkpoint representations",
            ),
            remainder_sha256=_tensor_sha256(remainder),
            bounded_modal_sha256=_tensor_sha256(bounded_modal),
        )

    def restore_checkpoint(
        self,
        checkpoint: PlaneRobinScalarCheckpoint,
    ) -> None:
        """Validate all identity and payload gates before target mutation."""

        if not isinstance(checkpoint, PlaneRobinScalarCheckpoint):
            raise TypeError("checkpoint must be a PlaneRobinScalarCheckpoint")
        expected_identity = _canonical_json(
            self.checkpoint_identity_metadata(),
            "checkpoint identity",
        )
        if checkpoint.identity_json != expected_identity:
            raise ValueError("checkpoint Robin runtime identity does not match")
        self._validate_domain_tensor(checkpoint.remainder, "checkpoint remainder")
        self._validate_domain_tensor(
            checkpoint.bounded_modal,
            "checkpoint bounded modes",
        )
        if _tensor_sha256(checkpoint.remainder) != checkpoint.remainder_sha256:
            raise ValueError("checkpoint remainder payload identity does not match")
        if (
            _tensor_sha256(checkpoint.bounded_modal)
            != checkpoint.bounded_modal_sha256
        ):
            raise ValueError("checkpoint modal payload identity does not match")
        recomputed_modal = self.operator.to_modal(checkpoint.remainder)
        if not torch.equal(recomputed_modal, checkpoint.bounded_modal):
            raise ValueError("checkpoint representations are not synchronized")
        progress_metadata = json.loads(checkpoint.progress_json)
        spectral_refresh = progress_metadata["spectral_refresh"]
        progress = IntegratorProgress(
            dt=progress_metadata["dt"],
            completed_steps=progress_metadata["completed_steps"],
            refresh_interval=spectral_refresh["interval"],
            refresh_step_count=spectral_refresh["step_count"],
            refresh_count=spectral_refresh["refresh_count"],
        )
        representation_metadata = json.loads(checkpoint.representations_json)
        representations = RepresentationLedger(
            generation=representation_metadata["generation"],
            physical_generation=representation_metadata["physical_generation"],
            spectral_generation=representation_metadata["spectral_generation"],
        )
        representations.require_physical_current()
        representations.require_spectral_current()

        self.remainder.copy_(checkpoint.remainder)
        self.bounded_modal.copy_(checkpoint.bounded_modal)
        self.state.replace_progress(progress)
        self.state.replace_representations(representations)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "runtime_kind": "plane_robin_scalar_bounded_axis_pilot",
            "lowering_plan": self.plan.to_metadata(),
            "lowering_plan_sha256": self.plan.canonical_sha256(),
            "operator": self.operator.to_metadata(),
            "operator_cache_key": self.cache_key.to_metadata(),
            "operator_cache_entry_count": self.cache.entry_count,
            "state": self.state.to_metadata(),
            "observation": "physical_field_reconstructed_on_demand",
            "complete_timestep_connected": False,
            "public_runner_connected": False,
            "runtime_selector_connected": False,
        }


__all__ = [
    "PlaneRobinOperatorCache",
    "PlaneRobinOperatorCacheKey",
    "PlaneRobinScalarCheckpoint",
    "PlaneRobinScalarRuntime",
]
