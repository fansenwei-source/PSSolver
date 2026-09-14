"""Boundary-safe scheduling for projected transform batches.

This module is an adapter service: it knows how to batch compatible projected
transforms, but it does not own algebraic field lifetime or model equations.
One scheduler instance is bound to one geometry/runtime/dtype/device context;
one method call is bounded to one algebraic generation or explicit-RHS island.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Protocol

import torch

from .representations import (
    AlgebraicRepresentationCache,
    BatchedPhysicalMaterialization,
)


class ProjectedTransformDirection(str, Enum):
    FORWARD = "forward"
    INVERSE = "inverse"


class ProjectedTransformContext(Protocol):
    """Adapter operations required by the batching scheduler."""

    batch_size: int
    physical_shape: tuple[int, ...]
    spectral_shape: tuple[int, ...]
    real_dtype: torch.dtype
    device: torch.device

    @property
    def spectral_dtype(self) -> torch.dtype: ...

    def boundary_conditions(self, component_name: str) -> tuple[str, ...]: ...

    def forward_projected(
        self, component_name: str, value: torch.Tensor
    ) -> torch.Tensor: ...

    def inverse_projected(
        self, component_name: str, value: torch.Tensor
    ) -> torch.Tensor: ...

    def transform_projected_packed(
        self,
        boundary_conditions: tuple[str, ...],
        value: torch.Tensor,
        *,
        direction: ProjectedTransformDirection,
    ) -> torch.Tensor: ...


@dataclass(frozen=True, slots=True)
class ProjectedTransformBatchKey:
    """Complete compatibility identity for one scheduled transform batch."""

    boundary_signature: tuple[str, ...]
    direction: ProjectedTransformDirection
    physical_shape: tuple[int, ...]
    spectral_shape: tuple[int, ...]
    batch_size: int
    real_dtype: str
    spectral_dtype: str
    device: str

    def to_metadata(self) -> dict[str, object]:
        return {
            "boundary_signature": list(self.boundary_signature),
            "direction": self.direction.value,
            "physical_shape": list(self.physical_shape),
            "spectral_shape": list(self.spectral_shape),
            "batch_size": self.batch_size,
            "real_dtype": self.real_dtype,
            "spectral_dtype": self.spectral_dtype,
            "device": self.device,
        }


@dataclass(frozen=True, slots=True)
class ScheduledProjectedTransforms:
    """Ordered transform values plus an auditable batch partition."""

    values: tuple[torch.Tensor, ...]
    batch_sizes: tuple[int, ...]
    batch_keys: tuple[ProjectedTransformBatchKey, ...]

    def __post_init__(self) -> None:
        values = tuple(self.values)
        sizes = tuple(self.batch_sizes)
        keys = tuple(self.batch_keys)
        if not values or not all(isinstance(value, torch.Tensor) for value in values):
            raise ValueError("scheduled transforms must contain tensors")
        if len(sizes) != len(keys) or any(
            not isinstance(size, int)
            or isinstance(size, bool)
            or size <= 0
            for size in sizes
        ):
            raise ValueError("scheduled batch sizes and keys must align")
        if sum(sizes) != len(values):
            raise ValueError("scheduled batch sizes must cover every value")
        if not all(isinstance(key, ProjectedTransformBatchKey) for key in keys):
            raise TypeError("scheduled batch keys have an invalid type")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "batch_sizes", sizes)
        object.__setattr__(self, "batch_keys", keys)


@dataclass(frozen=True, slots=True)
class CompiledProjectedTransformPlan:
    """Immutable grouping plan for one stable component signature.

    Geometry, boundary signatures, tensor shapes, dtypes, and device are
    runtime invariants. Compiling their compatibility partition once keeps
    those checks on the scheduler's cold path instead of rebuilding the same
    Python objects for every timestep.
    """

    component_names: tuple[str, ...]
    direction: ProjectedTransformDirection
    batch_indices: tuple[tuple[int, ...], ...]
    batch_keys: tuple[ProjectedTransformBatchKey, ...]
    _batch_sizes: tuple[int, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        names = tuple(self.component_names)
        indices = tuple(tuple(group) for group in self.batch_indices)
        keys = tuple(self.batch_keys)
        if not names or any(
            not isinstance(name, str) or not name for name in names
        ):
            raise ValueError("compiled transform component names must be nonempty")
        if len(set(names)) != len(names):
            raise ValueError("compiled transform component names must be unique")
        if not isinstance(self.direction, ProjectedTransformDirection):
            raise TypeError("compiled transform direction is invalid")
        if not indices or len(indices) != len(keys):
            raise ValueError("compiled transform batches and keys must align")
        flattened = tuple(index for group in indices for index in group)
        if (
            any(not group for group in indices)
            or sorted(flattened) != list(range(len(names)))
            or len(flattened) != len(set(flattened))
        ):
            raise ValueError("compiled transform batches must partition inputs")
        if any(
            not isinstance(key, ProjectedTransformBatchKey)
            or key.direction is not self.direction
            for key in keys
        ):
            raise ValueError("compiled transform keys do not match direction")
        object.__setattr__(self, "component_names", names)
        object.__setattr__(self, "batch_indices", indices)
        object.__setattr__(self, "batch_keys", keys)
        object.__setattr__(self, "_batch_sizes", tuple(map(len, indices)))

    @property
    def batch_sizes(self) -> tuple[int, ...]:
        return self._batch_sizes


class BoundarySignatureTransformScheduler:
    """Group only transforms with an identical resolved execution key."""

    def __init__(self, context: ProjectedTransformContext) -> None:
        self._context = context
        self._plan_cache: dict[
            tuple[ProjectedTransformDirection, tuple[str, ...]],
            CompiledProjectedTransformPlan,
        ] = {}

    def _key(
        self,
        component_name: str,
        direction: ProjectedTransformDirection,
    ) -> ProjectedTransformBatchKey:
        context = self._context
        return ProjectedTransformBatchKey(
            boundary_signature=tuple(
                context.boundary_conditions(component_name)
            ),
            direction=direction,
            physical_shape=tuple(context.physical_shape),
            spectral_shape=tuple(context.spectral_shape),
            batch_size=context.batch_size,
            real_dtype=str(context.real_dtype),
            spectral_dtype=str(context.spectral_dtype),
            device=str(context.device),
        )

    @property
    def compiled_plan_count(self) -> int:
        """Number of invariant component/direction signatures compiled."""

        return len(self._plan_cache)

    def compile_plan(
        self,
        component_names: tuple[str, ...],
        *,
        direction: ProjectedTransformDirection,
    ) -> CompiledProjectedTransformPlan:
        """Return the runtime-local immutable plan for this signature."""

        if isinstance(component_names, str):
            raise TypeError("component_names must be an iterable")
        component_names = tuple(component_names)
        if not component_names:
            raise ValueError(
                "scheduled projected inputs must be nonempty and align"
            )
        if not isinstance(direction, ProjectedTransformDirection):
            raise TypeError("direction must be a ProjectedTransformDirection")
        cache_key = (direction, component_names)
        cached = self._plan_cache.get(cache_key)
        if cached is not None:
            return cached
        if len(set(component_names)) != len(component_names):
            raise ValueError("scheduled projected component names must be unique")

        groups: dict[ProjectedTransformBatchKey, list[int]] = {}
        for index, name in enumerate(component_names):
            groups.setdefault(self._key(name, direction), []).append(index)
        plan = CompiledProjectedTransformPlan(
            component_names=component_names,
            direction=direction,
            batch_indices=tuple(tuple(indices) for indices in groups.values()),
            batch_keys=tuple(groups),
        )
        self._plan_cache[cache_key] = plan
        return plan

    def execute_plan(
        self,
        plan: CompiledProjectedTransformPlan,
        values: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Execute an already compiled plan with timestep-local tensors."""

        if not isinstance(plan, CompiledProjectedTransformPlan):
            raise TypeError("plan must be a CompiledProjectedTransformPlan")
        cached = self._plan_cache.get((plan.direction, plan.component_names))
        if cached is not plan:
            raise ValueError(
                "compiled transform plan belongs to another scheduler"
            )
        values = tuple(values)
        if len(values) != len(plan.component_names):
            raise ValueError(
                "scheduled projected inputs must be nonempty and align"
            )
        return self._execute_cached_plan(plan, values)

    def _execute_cached_plan(
        self,
        plan: CompiledProjectedTransformPlan,
        values: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Execute a scheduler-owned plan without repeating cold-path checks."""

        result: list[torch.Tensor | None] = [None] * len(values)
        for key, indices in zip(
            plan.batch_keys,
            plan.batch_indices,
            strict=True,
        ):
            if len(indices) == 1:
                index = indices[0]
                result[index] = (
                    self._context.forward_projected(
                        plan.component_names[index], values[index]
                    )
                    if plan.direction is ProjectedTransformDirection.FORWARD
                    else self._context.inverse_projected(
                        plan.component_names[index], values[index]
                    )
                )
                continue
            packed = torch.cat(tuple(values[index] for index in indices), dim=0)
            transformed = self._context.transform_projected_packed(
                key.boundary_signature,
                packed,
                direction=plan.direction,
            )
            pieces = transformed.split(self._context.batch_size, dim=0)
            if len(pieces) != len(indices):
                raise RuntimeError("scheduled projected transform split is invalid")
            for index, piece in zip(indices, pieces, strict=True):
                result[index] = piece
        if any(value is None for value in result):
            raise RuntimeError("scheduled projected transform is incomplete")
        return tuple(value for value in result if value is not None)

    def transform_values_many(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
        *,
        direction: ProjectedTransformDirection,
    ) -> tuple[torch.Tensor, ...]:
        """Transform values through a cached invariant grouping plan."""

        component_names = tuple(component_names)
        values = tuple(values)
        if not component_names or len(component_names) != len(values):
            raise ValueError(
                "scheduled projected inputs must be nonempty and align"
            )
        plan = self.compile_plan(component_names, direction=direction)
        return self._execute_cached_plan(plan, values)

    def transform_many(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
        *,
        direction: ProjectedTransformDirection,
    ) -> ScheduledProjectedTransforms:
        component_names = tuple(component_names)
        values = tuple(values)
        if not component_names or len(component_names) != len(values):
            raise ValueError(
                "scheduled projected inputs must be nonempty and align"
            )
        plan = self.compile_plan(component_names, direction=direction)
        result = self._execute_cached_plan(plan, values)
        return ScheduledProjectedTransforms(
            result,
            plan.batch_sizes,
            plan.batch_keys,
        )

    def forward_values_many(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        return self.transform_values_many(
            component_names,
            values,
            direction=ProjectedTransformDirection.FORWARD,
        )

    def inverse_values_many(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        return self.transform_values_many(
            component_names,
            values,
            direction=ProjectedTransformDirection.INVERSE,
        )

    def forward_many(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
    ) -> ScheduledProjectedTransforms:
        return self.transform_many(
            component_names,
            values,
            direction=ProjectedTransformDirection.FORWARD,
        )

    def inverse_many(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
    ) -> ScheduledProjectedTransforms:
        return self.transform_many(
            component_names,
            values,
            direction=ProjectedTransformDirection.INVERSE,
        )


class AlgebraicPhysicalIslandScheduler:
    """Schedule physical islands without owning their generation lifetime."""

    def __init__(
        self,
        transforms: BoundarySignatureTransformScheduler,
        representation_cache: AlgebraicRepresentationCache,
    ) -> None:
        if not isinstance(transforms, BoundarySignatureTransformScheduler):
            raise TypeError("transforms must be a transform scheduler")
        if not isinstance(representation_cache, AlgebraicRepresentationCache):
            raise TypeError("representation_cache must be an algebraic cache")
        self._transforms = transforms
        self._cache = representation_cache

    def materialize_dependencies(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
    ) -> BatchedPhysicalMaterialization:
        component_names = tuple(component_names)
        values = tuple(values)
        if not component_names or len(component_names) != len(values):
            raise ValueError("materialization inputs must be nonempty and align")
        if len(set(component_names)) != len(component_names):
            raise ValueError("materialization component names must be unique")
        physical: dict[str, torch.Tensor] = {}
        misses: list[tuple[str, torch.Tensor]] = []
        for name, value in zip(component_names, values, strict=True):
            cached = self._cache.physical_for(name, value)
            if cached is None:
                misses.append((name, value))
            else:
                physical[name] = cached
        batch_sizes: tuple[int, ...] = ()
        if misses:
            miss_names = tuple(name for name, _ in misses)
            miss_values = tuple(value for _, value in misses)
            plan = self._transforms.compile_plan(
                miss_names,
                direction=ProjectedTransformDirection.INVERSE,
            )
            transformed = self._transforms._execute_cached_plan(
                plan,
                miss_values,
            )
            batch_sizes = plan.batch_sizes
            for name, spectral, value in zip(
                miss_names,
                miss_values,
                transformed,
                strict=True,
            ):
                physical[name] = value
                if self._cache.active:
                    self._cache.register(name, value, spectral)
        return BatchedPhysicalMaterialization(
            {name: physical[name] for name in component_names},
            batch_sizes,
        )

    def project_outputs(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
    ) -> Mapping[str, torch.Tensor]:
        transformed = self.project_output_values(component_names, values)
        return MappingProxyType(
            dict(zip(component_names, transformed, strict=True))
        )

    def project_output_values(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, ...]:
        """Return ordered outputs without constructing a mapping on hot paths."""

        return self._transforms.forward_values_many(component_names, values)


__all__ = [
    "AlgebraicPhysicalIslandScheduler",
    "BoundarySignatureTransformScheduler",
    "CompiledProjectedTransformPlan",
    "ProjectedTransformBatchKey",
    "ProjectedTransformDirection",
    "ScheduledProjectedTransforms",
]
