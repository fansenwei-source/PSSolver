"""Boundary-safe scheduling for projected transform batches.

This module is an adapter service: it knows how to batch compatible projected
transforms, but it does not own algebraic field lifetime or model equations.
One scheduler instance is bound to one geometry/runtime/dtype/device context;
one method call is bounded to one algebraic generation or explicit-RHS island.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Iterator, Protocol

import torch

from .performance import RuntimePerformanceRecorder, performance_region
from .representations import (
    AlgebraicRepresentationCache,
    BatchedPhysicalMaterialization,
)


class ProjectedTransformDirection(str, Enum):
    FORWARD = "forward"
    INVERSE = "inverse"


class ProjectedBatchAssemblyMode(str, Enum):
    """How compatible projected-transform inputs form one leading batch."""

    COPY_CAT = "copy_cat"
    CONTIGUOUS_STORAGE_VIEW = "contiguous_storage_view"
    PREALLOCATED_WORKSPACE = "preallocated_workspace"


@dataclass(frozen=True, slots=True)
class ProjectedBatchAssemblyPolicy:
    """Immutable, scheduler-local policy for assembling transform batches.

    The view mode is deliberately conservative.  It is allowed only when all
    selected tensors are consecutive contiguous views into one storage;
    otherwise the scheduler falls back to the historical ``torch.cat`` path.
    The policy never changes grouping, transform order, boundary signatures,
    or the lifetime of the tensors that it receives.
    """

    mode: ProjectedBatchAssemblyMode = ProjectedBatchAssemblyMode.COPY_CAT
    workspace_source_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.mode, ProjectedBatchAssemblyMode):
            raise TypeError("mode must be a ProjectedBatchAssemblyMode")
        prefixes = tuple(self.workspace_source_prefixes)
        if any(
            not isinstance(prefix, str)
            or not prefix
            or prefix.strip() != prefix
            for prefix in prefixes
        ):
            raise ValueError("workspace source prefixes must be nonempty strings")
        if len(set(prefixes)) != len(prefixes):
            raise ValueError("workspace source prefixes must be unique")
        if self.use_preallocated_workspace and not prefixes:
            raise ValueError(
                "preallocated workspaces require at least one source prefix"
            )
        if not self.use_preallocated_workspace and prefixes:
            raise ValueError(
                "workspace source prefixes require preallocated workspace mode"
            )
        object.__setattr__(self, "workspace_source_prefixes", prefixes)

    @property
    def allow_contiguous_storage_view(self) -> bool:
        return self.mode is ProjectedBatchAssemblyMode.CONTIGUOUS_STORAGE_VIEW

    @property
    def use_preallocated_workspace(self) -> bool:
        return self.mode is ProjectedBatchAssemblyMode.PREALLOCATED_WORKSPACE

    def uses_workspace_for_source(self, source: str) -> bool:
        return self.use_preallocated_workspace and source.startswith(
            self.workspace_source_prefixes
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "mode": self.mode.value,
            "allow_contiguous_storage_view": (
                self.allow_contiguous_storage_view
            ),
            "use_preallocated_workspace": self.use_preallocated_workspace,
            "workspace_source_prefixes": list(
                self.workspace_source_prefixes
            ),
            "fallback": "copy_cat",
            "changes_transform_order": False,
            "retains_tensor_references": False,
            "owns_runtime_workspace": self.use_preallocated_workspace,
            "workspace_retains_timestep_inputs": False,
            "workspace_lifetime": (
                "runtime"
                if self.use_preallocated_workspace
                else "not_applicable"
            ),
        }

    @classmethod
    def copy_cat(cls) -> "ProjectedBatchAssemblyPolicy":
        return cls(ProjectedBatchAssemblyMode.COPY_CAT)

    @classmethod
    def contiguous_storage_view(cls) -> "ProjectedBatchAssemblyPolicy":
        return cls(ProjectedBatchAssemblyMode.CONTIGUOUS_STORAGE_VIEW)

    @classmethod
    def preallocated_workspace(
        cls,
        *,
        source_prefixes: tuple[str, ...] = ("algebraic.",),
    ) -> "ProjectedBatchAssemblyPolicy":
        return cls(
            ProjectedBatchAssemblyMode.PREALLOCATED_WORKSPACE,
            source_prefixes,
        )


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


def _contiguous_storage_batch_view(
    values: tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor | None, str | None]:
    """Return a zero-copy leading-dimension view when it is provably safe."""

    if len(values) < 2:
        return None, "fewer_than_two_components"
    first = values[0]
    if first.layout is not torch.strided:
        return None, "non_strided_layout"
    if first.ndim == 0 or first.numel() == 0:
        return None, "empty_or_scalar"
    if torch.is_grad_enabled() and any(value.requires_grad for value in values):
        return None, "autograd_enabled"
    shape = tuple(first.shape)
    stride = tuple(first.stride())
    if not first.is_contiguous():
        return None, "non_contiguous"
    if first.is_conj() or first.is_neg():
        return None, "conjugate_or_negative_view"
    storage = first.untyped_storage()
    storage_pointer = storage.data_ptr()
    storage_nbytes = storage.nbytes()
    component_elements = first.numel()
    for position, value in enumerate(values):
        if value.layout is not torch.strided:
            return None, "non_strided_layout"
        if value.dtype != first.dtype or value.device != first.device:
            return None, "dtype_or_device_mismatch"
        if tuple(value.shape) != shape:
            return None, "shape_mismatch"
        if tuple(value.stride()) != stride or not value.is_contiguous():
            return None, "stride_or_contiguity_mismatch"
        if value.is_conj() or value.is_neg():
            return None, "conjugate_or_negative_view"
        value_storage = value.untyped_storage()
        if (
            value_storage.data_ptr() != storage_pointer
            or value_storage.nbytes() != storage_nbytes
        ):
            return None, "distinct_storage"
        expected_offset = first.storage_offset() + position * component_elements
        if value.storage_offset() != expected_offset:
            return None, "non_adjacent_storage"
    merged_shape = (len(values) * shape[0], *shape[1:])
    return (
        first.as_strided(
            merged_shape,
            stride,
            storage_offset=first.storage_offset(),
        ),
        None,
    )


class BoundarySignatureTransformScheduler:
    """Group only transforms with an identical resolved execution key."""

    def __init__(
        self,
        context: ProjectedTransformContext,
        *,
        batch_assembly_policy: ProjectedBatchAssemblyPolicy | None = None,
        enable_batch_assembly_diagnostics: bool = False,
        performance_recorder: RuntimePerformanceRecorder | None = None,
    ) -> None:
        if batch_assembly_policy is None:
            batch_assembly_policy = ProjectedBatchAssemblyPolicy.copy_cat()
        if not isinstance(
            batch_assembly_policy,
            ProjectedBatchAssemblyPolicy,
        ):
            raise TypeError(
                "batch_assembly_policy must be a "
                "ProjectedBatchAssemblyPolicy or None"
            )
        if not isinstance(enable_batch_assembly_diagnostics, bool):
            raise TypeError("enable_batch_assembly_diagnostics must be a bool")
        if performance_recorder is not None and not isinstance(
            performance_recorder,
            RuntimePerformanceRecorder,
        ):
            raise TypeError(
                "performance_recorder must be a RuntimePerformanceRecorder "
                "or None"
            )
        self._context = context
        self._batch_assembly_policy = batch_assembly_policy
        self._enable_batch_assembly_diagnostics = (
            enable_batch_assembly_diagnostics
        )
        self._performance_recorder = performance_recorder
        self._attribution_stack: list[str] = []
        self._plan_cache: dict[
            tuple[ProjectedTransformDirection, tuple[str, ...]],
            CompiledProjectedTransformPlan,
        ] = {}
        self._workspace_cache: dict[
            tuple[
                ProjectedTransformDirection,
                tuple[str, ...],
                int,
            ],
            torch.Tensor,
        ] = {}
        self._active_workspaces: set[
            tuple[
                ProjectedTransformDirection,
                tuple[str, ...],
                int,
            ]
        ] = set()
        self.reset_batch_assembly_diagnostics()

    @property
    def batch_assembly_policy(self) -> ProjectedBatchAssemblyPolicy:
        return self._batch_assembly_policy

    def reset_batch_assembly_diagnostics(self) -> None:
        """Reset integer-only counters without retaining timestep tensors."""

        self._batch_assembly_counters: dict[str, int] = {
            "singleton_batches": 0,
            "singleton_components": 0,
            "singleton_logical_input_bytes": 0,
            "copy_cat_batches": 0,
            "copy_cat_components": 0,
            "copy_cat_logical_input_bytes": 0,
            "copy_cat_materialized_output_bytes": 0,
            "contiguous_view_batches": 0,
            "contiguous_view_components": 0,
            "contiguous_view_logical_input_bytes": 0,
            "contiguous_view_materialized_output_bytes": 0,
            "preallocated_workspace_batches": 0,
            "preallocated_workspace_components": 0,
            "preallocated_workspace_logical_input_bytes": 0,
            "preallocated_workspace_materialized_output_bytes": 0,
        }
        self._batch_assembly_fallback_reasons: dict[str, int] = {}
        self._batch_assembly_sources: dict[str, dict[str, object]] = {}

    @staticmethod
    def _validate_attribution_source(source: str) -> str:
        if (
            not isinstance(source, str)
            or not source
            or source.strip() != source
            or any(
                not (character.isalnum() or character in "._-")
                for character in source
            )
        ):
            raise ValueError(
                "batch-assembly attribution source must contain only letters, "
                "digits, '.', '_' or '-'"
            )
        return source

    @contextmanager
    def attribution_scope(self, source: str) -> Iterator[None]:
        """Assign a semantic caller to nested diagnostic-only assemblies."""

        source = self._validate_attribution_source(source)
        if (
            not self._enable_batch_assembly_diagnostics
            and self._performance_recorder is None
        ):
            yield
            return
        self._attribution_stack.append(source)
        try:
            yield
        finally:
            popped = self._attribution_stack.pop()
            if popped != source:
                raise RuntimeError("batch-assembly attribution stack is corrupt")

    def _resolved_attribution_source(
        self,
        source: str | None,
    ) -> str:
        if source is not None:
            return self._validate_attribution_source(source)
        if self._attribution_stack:
            return self._attribution_stack[-1]
        return "unattributed"

    @staticmethod
    def _empty_source_counters() -> dict[str, object]:
        return {
            "singleton_batches": 0,
            "singleton_components": 0,
            "singleton_logical_input_bytes": 0,
            "copy_cat_batches": 0,
            "copy_cat_components": 0,
            "copy_cat_logical_input_bytes": 0,
            "copy_cat_materialized_output_bytes": 0,
            "contiguous_view_batches": 0,
            "contiguous_view_components": 0,
            "contiguous_view_logical_input_bytes": 0,
            "contiguous_view_materialized_output_bytes": 0,
            "preallocated_workspace_batches": 0,
            "preallocated_workspace_components": 0,
            "preallocated_workspace_logical_input_bytes": 0,
            "preallocated_workspace_materialized_output_bytes": 0,
            "fallback_reasons": {},
            "retained_tensor_references": 0,
        }

    def _source_counters(self, source: str) -> dict[str, object]:
        counters = self._batch_assembly_sources.get(source)
        if counters is None:
            counters = self._empty_source_counters()
            counters["timing_region"] = (
                f"projected_batch_assembly.source.{source}"
            )
            self._batch_assembly_sources[source] = counters
        return counters

    def batch_assembly_diagnostics(self) -> Mapping[str, object]:
        """Return JSON-safe counters; the snapshot contains no tensors."""

        return MappingProxyType(
            {
                "schema_version": 2,
                "enabled": self._enable_batch_assembly_diagnostics,
                "policy": self._batch_assembly_policy.to_metadata(),
                **self._batch_assembly_counters,
                "fallback_reasons": dict(
                    self._batch_assembly_fallback_reasons
                ),
                "source_attribution": {
                    source: {
                        **counters,
                        "fallback_reasons": dict(
                            counters["fallback_reasons"]
                        ),
                    }
                    for source, counters in sorted(
                        self._batch_assembly_sources.items()
                    )
                },
                "retained_tensor_references": 0,
                "workspace_count": len(self._workspace_cache),
                "workspace_allocated_bytes": sum(
                    workspace.numel() * workspace.element_size()
                    for workspace in self._workspace_cache.values()
                ),
                "workspace_active_count": len(self._active_workspaces),
                "workspace_retains_timestep_inputs": False,
            }
        )

    def _record_batch_assembly(
        self,
        *,
        mode: str,
        values: tuple[torch.Tensor, ...],
        source: str,
        fallback_reason: str | None = None,
    ) -> None:
        if not self._enable_batch_assembly_diagnostics:
            return
        logical_bytes = sum(
            value.numel() * value.element_size() for value in values
        )
        self._batch_assembly_counters[f"{mode}_batches"] += 1
        self._batch_assembly_counters[f"{mode}_components"] += len(values)
        self._batch_assembly_counters[
            f"{mode}_logical_input_bytes"
        ] += logical_bytes
        materialized_bytes = (
            logical_bytes
            if mode in {"copy_cat", "preallocated_workspace"}
            else 0
        )
        self._batch_assembly_counters[
            f"{mode}_materialized_output_bytes"
        ] += materialized_bytes
        source_counters = self._source_counters(source)
        source_counters[f"{mode}_batches"] += 1
        source_counters[f"{mode}_components"] += len(values)
        source_counters[f"{mode}_logical_input_bytes"] += logical_bytes
        source_counters[
            f"{mode}_materialized_output_bytes"
        ] += materialized_bytes
        if fallback_reason is not None:
            reasons = self._batch_assembly_fallback_reasons
            reasons[fallback_reason] = reasons.get(fallback_reason, 0) + 1
            source_reasons = source_counters["fallback_reasons"]
            if not isinstance(source_reasons, dict):
                raise RuntimeError("source fallback counters are invalid")
            source_reasons[fallback_reason] = (
                source_reasons.get(fallback_reason, 0) + 1
            )

    def _record_singleton_batch(
        self,
        *,
        value: torch.Tensor,
        source: str,
    ) -> None:
        if not self._enable_batch_assembly_diagnostics:
            return
        logical_bytes = value.numel() * value.element_size()
        self._batch_assembly_counters["singleton_batches"] += 1
        self._batch_assembly_counters["singleton_components"] += 1
        self._batch_assembly_counters[
            "singleton_logical_input_bytes"
        ] += logical_bytes
        source_counters = self._source_counters(source)
        source_counters["singleton_batches"] += 1
        source_counters["singleton_components"] += 1
        source_counters["singleton_logical_input_bytes"] += logical_bytes

    def _assemble_batch(
        self,
        values: tuple[torch.Tensor, ...],
        *,
        source: str,
        workspace: torch.Tensor | None = None,
    ) -> torch.Tensor:
        timing_region = f"projected_batch_assembly.source.{source}"
        with performance_region(self._performance_recorder, timing_region):
            fallback_reason = (
                "source_out_of_scope"
                if self._batch_assembly_policy.use_preallocated_workspace
                else "policy_copy_cat"
            )
            if self._batch_assembly_policy.allow_contiguous_storage_view:
                packed, fallback_reason = _contiguous_storage_batch_view(values)
                if packed is not None:
                    self._record_batch_assembly(
                        mode="contiguous_view",
                        values=values,
                        source=source,
                    )
                    return packed
            if self._batch_assembly_policy.uses_workspace_for_source(source):
                if workspace is None:
                    raise RuntimeError(
                        "preallocated batch assembly lacks a compiled workspace"
                    )
                if torch.is_grad_enabled() and any(
                    value.requires_grad for value in values
                ):
                    packed = torch.cat(values, dim=0)
                    self._record_batch_assembly(
                        mode="copy_cat",
                        values=values,
                        source=source,
                        fallback_reason="autograd_enabled",
                    )
                    return packed
                expected_shape = (
                    len(values) * self._context.batch_size,
                    *values[0].shape[1:],
                )
                if tuple(workspace.shape) != expected_shape:
                    raise RuntimeError(
                        "compiled batch workspace has an invalid shape"
                    )
                if any(
                    tuple(value.shape)
                    != (self._context.batch_size, *values[0].shape[1:])
                    or value.dtype != workspace.dtype
                    or value.device != workspace.device
                    for value in values
                ):
                    raise ValueError(
                        "preallocated batch inputs do not match the workspace"
                    )
                packed = torch.cat(values, dim=0, out=workspace)
                self._record_batch_assembly(
                    mode="preallocated_workspace",
                    values=values,
                    source=source,
                )
                return packed
            packed = torch.cat(values, dim=0)
            self._record_batch_assembly(
                mode="copy_cat",
                values=values,
                source=source,
                fallback_reason=fallback_reason,
            )
            return packed

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

    @property
    def compiled_workspace_count(self) -> int:
        """Number of runtime-owned, input-free batch workspaces."""

        return len(self._workspace_cache)

    def _workspace_identity(
        self,
        plan: CompiledProjectedTransformPlan,
        batch_index: int,
    ) -> tuple[ProjectedTransformDirection, tuple[str, ...], int]:
        return (plan.direction, plan.component_names, batch_index)

    def _workspace_for_plan_batch(
        self,
        plan: CompiledProjectedTransformPlan,
        batch_index: int,
    ) -> torch.Tensor:
        identity = self._workspace_identity(plan, batch_index)
        cached = self._workspace_cache.get(identity)
        if cached is not None:
            return cached
        context = self._context
        indices = plan.batch_indices[batch_index]
        if len(indices) < 2:
            raise RuntimeError("singleton batches do not own a workspace")
        if plan.direction is ProjectedTransformDirection.FORWARD:
            shape = context.physical_shape
            dtype = context.real_dtype
        else:
            shape = context.spectral_shape
            dtype = context.spectral_dtype
        workspace = torch.empty(
            (len(indices) * context.batch_size, *shape),
            dtype=dtype,
            device=context.device,
        )
        self._workspace_cache[identity] = workspace
        return workspace

    @staticmethod
    def _shares_storage(left: torch.Tensor, right: torch.Tensor) -> bool:
        if left.layout is not torch.strided or right.layout is not torch.strided:
            return False
        return (
            left.untyped_storage().data_ptr()
            == right.untyped_storage().data_ptr()
        )

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
        *,
        attribution_source: str | None = None,
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
        return self._execute_cached_plan(
            plan,
            values,
            attribution_source=attribution_source,
        )

    def storage_compatible_order(
        self,
        component_names: tuple[str, ...],
        *,
        direction: ProjectedTransformDirection,
    ) -> tuple[str, ...]:
        """Order storage by scheduler compatibility without retaining tensors."""

        plan = self.compile_plan(component_names, direction=direction)
        return tuple(
            plan.component_names[index]
            for indices in plan.batch_indices
            for index in indices
        )

    def _execute_cached_plan(
        self,
        plan: CompiledProjectedTransformPlan,
        values: tuple[torch.Tensor, ...],
        *,
        attribution_source: str | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Execute a scheduler-owned plan without repeating cold-path checks."""

        source = self._resolved_attribution_source(attribution_source)
        result: list[torch.Tensor | None] = [None] * len(values)
        for batch_index, (key, indices) in enumerate(
            zip(
                plan.batch_keys,
                plan.batch_indices,
                strict=True,
            )
        ):
            if len(indices) == 1:
                index = indices[0]
                self._record_singleton_batch(
                    value=values[index],
                    source=source,
                )
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
            workspace_identity = self._workspace_identity(plan, batch_index)
            workspace = None
            if self._batch_assembly_policy.uses_workspace_for_source(source):
                workspace = self._workspace_for_plan_batch(plan, batch_index)
            if workspace is not None:
                if workspace_identity in self._active_workspaces:
                    raise RuntimeError(
                        "preallocated batch workspace does not support reentry"
                    )
                self._active_workspaces.add(workspace_identity)
            try:
                packed = self._assemble_batch(
                    tuple(values[index] for index in indices),
                    source=source,
                    workspace=workspace,
                )
                transformed = self._context.transform_projected_packed(
                    key.boundary_signature,
                    packed,
                    direction=plan.direction,
                )
                if workspace is not None and self._shares_storage(
                    transformed,
                    workspace,
                ):
                    raise RuntimeError(
                        "projected transform output aliases its input workspace"
                    )
            finally:
                if workspace is not None:
                    self._active_workspaces.remove(workspace_identity)
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
        attribution_source: str | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Transform values through a cached invariant grouping plan."""

        component_names = tuple(component_names)
        values = tuple(values)
        if not component_names or len(component_names) != len(values):
            raise ValueError(
                "scheduled projected inputs must be nonempty and align"
            )
        plan = self.compile_plan(component_names, direction=direction)
        return self._execute_cached_plan(
            plan,
            values,
            attribution_source=attribution_source,
        )

    def transform_many(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
        *,
        direction: ProjectedTransformDirection,
        attribution_source: str | None = None,
    ) -> ScheduledProjectedTransforms:
        component_names = tuple(component_names)
        values = tuple(values)
        if not component_names or len(component_names) != len(values):
            raise ValueError(
                "scheduled projected inputs must be nonempty and align"
            )
        plan = self.compile_plan(component_names, direction=direction)
        result = self._execute_cached_plan(
            plan,
            values,
            attribution_source=attribution_source,
        )
        return ScheduledProjectedTransforms(
            result,
            plan.batch_sizes,
            plan.batch_keys,
        )

    def forward_values_many(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
        *,
        attribution_source: str | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return self.transform_values_many(
            component_names,
            values,
            direction=ProjectedTransformDirection.FORWARD,
            attribution_source=attribution_source,
        )

    def inverse_values_many(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
        *,
        attribution_source: str | None = None,
    ) -> tuple[torch.Tensor, ...]:
        return self.transform_values_many(
            component_names,
            values,
            direction=ProjectedTransformDirection.INVERSE,
            attribution_source=attribution_source,
        )

    def forward_many(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
        *,
        attribution_source: str | None = None,
    ) -> ScheduledProjectedTransforms:
        return self.transform_many(
            component_names,
            values,
            direction=ProjectedTransformDirection.FORWARD,
            attribution_source=attribution_source,
        )

    def inverse_many(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
        *,
        attribution_source: str | None = None,
    ) -> ScheduledProjectedTransforms:
        return self.transform_many(
            component_names,
            values,
            direction=ProjectedTransformDirection.INVERSE,
            attribution_source=attribution_source,
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
        *,
        attribution_source: str | None = None,
    ) -> Mapping[str, torch.Tensor]:
        transformed = self.project_output_values(
            component_names,
            values,
            attribution_source=attribution_source,
        )
        return MappingProxyType(
            dict(zip(component_names, transformed, strict=True))
        )

    def project_output_values(
        self,
        component_names: tuple[str, ...],
        values: tuple[torch.Tensor, ...],
        *,
        attribution_source: str | None = None,
    ) -> tuple[torch.Tensor, ...]:
        """Return ordered outputs without constructing a mapping on hot paths."""

        return self._transforms.forward_values_many(
            component_names,
            values,
            attribution_source=attribution_source,
        )


__all__ = [
    "AlgebraicPhysicalIslandScheduler",
    "BoundarySignatureTransformScheduler",
    "CompiledProjectedTransformPlan",
    "ProjectedBatchAssemblyMode",
    "ProjectedBatchAssemblyPolicy",
    "ProjectedTransformBatchKey",
    "ProjectedTransformDirection",
    "ScheduledProjectedTransforms",
]
