"""Bounded generation-local workspace contracts for Phase 3.

Workspace buffers are allocated once from an immutable plan.  Generation
tokens prevent consumers from treating stale contents as current state.  The
module is provisional and remains disconnected from qualified runtimes until
the separately gated canary connection stage.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from collections.abc import Mapping

import torch


@dataclass(frozen=True, slots=True)
class WorkspaceSlotSpec:
    """One fixed-shape, fixed-dtype generation-local buffer."""

    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    purpose: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.isidentifier():
            raise ValueError("workspace slot name must be a Python identifier")
        if not isinstance(self.shape, tuple) or any(
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < 0
            for value in self.shape
        ):
            raise ValueError("workspace slot shape must contain non-negative integers")
        if not isinstance(self.dtype, torch.dtype):
            raise TypeError("workspace slot dtype must be a torch.dtype")
        if not isinstance(self.purpose, str) or not self.purpose.strip():
            raise ValueError("workspace slot purpose must be non-empty")

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)

    @property
    def size_bytes(self) -> int:
        return self.element_count * torch.empty((), dtype=self.dtype).element_size()

    def to_metadata(self) -> dict[str, object]:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype": str(self.dtype),
            "purpose": self.purpose,
            "size_bytes": self.size_bytes,
        }


class WorkspacePlan:
    """Immutable allocation plan for one bounded runtime workspace."""

    __slots__ = (
        "_device",
        "_index",
        "_maximum_bytes",
        "_slots",
    )

    def __init__(
        self,
        *,
        slots: tuple[WorkspaceSlotSpec, ...],
        device: torch.device | str,
        maximum_bytes: int | None = None,
    ) -> None:
        if not isinstance(slots, tuple):
            raise TypeError("workspace slots must be a tuple")
        if any(not isinstance(slot, WorkspaceSlotSpec) for slot in slots):
            raise TypeError("workspace slots must be WorkspaceSlotSpec values")
        names = tuple(slot.name for slot in slots)
        if len(set(names)) != len(names):
            raise ValueError("workspace slot names must be unique")
        try:
            normalized_device = torch.device(device)
        except (TypeError, RuntimeError) as exc:
            raise ValueError("workspace device is invalid") from exc
        required_bytes = sum(slot.size_bytes for slot in slots)
        if maximum_bytes is not None:
            if (
                not isinstance(maximum_bytes, int)
                or isinstance(maximum_bytes, bool)
                or maximum_bytes < 0
            ):
                raise ValueError("maximum_bytes must be non-negative or None")
            if required_bytes > maximum_bytes:
                raise ValueError("workspace plan exceeds maximum_bytes")
        self._slots = slots
        self._device = normalized_device
        self._maximum_bytes = maximum_bytes
        self._index = MappingProxyType(
            {name: index for index, name in enumerate(names)}
        )

    @property
    def slots(self) -> tuple[WorkspaceSlotSpec, ...]:
        return self._slots

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def required_bytes(self) -> int:
        return sum(slot.size_bytes for slot in self._slots)

    @property
    def maximum_bytes(self) -> int | None:
        return self._maximum_bytes

    def slot_index(self, name: str) -> int:
        try:
            return self._index[name]
        except KeyError as exc:
            raise KeyError(f"unknown workspace slot {name!r}") from exc

    def allocate(self) -> "RuntimeWorkspace":
        return RuntimeWorkspace(
            self,
            tuple(
                torch.empty(slot.shape, dtype=slot.dtype, device=self._device)
                for slot in self._slots
            ),
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "device": str(self._device),
            "slot_count": len(self._slots),
            "required_bytes": self.required_bytes,
            "maximum_bytes": self._maximum_bytes,
            "slots": [slot.to_metadata() for slot in self._slots],
        }


class RuntimeWorkspace:
    """Preallocated buffers with explicit generation validity."""

    __slots__ = (
        "_active",
        "_buffers",
        "_generation",
        "_plan",
        "_valid_generation",
    )

    def __init__(
        self,
        plan: WorkspacePlan,
        buffers: tuple[torch.Tensor, ...],
    ) -> None:
        if not isinstance(plan, WorkspacePlan):
            raise TypeError("plan must be a WorkspacePlan")
        if not isinstance(buffers, tuple) or len(buffers) != len(plan.slots):
            raise ValueError("workspace buffers do not match plan")
        for buffer, slot in zip(buffers, plan.slots, strict=True):
            if not isinstance(buffer, torch.Tensor):
                raise TypeError("workspace buffers must be tensors")
            if (
                tuple(buffer.shape) != slot.shape
                or buffer.dtype != slot.dtype
                or buffer.device != plan.device
            ):
                raise ValueError("workspace buffer does not match slot")
        self._plan = plan
        self._buffers = buffers
        self._generation = 0
        self._active = False
        self._valid_generation = [-1] * len(buffers)

    @property
    def plan(self) -> WorkspacePlan:
        return self._plan

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def active(self) -> bool:
        return self._active

    def begin_generation(self) -> int:
        if self._active:
            raise RuntimeError("workspace generation is already active")
        self._generation += 1
        self._active = True
        return self._generation

    def _require_token(self, token: object) -> int:
        if (
            not isinstance(token, int)
            or isinstance(token, bool)
            or not self._active
            or token != self._generation
        ):
            raise RuntimeError("workspace generation token is stale or inactive")
        return token

    def buffer_at(self, index: int, *, token: int) -> torch.Tensor:
        self._require_token(token)
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or index < 0
            or index >= len(self._buffers)
        ):
            raise IndexError("workspace slot index is out of range")
        return self._buffers[index]

    def buffer(self, name: str, *, token: int) -> torch.Tensor:
        return self.buffer_at(self._plan.slot_index(name), token=token)

    def publish_at(self, index: int, *, token: int) -> None:
        self.buffer_at(index, token=token)
        self._valid_generation[index] = self._generation

    def publish(self, name: str, *, token: int) -> None:
        self.publish_at(self._plan.slot_index(name), token=token)

    def require_at(self, index: int, *, token: int) -> torch.Tensor:
        value = self.buffer_at(index, token=token)
        if self._valid_generation[index] != self._generation:
            raise RuntimeError("workspace slot is not valid in this generation")
        return value

    def require(self, name: str, *, token: int) -> torch.Tensor:
        return self.require_at(self._plan.slot_index(name), token=token)

    def end_generation(self, *, token: int) -> None:
        self._require_token(token)
        self._active = False

    def abort_generation(self, *, token: int) -> None:
        self.end_generation(token=token)

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "plan": self._plan.to_metadata(),
            "generation": self._generation,
            "active": self._active,
            "published_slots": [
                slot.name
                for index, slot in enumerate(self._plan.slots)
                if self._active
                and self._valid_generation[index] == self._generation
            ],
        }


__all__ = [
    "RuntimeWorkspace",
    "WorkspacePlan",
    "WorkspaceSlotSpec",
]
