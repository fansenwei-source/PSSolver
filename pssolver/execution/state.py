"""Provisional runtime-state contracts for the v0.2 migration.

The objects in this module are deliberately disconnected from the qualified
Plane runtimes.  They make ownership and representation transitions
executable before Phase 3 changes a production timestep.  Importing this
module therefore allocates no tensor, installs no adapter, and changes no
default.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
import math
from types import MappingProxyType

import torch


class CurrentRepresentation(str, Enum):
    """Which evolved representation is current at one logical generation."""

    SYNCHRONIZED = "synchronized"
    PHYSICAL = "physical"
    SPECTRAL = "spectral"


class RepresentationLedger:
    """Track physical/spectral validity without owning or transforming data.

    A logical update creates a new generation in exactly one representation.
    Synchronizing the other representation never creates a new generation.
    This distinction is needed to replace the implicit validity assumptions in
    the legacy ``Fields`` container without putting checks in the final hot
    loop.
    """

    __slots__ = (
        "_generation",
        "_physical_generation",
        "_spectral_generation",
    )

    def __init__(
        self,
        *,
        generation: int = 0,
        physical_generation: int | None = 0,
        spectral_generation: int | None = 0,
    ) -> None:
        self._generation = self._require_generation(generation, "generation")
        self._physical_generation = self._require_optional_generation(
            physical_generation,
            "physical_generation",
        )
        self._spectral_generation = self._require_optional_generation(
            spectral_generation,
            "spectral_generation",
        )
        self._validate()

    @staticmethod
    def _require_generation(value: object, description: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{description} must be a non-negative integer")
        return value

    @classmethod
    def _require_optional_generation(
        cls,
        value: object,
        description: str,
    ) -> int | None:
        if value is None:
            return None
        return cls._require_generation(value, description)

    def _validate(self) -> None:
        generations = (
            self._physical_generation,
            self._spectral_generation,
        )
        if any(
            value is not None and value > self._generation
            for value in generations
        ):
            raise ValueError("representation generation cannot lead state")
        if self._generation not in generations:
            raise ValueError("at least one representation must be current")

    @property
    def generation(self) -> int:
        return self._generation

    @property
    def physical_generation(self) -> int | None:
        return self._physical_generation

    @property
    def spectral_generation(self) -> int | None:
        return self._spectral_generation

    @property
    def current(self) -> CurrentRepresentation:
        physical = self._physical_generation == self._generation
        spectral = self._spectral_generation == self._generation
        if physical and spectral:
            return CurrentRepresentation.SYNCHRONIZED
        if physical:
            return CurrentRepresentation.PHYSICAL
        if spectral:
            return CurrentRepresentation.SPECTRAL
        raise RuntimeError("representation ledger has no current value")

    def require_physical_current(self) -> None:
        if self._physical_generation != self._generation:
            raise RuntimeError("physical representation is stale")

    def require_spectral_current(self) -> None:
        if self._spectral_generation != self._generation:
            raise RuntimeError("spectral representation is stale")

    def mark_physical_updated(self) -> int:
        """Record a new physical value and make the old spectrum stale."""

        self.require_physical_current()
        self._generation += 1
        self._physical_generation = self._generation
        return self._generation

    def mark_spectral_updated(self) -> int:
        """Record a new spectral value and make the old physical state stale."""

        self.require_spectral_current()
        self._generation += 1
        self._spectral_generation = self._generation
        return self._generation

    def mark_physical_synchronized(self) -> None:
        """Record an inverse transform of the current spectrum."""

        self.require_spectral_current()
        self._physical_generation = self._generation

    def mark_spectral_synchronized(self) -> None:
        """Record a forward transform of the current physical state."""

        self.require_physical_current()
        self._spectral_generation = self._generation

    def to_metadata(self) -> dict[str, object]:
        return {
            "generation": self._generation,
            "physical_generation": self._physical_generation,
            "spectral_generation": self._spectral_generation,
            "current": self.current.value,
        }


class IntegratorProgress:
    """Persistent clock and refresh phase for one runtime state."""

    __slots__ = (
        "_completed_steps",
        "_dt",
        "_refresh_count",
        "_refresh_interval",
        "_refresh_step_count",
    )

    def __init__(
        self,
        *,
        dt: float,
        completed_steps: int = 0,
        refresh_interval: int | None = 20,
        refresh_step_count: int = 0,
        refresh_count: int = 0,
    ) -> None:
        if (
            not isinstance(dt, (int, float))
            or isinstance(dt, bool)
            or not math.isfinite(float(dt))
            or float(dt) <= 0.0
        ):
            raise ValueError("dt must be positive and finite")
        self._dt = float(dt)
        self._completed_steps = self._require_count(
            completed_steps,
            "completed_steps",
        )
        self._refresh_interval = self._require_interval(refresh_interval)
        self._refresh_step_count = self._require_count(
            refresh_step_count,
            "refresh_step_count",
        )
        self._refresh_count = self._require_count(
            refresh_count,
            "refresh_count",
        )
        self._validate_counters()

    @staticmethod
    def _require_count(value: object, description: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{description} must be a non-negative integer")
        return value

    @staticmethod
    def _require_interval(value: object) -> int | None:
        if value is None:
            return None
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("refresh_interval must be positive or None")
        return value

    def _validate_counters(self) -> None:
        interval = self._refresh_interval
        if interval is None:
            valid = (
                self._refresh_count == 0
                and self._refresh_step_count == self._completed_steps
            )
        else:
            valid = (
                self._refresh_step_count < interval
                and self._refresh_count * interval
                + self._refresh_step_count
                == self._completed_steps
            )
        if not valid:
            raise ValueError("spectral-refresh counters are inconsistent")

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def completed_steps(self) -> int:
        return self._completed_steps

    @property
    def time(self) -> float:
        return self._completed_steps * self._dt

    @property
    def refresh_interval(self) -> int | None:
        return self._refresh_interval

    @property
    def refresh_step_count(self) -> int:
        return self._refresh_step_count

    @property
    def refresh_count(self) -> int:
        return self._refresh_count

    @property
    def refresh_due_after_next_step(self) -> bool:
        interval = self._refresh_interval
        return (
            interval is not None
            and self._refresh_step_count + 1 >= interval
        )

    def commit_step(self, *, refreshed: bool) -> None:
        """Commit one successful step and its required refresh outcome."""

        if not isinstance(refreshed, bool):
            raise TypeError("refreshed must be a bool")
        refresh_due = self.refresh_due_after_next_step
        if refreshed != refresh_due:
            raise RuntimeError("refresh outcome does not match refresh clock")
        self._completed_steps += 1
        self._refresh_step_count += 1
        if refresh_due:
            self._refresh_step_count = 0
            self._refresh_count += 1

    def to_metadata(self) -> dict[str, object]:
        return {
            "dt": self._dt,
            "completed_steps": self._completed_steps,
            "time": self.time,
            "spectral_refresh": {
                "interval": self._refresh_interval,
                "step_count": self._refresh_step_count,
                "refresh_count": self._refresh_count,
            },
        }


class RuntimeState:
    """Sole provisional owner of evolved tensors and persistent progress.

    The constructor stores the supplied tensors by identity.  It never clones,
    converts, transforms, or reallocates them.  This permits later adapters to
    expose qualified legacy storage while ownership is migrated explicitly.
    """

    __slots__ = (
        "_component_index",
        "_component_names",
        "_persistent_algebraic",
        "_physical",
        "_progress",
        "_representations",
        "_spectral",
    )

    def __init__(
        self,
        *,
        component_names: tuple[str, ...],
        physical: torch.Tensor,
        spectral: torch.Tensor,
        progress: IntegratorProgress,
        representations: RepresentationLedger | None = None,
        persistent_algebraic: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        self._component_names = self._validate_component_names(component_names)
        self._component_index = MappingProxyType(
            {name: index for index, name in enumerate(self._component_names)}
        )
        self._physical = self._validate_storage(
            physical,
            "physical",
            len(self._component_names),
        )
        self._spectral = self._validate_storage(
            spectral,
            "spectral",
            len(self._component_names),
        )
        if self._physical.device != self._spectral.device:
            raise ValueError("physical and spectral storage devices differ")
        if self._physical.ndim > 1 and self._spectral.ndim > 1:
            if self._physical.shape[1] != self._spectral.shape[1]:
                raise ValueError("physical and spectral batch dimensions differ")
        if not isinstance(progress, IntegratorProgress):
            raise TypeError("progress must be an IntegratorProgress")
        if representations is None:
            representations = RepresentationLedger()
        if not isinstance(representations, RepresentationLedger):
            raise TypeError("representations must be a RepresentationLedger")
        self._progress = progress
        self._representations = representations
        self._persistent_algebraic = self._validate_persistent_algebraic(
            persistent_algebraic or {}
        )

    @staticmethod
    def _validate_component_names(value: object) -> tuple[str, ...]:
        if not isinstance(value, tuple) or not value:
            raise ValueError("component_names must be a non-empty tuple")
        if any(
            not isinstance(name, str) or not name or not name.isidentifier()
            for name in value
        ):
            raise ValueError("component names must be Python identifiers")
        if len(set(value)) != len(value):
            raise ValueError("component names must be unique")
        return value

    @staticmethod
    def _validate_storage(
        value: object,
        description: str,
        component_count: int,
    ) -> torch.Tensor:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{description} storage must be a tensor")
        if value.ndim < 1 or value.shape[0] != component_count:
            raise ValueError(
                f"{description} storage leading dimension must equal "
                "component count"
            )
        return value

    def _validate_persistent_algebraic(
        self,
        values: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if not isinstance(values, Mapping):
            raise TypeError("persistent_algebraic must be a mapping")
        validated: dict[str, torch.Tensor] = {}
        for name, value in values.items():
            if not isinstance(name, str) or not name or not name.isidentifier():
                raise ValueError("persistent algebraic names must be identifiers")
            if not isinstance(value, torch.Tensor):
                raise TypeError("persistent algebraic values must be tensors")
            if value.device != self._physical.device:
                raise ValueError("persistent algebraic tensor device differs")
            validated[name] = value
        return validated

    @property
    def component_names(self) -> tuple[str, ...]:
        return self._component_names

    @property
    def physical(self) -> torch.Tensor:
        return self._physical

    @property
    def spectral(self) -> torch.Tensor:
        return self._spectral

    @property
    def progress(self) -> IntegratorProgress:
        return self._progress

    @property
    def representations(self) -> RepresentationLedger:
        return self._representations

    @property
    def persistent_algebraic(self) -> Mapping[str, torch.Tensor]:
        return MappingProxyType(self._persistent_algebraic)

    def physical_component(self, name: str) -> torch.Tensor:
        try:
            return self._physical[self._component_index[name]]
        except KeyError as exc:
            raise KeyError(f"unknown evolved component {name!r}") from exc

    def spectral_component(self, name: str) -> torch.Tensor:
        try:
            return self._spectral[self._component_index[name]]
        except KeyError as exc:
            raise KeyError(f"unknown evolved component {name!r}") from exc

    def set_persistent_algebraic(
        self,
        name: str,
        value: torch.Tensor,
    ) -> None:
        validated = self._validate_persistent_algebraic({name: value})
        self._persistent_algebraic[name] = validated[name]

    def replace_progress(self, progress: IntegratorProgress) -> None:
        """Replace restored progress without changing tensor ownership."""

        if not isinstance(progress, IntegratorProgress):
            raise TypeError("progress must be an IntegratorProgress")
        self._progress = progress

    def replace_representations(
        self,
        representations: RepresentationLedger,
    ) -> None:
        """Replace validity after an exact synchronized restore."""

        if not isinstance(representations, RepresentationLedger):
            raise TypeError("representations must be a RepresentationLedger")
        self._representations = representations

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "component_names": list(self._component_names),
            "physical": {
                "shape": list(self._physical.shape),
                "dtype": str(self._physical.dtype),
                "device": str(self._physical.device),
            },
            "spectral": {
                "shape": list(self._spectral.shape),
                "dtype": str(self._spectral.dtype),
                "device": str(self._spectral.device),
            },
            "progress": self._progress.to_metadata(),
            "representations": self._representations.to_metadata(),
            "persistent_algebraic_names": sorted(self._persistent_algebraic),
        }


__all__ = [
    "CurrentRepresentation",
    "IntegratorProgress",
    "RepresentationLedger",
    "RuntimeState",
]
