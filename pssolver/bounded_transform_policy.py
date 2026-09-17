"""Auditable opt-in policy for bounded-axis transform implementation choice."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


_KINDS = ("dct", "dst")
_DIRECTIONS = ("forward", "inverse")
_EXECUTION_MODES = ("full", "truncated")
_VALUE_TYPES = ("real", "complex")
_ALGORITHMS = ("dense", "fft")


@dataclass(frozen=True)
class BoundedTransformSelectionContext:
    """Complete runtime identity used to make one bounded-axis decision."""

    geometry: str
    local_axis: int
    kind: str
    direction: str
    execution_mode: str
    physical_size: int
    retained_count: int
    line_count: int
    device_type: str
    device_name: str
    real_dtype: str
    value_type: str

    def __post_init__(self) -> None:
        if not self.geometry:
            raise ValueError("geometry must be non-empty")
        if self.local_axis < 0:
            raise ValueError("local_axis must be non-negative")
        if self.kind not in _KINDS:
            raise ValueError(f"kind must be one of {_KINDS}")
        if self.direction not in _DIRECTIONS:
            raise ValueError(f"direction must be one of {_DIRECTIONS}")
        if self.execution_mode not in _EXECUTION_MODES:
            raise ValueError(
                f"execution_mode must be one of {_EXECUTION_MODES}"
            )
        if self.physical_size <= 0:
            raise ValueError("physical_size must be positive")
        if not 0 < self.retained_count <= self.physical_size:
            raise ValueError(
                "retained_count must be in [1, physical_size]"
            )
        if self.line_count <= 0:
            raise ValueError("line_count must be positive")
        if not self.device_type or not self.device_name:
            raise ValueError("device identity must be non-empty")
        if self.real_dtype not in {"float32", "float64"}:
            raise ValueError("real_dtype must be float32 or float64")
        if self.value_type not in _VALUE_TYPES:
            raise ValueError(f"value_type must be one of {_VALUE_TYPES}")
        expected_mode = (
            "full"
            if self.retained_count == self.physical_size
            else "truncated"
        )
        if self.execution_mode != expected_mode:
            raise ValueError(
                "execution_mode disagrees with retained_count and "
                "physical_size"
            )

    @property
    def retained_fraction(self) -> float:
        return self.retained_count / self.physical_size

    def to_metadata(self) -> dict[str, object]:
        return {
            **asdict(self),
            "retained_fraction": self.retained_fraction,
        }


@dataclass(frozen=True)
class BoundedTransformQualificationCell:
    """One exact, evidence-bound permission to use the FFT implementation."""

    cell_id: str
    geometry: str
    local_axis: int | None
    kind: str
    direction: str
    execution_mode: str
    physical_size: int
    retained_count: int
    minimum_line_count: int
    device_type: str
    device_name: str
    real_dtype: str
    value_type: str
    selected_algorithm: str
    evidence_id: str

    def __post_init__(self) -> None:
        if not self.cell_id or not self.evidence_id:
            raise ValueError("cell_id and evidence_id must be non-empty")
        if self.local_axis is not None and self.local_axis < 0:
            raise ValueError("local_axis must be non-negative or None")
        context = BoundedTransformSelectionContext(
            geometry=self.geometry,
            local_axis=0 if self.local_axis is None else self.local_axis,
            kind=self.kind,
            direction=self.direction,
            execution_mode=self.execution_mode,
            physical_size=self.physical_size,
            retained_count=self.retained_count,
            line_count=self.minimum_line_count,
            device_type=self.device_type,
            device_name=self.device_name,
            real_dtype=self.real_dtype,
            value_type=self.value_type,
        )
        del context
        if self.selected_algorithm != "fft":
            raise ValueError(
                "qualification cells may only authorize the opt-in FFT "
                "algorithm"
            )

    def matches(self, context: BoundedTransformSelectionContext) -> bool:
        return (
            context.geometry == self.geometry
            and (
                self.local_axis is None
                or context.local_axis == self.local_axis
            )
            and context.kind == self.kind
            and context.direction == self.direction
            and context.execution_mode == self.execution_mode
            and context.physical_size == self.physical_size
            and context.retained_count == self.retained_count
            and context.line_count >= self.minimum_line_count
            and context.device_type == self.device_type
            and context.device_name == self.device_name
            and context.real_dtype == self.real_dtype
            and context.value_type == self.value_type
        )

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, object],
    ) -> "BoundedTransformQualificationCell":
        return cls(**dict(metadata))


@dataclass(frozen=True)
class BoundedTransformSelection:
    """Auditable result of applying a qualified policy to one context."""

    algorithm: str
    reason: str
    matched_cell_id: str | None
    evidence_id: str | None

    def __post_init__(self) -> None:
        if self.algorithm not in _ALGORITHMS:
            raise ValueError(f"algorithm must be one of {_ALGORITHMS}")
        if not self.reason:
            raise ValueError("reason must be non-empty")

    def to_metadata(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QualifiedBoundedTransformPolicy:
    """Immutable allow-list policy with a mandatory dense fallback."""

    name: str
    cells: tuple[BoundedTransformQualificationCell, ...]
    fallback_algorithm: str = "dense"
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("policy name must be non-empty")
        if self.schema_version != 1:
            raise ValueError("unsupported bounded-transform policy schema")
        if self.fallback_algorithm != "dense":
            raise ValueError("qualified policies must fall back to dense")
        cell_ids = [cell.cell_id for cell in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("qualification cell IDs must be unique")

    def select(
        self,
        context: BoundedTransformSelectionContext,
    ) -> BoundedTransformSelection:
        matches = [cell for cell in self.cells if cell.matches(context)]
        if len(matches) > 1:
            raise RuntimeError(
                "bounded-transform policy is ambiguous for context: "
                + json.dumps(context.to_metadata(), sort_keys=True)
            )
        if not matches:
            return BoundedTransformSelection(
                algorithm=self.fallback_algorithm,
                reason="no_matching_qualification_cell",
                matched_cell_id=None,
                evidence_id=None,
            )
        cell = matches[0]
        return BoundedTransformSelection(
            algorithm=cell.selected_algorithm,
            reason="matched_qualification_cell",
            matched_cell_id=cell.cell_id,
            evidence_id=cell.evidence_id,
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "fallback_algorithm": self.fallback_algorithm,
            "cells": [cell.to_metadata() for cell in self.cells],
        }

    @classmethod
    def from_metadata(
        cls,
        metadata: Mapping[str, object],
    ) -> "QualifiedBoundedTransformPolicy":
        payload = dict(metadata)
        raw_cells = payload.pop("cells")
        if not isinstance(raw_cells, list):
            raise TypeError("policy cells must be a list")
        return cls(
            cells=tuple(
                BoundedTransformQualificationCell.from_metadata(cell)
                for cell in raw_cells
            ),
            **payload,
        )


def load_bounded_transform_policy(
    path: str | Path,
) -> QualifiedBoundedTransformPolicy:
    """Load a policy or a policy-bearing qualification artifact."""

    resolved = Path(path).resolve()
    payload: Any = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("bounded-transform policy file must contain an object")
    policy_payload = payload.get("policy", payload)
    if not isinstance(policy_payload, dict):
        raise TypeError("policy entry must contain an object")
    policy = QualifiedBoundedTransformPolicy.from_metadata(policy_payload)
    for cell in policy.cells:
        if not math.isfinite(cell.retained_count / cell.physical_size):
            raise ValueError("policy contains a non-finite retained fraction")
    return policy
