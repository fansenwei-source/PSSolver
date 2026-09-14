"""Immutable policy for algebraic representation execution.

The policy replaces the historical cascade of Stage N.1--N.3 feature flags
with one validated architectural choice.  Legacy flags remain accepted at the
experimental runtime boundary, where they are translated into this object.
They do not propagate into lifecycle, scheduling, or model code.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class AlgebraicExecutionMode(str, Enum):
    """Supported generation-local representation strategies."""

    EAGER_COMPONENTWISE = "eager_componentwise"
    GENERATION_REUSE = "generation_reuse"
    LAZY_COMPONENTWISE = "lazy_componentwise"
    BATCHED_PHYSICAL_ISLANDS = "batched_physical_islands"


@dataclass(frozen=True, slots=True)
class AlgebraicExecutionPolicy:
    """One coherent representation, lifetime, and scheduling policy."""

    mode: AlgebraicExecutionMode = AlgebraicExecutionMode.EAGER_COMPONENTWISE

    def __post_init__(self) -> None:
        if not isinstance(self.mode, AlgebraicExecutionMode):
            raise TypeError("mode must be an AlgebraicExecutionMode")

    @property
    def representation_reuse(self) -> bool:
        return self.mode is not AlgebraicExecutionMode.EAGER_COMPONENTWISE

    @property
    def lazy_physical_materialization(self) -> bool:
        return self.mode in (
            AlgebraicExecutionMode.LAZY_COMPONENTWISE,
            AlgebraicExecutionMode.BATCHED_PHYSICAL_ISLANDS,
        )

    @property
    def batched_physical_islands(self) -> bool:
        return self.mode is AlgebraicExecutionMode.BATCHED_PHYSICAL_ISLANDS

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "mode": self.mode.value,
            "representation_reuse": self.representation_reuse,
            "lazy_physical_materialization": (
                self.lazy_physical_materialization
            ),
            "batched_physical_islands": self.batched_physical_islands,
            "lifetime": "one_pre_explicit_rhs_generation",
            "cross_generation_reuse": False,
            "checkpointed": False,
        }

    @classmethod
    def eager(cls) -> "AlgebraicExecutionPolicy":
        return cls(AlgebraicExecutionMode.EAGER_COMPONENTWISE)

    @classmethod
    def reuse(cls) -> "AlgebraicExecutionPolicy":
        return cls(AlgebraicExecutionMode.GENERATION_REUSE)

    @classmethod
    def lazy(cls) -> "AlgebraicExecutionPolicy":
        return cls(AlgebraicExecutionMode.LAZY_COMPONENTWISE)

    @classmethod
    def batched(cls) -> "AlgebraicExecutionPolicy":
        return cls(AlgebraicExecutionMode.BATCHED_PHYSICAL_ISLANDS)

    @classmethod
    def from_legacy_flags(
        cls,
        *,
        representation_reuse: bool = False,
        lazy_physical_materialization: bool = False,
        batched_physical_islands: bool = False,
    ) -> "AlgebraicExecutionPolicy":
        """Translate the monotone N.1--N.3 flag chain at one boundary."""

        values = {
            "representation_reuse": representation_reuse,
            "lazy_physical_materialization": lazy_physical_materialization,
            "batched_physical_islands": batched_physical_islands,
        }
        for name, value in values.items():
            if not isinstance(value, bool):
                raise TypeError(f"{name} must be a bool")
        if lazy_physical_materialization and not representation_reuse:
            raise ValueError(
                "lazy algebraic materialization requires representation reuse"
            )
        if batched_physical_islands and not (
            representation_reuse and lazy_physical_materialization
        ):
            raise ValueError(
                "batched physical islands require representation reuse and "
                "lazy algebraic materialization"
            )
        if batched_physical_islands:
            return cls.batched()
        if lazy_physical_materialization:
            return cls.lazy()
        if representation_reuse:
            return cls.reuse()
        return cls.eager()


def resolve_algebraic_execution_policy(
    policy: AlgebraicExecutionPolicy | None,
    *,
    enable_algebraic_representation_reuse: bool | None = None,
    enable_lazy_algebraic_materialization: bool | None = None,
    enable_batched_physical_islands: bool | None = None,
) -> AlgebraicExecutionPolicy:
    """Resolve a policy or translate legacy flags, never both.

    ``None`` distinguishes an omitted compatibility flag from an explicitly
    supplied value.  This prevents a caller from accidentally mixing two
    configuration authorities.
    """

    flags = (
        enable_algebraic_representation_reuse,
        enable_lazy_algebraic_materialization,
        enable_batched_physical_islands,
    )
    if policy is not None:
        if not isinstance(policy, AlgebraicExecutionPolicy):
            raise TypeError("policy must be an AlgebraicExecutionPolicy or None")
        if any(value is not None for value in flags):
            raise ValueError(
                "algebraic_execution_policy cannot be combined with legacy "
                "Stage N feature flags"
            )
        return policy
    normalized = tuple(False if value is None else value for value in flags)
    return AlgebraicExecutionPolicy.from_legacy_flags(
        representation_reuse=normalized[0],
        lazy_physical_materialization=normalized[1],
        batched_physical_islands=normalized[2],
    )


__all__ = [
    "AlgebraicExecutionMode",
    "AlgebraicExecutionPolicy",
    "resolve_algebraic_execution_policy",
]
