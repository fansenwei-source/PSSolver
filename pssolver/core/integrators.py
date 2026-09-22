"""Tensor-free time-integrator declarations.

Phase 4 introduces these provisional declarations without connecting them to
the qualified Plane runtime.  They use only the Python standard library and
describe mathematical time-discretization identity rather than tensor
storage, backend policy, or execution machinery.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math


class IntegratorScheme(str, Enum):
    """Time-integration schemes admitted by the Phase 4 declaration."""

    PROJECTED_SEMI_IMPLICIT_EULER = "projected_semi_implicit_euler"
    SBDF2 = "sbdf2"


@dataclass(frozen=True, slots=True)
class IntegratorSpec:
    """Immutable, tensor-free integrator identity and startup policy."""

    scheme: IntegratorScheme
    dt: float
    startup_scheme: IntegratorScheme | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.scheme, IntegratorScheme):
            raise TypeError("scheme must be an IntegratorScheme")
        if (
            not isinstance(self.dt, (int, float))
            or isinstance(self.dt, bool)
            or not math.isfinite(float(self.dt))
            or float(self.dt) <= 0.0
        ):
            raise ValueError("dt must be positive and finite")
        object.__setattr__(self, "dt", float(self.dt))
        if self.startup_scheme is not None and not isinstance(
            self.startup_scheme,
            IntegratorScheme,
        ):
            raise TypeError(
                "startup_scheme must be an IntegratorScheme or None"
            )
        if self.scheme is IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER:
            if self.startup_scheme is not None:
                raise ValueError(
                    "projected semi-implicit Euler has no startup scheme"
                )
        elif self.startup_scheme is not (
            IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
        ):
            raise ValueError(
                "SBDF2 requires projected semi-implicit Euler startup"
            )

    @classmethod
    def projected_semi_implicit_euler(cls, *, dt: float) -> "IntegratorSpec":
        """Construct the qualified one-step reference declaration."""

        return cls(
            scheme=IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER,
            dt=dt,
        )

    @classmethod
    def sbdf2(cls, *, dt: float) -> "IntegratorSpec":
        """Construct constant-step SBDF2 with its frozen startup scheme."""

        return cls(
            scheme=IntegratorScheme.SBDF2,
            dt=dt,
            startup_scheme=(
                IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
            ),
        )

    @property
    def formal_order(self) -> int:
        """Formal temporal order after any required startup."""

        if self.scheme is IntegratorScheme.SBDF2:
            return 2
        return 1

    @property
    def history_depth(self) -> int:
        """Number of completed earlier levels required by the scheme."""

        if self.scheme is IntegratorScheme.SBDF2:
            return 1
        return 0

    @property
    def constant_step(self) -> bool:
        """Whether the current declaration admits only one fixed dt."""

        return True

    def to_metadata(self) -> dict[str, object]:
        """Return the complete tensor-free numerical identity."""

        return {
            "schema_version": 1,
            "scheme": self.scheme.value,
            "dt": self.dt,
            "formal_order": self.formal_order,
            "history_depth": self.history_depth,
            "constant_step": self.constant_step,
            "startup_scheme": (
                None
                if self.startup_scheme is None
                else self.startup_scheme.value
            ),
        }


__all__ = ["IntegratorScheme", "IntegratorSpec"]
