"""Private control state shared by the four Channel pressure PCG paths.

The numerical paths retain ownership of every tensor operation.  This module
only records fixed-work floor states and makes the common stop, rollback, and
postcondition decisions.  Keeping tensor arithmetic at the call sites makes
the production, transpose, implicit, and qualification-oracle graphs retain
their established operation order.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, TypeVar, cast


CHANNEL_PCG_FIXED_WORK_ROUNDOFF_SATURATION = "fixed_work_roundoff_saturation"
CHANNEL_PCG_ITERATION_LIMIT = "iteration_limit_reached_without_postcondition"
CHANNEL_PCG_FIXED_WORK_COMPLETED = "fixed_work_completed"


_ValueT = TypeVar("_ValueT")


@dataclass(frozen=True, slots=True)
class ChannelPCGStop(Generic[_ValueT]):
    """A path-neutral stop decision with an optional floor rollback."""

    value: _ValueT
    iterations: int
    termination_reason: str
    acceptable: bool
    saturation_failure_reason: str
    rolled_back_to_floor: bool


@dataclass(slots=True)
class ChannelPCGControl(Generic[_ValueT]):
    """Track fixed-work control without performing tensor arithmetic."""

    fixed_work: bool
    floor_value: _ValueT | None = None
    floor_iterations: int | None = None
    floor_relative_residual: float | None = None
    saturation_failure_reason: str = CHANNEL_PCG_ITERATION_LIMIT

    @property
    def has_floor(self) -> bool:
        """Return whether a represented floor candidate has been retained."""

        return self.floor_value is not None

    def should_capture_floor(
        self,
        *,
        represented_at_floor: bool,
        relative_residual: float,
    ) -> bool:
        """Return whether the caller should clone the current path state."""

        if not self.fixed_work or not represented_at_floor:
            return False
        candidate = float(relative_residual)
        return (
            self.floor_relative_residual is None
            or candidate < self.floor_relative_residual
        )

    def capture_floor(
        self,
        value: _ValueT,
        *,
        iterations: int,
        relative_residual: float,
    ) -> None:
        """Retain a caller-created snapshot of the best recurrence floor."""

        self.floor_value = value
        self.floor_iterations = int(iterations)
        self.floor_relative_residual = float(relative_residual)

    def should_rollback_after_floor_exit(
        self,
        *,
        represented_at_floor: bool,
    ) -> bool:
        """Return whether a recurrence rose above an already observed floor."""

        return self.fixed_work and not represented_at_floor and self.has_floor

    def stop_for_failure(
        self,
        value: _ValueT,
        *,
        iterations: int,
        failure_reason: str,
    ) -> ChannelPCGStop[_ValueT]:
        """Resolve a breakdown/nonfinite event, rolling back when permitted."""

        if self.fixed_work and self.has_floor:
            self.saturation_failure_reason = failure_reason
            return ChannelPCGStop(
                value=cast(_ValueT, self.floor_value),
                iterations=cast(int, self.floor_iterations),
                termination_reason=(CHANNEL_PCG_FIXED_WORK_ROUNDOFF_SATURATION),
                acceptable=True,
                saturation_failure_reason=failure_reason,
                rolled_back_to_floor=True,
            )
        return ChannelPCGStop(
            value=value,
            iterations=int(iterations),
            termination_reason=failure_reason,
            acceptable=False,
            saturation_failure_reason=self.saturation_failure_reason,
            rolled_back_to_floor=False,
        )

    def stop_after_floor_exit(
        self,
        value: _ValueT,
        *,
        iterations: int,
    ) -> ChannelPCGStop[_ValueT]:
        """Roll back after the recurrence rises above its dtype floor."""

        return self.stop_for_failure(
            value,
            iterations=iterations,
            failure_reason=CHANNEL_PCG_ITERATION_LIMIT,
        )

    def stop_at_iteration_limit(
        self,
        value: _ValueT,
        *,
        iterations: int,
    ) -> ChannelPCGStop[_ValueT]:
        """Return the established tolerance/fixed-work limit decision."""

        if self.fixed_work:
            return ChannelPCGStop(
                value=value,
                iterations=int(iterations),
                termination_reason=CHANNEL_PCG_FIXED_WORK_COMPLETED,
                acceptable=True,
                saturation_failure_reason=self.saturation_failure_reason,
                rolled_back_to_floor=False,
            )
        return ChannelPCGStop(
            value=value,
            iterations=int(iterations),
            termination_reason=CHANNEL_PCG_ITERATION_LIMIT,
            acceptable=False,
            saturation_failure_reason=self.saturation_failure_reason,
            rolled_back_to_floor=False,
        )

    def requires_fixed_work_postcondition(
        self,
        *,
        acceptable: bool,
        termination_reason: str,
    ) -> bool:
        """Return whether the represented fixed-work state must be checked."""

        return (
            bool(acceptable)
            and self.has_floor
            and termination_reason
            in {
                CHANNEL_PCG_FIXED_WORK_COMPLETED,
                CHANNEL_PCG_FIXED_WORK_ROUNDOFF_SATURATION,
            }
        )

    def should_retry_floor_postcondition(
        self,
        *,
        represented_at_floor: bool,
        completed_value_available: bool,
        completed_equals_floor: bool,
    ) -> bool:
        """Return whether postcondition validation should retry the floor."""

        return (
            not represented_at_floor
            and completed_value_available
            and not completed_equals_floor
        )

    @staticmethod
    def may_keep_completed_with_roundoff_margin(
        *,
        represented_at_floor: bool,
        completed_value_available: bool,
        completed_relative_residual_available: bool,
        completed_within_margin: bool,
    ) -> bool:
        """Return whether normal completion may use its two-floor margin."""

        return (
            not represented_at_floor
            and completed_value_available
            and completed_relative_residual_available
            and completed_within_margin
        )

    def finish_fixed_work_postcondition(
        self,
        *,
        acceptable: bool,
        represented_at_floor: bool,
        termination_reason: str,
        saturation_reason_still_at_floor: bool,
    ) -> tuple[bool, str]:
        """Return the final acceptable flag and termination reason."""

        if not represented_at_floor:
            return False, self.saturation_failure_reason
        if (
            termination_reason == CHANNEL_PCG_FIXED_WORK_ROUNDOFF_SATURATION
            and not saturation_reason_still_at_floor
        ):
            return False, self.saturation_failure_reason
        # A failed floor retry may already have rejected this completed state.
        # The roundoff-margin check must not reverse that earlier decision.
        return acceptable, termination_reason


__all__ = [
    "CHANNEL_PCG_FIXED_WORK_COMPLETED",
    "CHANNEL_PCG_FIXED_WORK_ROUNDOFF_SATURATION",
    "CHANNEL_PCG_ITERATION_LIMIT",
    "ChannelPCGControl",
    "ChannelPCGStop",
]
