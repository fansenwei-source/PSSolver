"""Opt-in semantic timing for the experimental execution runtime.

The recorder is intentionally absent unless a diagnostic caller requests it.
CUDA regions use deferred events and synchronize only when a snapshot is
requested; normal shadow and production runs therefore acquire no events and
no additional synchronization points.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math
import time
from typing import Iterator

import torch


@dataclass(slots=True)
class _RegionAggregate:
    calls: int = 0
    total_seconds: float = 0.0
    peak_allocated_bytes_observed: int = 0
    peak_reserved_bytes_observed: int = 0


class RuntimePerformanceRecorder:
    """Collect nested semantic regions without changing tensor operations."""

    def __init__(self, device: object) -> None:
        resolved = torch.device(device)
        if resolved.type == "cuda" and resolved.index is None:
            raise ValueError(
                "CUDA performance recorder requires a concrete device index"
            )
        self.device = resolved
        self._regions: dict[str, _RegionAggregate] = {}
        self._pending_cuda_events: list[
            tuple[str, torch.cuda.Event, torch.cuda.Event]
        ] = []

    @staticmethod
    def _validate_name(name: str) -> str:
        if not isinstance(name, str) or not name or name.strip() != name:
            raise ValueError("performance region name must be non-empty")
        return name

    def _memory_snapshot(self) -> tuple[int, int]:
        if self.device.type != "cuda":
            return 0, 0
        return (
            int(torch.cuda.memory_allocated(self.device)),
            int(torch.cuda.memory_reserved(self.device)),
        )

    def _observe_memory(self, aggregate: _RegionAggregate) -> None:
        allocated, reserved = self._memory_snapshot()
        aggregate.peak_allocated_bytes_observed = max(
            aggregate.peak_allocated_bytes_observed,
            allocated,
        )
        aggregate.peak_reserved_bytes_observed = max(
            aggregate.peak_reserved_bytes_observed,
            reserved,
        )

    @contextmanager
    def region(self, name: str) -> Iterator[None]:
        """Record one possibly nested semantic region."""

        name = self._validate_name(name)
        aggregate = self._regions.setdefault(name, _RegionAggregate())
        aggregate.calls += 1
        if self.device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            try:
                yield
            finally:
                stop.record()
                self._pending_cuda_events.append((name, start, stop))
                self._observe_memory(aggregate)
            return

        start_seconds = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start_seconds
            if not math.isfinite(elapsed) or elapsed < 0.0:
                raise RuntimeError("performance clock returned an invalid value")
            aggregate.total_seconds += elapsed

    def _resolve_pending_events(self) -> None:
        if not self._pending_cuda_events:
            return
        torch.cuda.synchronize(self.device)
        pending, self._pending_cuda_events = self._pending_cuda_events, []
        for name, start, stop in pending:
            elapsed = float(start.elapsed_time(stop)) / 1000.0
            if not math.isfinite(elapsed) or elapsed < 0.0:
                raise RuntimeError("CUDA event returned an invalid duration")
            self._regions[name].total_seconds += elapsed

    def reset(self) -> None:
        """Discard accumulated observations after pending CUDA work completes."""

        self._resolve_pending_events()
        self._regions.clear()

    def snapshot(self) -> dict[str, object]:
        """Synchronize pending events and return JSON-compatible aggregates."""

        self._resolve_pending_events()
        timestep = self._regions.get("timestep.total")
        timestep_seconds = (
            timestep.total_seconds if timestep is not None else None
        )
        regions: dict[str, object] = {}
        for name in sorted(self._regions):
            aggregate = self._regions[name]
            if aggregate.calls <= 0:
                raise RuntimeError("recorded performance region has no calls")
            value: dict[str, object] = {
                "calls": aggregate.calls,
                "total_seconds": aggregate.total_seconds,
                "mean_seconds_per_call": (
                    aggregate.total_seconds / aggregate.calls
                ),
                "peak_allocated_bytes_observed": (
                    aggregate.peak_allocated_bytes_observed
                ),
                "peak_reserved_bytes_observed": (
                    aggregate.peak_reserved_bytes_observed
                ),
            }
            value["fraction_of_timestep_total"] = (
                aggregate.total_seconds / timestep_seconds
                if timestep_seconds is not None and timestep_seconds > 0.0
                else None
            )
            regions[name] = value
        return {
            "schema_version": 1,
            "enabled": True,
            "device": str(self.device),
            "timing_backend": (
                "deferred_cuda_events"
                if self.device.type == "cuda"
                else "perf_counter"
            ),
            "nested_regions_overlap": True,
            "regions": regions,
        }


def performance_region(
    recorder: RuntimePerformanceRecorder | None,
    name: str,
):
    """Return a no-op context when instrumentation was not requested."""

    if recorder is None:
        return nullcontext()
    if not isinstance(recorder, RuntimePerformanceRecorder):
        raise TypeError("recorder must be a RuntimePerformanceRecorder or None")
    return recorder.region(name)


def instrument_transform_backend(
    backend: object,
    recorder: RuntimePerformanceRecorder,
) -> None:
    """Wrap one experimental backend to count every forward and inverse."""

    if not isinstance(recorder, RuntimePerformanceRecorder):
        raise TypeError("recorder must be a RuntimePerformanceRecorder")
    if getattr(backend, "_stage_n_transform_instrumented", False):
        raise RuntimeError("transform backend is already instrumented")
    original_forward = getattr(backend, "forward", None)
    original_inverse = getattr(backend, "inverse", None)
    if not callable(original_forward) or not callable(original_inverse):
        raise TypeError("transform backend must provide forward and inverse")

    def timed_forward(tensor, boundary_conditions, **kwargs):
        with recorder.region("transform.forward"):
            return original_forward(tensor, boundary_conditions, **kwargs)

    def timed_inverse(spectral, boundary_conditions, **kwargs):
        with recorder.region("transform.inverse"):
            return original_inverse(spectral, boundary_conditions, **kwargs)

    backend.forward = timed_forward
    backend.inverse = timed_inverse
    backend._stage_n_transform_instrumented = True


__all__ = [
    "RuntimePerformanceRecorder",
    "instrument_transform_backend",
    "performance_region",
]
