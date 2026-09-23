"""Explicit P7.4 dual-path runtime facade for the rectangular Channel.

The legacy solver remains the default.  ``compiled_channel_v2`` is selected
only by the immutable Channel run specification and requires an explicit
builder; construction failure never falls back to the legacy implementation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.channel_active_nematics_declarations import (
    ChannelRuntimePath,
)
from pssolver.execution.state import RuntimeState
from pssolver.models.active_nematics import Q_COMPONENTS


_VELOCITY_COMPONENTS = ("ux", "uy", "uz")


def _storage_identity(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), int(storage.data_ptr()), int(storage.nbytes())


@dataclass(frozen=True, slots=True)
class ChannelOutputViews:
    """Zero-copy Q/u/p views owned by one Channel field allocation."""

    q: torch.Tensor
    velocity: torch.Tensor
    pressure: torch.Tensor
    owner: torch.Tensor

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, torch.Tensor)
            for value in (self.q, self.velocity, self.pressure, self.owner)
        ):
            raise TypeError("Channel output views must be tensors")
        if self.q.shape[0] != 5 or self.velocity.shape[0] != 3:
            raise ValueError("Channel output component counts are invalid")
        if self.pressure.shape != self.owner.shape[1:]:
            raise ValueError("Channel pressure view shape is invalid")
        owner_identity = _storage_identity(self.owner)
        if any(
            _storage_identity(value) != owner_identity
            for value in (self.q, self.velocity, self.pressure)
        ):
            raise ValueError("Channel outputs must be zero-copy field views")

    def to_metadata(self) -> dict[str, object]:
        return {
            "q_components": list(Q_COMPONENTS),
            "velocity_components": list(_VELOCITY_COMPONENTS),
            "pressure_component": "p",
            "q_shape": list(self.q.shape),
            "velocity_shape": list(self.velocity.shape),
            "pressure_shape": list(self.pressure.shape),
            "dtype": str(self.owner.dtype),
            "device": str(self.owner.device),
            "zero_copy": True,
        }


@dataclass(frozen=True, slots=True)
class ChannelRuntimeBuildRequest:
    """Single-authority request for one selected Channel runtime."""

    run_spec: ChannelActiveNematicRunSpec
    production_metadata: Mapping[str, object]
    initial_q: Mapping[str, torch.Tensor]
    device: object

    def __post_init__(self) -> None:
        if not isinstance(self.run_spec, ChannelActiveNematicRunSpec):
            raise TypeError("run_spec must be a ChannelActiveNematicRunSpec")
        if not isinstance(self.production_metadata, Mapping):
            raise TypeError("production_metadata must be a mapping")
        if not isinstance(self.initial_q, Mapping):
            raise TypeError("initial_q must be a mapping")
        if tuple(self.initial_q) != Q_COMPONENTS:
            raise ValueError("initial_q must contain ordered Q components")
        metadata = dict(self.production_metadata)
        if metadata.get("configuration") != self.run_spec.identity_metadata():
            raise ValueError(
                "mixed Channel runtime configuration authorities are forbidden"
            )
        if (
            metadata.get("runtime_selection")
            != self.run_spec.runtime_selection_metadata()
        ):
            raise ValueError(
                "runtime selection metadata must come from the resolved run spec"
            )


@runtime_checkable
class ChannelRuntimeAdapterProtocol(Protocol):
    """Narrow P7.4 surface shared by legacy and compiled Channel paths."""

    @property
    def runtime_path(self) -> ChannelRuntimePath: ...

    @property
    def solver(self) -> object: ...

    @property
    def fields(self) -> object: ...

    @property
    def output_views(self) -> ChannelOutputViews: ...

    @property
    def completed_steps(self) -> int: ...

    def advance(
        self,
        steps: int,
        *,
        pre_update_callback: Callable[[object, int], None] | None = None,
    ) -> None: ...

    def synchronize_for_observation(self) -> None: ...

    def capture_pressure_guess(self) -> torch.Tensor: ...

    def restore_pressure_guess(self, pressure_guess: torch.Tensor) -> None: ...

    def backend_restart_metadata(self) -> dict[str, object]: ...

    def restore_progress(
        self,
        *,
        completed_steps: int,
        spectral_refresh_interval: int | None,
        integrator_step_count: int,
        integrator_refresh_count: int,
    ) -> None: ...

    def to_metadata(self) -> dict[str, object]: ...


def _require_solver_surface(solver: object) -> None:
    missing = tuple(
        name
        for name in ("fields", "integrator", "model", "run", "refresh_static_fields")
        if not hasattr(solver, name)
    )
    if missing:
        raise TypeError(f"Channel runtime solver is missing {missing!r}")


def _output_views(solver: object) -> ChannelOutputViews:
    owner = solver.fields.spatial
    return ChannelOutputViews(
        q=owner[:5],
        velocity=owner[5:8],
        pressure=owner[8],
        owner=owner,
    )


def _completed_steps(integrator: object) -> int:
    state = getattr(integrator, "runtime_state", None)
    if isinstance(state, RuntimeState):
        return int(state.progress.completed_steps)
    interval = integrator.spectral_refresh_interval
    if interval is None:
        return int(integrator.step_count)
    return int(integrator.refresh_count * interval + integrator.step_count)


def _validate_pressure_guess(
    solver: object,
    pressure_guess: torch.Tensor,
) -> torch.Tensor:
    current = solver.model.static_model.pressure_guess
    if not isinstance(pressure_guess, torch.Tensor):
        raise TypeError("pressure_guess must be a tensor")
    if current is None:
        current = solver.fields["p.hat"]
    elif not isinstance(current, torch.Tensor):
        raise RuntimeError("Channel pressure state has an invalid type")
    if (
        pressure_guess.shape != current.shape
        or pressure_guess.dtype != current.dtype
        or pressure_guess.device != current.device
    ):
        raise ValueError("pressure_guess does not match the Channel runtime")
    if not bool(torch.isfinite(pressure_guess).all().item()):
        raise ValueError("pressure_guess must contain only finite values")
    return pressure_guess.detach().clone()


def _restore_progress(
    integrator: object,
    *,
    completed_steps: int,
    spectral_refresh_interval: int | None,
    integrator_step_count: int,
    integrator_refresh_count: int,
) -> None:
    integrator.set_spectral_refresh_interval(spectral_refresh_interval)
    integrator.restore_progress(completed_steps, static_fields_are_current=True)
    if (
        int(integrator.step_count) != integrator_step_count
        or int(integrator.refresh_count) != integrator_refresh_count
    ):
        raise RuntimeError("restored Channel refresh counters are inconsistent")


@dataclass(frozen=True, slots=True)
class LegacyChannelRuntimeAdapter:
    """Adapter around the unchanged legacy Channel solver."""

    _solver: object
    _views: ChannelOutputViews

    def __post_init__(self) -> None:
        _require_solver_surface(self._solver)
        if self._views.owner is not self._solver.fields.spatial:
            raise ValueError("Channel output views have the wrong owner")

    @classmethod
    def from_solver(cls, solver: object) -> "LegacyChannelRuntimeAdapter":
        _require_solver_surface(solver)
        return cls(solver, _output_views(solver))

    @property
    def runtime_path(self) -> ChannelRuntimePath:
        return ChannelRuntimePath.LEGACY_CHANNEL

    @property
    def solver(self) -> object:
        return self._solver

    @property
    def fields(self) -> object:
        return self._solver.fields

    @property
    def output_views(self) -> ChannelOutputViews:
        return self._views

    @property
    def completed_steps(self) -> int:
        return _completed_steps(self._solver.integrator)

    def advance(self, steps: int, *, pre_update_callback=None) -> None:
        self._solver.run(steps, pre_update_callback=pre_update_callback)

    def synchronize_for_observation(self) -> None:
        self._solver.refresh_static_fields()

    def capture_pressure_guess(self) -> torch.Tensor:
        value = self._solver.model.static_model.pressure_guess
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("Channel pressure state is not initialized")
        return value.detach().clone()

    def restore_pressure_guess(self, pressure_guess: torch.Tensor) -> None:
        restored = _validate_pressure_guess(self._solver, pressure_guess)
        self._solver.model.static_model.pressure_guess = restored

    def backend_restart_metadata(self) -> dict[str, object]:
        return {
            "kind": "legacy_channel_pressure_pcg",
            "state_keys": ["pressure_guess"],
        }

    def restore_progress(self, **values) -> None:
        _restore_progress(self._solver.integrator, **values)

    def to_metadata(self) -> dict[str, object]:
        return {
            "requested": self.runtime_path.value,
            "effective": self.runtime_path.value,
            "adapter": type(self).__name__,
            "fallback_used": False,
            "output_ownership": self.output_views.to_metadata(),
        }


@dataclass(frozen=True, slots=True)
class _CompiledChannelRuntimeAdapter:
    """Private adapter around the P7.3 direct compiled runtime."""

    _runtime: object
    _views: ChannelOutputViews

    def __post_init__(self) -> None:
        solver = getattr(self._runtime, "solver", None)
        _require_solver_surface(solver)
        if self._views.owner is not solver.fields.spatial:
            raise ValueError("Channel output views have the wrong owner")
        for name in (
            "capture_pressure_guess",
            "restore_pressure_guess",
        ):
            if not callable(getattr(self._runtime, name, None)):
                raise TypeError(f"compiled Channel runtime lacks {name}")

    @classmethod
    def from_runtime(cls, runtime: object) -> "_CompiledChannelRuntimeAdapter":
        solver = getattr(runtime, "solver", None)
        _require_solver_surface(solver)
        return cls(runtime, _output_views(solver))

    @property
    def runtime_path(self) -> ChannelRuntimePath:
        return ChannelRuntimePath.COMPILED_CHANNEL_V2

    @property
    def solver(self) -> object:
        return self._runtime.solver

    @property
    def fields(self) -> object:
        return self.solver.fields

    @property
    def output_views(self) -> ChannelOutputViews:
        return self._views

    @property
    def completed_steps(self) -> int:
        return int(self._runtime.completed_steps)

    def advance(self, steps: int, *, pre_update_callback=None) -> None:
        self.solver.run(
            steps,
            pre_update_callback=pre_update_callback,
        )

    def synchronize_for_observation(self) -> None:
        if self._runtime.workspace.active:
            raise RuntimeError("cannot observe during an active timestep")
        self.solver.refresh_static_fields()
        self.solver.integrator._synchronize_pressure_state()

    def capture_pressure_guess(self) -> torch.Tensor:
        return self._runtime.capture_pressure_guess()

    def restore_pressure_guess(self, pressure_guess: torch.Tensor) -> None:
        self._runtime.restore_pressure_guess(pressure_guess)

    def backend_restart_metadata(self) -> dict[str, object]:
        return {
            "kind": "compiled_channel_v2_pressure_pcg",
            "state_keys": ["pressure_guess"],
        }

    def restore_progress(self, **values) -> None:
        _restore_progress(self.solver.integrator, **values)

    def to_metadata(self) -> dict[str, object]:
        return {
            "requested": self.runtime_path.value,
            "effective": self.runtime_path.value,
            "adapter": type(self).__name__,
            "fallback_used": False,
            "output_ownership": self.output_views.to_metadata(),
            "compiled_runtime": self._runtime.to_metadata(),
        }


LegacyChannelBuilder = Callable[[], object]
CompiledChannelBuilder = Callable[[], object]


def build_channel_active_nematic_runtime(
    request: ChannelRuntimeBuildRequest,
    *,
    legacy_builder: LegacyChannelBuilder | None = None,
    compiled_builder: CompiledChannelBuilder | None = None,
) -> ChannelRuntimeAdapterProtocol:
    """Construct exactly the selected Channel runtime without fallback."""

    if not isinstance(request, ChannelRuntimeBuildRequest):
        raise TypeError("request must be a ChannelRuntimeBuildRequest")
    path = request.run_spec.runtime_path
    if path is ChannelRuntimePath.LEGACY_CHANNEL:
        if legacy_builder is None:
            raise RuntimeError(
                "legacy_channel requires an application-owned legacy builder; "
                "runtime fallback is forbidden"
            )
        if not callable(legacy_builder):
            raise TypeError("legacy_builder must be callable")
        return LegacyChannelRuntimeAdapter.from_solver(legacy_builder())
    if path is not ChannelRuntimePath.COMPILED_CHANNEL_V2:
        raise ValueError("unsupported Channel runtime path")
    if compiled_builder is None:
        raise RuntimeError(
            "compiled_channel_v2 requires an explicit compiled builder; "
            "runtime fallback is forbidden"
        )
    if not callable(compiled_builder):
        raise TypeError("compiled_builder must be callable")
    runtime = compiled_builder()
    adapter = _CompiledChannelRuntimeAdapter.from_runtime(runtime)
    if adapter.runtime_path is not path:
        raise ValueError("compiled builder returned the wrong runtime path")
    return adapter


__all__ = [
    "ChannelOutputViews",
    "ChannelRuntimeAdapterProtocol",
    "ChannelRuntimeBuildRequest",
    "LegacyChannelRuntimeAdapter",
    "build_channel_active_nematic_runtime",
]
