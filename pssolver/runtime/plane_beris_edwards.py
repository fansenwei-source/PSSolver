"""Explicit dual-path runtime edge for the Plane Beris--Edwards model.

The legacy adapter remains the default and receives an already-authorized
builder from the production driver.  The separated implementation is imported
only after the immutable run specification explicitly selects its canary path.
This module owns selection; equations, geometry objects, and the generic
spectral solver do not reinterpret the choice.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from pssolver.configuration import (
    PlaneBerisEdwardsRunSpec,
    PlaneRuntimePath,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.execution.state import RuntimeState


def _require_runtime_surface(solver: object, projector: object) -> None:
    required_solver = ("fields", "integrator", "run", "refresh_static_fields")
    missing = tuple(name for name in required_solver if not hasattr(solver, name))
    if missing:
        raise TypeError(f"Plane runtime solver is missing {missing!r}")
    if not hasattr(projector, "retained_axis_counts"):
        raise TypeError("Plane runtime projector lacks retained-axis metadata")


@dataclass(frozen=True, slots=True)
class PlaneRuntimeBuildRequest:
    """One fully resolved, single-authority Plane runtime request."""

    run_spec: PlaneBerisEdwardsRunSpec
    production_metadata: Mapping[str, object]
    initial_values: Mapping[str, object]
    device: object

    def __post_init__(self) -> None:
        if not isinstance(self.run_spec, PlaneBerisEdwardsRunSpec):
            raise TypeError("run_spec must be a PlaneBerisEdwardsRunSpec")
        if not isinstance(self.production_metadata, Mapping):
            raise TypeError("production_metadata must be a mapping")
        if not isinstance(self.initial_values, Mapping):
            raise TypeError("initial_values must be a mapping")
        metadata = dict(self.production_metadata)
        configuration = metadata.get("configuration")
        if not isinstance(configuration, Mapping):
            raise ValueError("production metadata lacks configuration identity")
        expected = self.run_spec.identity_metadata()
        actual = dict(configuration)
        identity_keys = (
            "schema_version",
            "authority",
            "runtime_path",
            "canonical_sha256",
        )
        if any(actual.get(key) != expected[key] for key in identity_keys):
            raise ValueError(
                "mixed Plane runtime configuration authorities are forbidden"
            )
        runtime_selection = metadata.get("runtime_selection")
        expected_selection = self.run_spec.runtime_selection_metadata()
        if (
            not isinstance(runtime_selection, Mapping)
            or dict(runtime_selection) != expected_selection
        ):
            raise ValueError(
                "runtime selection metadata must come from the resolved run spec"
            )


@runtime_checkable
class PlaneRuntimeAdapterProtocol(Protocol):
    """Narrow surface needed before the shared Stage O.3 workflow exists."""

    @property
    def runtime_path(self) -> PlaneRuntimePath: ...

    @property
    def solver(self) -> object: ...

    @property
    def projector(self) -> object: ...

    @property
    def fields(self) -> object: ...

    @property
    def completed_steps(self) -> int: ...

    def advance(
        self,
        steps: int,
        *,
        pre_update_callback: Callable[[object, int], None] | None = None,
    ) -> None: ...

    def synchronize_for_observation(self) -> None: ...

    def projected_normal_force(self) -> object: ...

    def flow_diagnostics(self) -> Mapping[str, object]: ...

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


@dataclass(frozen=True, slots=True)
class LegacyPlaneRuntimeAdapter:
    """Adapter around the unchanged Plane production solver assembly."""

    _solver: object
    _projector: object

    def __post_init__(self) -> None:
        _require_runtime_surface(self._solver, self._projector)

    @property
    def runtime_path(self) -> PlaneRuntimePath:
        return PlaneRuntimePath.LEGACY_PRODUCTION

    @property
    def solver(self) -> object:
        return self._solver

    @property
    def projector(self) -> object:
        return self._projector

    @property
    def fields(self) -> object:
        return self._solver.fields

    @property
    def completed_steps(self) -> int:
        integrator = self._solver.integrator
        state = getattr(integrator, "runtime_state", None)
        if isinstance(state, RuntimeState):
            return int(state.progress.completed_steps)
        interval = integrator.spectral_refresh_interval
        if interval is None:
            return int(integrator.step_count)
        return int(integrator.refresh_count * interval + integrator.step_count)

    def advance(
        self,
        steps: int,
        *,
        pre_update_callback: Callable[[object, int], None] | None = None,
    ) -> None:
        self._solver.run(
            steps,
            pre_update_callback=pre_update_callback,
        )

    def synchronize_for_observation(self) -> None:
        self._solver.refresh_static_fields()

    def projected_normal_force(self) -> object:
        value = self._solver.model.static_model.last_projected_normal_force
        if value is None:
            raise RuntimeError("legacy normal-force diagnostics are unavailable")
        return value

    def flow_diagnostics(self) -> Mapping[str, object]:
        model = self._solver.model.static_model
        return {
            "last_pressure_iterations": int(model.last_pressure_iterations),
            "last_pressure_residual": float(model.last_pressure_residual),
            "last_pressure_relative_residual": float(
                model.last_pressure_relative_residual
            ),
        }

    def backend_restart_metadata(self) -> dict[str, object]:
        return {"kind": "legacy_plane_stateless", "state_keys": []}

    def restore_progress(
        self,
        *,
        completed_steps: int,
        spectral_refresh_interval: int | None,
        integrator_step_count: int,
        integrator_refresh_count: int,
    ) -> None:
        _restore_integrator_progress(
            self._solver.integrator,
            completed_steps=completed_steps,
            spectral_refresh_interval=spectral_refresh_interval,
            integrator_step_count=integrator_step_count,
            integrator_refresh_count=integrator_refresh_count,
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "requested": self.runtime_path.value,
            "effective": self.runtime_path.value,
            "adapter": type(self).__name__,
            "fallback_used": False,
            "separated_architecture": None,
        }


@dataclass(frozen=True, slots=True)
class SeparatedCanaryPlaneRuntimeAdapter:
    """Adapter around the opt-in N.4.1 separated Plane runtime."""

    _runtime: object

    def __post_init__(self) -> None:
        solver = getattr(self._runtime, "solver", None)
        projector = getattr(self._runtime, "projector", None)
        _require_runtime_surface(solver, projector)
        if not hasattr(self._runtime, "synchronize_algebraic_for_observation"):
            raise TypeError("separated Plane runtime lacks synchronization")
        if not hasattr(self._runtime, "to_metadata"):
            raise TypeError("separated Plane runtime lacks metadata")

    @property
    def runtime_path(self) -> PlaneRuntimePath:
        return PlaneRuntimePath.SEPARATED_CANARY

    @property
    def solver(self) -> object:
        return self._runtime.solver

    @property
    def projector(self) -> object:
        return self._runtime.projector

    @property
    def fields(self) -> object:
        return self._runtime.solver.fields

    @property
    def completed_steps(self) -> int:
        integrator = self._runtime.solver.integrator
        state = getattr(integrator, "runtime_state", None)
        if isinstance(state, RuntimeState):
            return int(state.progress.completed_steps)
        interval = integrator.spectral_refresh_interval
        if interval is None:
            return int(integrator.step_count)
        return int(integrator.refresh_count * interval + integrator.step_count)

    def advance(
        self,
        steps: int,
        *,
        pre_update_callback: Callable[[object, int], None] | None = None,
    ) -> None:
        self._runtime.solver.run(
            steps,
            pre_update_callback=pre_update_callback,
        )

    def synchronize_for_observation(self) -> None:
        self._runtime.synchronize_algebraic_for_observation()

    def projected_normal_force(self) -> object:
        transient = self._runtime.transient_algebraic_state()
        try:
            return transient["force_z"]
        except KeyError as exc:
            raise RuntimeError(
                "separated normal-force diagnostics are unavailable"
            ) from exc

    def flow_diagnostics(self) -> Mapping[str, object]:
        diagnostics = self._runtime.algebraic_diagnostics()
        try:
            return diagnostics["flow"]
        except KeyError as exc:
            raise RuntimeError(
                "separated flow diagnostics are unavailable"
            ) from exc

    def backend_restart_metadata(self) -> dict[str, object]:
        restart = self._runtime.capture_algebraic_restart_state()
        if any(system.tensors for system in restart.systems):
            raise RuntimeError(
                "Plane separated runtime unexpectedly has persistent "
                "algebraic restart tensors"
            )
        return {
            "kind": "separated_plane_stateless_algebraic",
            "state_keys": [],
            "algebraic_restart": restart.to_metadata(),
        }

    def restore_progress(
        self,
        *,
        completed_steps: int,
        spectral_refresh_interval: int | None,
        integrator_step_count: int,
        integrator_refresh_count: int,
    ) -> None:
        _restore_integrator_progress(
            self._runtime.solver.integrator,
            completed_steps=completed_steps,
            spectral_refresh_interval=spectral_refresh_interval,
            integrator_step_count=integrator_step_count,
            integrator_refresh_count=integrator_refresh_count,
        )

    def to_metadata(self) -> dict[str, object]:
        return {
            "requested": self.runtime_path.value,
            "effective": self.runtime_path.value,
            "adapter": type(self).__name__,
            "fallback_used": False,
            "separated_architecture": self._runtime.to_metadata(),
        }


LegacyRuntimeBuilder = Callable[[], tuple[object, object]]


def _restore_integrator_progress(
    integrator: object,
    *,
    completed_steps: int,
    spectral_refresh_interval: int | None,
    integrator_step_count: int,
    integrator_refresh_count: int,
) -> None:
    integrator.set_spectral_refresh_interval(spectral_refresh_interval)
    integrator.restore_progress(
        completed_steps,
        static_fields_are_current=True,
    )
    if (
        int(integrator.step_count) != integrator_step_count
        or int(integrator.refresh_count) != integrator_refresh_count
    ):
        raise RuntimeError("restored spectral-refresh counters are inconsistent")


def build_plane_beris_edwards_runtime(
    request: PlaneRuntimeBuildRequest,
    *,
    legacy_builder: LegacyRuntimeBuilder | None = None,
) -> PlaneRuntimeAdapterProtocol:
    """Build exactly the runtime selected by the immutable run specification.

    The package-owned legacy builder is the production path.  The optional
    injection point remains available for characterization tests and explicit
    rollback checks; applications no longer need to own numerical assembly.
    """

    if not isinstance(request, PlaneRuntimeBuildRequest):
        raise TypeError("request must be a PlaneRuntimeBuildRequest")
    if legacy_builder is not None and not callable(legacy_builder):
        raise TypeError("legacy_builder must be callable")
    components = decompose_plane_beris_edwards_run_spec(request.run_spec)
    execution = components.execution
    if execution.runtime_path is PlaneRuntimePath.LEGACY_PRODUCTION:
        if legacy_builder is None:
            from .plane_legacy import build_legacy_plane_runtime

            solver, projector = build_legacy_plane_runtime(
                request.run_spec,
                device=request.device,
                initial_values=request.initial_values,
            )
        else:
            solver, projector = legacy_builder()
        return LegacyPlaneRuntimeAdapter(solver, projector)

    if execution.disable_q_gradient_reuse:
        raise ValueError(
            "separated_canary does not accept legacy Q-gradient cache flags"
        )

    # Deliberately lazy: omitted/default selection never imports experimental
    # architecture modules into the production process.
    from pssolver.experimental.plane_shadow_driver import (
        build_plane_separated_canary_runtime_from_production_metadata,
    )

    runtime, comparison = (
        build_plane_separated_canary_runtime_from_production_metadata(
            request.production_metadata,
            device=request.device,
        )
    )
    comparison.require_compatible()
    runtime.reset(request.initial_values)
    return SeparatedCanaryPlaneRuntimeAdapter(runtime)


__all__ = [
    "LegacyPlaneRuntimeAdapter",
    "PlaneRuntimeAdapterProtocol",
    "PlaneRuntimeBuildRequest",
    "SeparatedCanaryPlaneRuntimeAdapter",
    "build_plane_beris_edwards_runtime",
]
