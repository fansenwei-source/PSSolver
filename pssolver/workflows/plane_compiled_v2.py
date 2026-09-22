"""Private observation and checkpoint surface for compiled Plane.

P5.4 introduced this adapter while the compiled runtime was disconnected.
P5.5 keeps it private and composes it behind the explicit ``compiled_v2``
runtime adapter.  The existing Plane observation and checkpoint-v1 schemas
remain unchanged.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math

import numpy as np
import torch

from pssolver.configuration import PlaneBerisEdwardsRunSpec, PlaneRuntimePath
from pssolver.models.active_nematics import Q_COMPONENTS
from pssolver.runtime.plane_compiled_v2_binding import (
    PlaneCompiledV2BindingPlan,
    bind_plane_compiled_v2,
)
from pssolver.runtime.plane_compiled_v2_step import (
    PlaneCompiledEulerStepProgram,
    build_plane_compiled_v2_step_program,
)
from pssolver.runtime.plane_legacy import build_legacy_plane_runtime
from pssolver.workflows.plane_checkpoint import (
    PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
    PlaneWorkflowCheckpoint,
)
from pssolver.workflows.plane_observation import (
    PlaneDiagnostic,
    PlaneObservation,
)


_VELOCITY_COMPONENTS = ("ux", "uy", "uz")
_CHECKPOINT_PATH_CARRIER = PlaneRuntimePath.COMPILED_V2


def _storage_identity(tensor: torch.Tensor) -> tuple[str, int, int]:
    storage = tensor.untyped_storage()
    return str(tensor.device), int(storage.data_ptr()), int(storage.nbytes())


@dataclass(frozen=True, slots=True)
class PlaneCompiledOutputViews:
    """Zero-copy physical views in canonical Q/u/p component order."""

    q: torch.Tensor
    velocity: torch.Tensor
    pressure: torch.Tensor
    owner: torch.Tensor

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, torch.Tensor)
            for value in (self.q, self.velocity, self.pressure, self.owner)
        ):
            raise TypeError("compiled output views must be tensors")
        if self.q.shape[0] != 5 or self.velocity.shape[0] != 3:
            raise ValueError("compiled output component counts are invalid")
        if self.pressure.shape != self.owner.shape[1:]:
            raise ValueError("compiled pressure view shape is invalid")
        if (
            self.q.shape[1:] != self.owner.shape[1:]
            or self.velocity.shape[1:] != self.owner.shape[1:]
        ):
            raise ValueError("compiled output grids are incompatible")
        owner_identity = _storage_identity(self.owner)
        if any(
            _storage_identity(value) != owner_identity
            for value in (self.q, self.velocity, self.pressure)
        ):
            raise ValueError("compiled outputs must be zero-copy field views")

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
class PlaneCompiledCheckpointRestorePlan:
    """Fully validated, mutation-free checkpoint restore plan."""

    checkpoint: PlaneWorkflowCheckpoint
    copies: tuple[tuple[torch.Tensor, torch.Tensor], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.checkpoint, PlaneWorkflowCheckpoint):
            raise TypeError("checkpoint must be a PlaneWorkflowCheckpoint")
        if len(self.copies) != 2 * len(Q_COMPONENTS):
            raise ValueError("compiled restore plan is incomplete")


class PlaneCompiledV2WorkflowAdapter:
    """Private observation/checkpoint adapter for one compiled program."""

    __slots__ = (
        "_binding",
        "_integrator",
        "_program",
        "_views",
    )

    connected_runtime = None

    def __init__(
        self,
        binding: PlaneCompiledV2BindingPlan,
        program: PlaneCompiledEulerStepProgram,
    ) -> None:
        if not isinstance(binding, PlaneCompiledV2BindingPlan):
            raise TypeError("binding must be a PlaneCompiledV2BindingPlan")
        if not isinstance(program, PlaneCompiledEulerStepProgram):
            raise TypeError("program must be a PlaneCompiledEulerStepProgram")
        if program.state is not binding.state:
            raise ValueError("compiled workflow state identity is incompatible")
        if (
            binding.connected_runtime is not None
            or program.connected_runtime is not None
        ):
            raise ValueError("compiled workflow must remain disconnected")
        integrator = getattr(binding.operators.prepare_algebraic, "__self__", None)
        if integrator is None or getattr(integrator, "runtime_state", None) is not (
            binding.state
        ):
            raise ValueError("compiled workflow integrator identity is incompatible")
        owner = binding.fields.spatial
        views = PlaneCompiledOutputViews(
            q=owner[:5],
            velocity=owner[5:8],
            pressure=owner[8],
            owner=owner,
        )
        self._binding = binding
        self._program = program
        self._integrator = integrator
        self._views = views

    @property
    def completed_steps(self) -> int:
        return int(self._program.state.progress.completed_steps)

    @property
    def output_views(self) -> PlaneCompiledOutputViews:
        self._program.state.representations.require_physical_current()
        return self._views

    def step(
        self,
        *,
        pre_update_callback: Callable[[], None] | None = None,
    ) -> None:
        """Advance one pre-bound step without runtime fallback."""

        if pre_update_callback is not None and not callable(
            pre_update_callback
        ):
            raise TypeError("pre_update_callback must be callable or None")
        self._program.step(pre_update_callback=pre_update_callback)

    def synchronize_for_observation(self) -> None:
        """Materialize algebraic fields for the current physical Q state."""

        if self._program.workspace.active:
            raise RuntimeError("cannot observe during an active timestep")
        self._program.state.representations.require_physical_current()
        self._integrator.model.update_static_fields()

    def capture_observation(
        self,
        *,
        step: int | None = None,
        synchronize: bool = False,
    ) -> PlaneObservation:
        if synchronize:
            self.synchronize_for_observation()
        views = self.output_views
        actual_step = self.completed_steps if step is None else step
        q = (
            views.q[:, 0]
            .movedim(0, -1)
            .detach()
            .to(device="cpu")
            .contiguous()
            .numpy()
        )
        velocity = (
            views.velocity[:, 0]
            .movedim(0, -1)
            .detach()
            .to(device="cpu")
            .contiguous()
            .numpy()
        )
        pressure = (
            views.pressure[0]
            .detach()
            .to(device="cpu")
            .contiguous()
            .numpy()
        )
        return PlaneObservation(
            step=actual_step,
            q=q,
            velocity=velocity,
            pressure=pressure,
        )

    def projected_normal_force(self) -> torch.Tensor:
        stokes = self._binding.operators.stokes_kernel
        if not stokes.cache_force_diagnostics:
            raise RuntimeError("compiled normal-force diagnostics are disabled")
        value = stokes.last_projected_normal_force
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("compiled normal-force diagnostics are unavailable")
        if not bool(torch.isfinite(value).all().item()):
            raise RuntimeError("compiled normal-force diagnostics are non-finite")
        return value

    def flow_diagnostics(self) -> Mapping[str, object]:
        model = self._binding.operators.stokes_kernel
        if not model.pressure_diagnostics:
            raise RuntimeError("compiled pressure diagnostics are disabled")
        values = {
            "last_pressure_iterations": int(model.last_pressure_iterations),
            "last_pressure_residual": float(model.last_pressure_residual),
            "last_pressure_relative_residual": float(
                model.last_pressure_relative_residual
            ),
        }
        if not all(
            math.isfinite(float(value)) for value in values.values()
        ):
            raise RuntimeError("compiled pressure diagnostics are non-finite")
        return values

    def capture_diagnostic(
        self,
        *,
        step: int,
        viscosity: float,
        friction: float,
    ) -> PlaneDiagnostic:
        fields = self._binding.fields
        self._program.state.representations.require_physical_current()
        div_u = (
            fields.gradient("ux", axis=0)
            + fields.gradient("uy", axis=1)
            + fields.gradient("uz", axis=2)
        )
        div_abs = div_u.abs()
        div_max = div_abs.max().item()
        div_rms = torch.sqrt(torch.mean(div_abs.square())).item()
        grad_u_sq = sum(
            fields.gradient(name, axis=axis).abs().square()
            for name in _VELOCITY_COMPONENTS
            for axis in range(3)
        )
        grad_u_rms = torch.sqrt(torch.mean(grad_u_sq)).item()
        div_rel = div_rms / max(grad_u_rms, 1.0e-30)
        normal_force = self.projected_normal_force()
        residual = fields.gradient("p", axis=2) - (
            normal_force
            + viscosity * fields.laplacian("uz")
            - friction * fields["uz"]
        )
        wall = torch.stack(
            (residual[..., 0], residual[..., -1]),
            dim=-1,
        ).abs()
        flow = self.flow_diagnostics()
        return PlaneDiagnostic(
            step=step,
            div_max=div_max,
            div_rms=div_rms,
            div_rel=div_rel,
            schur_iterations=float(flow["last_pressure_iterations"]),
            schur_abs_residual=float(flow["last_pressure_residual"]),
            schur_rel_residual=float(
                flow["last_pressure_relative_residual"]
            ),
            wall_normal_momentum_max=wall.max().item(),
            wall_normal_momentum_rms=torch.sqrt(
                torch.mean(wall.square())
            ).item(),
        )

    def backend_restart_metadata(self) -> dict[str, object]:
        return {
            "kind": "compiled_v2_plane_stateless",
            "state_keys": [],
            "compiled_runtime_identity": "compiled_v2",
            "runtime_identity_sha256": self._binding.runtime_identity_sha256,
            "format_v1_runtime_path_carrier": (
                _CHECKPOINT_PATH_CARRIER.value
            ),
        }

    def capture_checkpoint(self) -> PlaneWorkflowCheckpoint:
        self.synchronize_for_observation()
        fields = self._binding.fields
        progress = self._program.state.progress
        return PlaneWorkflowCheckpoint(
            format_version=PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
            runtime_path=_CHECKPOINT_PATH_CARRIER,
            runtime_identity_sha256=self._binding.runtime_identity_sha256,
            completed_steps=progress.completed_steps,
            spectral_refresh_interval=progress.refresh_interval,
            integrator_step_count=progress.refresh_step_count,
            integrator_refresh_count=progress.refresh_count,
            evolved_spatial={name: fields[name] for name in Q_COMPONENTS},
            evolved_spectral={
                name: fields[f"{name}.hat"] for name in Q_COMPONENTS
            },
            backend_restart=self.backend_restart_metadata(),
        )

    def validate_checkpoint(
        self,
        checkpoint: PlaneWorkflowCheckpoint,
    ) -> PlaneCompiledCheckpointRestorePlan:
        """Reject every identity/layout mismatch before mutating state."""

        if not isinstance(checkpoint, PlaneWorkflowCheckpoint):
            raise TypeError("checkpoint must be a PlaneWorkflowCheckpoint")
        if checkpoint.format_version != PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION:
            raise ValueError("unsupported Plane workflow checkpoint version")
        if checkpoint.runtime_path is not _CHECKPOINT_PATH_CARRIER:
            raise ValueError("cross-runtime Plane checkpoint restart is unsupported")
        if checkpoint.runtime_identity_sha256 != (
            self._binding.runtime_identity_sha256
        ):
            raise ValueError("checkpoint runtime identity does not match target")
        if dict(checkpoint.backend_restart) != self.backend_restart_metadata():
            raise ValueError(
                "checkpoint backend restart contract does not match target"
            )
        if self._program.workspace.active:
            raise RuntimeError("cannot restore during an active timestep")

        copies: list[tuple[torch.Tensor, torch.Tensor]] = []
        fields = self._binding.fields
        for name in Q_COMPONENTS:
            for suffix, values in (
                ("", checkpoint.evolved_spatial),
                (".hat", checkpoint.evolved_spectral),
            ):
                target = fields[f"{name}{suffix}"]
                source = values[name]
                if target.shape != source.shape:
                    raise ValueError(f"checkpoint shape for {name}{suffix} differs")
                if target.dtype != source.dtype:
                    raise ValueError(f"checkpoint dtype for {name}{suffix} differs")
                if not bool(torch.isfinite(source).all().item()):
                    raise ValueError(
                        f"checkpoint tensor for {name}{suffix} is non-finite"
                    )
                copies.append((target, source))
        return PlaneCompiledCheckpointRestorePlan(
            checkpoint=checkpoint,
            copies=tuple(copies),
        )

    def restore_checkpoint(
        self,
        checkpoint: PlaneWorkflowCheckpoint,
    ) -> int:
        plan = self.validate_checkpoint(checkpoint)
        for target, source in plan.copies:
            target.copy_(source.to(device=target.device))
        self.synchronize_for_observation()
        progress = plan.checkpoint
        self._integrator.set_spectral_refresh_interval(
            progress.spectral_refresh_interval
        )
        self._integrator.restore_progress(
            progress.completed_steps,
            static_fields_are_current=True,
        )
        state_progress = self._program.state.progress
        if (
            state_progress.refresh_step_count
            != progress.integrator_step_count
            or state_progress.refresh_count
            != progress.integrator_refresh_count
        ):
            raise RuntimeError("restored spectral-refresh counters are inconsistent")
        return state_progress.completed_steps

    def to_metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "identity": "compiled_v2_observation_checkpoint_adapter",
            "compiled_runtime_identity": "compiled_v2",
            "runtime_identity_sha256": self._binding.runtime_identity_sha256,
            "output_views": self._views.to_metadata(),
            "checkpoint": {
                "format_version": PLANE_WORKFLOW_CHECKPOINT_FORMAT_VERSION,
                "runtime_path_carrier": _CHECKPOINT_PATH_CARRIER.value,
                "backend_identity_enforced": True,
                "preflight_before_mutation": True,
                "cross_runtime_restore": False,
            },
            "diagnostics": {
                "projected_normal_force": bool(
                    self._binding.operators.stokes_kernel.cache_force_diagnostics
                ),
                "pressure": bool(
                    self._binding.operators.stokes_kernel.pressure_diagnostics
                ),
            },
            "runtime_selector_added": True,
            "application_import_added": True,
            "implicit_fallback": False,
            "connected_runtime": self.connected_runtime,
        }


@dataclass(frozen=True, slots=True)
class _CompiledV2PlaneRuntimeAdapter:
    """Private application-facing composition of the P5.2--P5.4 pieces."""

    _solver: object
    _projector: object
    _workflow: PlaneCompiledV2WorkflowAdapter

    @property
    def runtime_path(self) -> PlaneRuntimePath:
        return PlaneRuntimePath.COMPILED_V2

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
        return self._workflow.completed_steps

    def advance(
        self,
        steps: int,
        *,
        pre_update_callback: Callable[[object, int], None] | None = None,
    ) -> None:
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
            raise ValueError("compiled-v2 steps must be a non-negative integer")
        if pre_update_callback is not None and not callable(
            pre_update_callback
        ):
            raise TypeError("pre_update_callback must be callable or None")
        for local_step in range(steps):
            callback = (
                None
                if pre_update_callback is None
                else lambda local_step=local_step: pre_update_callback(
                    self._solver,
                    local_step,
                )
            )
            self._workflow.step(pre_update_callback=callback)

    def synchronize_for_observation(self) -> None:
        self._workflow.synchronize_for_observation()

    def projected_normal_force(self) -> torch.Tensor:
        return self._workflow.projected_normal_force()

    def flow_diagnostics(self) -> Mapping[str, object]:
        return self._workflow.flow_diagnostics()

    def backend_restart_metadata(self) -> dict[str, object]:
        return self._workflow.backend_restart_metadata()

    def restore_progress(
        self,
        *,
        completed_steps: int,
        spectral_refresh_interval: int | None,
        integrator_step_count: int,
        integrator_refresh_count: int,
    ) -> None:
        integrator = self._solver.integrator
        integrator.set_spectral_refresh_interval(spectral_refresh_interval)
        integrator.restore_progress(
            completed_steps,
            static_fields_are_current=True,
        )
        if (
            int(integrator.step_count) != integrator_step_count
            or int(integrator.refresh_count) != integrator_refresh_count
        ):
            raise RuntimeError(
                "restored spectral-refresh counters are inconsistent"
            )

    def to_metadata(self) -> dict[str, object]:
        return {
            "requested": self.runtime_path.value,
            "effective": self.runtime_path.value,
            "adapter": type(self).__name__,
            "fallback_used": False,
            "separated_architecture": None,
            "compiled_architecture": self._workflow.to_metadata(),
        }


def build_plane_compiled_v2_runtime(
    run_spec: PlaneBerisEdwardsRunSpec,
    *,
    device: object,
    initial_values: Mapping[str, object],
) -> _CompiledV2PlaneRuntimeAdapter:
    """Build the private compiled runtime selected by the application."""

    if not isinstance(run_spec, PlaneBerisEdwardsRunSpec):
        raise TypeError("run_spec must be a PlaneBerisEdwardsRunSpec")
    if run_spec.runtime_path is not PlaneRuntimePath.COMPILED_V2:
        raise ValueError("compiled runtime builder requires compiled_v2")
    solver, projector = build_legacy_plane_runtime(
        run_spec,
        device=device,
        initial_values=initial_values,
    )
    binding = bind_plane_compiled_v2(
        run_spec,
        solver=solver,
        projector=projector,
    )
    program = build_plane_compiled_v2_step_program(binding)
    workflow = PlaneCompiledV2WorkflowAdapter(binding, program)
    return _CompiledV2PlaneRuntimeAdapter(solver, projector, workflow)


__all__ = [
    "PlaneCompiledCheckpointRestorePlan",
    "PlaneCompiledOutputViews",
    "PlaneCompiledV2WorkflowAdapter",
    "build_plane_compiled_v2_runtime",
]
