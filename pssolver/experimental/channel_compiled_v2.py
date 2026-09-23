"""Direct-import P7.3 execution adapter for the compiled Channel canary.

This module connects the P7.1 run declaration to explicit ``RuntimeState``, a
bounded ``RuntimeWorkspace`` and a pre-bound ``StepProgram``.  It is not
exported from a package root and is not selectable by ``Channel.py``.  The
legacy Channel runtime remains the production default and rollback oracle.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch

from pssolver.channel import Q_COMPONENTS, build_active_nematic_channel
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.execution.state import RuntimeState
from pssolver.execution.workspace import RuntimeWorkspace
from pssolver.integrator import SemiImplicitEulerIntegrator
from pssolver.integrators.state_backed import (
    StateBackedProjectedIntegratorMixin,
)
from pssolver.integrators.step_program import (
    ProjectedSemiImplicitEulerStepProgram,
)
from pssolver.planning.channel_compiled_v2 import (
    CHANNEL_COMPILED_V2_IDENTITY,
    ChannelCompiledV2Declaration,
    channel_compiled_v2_declaration,
)


class _ChannelCompiledV2EulerIntegrator(
    StateBackedProjectedIntegratorMixin,
    SemiImplicitEulerIntegrator,
):
    """Channel oracle update order with explicit state ownership."""

    _phase3_connection_stage = "P7.3_channel_compiled_v2_direct"

    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(model, dt, qx, qy, q2)
        self._synchronize_pressure_state()

    @property
    def _pressure_solver(self):
        solver = self.model.static_model
        if not hasattr(solver, "pressure_guess"):
            raise TypeError(
                "compiled Channel static model lacks pressure warm-start state"
            )
        return solver

    def _synchronize_pressure_state(self) -> None:
        pressure_guess = self._pressure_solver.pressure_guess
        if pressure_guess is None:
            pressure_guess = torch.zeros_like(self.model.fields["p.hat"])
            self._pressure_solver.pressure_guess = pressure_guess
        if not isinstance(pressure_guess, torch.Tensor):
            raise TypeError("Channel pressure guess must be a tensor or None")
        self.runtime_state.set_persistent_algebraic(
            "pressure_guess",
            pressure_guess,
        )

    def _prepare_algebraic(self, state, workspace, generation):
        super()._prepare_algebraic(state, workspace, generation)
        self._synchronize_pressure_state()

    def _project_dynamic_spectra(self, state, workspace, generation):
        # The frozen Channel oracle uses dealias=none, so this named stage is
        # intentionally a pre-bound no-op rather than a runtime branch.
        del state, workspace, generation

    def _inverse_dynamic_spectra(self, state, workspace, generation):
        del state, workspace, generation
        for group in self.dynamic_transform_groups:
            self.model.fields.store_spatial_group(
                group,
                self.model.fields.inverse_transform_group(group),
            )

    def _refresh_dynamic_spectra(self):
        for group in self.dynamic_transform_groups:
            self.model.fields.store_spectral_group(
                group,
                self.model.fields.forward_transform_group(group),
            )

    def rebind_runtime_state(self) -> None:
        super().rebind_runtime_state()
        self._synchronize_pressure_state()

    def restore_pressure_guess(self, pressure_guess: torch.Tensor) -> None:
        current = self._pressure_solver.pressure_guess
        if not isinstance(pressure_guess, torch.Tensor):
            raise TypeError("pressure_guess must be a tensor")
        if (
            not isinstance(current, torch.Tensor)
            or pressure_guess.shape != current.shape
            or pressure_guess.dtype != current.dtype
            or pressure_guess.device != current.device
        ):
            raise ValueError(
                "pressure_guess does not match the bound Channel runtime"
            )
        if not bool(torch.isfinite(pressure_guess).all().item()):
            raise ValueError("pressure_guess must contain only finite values")
        restored = pressure_guess.detach().clone()
        self._pressure_solver.pressure_guess = restored
        self.runtime_state.set_persistent_algebraic(
            "pressure_guess",
            restored,
        )


@dataclass(frozen=True, slots=True)
class ChannelCompiledV2Runtime:
    """Explicit P7.3 owner of one direct-import compiled Channel runtime."""

    solver: object
    run_spec: ChannelActiveNematicRunSpec
    declaration: ChannelCompiledV2Declaration

    def __post_init__(self) -> None:
        if not isinstance(self.run_spec, ChannelActiveNematicRunSpec):
            raise TypeError("run_spec must be a ChannelActiveNematicRunSpec")
        if not isinstance(self.declaration, ChannelCompiledV2Declaration):
            raise TypeError("declaration must be a ChannelCompiledV2Declaration")
        if not isinstance(
            getattr(self.solver, "integrator", None),
            _ChannelCompiledV2EulerIntegrator,
        ):
            raise TypeError("solver lacks the P7.3 Channel integrator")

    @property
    def state(self) -> RuntimeState:
        return self.solver.integrator.runtime_state

    @property
    def workspace(self) -> RuntimeWorkspace:
        return self.solver.integrator.runtime_workspace

    @property
    def step_program(self) -> ProjectedSemiImplicitEulerStepProgram:
        return self.solver.integrator._step_program

    @property
    def completed_steps(self) -> int:
        return self.state.progress.completed_steps

    def advance(self, steps: int) -> None:
        if (
            not isinstance(steps, int)
            or isinstance(steps, bool)
            or steps < 0
        ):
            raise ValueError("steps must be a non-negative integer")
        self.solver.run(steps)

    def capture_pressure_guess(self) -> torch.Tensor:
        return self.state.persistent_algebraic["pressure_guess"].detach().clone()

    def restore_pressure_guess(self, pressure_guess: torch.Tensor) -> None:
        self.solver.integrator.restore_pressure_guess(pressure_guess)

    def to_metadata(self) -> dict[str, object]:
        pressure_solver = self.solver.model.static_model
        pressure_guess = self.state.persistent_algebraic["pressure_guess"]
        return {
            "schema_version": 1,
            "identity": CHANNEL_COMPILED_V2_IDENTITY,
            "runtime_selection": {
                "requested": CHANNEL_COMPILED_V2_IDENTITY,
                "effective": CHANNEL_COMPILED_V2_IDENTITY,
                "fallback_used": False,
                "production_connection": False,
            },
            "declaration": self.declaration.to_metadata(),
            "state": self.state.to_metadata(),
            "workspace": self.workspace.to_metadata(),
            "step_program": self.step_program.to_metadata(),
            "pressure_state": {
                "name": "pressure_guess",
                "shape": list(pressure_guess.shape),
                "dtype": str(pressure_guess.dtype),
                "device": str(pressure_guess.device),
                "iterations": int(pressure_solver.last_pressure_iterations),
                "residual": float(pressure_solver.last_pressure_residual),
                "relative_residual": float(
                    pressure_solver.last_pressure_relative_residual
                ),
            },
            "construction": {
                "registry_lookup_in_step": False,
                "capability_lookup_in_step": False,
                "configuration_parsing_in_step": False,
                "runtime_fallback": False,
            },
        }


def _validate_initial_q(
    initial_q: Mapping[str, torch.Tensor],
    shape: tuple[int, int, int],
) -> dict[str, torch.Tensor]:
    if not isinstance(initial_q, Mapping):
        raise TypeError("initial_q must be a mapping")
    values = dict(initial_q)
    if set(values) != set(Q_COMPONENTS):
        raise ValueError("initial_q must contain exactly the five Q components")
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"initial_q[{name!r}] must be a tensor")
        if value.shape not in {shape, (1, *shape)}:
            raise ValueError(
                f"initial_q[{name!r}] has an incompatible shape"
            )
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError(f"initial_q[{name!r}] must be finite")
    return values


def build_channel_compiled_v2_runtime(
    run_spec: ChannelActiveNematicRunSpec,
    *,
    initial_q: Mapping[str, torch.Tensor],
    device: object = "cpu",
) -> ChannelCompiledV2Runtime:
    """Build the direct-import P7.3 runtime without changing a selector."""

    if not isinstance(run_spec, ChannelActiveNematicRunSpec):
        raise TypeError("run_spec must be a ChannelActiveNematicRunSpec")
    components = run_spec.components
    if run_spec.dtype != "float32":
        raise ValueError(
            "P7.3 preserves the float32 legacy Channel construction contract"
        )
    values = _validate_initial_q(initial_q, run_spec.shape)
    material = components.material
    pressure = components.pressure_solver
    solver = build_active_nematic_channel(
        run_spec.shape,
        run_spec.lengths,
        run_spec.dt,
        values,
        device=device,
        batchsize=run_spec.batch_size,
        rho=material.rho,
        elastic_constant=material.elastic_constant,
        beta=material.beta,
        friction=material.friction,
        viscosity=material.viscosity,
        pressure_rel_tol=pressure.relative_tolerance,
        pressure_max_iter=pressure.max_iterations,
        pressure_fixed_iterations=pressure.fixed_iterations,
    )
    solver.parameters["alpha"] = torch.full(
        (run_spec.batch_size, *run_spec.shape),
        material.activity,
        dtype=solver.dtype,
        device=solver.device,
    )
    old_integrator = solver.integrator
    solver.integrator_cl = _ChannelCompiledV2EulerIntegrator
    solver.integrator = _ChannelCompiledV2EulerIntegrator(
        solver.model,
        solver.dt,
        solver.qx,
        solver.qy,
        solver.q2,
    )
    solver.integrator.spectral_refresh_interval = (
        old_integrator.spectral_refresh_interval
    )
    declaration = channel_compiled_v2_declaration()
    return ChannelCompiledV2Runtime(
        solver=solver,
        run_spec=run_spec,
        declaration=declaration,
    )


__all__ = [
    "ChannelCompiledV2Runtime",
    "build_channel_compiled_v2_runtime",
]
