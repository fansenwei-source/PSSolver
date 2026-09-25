"""P8.2 complete-stress Beris--Edwards runtime for a periodic box."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

import torch

from pssolver.configuration.periodic_beris_edwards import (
    PERIODIC_RUNTIME_PATH,
    PeriodicBerisEdwardsRunSpec,
)
from pssolver.integrators.state_backed import (
    StateBackedProjectedIntegratorMixin,
)
from pssolver.integrator import SemiImplicitEulerIntegrator
from pssolver.linear_solvers.stokes import PeriodicModalStokesSolver
from pssolver.models.active_nematics import (
    BerisEdwardsFreeSlipStokes,
    BerisEdwardsPointwiseKernels,
    BerisEdwardsQGradientCache,
    BerisEdwardsQNonlinearModel,
    Q_COMPONENTS,
    beris_edwards_linear_operator,
)
from pssolver.operators.projection import BasisAwareSpectralProjector
from pssolver.solver import SpectralSolver


PERIODIC_BOUNDARIES = ("periodic", "periodic", "periodic")


class _BerisEdwardsPeriodicStokes(
    PeriodicModalStokesSolver,
    BerisEdwardsFreeSlipStokes,
):
    """Runtime composition of complete nematic force and periodic Stokes."""

    def __init__(
        self,
        solver,
        spectral_projector,
        *,
        beta_value,
        friction,
        viscosity,
        ldg_a,
        ldg_b,
        ldg_c,
        ldg_l1,
        flow_alignment,
        molecular_field_linear_space,
        stress_divergence_sum_space,
        cache_force_diagnostics,
        cache_pressure_diagnostics,
        q_gradient_cache,
        pointwise_kernels,
        zero_mode_policy,
    ):
        numeric_values = (
            beta_value,
            friction,
            viscosity,
            ldg_a,
            ldg_b,
            ldg_c,
            ldg_l1,
            flow_alignment,
        )
        if not all(math.isfinite(float(value)) for value in numeric_values):
            raise ValueError("Stokes and nematic coefficients must be finite")
        if ldg_l1 <= 0:
            raise ValueError("the one-constant L1 coefficient must be positive")
        if molecular_field_linear_space not in {"physical", "spectral"}:
            raise ValueError("invalid molecular-field linear space")
        if stress_divergence_sum_space not in {"physical", "spectral"}:
            raise ValueError("invalid stress-divergence sum space")

        PeriodicModalStokesSolver.__init__(
            self,
            solver.transform_backend,
            boundary_conditions=PERIODIC_BOUNDARIES,
            friction=friction,
            viscosity=viscosity,
            zero_mode_policy=zero_mode_policy,
            pressure_diagnostics=cache_pressure_diagnostics,
        )
        self.beta = float(beta_value)
        self.ldg_a = float(ldg_a)
        self.ldg_b = float(ldg_b)
        self.ldg_c = float(ldg_c)
        self.ldg_l1 = float(ldg_l1)
        self.flow_alignment = float(flow_alignment)
        self.molecular_field_linear_space = molecular_field_linear_space
        self.stress_divergence_sum_space = stress_divergence_sum_space
        self.cache_force_diagnostics = bool(cache_force_diagnostics)
        self.pointwise_kernels = pointwise_kernels
        self.q_gradient_cache = q_gradient_cache
        self.spectral_projector = spectral_projector
        self.q_boundary_conditions = PERIODIC_BOUNDARIES
        self.distortion_odd_boundary_conditions = PERIODIC_BOUNDARIES
        self.last_tangential_force_mean = None
        self.last_total_tangential_force_mean = None
        self.last_active_tangential_force_mean = None
        self.last_passive_tangential_force_mean = None
        self.last_removed_tangential_force_mean = None
        self.last_projected_normal_force = None
        self.last_uniform_velocity_mode_action = (
            "remove_all_uniform_velocity_and_force"
            if zero_mode_policy == "zero_mean"
            else "retain_all_uniform_velocity_resolved_by_friction"
        )


class _PeriodicProjectedEuler(
    StateBackedProjectedIntegratorMixin,
    SemiImplicitEulerIntegrator,
):
    _phase3_connection_stage = "P8.2_periodic_complete_stress"

    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(model, dt, qx, qy, q2)
        self.spectral_projector = model.spectral_projector

    def _refresh_dynamic_spectra(self):
        self.spectral_projector.refresh_dynamic_fields(
            self.model.fields,
            sync_spatial=True,
        )

    def _inverse_dynamic_spectra(self, state, workspace, generation):
        for group in self.dynamic_transform_groups:
            bcs = self.model.fields.get_boundary_conditions(group[0])
            self.model.fields.store_spatial_group(
                group,
                self.spectral_projector.inverse_transform(
                    self.model.fields.select_spectral_group(group),
                    bcs,
                ),
            )


@dataclass(frozen=True, slots=True)
class PeriodicRuntimeBuildRequest:
    run_spec: PeriodicBerisEdwardsRunSpec
    initial_values: Mapping[str, object]
    device: object

    def __post_init__(self) -> None:
        if not isinstance(self.run_spec, PeriodicBerisEdwardsRunSpec):
            raise TypeError("run_spec must be PeriodicBerisEdwardsRunSpec")
        if not isinstance(self.initial_values, Mapping):
            raise TypeError("initial_values must be a mapping")
        if tuple(self.initial_values) != Q_COMPONENTS:
            raise ValueError("initial_values must contain ordered Q components")


@runtime_checkable
class PeriodicRuntimeAdapterProtocol(Protocol):
    @property
    def runtime_path(self) -> str: ...
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
    def restore_progress(
        self,
        *,
        completed_steps: int,
        spectral_refresh_interval: int | None,
        integrator_step_count: int,
        integrator_refresh_count: int,
    ) -> None: ...
    def backend_restart_metadata(self) -> dict[str, object]: ...
    def to_metadata(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class PeriodicRuntimeAdapter:
    _solver: SpectralSolver
    _projector: BasisAwareSpectralProjector

    @property
    def runtime_path(self) -> str:
        return PERIODIC_RUNTIME_PATH

    @property
    def solver(self):
        return self._solver

    @property
    def projector(self):
        return self._projector

    @property
    def fields(self):
        return self._solver.fields

    @property
    def completed_steps(self) -> int:
        integrator = self._solver.integrator
        interval = integrator.spectral_refresh_interval
        if interval is None:
            return int(integrator.step_count)
        return int(integrator.refresh_count * interval + integrator.step_count)

    def advance(self, steps, *, pre_update_callback=None):
        self._solver.run(steps, pre_update_callback=pre_update_callback)

    def synchronize_for_observation(self):
        self._solver.refresh_static_fields()

    def restore_progress(
        self,
        *,
        completed_steps,
        spectral_refresh_interval,
        integrator_step_count,
        integrator_refresh_count,
    ):
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
            raise RuntimeError("restored refresh counters are inconsistent")

    def backend_restart_metadata(self):
        return {"kind": "periodic_stokes_stateless", "state_keys": []}

    def flow_diagnostics(self):
        flow = self._solver.model.static_model
        return {
            "last_pressure_iterations": int(flow.last_pressure_iterations),
            "last_pressure_residual": float(flow.last_pressure_residual),
            "last_pressure_relative_residual": float(
                flow.last_pressure_relative_residual
            ),
        }

    def to_metadata(self):
        return {
            "requested": PERIODIC_RUNTIME_PATH,
            "effective": PERIODIC_RUNTIME_PATH,
            "adapter": type(self).__name__,
            "fallback_used": False,
            "pressure_gauge": "zero_mean",
            "uniform_velocity_mode_action": (
                "remove_all_uniform_velocity_and_force"
            ),
        }


def _dtype(value: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "float64": torch.float64}[value]
    except KeyError as exc:
        raise ValueError("periodic dtype must be float32 or float64") from exc


def build_periodic_beris_edwards_runtime(
    request: PeriodicRuntimeBuildRequest,
) -> PeriodicRuntimeAdapter:
    """Build the one qualified P8.2 runtime without runtime fallback."""

    if not isinstance(request, PeriodicRuntimeBuildRequest):
        raise TypeError("request must be PeriodicRuntimeBuildRequest")
    spec = request.run_spec.simulation
    domain = spec.geometry.domain
    numerics = spec.numerics
    parameters = spec.equation_system.parameters
    material = parameters["material"]
    stokes = parameters["stokes"]["parameters"]
    execution = spec.execution.options
    real_dtype = _dtype(numerics.precision.value)

    solver = SpectralSolver(
        shape=domain.shape,
        L=domain.lengths,
        dt=spec.time_integration.integrator.dt,
        device=request.device,
        batchsize=1,
        dtype=real_dtype,
        transform_execution_order=numerics.transform_execution_order.value,
        spectral_storage=numerics.spectral_storage.value,
        hermitian_axis=numerics.hermitian_axis,
    )
    projector = BasisAwareSpectralProjector(
        solver,
        rule=numerics.dealias_rule.value,
        transform_execution=numerics.projected_transform_execution.value,
    )
    solver.model.spectral_projector = projector
    solver.model.set_static_inverse_transform(projector.inverse_transform)
    solver.integrator_cl = _PeriodicProjectedEuler
    pointwise = BerisEdwardsPointwiseKernels(execution["pointwise_execution"])
    q2 = solver.get_q2(PERIODIC_BOUNDARIES)
    linear = beris_edwards_linear_operator(
        q2,
        ldg_a=material["ldg_a"],
        ldg_l1=parameters["ldg_l1"],
        rotational_viscosity=material["gamma"],
    )
    for name in Q_COMPONENTS:
        solver.model.add_dynamic_field(
            name,
            init=request.initial_values[name],
            L_hat=linear,
            boundary_conditions=PERIODIC_BOUNDARIES,
        )
    for name in ("ux", "uy", "uz", "p"):
        solver.model.add_static_field(
            name,
            boundary_conditions=PERIODIC_BOUNDARIES,
        )
    cache = (
        None
        if execution["disable_q_gradient_reuse"]
        else BerisEdwardsQGradientCache()
    )
    solver.model.set_nonlinear_model(
        BerisEdwardsQNonlinearModel(
            projector,
            PERIODIC_BOUNDARIES,
            ldg_b=material["ldg_b"],
            ldg_c=material["ldg_c"],
            rotational_viscosity=material["gamma"],
            flow_alignment=material["flow_alignment"],
            q_gradient_cache=cache,
            pointwise_kernels=pointwise,
        )
    )
    solver.model.set_static_compute_model(
        _BerisEdwardsPeriodicStokes(
            solver,
            spectral_projector=projector,
            beta_value=material["beta"],
            friction=stokes["friction"],
            viscosity=stokes["viscosity"],
            ldg_a=material["ldg_a"],
            ldg_b=material["ldg_b"],
            ldg_c=material["ldg_c"],
            ldg_l1=parameters["ldg_l1"],
            flow_alignment=material["flow_alignment"],
            molecular_field_linear_space=execution[
                "molecular_field_linear_space"
            ],
            stress_divergence_sum_space=execution[
                "stress_divergence_sum_space"
            ],
            cache_force_diagnostics=False,
            cache_pressure_diagnostics=True,
            q_gradient_cache=cache,
            pointwise_kernels=pointwise,
            zero_mode_policy=stokes["tangential_zero_mode_policy"],
        )
    )
    solver.model.parameters.new_param(
        "alpha",
        torch.tensor(
            parameters["activity_amplitude"],
            device=request.device,
            dtype=real_dtype,
        ),
    )
    solver.build()
    solver.integrator.set_spectral_refresh_interval(None)
    projector.project_dynamic_fields(solver.model.fields, sync_spatial=True)
    if projector.enabled:
        solver.integrator._static_fields_are_current = False
    return PeriodicRuntimeAdapter(solver, projector)


__all__ = [
    "PERIODIC_BOUNDARIES",
    "PeriodicRuntimeAdapter",
    "PeriodicRuntimeAdapterProtocol",
    "PeriodicRuntimeBuildRequest",
    "build_periodic_beris_edwards_runtime",
]
