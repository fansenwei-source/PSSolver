"""P8.3 complete-stress Beris--Edwards runtime for a no-slip Channel."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from typing import Protocol, runtime_checkable

import torch

from pssolver.configuration.channel_beris_edwards import (
    CHANNEL_COMPLETE_STRESS_RUNTIME_PATH,
    ChannelBerisEdwardsRunSpec,
)
from pssolver.integrator import SemiImplicitEulerIntegrator
from pssolver.integrators.state_backed import (
    StateBackedProjectedIntegratorMixin,
)
from pssolver.linear_solvers.stokes.channel_no_slip import (
    CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    CHANNEL_VELOCITY_BOUNDARY_CONDITIONS,
    ChannelNoSlipModalStokesSolver,
)
from pssolver.models.active_nematics import (
    BerisEdwardsPointwiseKernels,
    BerisEdwardsQGradientCache,
    BerisEdwardsQNonlinearModel,
    CHANNEL_Q_BOUNDARY_CONDITIONS,
    CHANNEL_DISTORTION_STRESS_BOUNDARY_CONDITIONS,
    Q_COMPONENTS,
    beris_edwards_molecular_field_components,
    beris_edwards_linear_operator,
)
from pssolver.operators.tensor_divergence import (
    projected_common_basis_stress_divergence,
    projected_component_basis_stress_divergence,
)
from pssolver.operators.projection import BasisAwareSpectralProjector
from pssolver.solver import SpectralSolver


class BerisEdwardsChannelStokes(ChannelNoSlipModalStokesSolver):
    """Complete nematic stress followed by the Channel Schur/PCG solve."""

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
        pressure_relative_tolerance,
        pressure_max_iterations,
        pressure_fixed_iterations,
        q_gradient_cache,
        pointwise_kernels,
    ) -> None:
        values = (
            beta_value,
            friction,
            viscosity,
            ldg_a,
            ldg_b,
            ldg_c,
            ldg_l1,
            flow_alignment,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Channel nematic coefficients must be finite")
        if ldg_l1 <= 0:
            raise ValueError("the one-constant L1 coefficient must be positive")
        if molecular_field_linear_space not in {"physical", "spectral"}:
            raise ValueError("invalid molecular-field linear space")
        if stress_divergence_sum_space != "physical":
            raise ValueError(
                "two-bounded-axis Channel stress divergence requires "
                "stress_divergence_sum_space='physical'"
            )
        if not isinstance(pointwise_kernels, BerisEdwardsPointwiseKernels):
            raise TypeError("pointwise_kernels has the wrong type")
        if q_gradient_cache is not None and not isinstance(
            q_gradient_cache,
            BerisEdwardsQGradientCache,
        ):
            raise TypeError("q_gradient_cache has the wrong type")
        super().__init__(
            solver.transform_backend,
            friction=friction,
            viscosity=viscosity,
            pressure_relative_tolerance=pressure_relative_tolerance,
            pressure_max_iterations=pressure_max_iterations,
            pressure_fixed_iterations=pressure_fixed_iterations,
        )
        self.transform_backend = solver.transform_backend
        self.beta = float(beta_value)
        self.ldg_a = float(ldg_a)
        self.ldg_b = float(ldg_b)
        self.ldg_c = float(ldg_c)
        self.ldg_l1 = float(ldg_l1)
        self.flow_alignment = float(flow_alignment)
        self.molecular_field_linear_space = molecular_field_linear_space
        self.stress_divergence_sum_space = stress_divergence_sum_space
        self.spectral_projector = spectral_projector
        self.q_gradient_cache = q_gradient_cache
        self.pointwise_kernels = pointwise_kernels
        self.q_boundary_conditions = CHANNEL_Q_BOUNDARY_CONDITIONS
        self.velocity_boundary_conditions = CHANNEL_VELOCITY_BOUNDARY_CONDITIONS
        self.pressure_boundary_conditions = CHANNEL_PRESSURE_BOUNDARY_CONDITIONS
        self.distortion_boundary_conditions = (
            CHANNEL_DISTORTION_STRESS_BOUNDARY_CONDITIONS
        )

    def _project_physical_tensor(self, tensor, boundary_conditions):
        return self.spectral_projector.inverse_transform(
            self.spectral_projector.forward_transform(
                tensor,
                boundary_conditions,
            ),
            boundary_conditions,
        )

    def compute_nematic_force(self, fields, alpha):
        if self.q_gradient_cache is not None:
            self.q_gradient_cache.clear()
        q_components = tuple(fields[name] for name in Q_COMPONENTS)
        if self.molecular_field_linear_space == "physical":
            laplacians = tuple(
                fields.laplacian(name, projector=self.spectral_projector)
                for name in Q_COMPONENTS
            )
            raw_h = beris_edwards_molecular_field_components(
                q_components,
                laplacians,
                ldg_a=self.ldg_a,
                ldg_b=self.ldg_b,
                ldg_c=self.ldg_c,
                ldg_l1=self.ldg_l1,
            )
            h_tensor = self._project_physical_tensor(
                torch.stack(raw_h),
                self.q_boundary_conditions,
            )
        else:
            bulk_h = self.pointwise_kernels.bulk_molecular_field_components(
                q_components,
                ldg_a=self.ldg_a,
                ldg_b=self.ldg_b,
                ldg_c=self.ldg_c,
            )
            h_hat = self.spectral_projector.forward_transform(
                torch.stack(bulk_h),
                self.q_boundary_conditions,
            )
            for index, name in enumerate(Q_COMPONENTS):
                h_hat[index].add_(
                    fields.laplacian_hat(name),
                    alpha=self.ldg_l1,
                )
            h_tensor = self.spectral_projector.inverse_transform(
                h_hat,
                self.q_boundary_conditions,
            )
        algebraic_stress = self.pointwise_kernels.algebraic_stress_components(
            q_components,
            tuple(h_tensor[index] for index in range(5)),
            flow_alignment=self.flow_alignment,
            active_prefactor=self.beta * alpha,
        )
        algebraic_force = projected_common_basis_stress_divergence(
            self.transform_backend,
            algebraic_stress,
            self.q_boundary_conditions,
            projector=self.spectral_projector,
            sum_space="physical",
        )
        q_gradients = tuple(
            tuple(
                fields.gradient(
                    name,
                    axis=axis,
                    projector=self.spectral_projector,
                )
                for name in Q_COMPONENTS
            )
            for axis in range(3)
        )
        distortion_stress = self.pointwise_kernels.distortion_stress_components(
            q_gradients,
            ldg_l1=self.ldg_l1,
        )
        distortion_force = projected_component_basis_stress_divergence(
            self.transform_backend,
            distortion_stress,
            self.distortion_boundary_conditions,
            output_boundary_conditions=self.velocity_boundary_conditions,
            projector=self.spectral_projector,
        )
        total_force = self._project_physical_tensor(
            algebraic_force + distortion_force,
            self.velocity_boundary_conditions,
        )
        if self.q_gradient_cache is not None:
            self.q_gradient_cache.stage(fields, q_gradients)
        return total_force

    def after_static_fields_updated(self, fields):
        if self.q_gradient_cache is not None:
            self.q_gradient_cache.publish(fields)

    def forward(self, fields, params):
        force = self.compute_nematic_force(fields, params["alpha"])
        force_hat = self.spectral_projector.forward_transform(
            force,
            self.velocity_boundary_conditions,
        )
        return torch.stack(self.solve_force_hats(*force_hat))


@dataclass(frozen=True, slots=True)
class ChannelBerisEdwardsOutputViews:
    q: torch.Tensor
    velocity: torch.Tensor
    pressure: torch.Tensor
    owner: torch.Tensor

    def __post_init__(self) -> None:
        if self.q.shape[0] != 5 or self.velocity.shape[0] != 3:
            raise ValueError("complete-stress Channel component counts differ")
        if self.pressure.shape != self.owner.shape[1:]:
            raise ValueError("complete-stress Channel pressure shape differs")
        owner = self.owner.untyped_storage().data_ptr()
        if any(
            value.untyped_storage().data_ptr() != owner
            for value in (self.q, self.velocity, self.pressure)
        ):
            raise ValueError("complete-stress Channel outputs must be zero-copy")


class _ChannelProjectedEuler(
    StateBackedProjectedIntegratorMixin,
    SemiImplicitEulerIntegrator,
):
    _phase3_connection_stage = "P8.3_channel_complete_stress"

    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(model, dt, qx, qy, q2)
        self.spectral_projector = model.spectral_projector

    def _refresh_dynamic_spectra(self):
        self.spectral_projector.refresh_dynamic_fields(
            self.model.fields,
            sync_spatial=True,
        )

    def _inverse_dynamic_spectra(self, state, workspace, generation):
        del state, workspace, generation
        for group in self.dynamic_transform_groups:
            boundaries = self.model.fields.get_boundary_conditions(group[0])
            self.model.fields.store_spatial_group(
                group,
                self.spectral_projector.inverse_transform(
                    self.model.fields.select_spectral_group(group),
                    boundaries,
                ),
            )


@dataclass(frozen=True, slots=True)
class ChannelBerisEdwardsRuntimeBuildRequest:
    run_spec: ChannelBerisEdwardsRunSpec
    initial_values: Mapping[str, object]
    device: object

    def __post_init__(self) -> None:
        if not isinstance(self.run_spec, ChannelBerisEdwardsRunSpec):
            raise TypeError("run_spec must be ChannelBerisEdwardsRunSpec")
        if not isinstance(self.initial_values, Mapping):
            raise TypeError("initial_values must be a mapping")
        if tuple(self.initial_values) != Q_COMPONENTS:
            raise ValueError("initial_values must contain ordered Q components")


@runtime_checkable
class ChannelBerisEdwardsRuntimeAdapterProtocol(Protocol):
    @property
    def runtime_path(self) -> str: ...
    @property
    def solver(self) -> object: ...
    @property
    def projector(self) -> object: ...
    @property
    def fields(self) -> object: ...
    @property
    def output_views(self) -> ChannelBerisEdwardsOutputViews: ...
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
    def restore_pressure_guess(self, value: torch.Tensor) -> None: ...
    def restore_progress(self, **values) -> None: ...
    def backend_restart_metadata(self) -> dict[str, object]: ...
    def to_metadata(self) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class ChannelBerisEdwardsRuntimeAdapter:
    _solver: SpectralSolver
    _projector: BasisAwareSpectralProjector
    _views: ChannelBerisEdwardsOutputViews

    @property
    def runtime_path(self) -> str:
        return CHANNEL_COMPLETE_STRESS_RUNTIME_PATH

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
    def output_views(self):
        return self._views

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

    def capture_pressure_guess(self):
        value = self._solver.model.static_model.pressure_guess
        if not isinstance(value, torch.Tensor):
            raise RuntimeError("Channel pressure warm-start state is unavailable")
        return value.detach().clone()

    def restore_pressure_guess(self, value):
        target = self._solver.fields["p.hat"]
        if not isinstance(value, torch.Tensor):
            raise TypeError("pressure_guess must be a tensor")
        if value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError("pressure_guess does not match the Channel runtime")
        if not bool(torch.isfinite(value).all().item()):
            raise ValueError("pressure_guess must be finite")
        self._solver.model.static_model.pressure_guess = value.detach().clone()

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
            raise RuntimeError("restored Channel refresh counters differ")

    def backend_restart_metadata(self):
        return {
            "kind": "channel_complete_stress_pressure_pcg",
            "state_keys": ["pressure_guess"],
        }

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
            "requested": self.runtime_path,
            "effective": self.runtime_path,
            "adapter": type(self).__name__,
            "fallback_used": False,
            "pressure_gauge": "zero_mean",
            "velocity_nullspace": "absent_due_to_two_bounded_no_slip_axes",
            "pressure_warm_start": True,
        }


def _dtype(value: str) -> torch.dtype:
    try:
        return {"float32": torch.float32, "float64": torch.float64}[value]
    except KeyError as exc:
        raise ValueError("Channel dtype must be float32 or float64") from exc


def build_channel_beris_edwards_runtime(
    request: ChannelBerisEdwardsRuntimeBuildRequest,
) -> ChannelBerisEdwardsRuntimeAdapter:
    """Build the one qualified P8.3 runtime without fallback."""

    if not isinstance(request, ChannelBerisEdwardsRuntimeBuildRequest):
        raise TypeError("request must be ChannelBerisEdwardsRuntimeBuildRequest")
    spec = request.run_spec.simulation
    domain = spec.geometry.domain
    numerics = spec.numerics
    parameters = spec.equation_system.parameters
    material = parameters["material"]
    stokes = parameters["stokes"]["parameters"]
    pressure = dict(spec.discretization_parameters["pressure_solver"])
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
    solver.integrator_cl = _ChannelProjectedEuler
    pointwise = BerisEdwardsPointwiseKernels(execution["pointwise_execution"])
    q2 = solver.get_q2(CHANNEL_Q_BOUNDARY_CONDITIONS)
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
            boundary_conditions=CHANNEL_Q_BOUNDARY_CONDITIONS,
        )
    for name in ("ux", "uy", "uz"):
        solver.model.add_static_field(
            name,
            boundary_conditions=CHANNEL_VELOCITY_BOUNDARY_CONDITIONS,
        )
    solver.model.add_static_field(
        "p",
        boundary_conditions=CHANNEL_PRESSURE_BOUNDARY_CONDITIONS,
    )
    cache = (
        None
        if execution["disable_q_gradient_reuse"]
        else BerisEdwardsQGradientCache()
    )
    solver.model.set_nonlinear_model(
        BerisEdwardsQNonlinearModel(
            projector,
            CHANNEL_Q_BOUNDARY_CONDITIONS,
            ldg_b=material["ldg_b"],
            ldg_c=material["ldg_c"],
            rotational_viscosity=material["gamma"],
            flow_alignment=material["flow_alignment"],
            q_gradient_cache=cache,
            pointwise_kernels=pointwise,
        )
    )
    solver.model.set_static_compute_model(
        BerisEdwardsChannelStokes(
            solver,
            projector,
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
            pressure_relative_tolerance=pressure["relative_tolerance"],
            pressure_max_iterations=pressure["max_iterations"],
            pressure_fixed_iterations=pressure["fixed_iterations"],
            q_gradient_cache=cache,
            pointwise_kernels=pointwise,
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
    owner = solver.fields.spatial
    views = ChannelBerisEdwardsOutputViews(
        q=owner[:5],
        velocity=owner[5:8],
        pressure=owner[8],
        owner=owner,
    )
    return ChannelBerisEdwardsRuntimeAdapter(solver, projector, views)


__all__ = [
    "ChannelBerisEdwardsOutputViews",
    "ChannelBerisEdwardsRuntimeAdapter",
    "ChannelBerisEdwardsRuntimeAdapterProtocol",
    "ChannelBerisEdwardsRuntimeBuildRequest",
    "build_channel_beris_edwards_runtime",
]
