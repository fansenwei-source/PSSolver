"""Stateless Periodic constraint adapter with caller-owned diagnostics."""

import math
from collections.abc import Mapping
from typing import Literal, Protocol

import torch
from torch.autograd.function import once_differentiable

from pssolver.linear_solvers.stokes.channel_no_slip import (
    CHANNEL_VELOCITY_BOUNDARY_CONDITIONS,
    ChannelNoSlipModalStokesSolver,
    PCGRole,
)
from pssolver.linear_solvers.stokes.plane_free_slip_stateless import solve_force_hats_stateless
from pssolver.models.active_nematics import (
    CHANNEL_DISTORTION_STRESS_BOUNDARY_CONDITIONS,
    CHANNEL_Q_BOUNDARY_CONDITIONS,
    Q_COMPONENTS,
    BerisEdwardsPointwiseKernels,
    BerisEdwardsQGradientCache,
    BerisEdwardsQNonlinearModel,
    beris_edwards_molecular_field_components,
)
from pssolver.operators.tensor_divergence import (
    projected_common_basis_stress_divergence,
    projected_component_basis_stress_divergence,
)


SolverState = Mapping[str, torch.Tensor | int | None]


class ConstraintSolver(Protocol):
    key: tuple[str, str]
    adjoint_mode: Literal["autograd", "custom_transpose"]

    def seed(self, state) -> SolverState: ...
    def solve(self, rhs, solver_state, *, out=None): ...
    def solve_transpose(self, rhs_bar, solver_state): ...


class PeriodicStokesConstraint:
    key = ("stokes", "periodic^3")
    adjoint_mode = "autograd"

    def __init__(self, term, *, pressure_diagnostics):
        self.term = term
        self.pressure_diagnostics = pressure_diagnostics

    def seed(self, state):
        return {}

    def constrained_hats(self, view, alpha, *, diagnostics):
        def solve(*forces):
            values, report = self.term.solve_force_hats_with_diagnostics(
                *forces,
                pressure_diagnostics=self.pressure_diagnostics,
            )
            if diagnostics is not None:
                diagnostics.update(report)
            return values

        return self.term.constrained_hats(view, alpha, solve=solve, record=None)

    def solve(self, rhs, solver_state, *, out=None):
        values, _ = self.term.solve_force_hats_with_diagnostics(
            *rhs,
            pressure_diagnostics=False,
        )
        result = torch.stack(values)
        if out is not None:
            out.copy_(result)
            result = out
        return result, {}

    def solve_transpose(self, rhs_bar, solver_state):
        with torch.enable_grad():
            rhs = torch.zeros_like(rhs_bar[:3], requires_grad=True)
            result, _ = self.solve(rhs, solver_state)
            return torch.autograd.grad(result, rhs, rhs_bar, create_graph=True)[0]


class _ChannelNematicForce:
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
        self.pressure_solver = ChannelNoSlipModalStokesSolver(
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

    def _q_gradients(self, fields):
        return tuple(
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

    def rebuild_q_gradient_cache(self, fields):
        """Reconstruct the derived one-use cache after exact restart.

        The cache is an optimization derived entirely from the restored Q
        spectra, so checkpointing its fifteen physical gradient tensors would
        duplicate large arrays.  Rebuilding it once preserves the execution
        path of an uninterrupted step without enlarging the checkpoint.
        """

        if self.q_gradient_cache is None:
            return False
        self.q_gradient_cache.stage(fields, self._q_gradients(fields))
        if not self.q_gradient_cache.publish(fields):
            raise RuntimeError("restored Channel Q-gradient cache is invalid")
        return True

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
        q_gradients = self._q_gradients(fields)
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


def channel_explicit_term(projector, plan):
    """Build the Channel explicit Q term with the original velocity gradient path."""
    cache = BerisEdwardsQGradientCache() if plan.q_gradient_reuse else None
    pointwise = BerisEdwardsPointwiseKernels(plan.pointwise_execution)
    return BerisEdwardsQNonlinearModel(
        projector,
        CHANNEL_Q_BOUNDARY_CONDITIONS,
        **plan.arguments,
        q_gradient_cache=cache,
        pointwise_kernels=pointwise,
        velocity_gradient_inverse_transform=projector.inverse_transform_full,
    )


def channel_stokes_term(host, projector, plan, explicit):
    """Compose the existing complete-stress force with the canonical Channel solver."""
    arguments = dict(plan.arguments)
    arguments.pop("zero_mode_policy")
    arguments.setdefault("pressure_relative_tolerance", 1.0e-10)
    arguments.setdefault("pressure_max_iterations", 500)
    arguments.setdefault("pressure_fixed_iterations", None)
    return _ChannelNematicForce(
        host, projector, **arguments,
        q_gradient_cache=explicit.q_gradient_cache,
        pointwise_kernels=explicit.pointwise_kernels,
    )


class _ImplicitPressureSolve(torch.autograd.Function):
    @staticmethod
    def forward(ctx, rhs_hat, operator):
        ctx.operator = operator
        value = operator._solve_primal_no_grad(rhs_hat)
        ctx.solve_sequence_id = operator._solve_sequence_id
        return value

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_pressure_hat):
        grad_rhs = ctx.operator._solve_transpose_no_grad(
            grad_pressure_hat,
            source_primal_sequence_id=ctx.solve_sequence_id,
        )
        return grad_rhs, None


class _FunctionalPressureRun:
    """Zero-warm-start Channel pressure solve with an implicit custom VJP."""

    def __init__(self, solver, *, make_diagnostics, error, schema_version):
        if not isinstance(solver, ChannelNoSlipModalStokesSolver):
            raise TypeError("solver must be a ChannelNoSlipModalStokesSolver")
        self._solver = solver
        self._make_diagnostics = make_diagnostics
        self._error = error
        self._schema_version = schema_version
        # The original transpose constructor performed this diagnostic bookkeeping.
        self._make_diagnostics(
            solver,
            operator_identity="channel_pressure_schur_conjugate_transpose",
            solve_phase="not_run",
        )
        self._solve_sequence_id = 0
        self.last_primal_diagnostics = self._make_diagnostics(
            solver,
            operator_identity="channel_pressure_schur_primal",
            solve_phase="not_run",
        )
        self.last_transpose_diagnostics = self._make_diagnostics(
            solver,
            operator_identity="channel_pressure_schur_conjugate_transpose",
            solve_phase="not_run",
        )

    def _validate_rhs(self, rhs_hat: torch.Tensor) -> None:
        if not isinstance(rhs_hat, torch.Tensor):
            raise TypeError("pressure right-hand side must be a torch.Tensor")
        expected_shape = tuple(self._solver.pressure_null_mask.shape)
        if tuple(rhs_hat.shape) != expected_shape:
            raise ValueError(
                "P9.7.3 pressure right-hand side shape must be "
                f"{expected_shape}, got {tuple(rhs_hat.shape)}"
            )
        if rhs_hat.dtype is not self._solver.ikx.dtype:
            raise TypeError(
                "pressure right-hand side dtype must match the solver "
                f"spectral dtype {self._solver.ikx.dtype}"
            )
        if rhs_hat.device != self._solver.ikx.device:
            raise ValueError("pressure right-hand side device must match the solver device")

    @torch.no_grad()
    def _solve_no_grad(
        self,
        rhs_hat,
        action,
        operator_identity,
        *,
        solve_phase,
        solve_sequence_id,
        source_primal_sequence_id,
    ):
        state = {
            "context": self,
            "action": action,
            "diagnostics": self._make_diagnostics,
            "error": self._error,
            "standalone": False,
            "operator_identity": operator_identity,
            "solve_phase": solve_phase,
            "solve_sequence_id": solve_sequence_id,
            "source_primal_sequence_id": source_primal_sequence_id,
        }
        role = (
            PCGRole.FUNCTIONAL_PRIMAL
            if solve_phase == "functional_primal"
            else PCGRole.FUNCTIONAL_TRANSPOSE
        )
        entry = (
            self._solver.solve
            if role is PCGRole.FUNCTIONAL_PRIMAL
            else self._solver.solve_transpose
        )
        return entry(rhs_hat, state, role=role)

    def _solve_primal_no_grad(self, rhs_hat):
        self._solve_sequence_id += 1
        sequence = self._solve_sequence_id
        self.last_transpose_diagnostics = self._make_diagnostics(
            self._solver,
            operator_identity="channel_pressure_schur_conjugate_transpose",
            solve_phase="not_run",
            solve_sequence_id=sequence,
            source_primal_sequence_id=sequence,
        )
        value, diagnostics = self._solve_no_grad(
            rhs_hat,
            self._solver._pressure_operator,
            "channel_pressure_schur_primal",
            solve_phase="functional_primal",
            solve_sequence_id=sequence,
            source_primal_sequence_id=None,
        )
        self.last_primal_diagnostics = diagnostics
        self._validate_solve_diagnostics(diagnostics, "primal")
        return value

    def _solve_transpose_no_grad(
        self,
        rhs_hat,
        *,
        source_primal_sequence_id=None,
    ):
        if source_primal_sequence_id is None:
            source_primal_sequence_id = self._solve_sequence_id
        value, diagnostics = self._solve_no_grad(
            rhs_hat,
            self._solver.apply_pressure_operator_transpose,
            "channel_pressure_schur_conjugate_transpose",
            solve_phase="functional_transpose",
            solve_sequence_id=source_primal_sequence_id,
            source_primal_sequence_id=source_primal_sequence_id,
        )
        if source_primal_sequence_id == self._solve_sequence_id:
            self.last_transpose_diagnostics = diagnostics
        self._validate_solve_diagnostics(diagnostics, "transpose")
        return value

    def _persist_solve_diagnostics(
        self,
        diagnostics: object,
    ) -> None:
        if diagnostics.solve_phase == "functional_primal":
            self.last_primal_diagnostics = diagnostics
        elif (
            diagnostics.solve_phase == "functional_transpose"
            and diagnostics.source_primal_sequence_id == self._solve_sequence_id
        ):
            self.last_transpose_diagnostics = diagnostics

    def _validate_solve_diagnostics(
        self,
        diagnostics: object,
        description: str,
    ) -> None:
        if not diagnostics.acceptable:
            raise self._error(
                f"functional pressure {description} solve is unacceptable",
                operation=f"channel_pressure_{description}_solve",
                details=diagnostics.to_metadata(),
            )

    def solve(self, rhs_hat: torch.Tensor) -> torch.Tensor:
        """Solve pressure and attach the graph-free implicit VJP."""

        self._validate_rhs(rhs_hat)
        return _ImplicitPressureSolve.apply(rhs_hat, self)

    def solve_force_hats(self, fx_hat, fy_hat, fz_hat):
        """Solve the surrounding Stokes map with functional pressure state."""

        force_hats = (fx_hat, fy_hat, fz_hat)
        if any(not isinstance(value, torch.Tensor) for value in force_hats):
            raise TypeError("Channel force coefficients must be tensors")
        expected = tuple(self._solver.a_inv.shape)
        for value in force_hats:
            if tuple(value.shape) != expected:
                raise ValueError(
                    f"Channel force coefficient shape must be {expected}, got {tuple(value.shape)}"
                )
            if value.dtype is not self._solver.ikx.dtype:
                raise TypeError("Channel force coefficient dtype must match the solver")
            if value.device != self._solver.ikx.device:
                raise ValueError("Channel force coefficient device must match the solver")
            if not bool(torch.isfinite(value.detach()).all().item()):
                raise ValueError("Channel force coefficients must be finite")
        free_velocity = tuple(self._solver._helmholtz_inverse(value) for value in force_hats)
        pressure_rhs = self._solver._project_pressure_gauge(-self._solver.divergence_hat(*free_velocity))
        pressure_hat = self.solve(pressure_rhs)
        pressure_gradient = self._solver.pressure_gradient_hats(pressure_hat)
        velocity_hat = tuple(
            free_velocity[axis] - self._solver._helmholtz_inverse(pressure_gradient[axis])
            for axis in range(3)
        )
        return (*velocity_hat, pressure_hat)

    def pressure_solve_metadata(self) -> dict[str, object]:
        """Return stable outcomes for the most recent primal/transpose solves."""

        return {
            "schema_version": self._schema_version,
            "solve_sequence_id": self._solve_sequence_id,
            "primal": self.last_primal_diagnostics.to_metadata(),
            "transpose": self.last_transpose_diagnostics.to_metadata(),
        }


class ChannelStokesConstraint:
    """Channel stress and pressure with the warm-start pair owned by each State."""

    key = ("stokes", "periodic×no_slip^2")
    adjoint_mode = "custom_transpose"

    def __init__(self, term, *, warm_start):
        self.term = term
        self.warm_start = warm_start

    def seed(self, state):
        return {
            "pressure_guess": None,
            "pressure_guess_action": None,
            "pressure_guess_action_exponent": None,
        }

    def constrained_hats(self, view, alpha, solver_state, *, diagnostics):
        force = self.term.compute_nematic_force(view, alpha)
        force_hat = self.term.spectral_projector.forward_transform(
            force, self.term.velocity_boundary_conditions
        )
        if isinstance(diagnostics, _FunctionalPressureRun):
            if self.warm_start != "zero":
                raise ValueError("functional Channel pressure requires zero warm start")
            *velocity, pressure = diagnostics.solve_force_hats(*force_hat)
            return (torch.stack(tuple(velocity)), pressure), self.seed(None)
        state = solver_state if self.warm_start == "previous" else self.seed(None)
        values, next_state, report = self.solve(force_hat, state)
        if diagnostics is not None:
            diagnostics.update(report)
        return values, next_state

    def solve(self, rhs, solver_state):
        values, next_state, report = self.term.pressure_solver.solve_force_hats(*rhs, solver_state=solver_state)
        return torch.stack(values), next_state, report

    def solve_transpose(self, rhs_bar, solver_state):
        return self.term.pressure_solver.solve_transpose(
            rhs_bar, solver_state, role=PCGRole.FUNCTIONAL_TRANSPOSE
        )

    def functional_pressure_diagnostics(self, *, make_diagnostics, error, schema_version):
        """Create the caller-owned diagnostics context for the zero-start path."""
        return _FunctionalPressureRun(
            self.term.pressure_solver,
            make_diagnostics=make_diagnostics,
            error=error,
            schema_version=schema_version,
        )


class PlaneStokesConstraint:
    key = ("stokes", "periodic^2×free_slip")
    adjoint_mode = "autograd"

    def __init__(self, term, *, pressure_diagnostics):
        self.term = term
        self.pressure_diagnostics = pressure_diagnostics

    def seed(self, state):
        return {}

    def constrained_hats(self, view, alpha, *, diagnostics):
        def solve(*forces):
            ux, uy, uz, pressure, report = solve_force_hats_stateless(
                self.term, *forces, pressure_diagnostics=self.pressure_diagnostics
            )
            if diagnostics is not None:
                diagnostics.update(report)
            return ux, uy, uz, pressure

        record = None if diagnostics is None else lambda name, value: diagnostics.__setitem__(name, value)
        return self.term.constrained_hats(view, alpha, solve=solve, record=record)

    def solve(self, rhs, solver_state, *, out=None):
        *values, _ = solve_force_hats_stateless(self.term, *rhs, pressure_diagnostics=False)
        result = torch.stack(values)
        if out is not None:
            out.copy_(result)
            result = out
        return result, {}

    def solve_transpose(self, rhs_bar, solver_state):
        with torch.enable_grad():
            rhs = torch.zeros_like(rhs_bar[:3], requires_grad=True)
            result, _ = self.solve(rhs, solver_state)
            return torch.autograd.grad(result, rhs, rhs_bar, create_graph=True)[0]
