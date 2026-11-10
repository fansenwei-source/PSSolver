"""Materialized Periodic execution with caller-owned state and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType

import torch

from pssolver.backends.tensor_product import TensorProductTransformBackend
from pssolver.operators.projection import BasisAwareSpectralProjector
from pssolver.planning.engine_plan import Plan

from . import program
from .constraints import PeriodicStokesConstraint
from .field_view import FieldView
from .imex import SemiImplicitEuler
from .linear import DiagonalOperator
from .progress import _refresh_counters_from_completed_steps
from .resolve import resolve
from .state import State


@dataclass(frozen=True, slots=True)
class _ProjectorHost:
    shape: tuple[int, ...]
    transform_backend: TensorProductTransformBackend


class Engine:
    """One immutable execution configuration, reusable by independent states."""

    def __init__(self, plan: Plan, *, device, dtype, batch=1):
        if not isinstance(plan, Plan) or plan.schema_version != 1:
            raise ValueError("unsupported engine plan")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("engine dtype must be torch.float32 or torch.float64")
        if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
            raise ValueError("engine batch must be a positive integer")
        if (
            plan.geometry.kind != "periodic_box"
            or len(plan.constraints) != 1
            or len(plan.explicit) != 1
            or len(plan.linear) != 1
            or plan.integrator.scheme != "semi_implicit_euler"
            or plan.integrator.history_length != 0
            or plan.integrator.refresh_interval is not None
            or plan.boundary_data
        ):
            raise ValueError("engine S-25 supports only the Periodic Euler plan")
        self.plan = plan
        self.device = torch.device(device)
        self.dtype = dtype
        self.batch = batch
        geometry = plan.geometry
        self.shape = geometry.shape
        self.dt = plan.integrator.dt
        self.transform_backend = TensorProductTransformBackend(
            shape=geometry.shape,
            lengths=geometry.lengths,
            device=self.device,
            dtype=dtype,
            execution_order=geometry.transform_execution_order,
            spectral_storage=geometry.spectral_storage,
            hermitian_axis=geometry.hermitian_axis,
            periodic_transform_execution=geometry.periodic_transform_execution,
        )
        host = _ProjectorHost(self.shape, self.transform_backend)
        self.projector = BasisAwareSpectralProjector(
            host,
            rule=plan.dealias.rule,
            transform_execution=plan.dealias.transform_execution,
        )
        self.components = tuple(name for field in plan.fields for name in field.components)
        self.component_boundaries = tuple(
            field.boundary_conditions for field in plan.fields for _ in field.components
        )
        self.evolved_count = sum(
            len(field.components) for field in plan.fields if field.role == "evolved"
        )
        self.boundaries = plan.geometry.axis_kinds
        self._observable_slices = MappingProxyType(
            {
                field.name: slice(
                    sum(len(item.components) for item in plan.fields[:index]),
                    sum(len(item.components) for item in plan.fields[: index + 1]),
                )
                for index, field in enumerate(plan.fields)
            }
        )
        recipe = resolve(plan.linear[0].recipe)
        linear = recipe(self.transform_backend.get_q2(self.boundaries), **plan.linear[0].arguments)
        # Fields.build adds the batch dimension before stacking components.
        linear = linear.unsqueeze(0).expand(batch, *linear.shape)
        self.linear = DiagonalOperator(torch.stack([linear] * self.evolved_count), self.dt)
        self.scheme = SemiImplicitEuler()
        self.explicit = resolve(plan.explicit[0].kernel)(self.projector, plan.explicit[0])
        term = resolve(plan.constraints[0].solver)(
            host, self.projector, plan.constraints[0], self.explicit
        )
        self.constraint = PeriodicStokesConstraint(
            term,
            pressure_diagnostics=plan.constraints[0].pressure_diagnostics,
        )
        self._default_controls = MappingProxyType(
            {
                item.name: torch.tensor(item.default, device=self.device, dtype=dtype)
                for item in plan.parameters
                if item.default is not None
            }
        )
        self._materialize_transform_caches()

    def _materialize_transform_caches(self):
        """Build lazy basis-only data before any pure engine entry is called."""
        backend = self.transform_backend
        bc = self.boundaries
        physical = torch.zeros(self.shape, dtype=self.dtype, device=self.device)
        spectral = backend.forward(physical, bc)
        for axis in range(len(self.shape)):
            backend.gradient_hat(spectral, bc, axis)
        self.projector.forward_transform(physical, bc)
        self.projector.inverse_transform(spectral, bc)
        self.projector.project_real_spectrum_(spectral, bc)

    def _clear_cache(self):
        cache = self.explicit.q_gradient_cache
        if cache is not None:
            cache.clear()

    def _alpha(self, controls):
        return (self._default_controls if controls is None else controls)["alpha"]

    def field_view(self, state):
        return FieldView(
            state.spatial,
            state.spectral,
            self.transform_backend,
            self.components,
            self.component_boundaries,
        )

    def initial_state(self, evolved, *, diagnostics=None):
        expected = self.components[: self.evolved_count]
        if tuple(evolved) != expected:
            raise ValueError("initial values must contain ordered evolved components")
        values = []
        for name in expected:
            value = evolved[name]
            if tuple(value.shape) == self.shape:
                value = value.unsqueeze(0).expand(self.batch, *self.shape)
            if (
                tuple(value.shape) != (self.batch, *self.shape)
                or value.dtype != self.dtype
                or value.device != self.device
            ):
                raise ValueError("initial value shape, dtype or device differs from engine")
            values.append(value)
        values.extend(torch.zeros_like(values[0]) for _ in self.components[self.evolved_count :])
        spatial = torch.stack(values)
        spectral = torch.stack(
            [
                self.transform_backend.forward(value, bc)
                for value, bc in zip(spatial, self.component_boundaries, strict=True)
            ]
        )
        state = State(
            spectral, spatial, (), {}, 0, _refresh_counters_from_completed_steps(0, None), False
        )
        self._clear_cache()
        try:
            # The construction solve validates the constraint; it does not
            # publish its fields into the initially zero constraint slots.
            alpha = self._default_controls.get("alpha")
            if alpha is None:
                alpha = torch.zeros(self.shape, device=self.device, dtype=self.dtype)
            self.constraint.constrained_hats(self.field_view(state), alpha, diagnostics=diagnostics)
            state.solver = self.constraint.seed(state)
            self.projector.project_real_spectrum_(spectral[: self.evolved_count], self.boundaries)
            spatial[: self.evolved_count].copy_(
                self.projector.inverse_transform(
                    spectral[: self.evolved_count],
                    self.boundaries,
                )
            )
            return state
        finally:
            self._clear_cache()

    def _execute(self, operation, state, controls, *, inplace, diagnostics):
        self._clear_cache()
        try:
            return operation(
                self,
                state,
                self.field_view(state),
                self._alpha(controls),
                out=state if inplace else None,
                diagnostics=diagnostics,
            )
        finally:
            self._clear_cache()

    def prepare_constraints(self, state, controls=None, *, diagnostics=None):
        return self._execute(
            program.prepare, state, controls, inplace=False, diagnostics=diagnostics
        )

    def prepare_constraints_(self, state, controls=None, *, diagnostics=None):
        self._execute(program.prepare, state, controls, inplace=True, diagnostics=diagnostics)

    def step(self, state, controls=None, *, diagnostics=None):
        return self._execute(
            program.advance, state, controls, inplace=False, diagnostics=diagnostics
        )

    def step_(self, state, controls=None, *, diagnostics=None):
        self._execute(program.advance, state, controls, inplace=True, diagnostics=diagnostics)

    def observe(self, state, names=None):
        if not state.constrained_current:
            raise RuntimeError("observe requires constrained_current")
        if names is None:
            names = self.plan.outputs.observables
        return {name: state.spatial[self._observable_slices[name]] for name in names}
