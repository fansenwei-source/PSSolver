"""Materialized Periodic execution with caller-owned state and diagnostics."""

from __future__ import annotations

from dataclasses import dataclass, replace
from importlib import import_module
from types import MappingProxyType

import torch

from pssolver.backends.tensor_product import TensorProductTransformBackend
from pssolver.operators.projection import BasisAwareSpectralProjector
from pssolver.planning.capabilities import CAPABILITIES, lookup
from pssolver.planning.engine_plan import Plan

from . import program
from .constraints import PeriodicStokesConstraint
from .field_view import FieldView
from .imex import SemiImplicitEuler
from .linear import DiagonalOperator
from .memberwise_transforms import _MemberwiseTransforms
from .progress import _refresh_counters_from_completed_steps
from .state import State


def _resolve(capability):
    if capability not in CAPABILITIES:
        raise ValueError("engine implementation capability is not allowed")
    module, name = capability.implementation.split(":")
    return getattr(import_module(module), name)


def _capability(kind, reference):
    item = next((value for value in CAPABILITIES
                 if value.kind == kind and value.implementation == reference), None)
    if item is None:
        raise ValueError(f"engine implementation reference is not allowed: {reference!r}")
    return item


@dataclass(frozen=True, slots=True)
class _ProjectorHost:
    shape: tuple[int, ...]
    transform_backend: TensorProductTransformBackend | _MemberwiseTransforms


class Engine:
    """One immutable execution configuration, reusable by independent states."""

    def __init__(self, plan: Plan, *, device, dtype, batch=1, kernels=None):
        if not isinstance(plan, Plan) or plan.schema_version != 1:
            raise ValueError("unsupported engine plan")
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("engine dtype must be torch.float32 or torch.float64")
        if not isinstance(batch, int) or isinstance(batch, bool) or batch < 1:
            raise ValueError("engine batch must be a positive integer")
        if (
            plan.geometry.kind != "periodic_box"
            or len(plan.constraints) > 1
            or plan.integrator.scheme != "semi_implicit_euler"
            or plan.integrator.history_length != 0
            or plan.integrator.refresh_interval is not None
            or plan.boundary_data
        ):
            raise ValueError("engine supports only registered Periodic Euler plans")
        kernels = {} if kernels is None else dict(kernels)
        package_explicit = {item.implementation for item in CAPABILITIES if item.kind == "explicit"}
        local = {item.kernel for item in plan.explicit if item.kernel not in package_explicit}
        if set(kernels) != local or not all(callable(value) for value in kernels.values()):
            raise ValueError("model kernels must match the plan's local identities exactly")
        linear_capabilities = [_capability("linear_recipe", item.recipe) for item in plan.linear]
        explicit_capabilities = [
            lookup("explicit", ("pointwise", "periodic^3")) if item.kernel in local
            else _capability("explicit", item.kernel) for item in plan.explicit
        ]
        constraint_capabilities = [_capability("constraint", item.solver) for item in plan.constraints]
        self._beris_edwards = (
            [item.key for item in linear_capabilities] == [("beris_edwards", "any")]
            and [item.key for item in explicit_capabilities] == [("beris_edwards_q", "periodic^3")]
            and [item.key for item in constraint_capabilities] == [("stokes", "periodic^3")]
        )
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
        if batch > 1:
            self.transform_backend = _MemberwiseTransforms(self.transform_backend, batch)
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
        if self._beris_edwards:
            recipe = _resolve(linear_capabilities[0])
            linear = recipe(self.transform_backend.get_q2(self.boundaries), **plan.linear[0].arguments)
            # Fields.build adds the batch dimension before stacking components.
            linear = linear.unsqueeze(0).expand(batch, *linear.shape)
            self.linear = DiagonalOperator(torch.stack([linear] * self.evolved_count), self.dt)
        else:
            self.linear = self._generic_linear(linear_capabilities)
        self.scheme = SemiImplicitEuler()
        if self._beris_edwards:
            self.explicit = _resolve(explicit_capabilities[0])(self.projector, plan.explicit[0])
            term = _resolve(constraint_capabilities[0])(
                host, self.projector, plan.constraints[0], self.explicit
            )
            self.constraint = PeriodicStokesConstraint(
                term,
                pressure_diagnostics=plan.constraints[0].pressure_diagnostics,
            )
        else:
            if plan.constraints or any(item.kernel not in local for item in plan.explicit):
                raise ValueError("package explicit/constraint terms require their complete registered composition")
            self.explicit = self._generic_explicit(explicit_capabilities, kernels)
            self.constraint = None
        self._default_controls = MappingProxyType(
            {
                item.name: torch.tensor(item.default, device=self.device, dtype=dtype)
                for item in plan.parameters
                if item.default is not None
            }
        )
        self._materialize_transform_caches()

    def _generic_linear(self, capabilities):
        symbols = {}
        for item, capability in zip(self.plan.linear, capabilities, strict=True):
            recipe = _resolve(capability)
            if capability.key[0] == "expression":
                symbol = recipe(self.transform_backend, self.boundaries,
                                item.arguments["monomials"], self.dtype)
            else:
                symbol = recipe(self.transform_backend.get_q2(self.boundaries), **item.arguments)
            symbols[item.target] = symbol if item.target not in symbols else symbols[item.target] + symbol
        components = []
        for field in self.plan.fields:
            if field.role != "evolved":
                continue
            symbol = symbols.get(field.name)
            if symbol is None:
                symbol = torch.zeros_like(self.transform_backend.get_q2(self.boundaries))
            symbol = symbol.unsqueeze(0).expand(self.batch, *symbol.shape)
            components.extend([symbol] * len(field.components))
        return DiagonalOperator(torch.stack(components), self.dt)

    def _generic_explicit(self, capabilities, kernels):
        terms = tuple((item, _resolve(capability)(self, item, kernels[item.kernel]))
                      for item, capability in zip(self.plan.explicit, capabilities, strict=True))

        def evaluate(view, controls, *, completed_steps):
            values = {}
            for item, term in terms:
                value = term(view, controls, completed_steps=completed_steps)
                values[item.target] = value if item.target not in values else values[item.target] + value
            result = []
            for field in self.plan.fields:
                if field.role == "evolved":
                    value = values.get(field.name)
                    if value is None:
                        value = torch.zeros_like(view.spectral[self._observable_slices[field.name]])
                    result.append(value)
            return torch.cat(result)

        return evaluate

    def _explicit_at(self, completed_steps):
        if self._beris_edwards:
            return self.explicit
        return lambda view, controls: self.explicit(view, controls, completed_steps=completed_steps)

    def _constraint_hats(self, view, controls, *, diagnostics):
        return self.constraint.constrained_hats(view, controls["alpha"], diagnostics=diagnostics)

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
        cache = self.explicit.q_gradient_cache if self._beris_edwards else None
        if cache is not None:
            cache.clear()

    @property
    def default_controls(self):
        """Return the existing immutable default-control mapping without copying."""
        return self._default_controls

    def _controls(self, controls):
        return dict(self._default_controls) | ({} if controls is None else dict(controls))

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
            if not self._beris_edwards:
                projected = self.projector.project_real_spectrum(spectral, self.boundaries)
                physical = self.projector.inverse_transform(projected, self.boundaries)
                return replace(state, spectral=projected, spatial=physical)
            # The construction solve validates the constraint; it does not
            # publish its fields into the initially zero constraint slots.
            if self.constraint is not None:
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
                self._controls(controls),
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
