"""Tensor-free validation and assembly of declarative model terms."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from pssolver.core.spectral_identity import periodic_hermitian_state_projection
from pssolver.core.terms import (
    Constraint,
    Coordinates,
    Explicit,
    Expr,
    FieldRef,
    Linear,
    Model,
    PackageTerm,
    Time,
)
from pssolver.planning.capabilities import basis_signature, lookup
from pssolver.planning.engine_plan import (
    ConstraintPlan,
    DealiasPlan,
    ExplicitTermPlan,
    FieldPlan,
    GeometryPlan,
    IntegratorPlan,
    LinearTermPlan,
    OutputPlan,
    ParameterPlan,
    Plan,
    TransformGroup,
    build_plan,
)
from pssolver.planning.term_analysis import LinearAnalysisError, normalize


class AssemblyRejection(ValueError):
    """A mathematical incompatibility or an explicitly deferred capability."""

    def __init__(self, kind, code, reason, *, source=None, opens_in=None):
        super().__init__(reason)
        self.kind = kind
        self.code = code
        self.reason = reason
        self.source = source
        self.opens_in = opens_in


def _reject(code, reason, *, source=None, opens_in=None):
    raise AssemblyRejection(
        "mathematical" if source is None else "scope",
        code,
        reason,
        source=source,
        opens_in=opens_in,
    )


@dataclass(frozen=True, slots=True)
class GeometryInput:
    kind: str
    shape: tuple[int, ...]
    lengths: tuple[float, ...]


@dataclass(frozen=True, slots=True)
class NumericsInput:
    spectral_storage: str
    hermitian_axis: int | None
    transform_execution_order: str
    dealias_rule: str
    projected_transform_execution: str


@dataclass(frozen=True, slots=True)
class EulerInput:
    dt: float
    refresh_interval: int | None = None


@dataclass(frozen=True, slots=True)
class ExecutionInput:
    pointwise_execution: str = "eager"
    q_gradient_reuse: bool = False
    constraint_diagnostics: bool = False
    options: Mapping[str, object] = field(default_factory=dict)
    constraint_warm_start: str | None = None
    channel_pressure_options: Mapping[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class OutputInput:
    observables: tuple[str, ...]
    capture: str | None = None


@dataclass(frozen=True, slots=True)
class AssemblyProblem:
    model: Model
    geometry: GeometryInput
    boundaries: Mapping[str, tuple[str, ...]]
    integrator: EulerInput
    numerics: NumericsInput
    execution: ExecutionInput
    outputs: OutputInput


@dataclass(frozen=True, slots=True)
class AssembledProblem:
    plan: Plan
    kernels: Mapping[str, Callable]

    def engine(self, *, device, dtype, batch=1):
        """Materialize the plan only when the caller explicitly requests it."""
        from pssolver.engine.engine import Engine

        return Engine(self.plan, device=device, dtype=dtype, batch=batch, kernels=self.kernels)


def _reference(value, fields, *, role=None):
    if not isinstance(value, FieldRef) or value.name not in fields:
        _reject("invalid_field_reference", "term references an undeclared field")
    declared = fields[value.name]
    if value != declared.reference(value.name) or (role is not None and declared.role != role):
        _reject("invalid_field_reference", "term field components or role differ from declaration")
    return value.name


def _input(value, fields):
    if isinstance(value, FieldRef):
        return ("field", _reference(value, fields))
    if isinstance(value, Coordinates):
        return ("coordinates",)
    if isinstance(value, Time):
        return ("time",)
    if isinstance(value, Expr) and value.operation == "derivative":
        ref, axis, order = value.arguments
        name = _reference(ref, fields)
        if (
            not isinstance(axis, int)
            or axis not in range(3)
            or not isinstance(order, int)
            or order < 1
        ):
            _reject("invalid_linear_expression", "invalid explicit derivative input")
        return ("derivative", name, axis, order)
    _reject("invalid_field_reference", "unsupported explicit input")


def _boundaries(problem):
    geometry = problem.geometry
    if geometry.kind not in {"periodic_box", "plane_slab", "rectangular_channel"}:
        _reject(
            "geometry_not_migrated",
            "geometry awaits migration",
            source="S-22 §4",
            opens_in="0.5 (S-27, S-28)",
        )
    if getattr(problem, "boundary_data", ()):
        _reject(
            "boundary_data_reserved",
            "nonempty boundary data is reserved",
            source="S-22 §10.2",
            opens_in="0.6",
        )
    if set(problem.boundaries) != set(problem.model.fields):
        _reject("incompatible_boundaries", "every field requires its declared boundary basis")
    for name, boundaries in problem.boundaries.items():
        if isinstance(boundaries, Mapping):
            components = problem.model.fields[name].reference(name).components
            if set(boundaries) != set(components):
                _reject("incompatible_boundaries", "every component requires a boundary basis")
            values = tuple(tuple(boundaries[item]) for item in components)
        else:
            values = (tuple(boundaries),)
        for value in values:
            _validate_basis(value, geometry.kind)
    return {
        "periodic_box": ("periodic",) * 3,
        "plane_slab": ("periodic", "periodic", "bounded"),
        "rectangular_channel": ("periodic", "bounded", "bounded"),
    }[geometry.kind]


def _validate_basis(boundaries, geometry_kind):
        if any("robin" in str(item) or "anchoring" in str(item) for item in boundaries):
            _reject(
                "robin_or_anchoring",
                "Robin and anchoring await general boundary support",
                source="PROJECT P4",
                opens_in="0.6",
            )
        if any("nonzero_dirichlet" in str(item) for item in boundaries):
            _reject(
                "nonzero_dirichlet_lifting",
                "nonzero Dirichlet needs lifting",
                source="B8/S2-07, D-0007",
                opens_in="0.6",
            )
        if geometry_kind == "periodic_box" and tuple(boundaries) != ("periodic",) * 3:
            _reject("incompatible_boundaries", "periodic_box requires three periodic axes")
        if geometry_kind == "plane_slab" and tuple(boundaries) not in {
            ("periodic", "periodic", "neumann"), ("periodic", "periodic", "dirichlet")
        }:
            _reject("incompatible_boundaries", "plane_slab requires a homogeneous mixed basis")
        if geometry_kind == "rectangular_channel" and (
            len(boundaries) != 3 or boundaries[0] != "periodic"
            or any(value not in {"dirichlet", "neumann"} for value in boundaries[1:])
        ):
            _reject("incompatible_boundaries", "Channel requires one periodic axis and two wall axes")


def _field_basis(problem, name):
    value = problem.boundaries[name]
    if isinstance(value, Mapping):
        return tuple(tuple(value[item]) for item in problem.model.fields[name].reference(name).components)
    return tuple(value)


def _arguments(term, execution, consumed):
    for key in term.execution_options:
        consumed[key] += 1
        if key not in execution.options or consumed[key] != 1:
            _reject(
                "unused_or_missing_execution_option",
                "execution option missing or consumed more than once",
            )
    return dict(term.arguments) | {key: execution.options[key] for key in term.execution_options}


def assemble(problem):
    """Validate declarations and build one immutable plan and local-kernel map."""
    model = problem.model
    bc = _boundaries(problem)
    if not isinstance(problem.integrator, EulerInput):
        _reject(
            "integrator_not_available", "only Euler is available", source="D-0039", opens_in="0.7"
        )
    if problem.geometry.kind == "periodic_box" and problem.integrator.refresh_interval is not None:
        _reject(
            "refresh_not_supported",
            "Periodic refresh must remain disabled",
            source="P2",
            opens_in="never",
        )
    if problem.integrator.refresh_interval is not None and (
        not isinstance(problem.integrator.refresh_interval, int)
        or isinstance(problem.integrator.refresh_interval, bool)
        or problem.integrator.refresh_interval <= 0
    ):
        _reject("refresh_not_supported", "refresh_interval must be positive or None")
    fields = model.fields
    execution = problem.execution
    channel = problem.geometry.kind == "rectangular_channel"
    if channel and problem.integrator.refresh_interval is not None:
        _reject("refresh_not_supported", "Channel refresh must remain disabled", source="P2")
    if not channel and (execution.constraint_warm_start is not None or execution.channel_pressure_options):
        _reject("unused_or_missing_execution_option", "Channel pressure options require a Channel")
    linear, explicit, constraints, kernels = [], [], [], {}
    consumed = Counter()
    supports_reuse = False
    for index, term in enumerate(model.terms(*model.namespaces())):
        if isinstance(term, (Linear, Explicit)):
            target = _reference(term.target, fields, role="evolved")
            sig = basis_signature(_field_basis(problem, target))
        if isinstance(term, Linear):
            if term.diagonal is not None:
                capability = lookup("linear_recipe", (term.diagonal.name, "any"))
                if capability is None:
                    _reject("invalid_linear_expression", "diagonal recipe is unavailable")
                arguments = term.diagonal.arguments
            else:
                capability = lookup("linear_recipe", ("expression", sig))
                try:
                    arguments = {"monomials": normalize(term.expr, term.target)}
                except LinearAnalysisError as error:
                    _reject(error.code, str(error))
            linear.append(LinearTermPlan(target, "diagonal", capability.implementation, arguments))
        elif isinstance(term, Explicit):
            arguments = _arguments(term, execution, consumed)
            if isinstance(term.kernel, PackageTerm):
                capability = lookup("explicit", (term.kernel.name, sig))
                if capability is None:
                    _reject(
                        "missing_explicit_implementation",
                        "no explicit implementation for this basis",
                    )
                reference = capability.implementation
                supports_reuse |= term.kernel.name == "beris_edwards_q"
                mode, reuse = execution.pointwise_execution, execution.q_gradient_reuse
            else:
                if execution.pointwise_execution != "eager":
                    _reject(
                        "compiled_model_kernel",
                        "model-local kernels require eager execution",
                        source="S-26 §5.6",
                        opens_in="uncommitted; 0.6 design decision",
                    )
                if not callable(term.kernel):
                    _reject("missing_explicit_implementation", "model-local kernel is not callable")
                qualified = getattr(term.kernel, "__qualname__", "")
                module = getattr(term.kernel, "__module__", "")
                model_name = f"{type(model).__module__}:{type(model).__qualname__}"
                reference = (
                    f"{module}:{qualified}"
                    if module and "<" not in qualified
                    else f"{model_name}.terms[{index}]"
                )
                if reference in kernels and kernels[reference] is not term.kernel:
                    _reject(
                        "missing_explicit_implementation",
                        "two different kernels have the same identity",
                    )
                kernels[reference] = term.kernel
                if arguments:
                    _reject(
                        "unused_or_missing_execution_option",
                        "local kernels receive their declared inputs, not package arguments",
                    )
                arguments = dict(
                    inputs=tuple(
                        _input(value, fields)
                        for value in ((term.target,) if term.inputs is None else term.inputs)
                    ),
                    model=model_name,
                    model_version=model.version,
                )
                mode, reuse = "eager", False
            explicit.append(ExplicitTermPlan(target, reference, arguments, mode, reuse))
        elif isinstance(term, Constraint):
            inputs = tuple(_reference(value, fields) for value in term.inputs)
            outputs = tuple(_reference(value, fields, role="constrained") for value in term.outputs)
            sig = basis_signature(bc)
            if problem.geometry.kind == "plane_slab":
                velocity = _field_basis(problem, "velocity") if "velocity" in fields else ()
                pressure = _field_basis(problem, "pressure") if "pressure" in fields else ()
                if velocity == (("periodic", "periodic", "neumann"),
                                ("periodic", "periodic", "neumann"),
                                ("periodic", "periodic", "dirichlet")) and pressure == ("periodic", "periodic", "neumann"):
                    sig = "periodic^2×free_slip"
            if channel and term.kind == "stokes":
                wall_neumann = ("periodic", "neumann", "neumann")
                wall_dirichlet = ("periodic", "dirichlet", "dirichlet")
                if (
                    inputs == ("Q",) and outputs == ("velocity", "pressure")
                    and _field_basis(problem, "Q") == wall_neumann
                    and _field_basis(problem, "velocity") == (wall_dirichlet,) * 3
                    and _field_basis(problem, "pressure") == wall_neumann
                ):
                    sig = "periodic×no_slip^2"
            capability = lookup("constraint", (term.kind, sig))
            if capability is None:
                _reject("missing_constraint_solver", "no constraint solver for this basis")
            arguments = _arguments(term, execution, consumed)
            warm_start = "stateless"
            if sig == "periodic×no_slip^2":
                warm_start = execution.constraint_warm_start or "previous"
                if warm_start not in {"previous", "zero"}:
                    _reject("unused_or_missing_execution_option", "invalid Channel warm-start policy")
                if set(execution.channel_pressure_options) - {
                    "pressure_relative_tolerance", "pressure_max_iterations", "pressure_fixed_iterations",
                }:
                    _reject("unused_or_missing_execution_option", "unknown Channel pressure option")
                arguments.update(execution.channel_pressure_options)
            constraints.append(
                ConstraintPlan(
                    (term.kind, sig),
                    capability.implementation,
                    inputs,
                    outputs,
                    arguments,
                    warm_start,
                    capability.adjoint_mode,
                    execution.constraint_diagnostics,
                )
            )
        else:
            _reject("invalid_linear_expression", "unknown model term")
    if channel and not any(value.key == ("stokes", "periodic×no_slip^2") for value in constraints):
        _reject("missing_constraint_solver", "Channel requires the registered no-slip Stokes term")
    if set(consumed) != set(execution.options) or (
        execution.q_gradient_reuse and not supports_reuse
    ):
        _reject(
            "unused_or_missing_execution_option",
            "unused execution option or unsupported gradient reuse",
        )
    plan_fields = tuple(
        FieldPlan(name, value.role, value.reference(name).components, _field_basis(problem, name))
        for name, value in fields.items()
    )
    groups = []
    for role in ("evolved", "constrained"):
        grouped = {}
        ordered = []
        for item in plan_fields:
            if item.role != role:
                continue
            bases = item.boundary_conditions
            if isinstance(bases[0], str):
                bases = (bases,) * len(item.components)
            for component, basis in zip(item.components, bases, strict=True):
                grouped.setdefault(basis, []).append(component)
                ordered.append(component)
        for basis, components in grouped.items():
            indices = [ordered.index(component) for component in components]
            indexing = "contiguous_slice" if indices == list(range(indices[0], indices[0] + len(indices))) else "advanced"
            groups.append(TransformGroup(tuple(components), basis, indexing))
    for name in problem.outputs.observables:
        if name not in fields:
            _reject("invalid_field_reference", "observable references an undeclared field")
    geometry, numerics = problem.geometry, problem.numerics
    plan = build_plan(
        schema_version=1,
        geometry=GeometryPlan(
            geometry.kind,
            geometry.shape,
            geometry.lengths,
            bc,
            numerics.spectral_storage,
            numerics.hermitian_axis,
            periodic_hermitian_state_projection(numerics.spectral_storage),
            numerics.transform_execution_order,
            "multidim",
        ),
        fields=plan_fields,
        groups=tuple(groups),
        linear=tuple(linear),
        explicit=tuple(explicit),
        constraints=tuple(constraints),
        dealias=DealiasPlan(numerics.dealias_rule, numerics.projected_transform_execution),
        integrator=IntegratorPlan("semi_implicit_euler", problem.integrator.dt, 0, problem.integrator.refresh_interval),
        parameters=tuple(
            ParameterPlan(name, value.kind, value.controllable, value.default)
            for name, value in model.parameters.items()
        ),
        boundary_data=(),
        outputs=OutputPlan(
            problem.outputs.observables, "after_constraints", problem.outputs.capture
        ),
    )
    return AssembledProblem(plan, MappingProxyType(kernels))
