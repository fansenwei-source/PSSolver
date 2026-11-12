"""Lower a public declaration to exactly one tensor-free assembly request."""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real

from pssolver.boundaries.homogeneous import HomogeneousBoundaryPolicy
from pssolver.boundaries.prescribed import StaticPrescribedDirichletPolicy
from pssolver.boundaries.robin import StaticRobinBoundaryPolicy
from pssolver.core.pressure import channel_pressure_minimum_relative_tolerance_for_precision
from pssolver.core.terms import Parameter
from pssolver.planning.assemble import (
    AssemblyProblem,
    EulerInput,
    ExecutionInput,
    GeometryInput,
    NumericsInput,
    OutputInput,
)
from pssolver.presets.beris_edwards import CompleteStressBerisEdwards as Model
from pssolver.presets.shendruk import (
    ShendrukPlanePreset,
    SpectralRefreshSpec,
    _resolve_spectral_refresh,
    positive_equilibrium_S,
    resolve_shendruk_plane_preset,
)

from .declarations import GeneratedInitialCondition, SnapshotInitialCondition
from .errors import PublicCompilationRejectionCode as Code
from .errors import PublicSimulationCompilationError
from .models import CompleteStressBerisEdwards, ShendrukPlaneBerisEdwards
from .snapshot_header import validate_snapshot_header


_FIELDS = ("Q", "velocity", "pressure")
_COMPONENTS = ("ux", "uy", "uz")
_BASIS = {
    "periodic_box": {
        "Q": {"neumann": ("periodic",) * 3},
        "velocity": {
            "free_slip_velocity": ("periodic",) * 3,
            "no_slip_velocity": ("periodic",) * 3,
        },
        "pressure": {"neumann": ("periodic",) * 3},
    },
    "plane_slab": {
        "Q": {"neumann": ("periodic", "periodic", "neumann")},
        "velocity": {
            "free_slip_velocity": {
                "ux": ("periodic", "periodic", "neumann"),
                "uy": ("periodic", "periodic", "neumann"),
                "uz": ("periodic", "periodic", "dirichlet"),
            },
        },
        "pressure": {"neumann": ("periodic", "periodic", "neumann")},
    },
    "rectangular_channel": {
        "Q": {"neumann": ("periodic", "neumann", "neumann")},
        "velocity": {
            "no_slip_velocity": dict.fromkeys(
                _COMPONENTS,
                (
                    "periodic",
                    "dirichlet",
                    "dirichlet",
                ),
            )
        },
        "pressure": {"neumann": ("periodic", "neumann", "neumann")},
    },
}
_CAPTURES = {
    "periodic_box": "pssolver.runtime.observations:capture_periodic_observation",
    "plane_slab": "pssolver.runtime.plane_observations:capture_plane_observation",
    "rectangular_channel": "pssolver.runtime.channel_observations:capture_channel_observation",
}
_EXECUTION_DEFAULTS = {
    "disable_q_gradient_reuse": False,
    "molecular_field_linear_space": "spectral",
    "stress_divergence_sum_space": "spectral",
    "pointwise_execution": "eager",
    "tf32": "off",
}


@dataclass(frozen=True, slots=True)
class EngineConstruction:
    """Execution inputs kept outside Plan identity until run time."""

    device: str
    dtype: str
    tf32: str
    batch: int = 1
    plane_preset: ShendrukPlanePreset | None = None
    refresh: SpectralRefreshSpec | None = None


def boundary_bases(geometry, boundaries):
    """Resolve public physical policies to the qualified per-component bases."""
    for policy in boundaries.values():
        if isinstance(policy, StaticRobinBoundaryPolicy):
            raise PublicSimulationCompilationError(
                Code.ROBIN_OR_ANCHORING,
                "Robin and anchoring await general boundary support",
                kind="scope",
                source="PROJECT P4",
                opens_in="0.6",
            )
        if isinstance(policy, StaticPrescribedDirichletPolicy):
            raise PublicSimulationCompilationError(
                Code.NONZERO_DIRICHLET_LIFTING,
                "nonzero Dirichlet needs lifting",
                kind="scope",
                source="B8/S2-07, D-0007",
                opens_in="0.6",
            )
    if geometry.name == "periodic_box" and not boundaries:
        return dict.fromkeys(_FIELDS, ("periodic",) * 3)
    if set(boundaries) != set(_FIELDS):
        raise PublicSimulationCompilationError(
            Code.INCOMPATIBLE_BOUNDARIES,
            "every field requires its declared boundary basis",
        )
    kinds = _BASIS[geometry.name]
    result = {}
    for name, policy in boundaries.items():
        if (
            not isinstance(policy, HomogeneousBoundaryPolicy)
            or policy.field_name != name
            or policy.kind not in kinds[name]
        ):
            raise PublicSimulationCompilationError(
                Code.INCOMPATIBLE_BOUNDARIES,
                f"{geometry.name} has no qualified basis for {name}: {policy.kind}",
            )
        result[name] = kinds[name][policy.kind]
    return result


ITERATIVE_PRESSURE_OPTION_KEYS = frozenset(
    {
        "algorithm",
        "fixed_iterations",
        "max_iterations",
        "relative_tolerance",
        "warm_start",
    }
)


def validate_iterative_pressure_options(
    value: object,
    *,
    description: str,
    required_algorithm: str = "preconditioned_conjugate_gradient",
    minimum_relative_tolerance: float | None = None,
) -> dict[str, object]:
    """Validate the common public iterative-pressure declaration."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{description} must be a mapping")
    observed = set(value)
    if observed != ITERATIVE_PRESSURE_OPTION_KEYS:
        raise ValueError(
            f"{description} must declare every qualified option explicitly; "
            f"missing={tuple(sorted(ITERATIVE_PRESSURE_OPTION_KEYS - observed))!r}, "
            f"extra={tuple(sorted(observed - ITERATIVE_PRESSURE_OPTION_KEYS))!r}"
        )
    pressure = dict(value)
    if pressure["algorithm"] != required_algorithm:
        raise ValueError(f"{description} algorithm must be {required_algorithm!r}")
    tolerance = pressure["relative_tolerance"]
    if (
        not isinstance(tolerance, Real)
        or isinstance(tolerance, bool)
        or not math.isfinite(float(tolerance))
        or float(tolerance) <= 0.0
    ):
        raise ValueError(f"{description} relative_tolerance must be finite and positive")
    maximum = pressure["max_iterations"]
    if not isinstance(maximum, Integral) or isinstance(maximum, bool) or int(maximum) <= 0:
        raise ValueError(f"{description} max_iterations must be a positive integer")
    fixed = pressure["fixed_iterations"]
    if fixed is not None:
        if not isinstance(fixed, Integral) or isinstance(fixed, bool) or int(fixed) <= 0:
            raise ValueError(f"{description} fixed_iterations must be null or a positive integer")
        if int(fixed) > int(maximum):
            raise ValueError(f"{description} fixed_iterations must not exceed max_iterations")
    if minimum_relative_tolerance is not None:
        if (
            not isinstance(minimum_relative_tolerance, Real)
            or isinstance(minimum_relative_tolerance, bool)
            or not math.isfinite(float(minimum_relative_tolerance))
            or float(minimum_relative_tolerance) <= 0.0
        ):
            raise ValueError("minimum_relative_tolerance must be finite and positive")
        if fixed is None and float(tolerance) < float(minimum_relative_tolerance):
            raise ValueError(
                f"{description} relative_tolerance is below the supported "
                f"tolerance-mode floor ({float(minimum_relative_tolerance)!r})"
            )
    if not isinstance(pressure["warm_start"], bool):
        raise ValueError(f"{description} warm_start must be a bool")
    return pressure


def _pressure_options(numerics, kind):
    if kind != "rectangular_channel":
        if numerics.pressure_solver is not None:
            raise PublicSimulationCompilationError(
                Code.UNUSED_OR_MISSING_EXECUTION_OPTION,
                "Channel pressure options require a Channel",
            )
        return None, {}
    description = "Channel pressure_solver"
    try:
        pressure = validate_iterative_pressure_options(
            numerics.pressure_solver, description=description
        )
    except ValueError as exc:
        raise PublicSimulationCompilationError(Code.DECLARATION_CONTRACT, str(exc)) from exc
    floor = channel_pressure_minimum_relative_tolerance_for_precision(numerics.dtype)
    if pressure["fixed_iterations"] is None and pressure["relative_tolerance"] < floor:
        raise PublicSimulationCompilationError(
            Code.PRESSURE_TOLERANCE_BELOW_FLOOR,
            f"{description} relative_tolerance is below the supported {numerics.dtype} "
            f"tolerance-mode floor ({floor!r})",
        )
    return "previous", {
        "pressure_relative_tolerance": pressure["relative_tolerance"],
        "pressure_max_iterations": pressure["max_iterations"],
        "pressure_fixed_iterations": pressure["fixed_iterations"],
    }


def _material(model):
    return {
        name: getattr(model, name)
        for name in (
            "ldg_a",
            "ldg_b",
            "ldg_c",
            "gamma",
            "flow_alignment",
            "beta",
        )
    }


def _plane_model(model, height, normal_basis):
    material = _material(model)
    if isinstance(model, ShendrukPlaneBerisEdwards):
        activity_number, frank_k = model.activity_number, model.frank_k
        parameterization = model.parameterization
        low, high = model.coefficient_min, model.coefficient_max
        viscosity, policy = model.eta, model.zero_mode_policy
        friction = model.friction_mode_fric if policy == "friction" else 0.0
    else:
        if model.tangential_zero_mode_policy != "zero_mean":
            raise PublicSimulationCompilationError(
                Code.DECLARATION_CONTRACT,
                "the public model currently connects only zero-mean tangential flow",
            )
        if model.activity <= 0:
            raise PublicSimulationCompilationError(
                Code.DECLARATION_CONTRACT,
                "the qualified Plane translation requires positive activity",
            )
        try:
            equilibrium_s = positive_equilibrium_S(model.ldg_a, model.ldg_b, model.ldg_c)
        except ValueError as exc:
            raise PublicSimulationCompilationError(
                Code.DECLARATION_CONTRACT,
                "the qualified Plane translation requires bulk coefficients "
                f"with a positive real uniaxial equilibrium: {exc}",
            ) from exc
        equilibrium_q = 1.5 * equilibrium_s
        frank_k = 2.0 * equilibrium_q**2 * model.ldg_l1
        activity_number = height * math.sqrt(model.activity / frank_k)
        parameterization, low, high = "fixed-k", 0.01, 0.05
        viscosity, friction, policy = model.viscosity, model.friction, "zero_mean"
    try:
        preset = resolve_shendruk_plane_preset(
            activity_number=activity_number,
            height=height,
            parameterization=parameterization,
            frank_k=frank_k,
            coefficient_min=low,
            coefficient_max=high,
            ldg_a=model.ldg_a,
            ldg_b=model.ldg_b,
            ldg_c=model.ldg_c,
            gamma=model.gamma,
        )
    except ValueError as exc:
        raise PublicSimulationCompilationError(Code.DECLARATION_CONTRACT, str(exc)) from exc
    material["gamma"] = preset.rotational_viscosity
    return Model(
        material=material,
        ldg_l1=preset.ldg_l1,
        stokes={
            "friction": friction,
            "viscosity": viscosity,
            "tangential_zero_mode_policy": policy,
        },
        alpha=Parameter(preset.zeta),
        distortion_odd_boundary_conditions=normal_basis,
    ), preset


def _validate_initial_condition(simulation, kind):
    initial = simulation.initial_condition
    if kind == "plane_slab":
        if (
            not isinstance(initial, GeneratedInitialCondition)
            or initial.family != "extruded_defect_gas"
        ):
            raise PublicSimulationCompilationError(
                Code.INITIAL_CONDITION_CONTRACT,
                "Plane requires generated extruded_defect_gas initial data",
            )
        p = initial.parameters
        nz = simulation.geometry.domain.shape[2]
        modes = tuple(p.get("twist_modes", ()))
        for bad, text in (
            (p.get("initial_s", 1) <= 0, "--initial-s must be positive"),
            (p.get("num_defect_pairs", 1) <= 0, "--num-defect-pairs must be positive"),
            (p.get("defect_min_separation", 1) <= 0, "--defect-min-separation must be positive"),
            (p.get("defect_core_radius", 1) <= 0, "--defect-core-radius must be positive"),
            (p.get("twist_amplitude", 0) < 0, "--twist-amplitude must be non-negative"),
            (len(set(modes)) != len(modes), "--twist-modes must be unique"),
            (any(m <= 0 or m >= nz for m in modes), "--twist-modes must satisfy 1 <= mode < nz"),
        ):
            if bad:
                raise PublicSimulationCompilationError(Code.INITIAL_CONDITION_CONTRACT, text)
        return
    stage = "P8.2" if kind == "periodic_box" else "P8.3"
    if not isinstance(initial, SnapshotInitialCondition):
        raise PublicSimulationCompilationError(
            Code.INITIAL_CONDITION_CONTRACT,
            f"{stage} requires an external Q snapshot",
        )
    if initial.mode != "branch":
        raise PublicSimulationCompilationError(
            Code.INITIAL_CONDITION_CONTRACT,
            f"{stage} accepts snapshot mode='branch'; restart uses checkpoints",
        )
    try:
        validate_snapshot_header(
            initial.directory, initial.step, simulation.geometry.domain.shape, 5
        )
    except ValueError as exc:
        raise PublicSimulationCompilationError(Code.INITIAL_CONDITION_CONTRACT, str(exc)) from exc


def lower_simulation(simulation) -> tuple[AssemblyProblem, EngineConstruction]:
    """Prepare the one assembly request and the independent run-time settings."""
    geometry = simulation.geometry
    domain = geometry.domain
    kind = geometry.name
    time = simulation.time
    numerical = simulation.numerics
    execution = simulation.execution
    _validate_initial_condition(simulation, kind)
    bc = boundary_bases(geometry, simulation.boundaries)
    unknown = set(execution.options) - set(_EXECUTION_DEFAULTS)
    if unknown:
        raise PublicSimulationCompilationError(
            Code.DECLARATION_CONTRACT,
            f"unsupported execution options: {tuple(sorted(unknown))!r}",
        )
    options = _EXECUTION_DEFAULTS | dict(execution.options)
    if options["tf32"] not in ("off", "on"):
        raise PublicSimulationCompilationError(
            Code.DECLARATION_CONTRACT, "TF32 policy must be 'off' or 'on'"
        )
    if type(options["disable_q_gradient_reuse"]) is not bool:
        raise PublicSimulationCompilationError(
            Code.DECLARATION_CONTRACT, "disable_q_gradient_reuse must be a bool"
        )
    model = simulation.model
    plane_preset = None
    if kind == "plane_slab":
        model, plane_preset = _plane_model(model, domain.lengths[2], bc["velocity"]["uz"])
    elif isinstance(model, CompleteStressBerisEdwards):
        model = Model(
            material=_material(model),
            ldg_l1=model.ldg_l1,
            stokes={
                "friction": model.friction,
                "viscosity": model.viscosity,
                "tangential_zero_mode_policy": model.tangential_zero_mode_policy,
            },
            alpha=Parameter(model.activity),
        )
    else:
        raise PublicSimulationCompilationError(
            Code.DECLARATION_CONTRACT,
            "ShendrukPlaneBerisEdwards requires PlaneSlab",
        )
    refresh = {} if time.refresh is None else time.refresh
    try:
        resolved_refresh = _resolve_spectral_refresh(
            dt=time.dt,
            spectral_refresh_time=refresh.get("physical_time"),
            spectral_refresh_steps=refresh.get("steps"),
            disable_spectral_refresh=time.refresh is None,
        )
    except ValueError as exc:
        raise PublicSimulationCompilationError(Code.DECLARATION_CONTRACT, str(exc)) from exc
    warm_start, pressure = _pressure_options(numerical, kind)
    problem = AssemblyProblem(
        model,
        GeometryInput(kind, tuple(domain.shape), tuple(domain.lengths)),
        bc,
        EulerInput(time.dt, resolved_refresh.effective_interval_steps),
        NumericsInput(
            numerical.spectral_storage.value,
            numerical.hermitian_axis,
            numerical.transform_execution_order.value,
            numerical.dealias_rule.value,
            numerical.projected_transform_execution.value,
        ),
        ExecutionInput(
            options["pointwise_execution"],
            not options["disable_q_gradient_reuse"],
            kind == "periodic_box" or simulation.output.diagnostics,
            {
                key: options[key]
                for key in ("molecular_field_linear_space", "stress_divergence_sum_space")
            },
            constraint_warm_start=warm_start,
            channel_pressure_options=pressure,
        ),
        OutputInput(_FIELDS, _CAPTURES[kind]),
    )
    return problem, EngineConstruction(
        execution.device,
        numerical.dtype,
        options["tf32"],
        plane_preset=plane_preset,
        refresh=resolved_refresh,
    )


__all__ = ["boundary_bases", "lower_simulation"]
