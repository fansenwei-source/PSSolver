"""Fail-closed compilation of the public :class:`Simulation` declaration.

The public API is not a hard-coded model/geometry switch.  A small immutable
registry selects a qualified application adapter from the equation variant and
geometry identity.  P7.7.9 contains the two combinations already qualified by
the migration (Plane complete-stress and Channel active-force); future
combinations extend the registry without changing :class:`Simulation`.

Selection and translation happen before allocation.  An unregistered pair is
reported as a structured capability gap rather than being coerced to a nearby
application or discovered by failure in the timestep.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import json
import math
from types import MappingProxyType

from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_plane_beris_edwards_simulation,
)
from pssolver.configuration.package_construction import (
    plan_package_runtime_construction,
)
from pssolver.configuration.plane_beris_edwards import (
    PlaneBerisEdwardsRunSpec,
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_component_graph import (
    build_plane_beris_edwards_run_components,
)
from pssolver.configuration.simulation import (
    InitialConditionSource,
    SimulationSpec,
)
from pssolver.configuration.simulation_lowering import lower_simulation_spec
from pssolver.core.integrators import IntegratorScheme
from pssolver.models.active_nematics.q_tensor import positive_equilibrium_S


PUBLIC_PLANE_APPLICATION = "plane_complete_stress_beris_edwards"
PUBLIC_CHANNEL_APPLICATION = "channel_legacy_active_force_active_nematics"
PUBLIC_PERIODIC_APPLICATION = "periodic_complete_stress_beris_edwards"
PUBLIC_CHANNEL_COMPLETE_APPLICATION = "channel_complete_stress_beris_edwards"

_PLANE_COMPILER_KEY = ("complete_stress_beris_edwards", "plane_slab")
_CHANNEL_COMPILER_KEY = (
    "legacy_active_force_active_nematics",
    "rectangular_channel",
)
_PERIODIC_COMPILER_KEY = (
    "complete_stress_beris_edwards",
    "periodic_box",
)
_CHANNEL_COMPLETE_COMPILER_KEY = (
    "complete_stress_beris_edwards",
    "rectangular_channel",
)

_INITIAL_KEYS = frozenset(
    {
        "background_angle",
        "defect_core_radius",
        "defect_min_separation",
        "initial_s",
        "num_defect_pairs",
        "seed",
        "twist_amplitude",
        "twist_modes",
    }
)
_EXECUTION_KEYS = frozenset(
    {
        "device",
        "disable_q_gradient_reuse",
        "fallback_allowed",
        "molecular_field_linear_space",
        "pointwise_execution",
        "stress_divergence_sum_space",
        "tf32",
    }
)
_WORKFLOW_KEYS = frozenset(
    {
        "checkpoint_interval",
        "diagnostic_interval",
        "diagnostics",
        "output_dir",
        "restart_from",
        "save_hydrodynamics",
        "save_interval",
        "save_start_step",
    }
)
_INVOCATION_KEYS = frozenset({"dry_run", "validation_config_sha256"})


class PublicCompilationRejectionCode(str, Enum):
    """Stable reason a public declaration cannot become an application."""

    UNREGISTERED_MODEL_GEOMETRY = "unregistered_model_geometry"
    APPLICATION_CONTRACT = "application_contract"


class PublicSimulationCompilationError(ValueError):
    """Raised before allocation when the public request is not qualified."""

    def __init__(
        self,
        message: str,
        *,
        code: PublicCompilationRejectionCode = (
            PublicCompilationRejectionCode.APPLICATION_CONTRACT
        ),
        context: Mapping[str, object] | None = None,
    ) -> None:
        self.code = code
        self.context = _json_mapping({} if context is None else context)
        super().__init__(f"{code.value}: {message}")

    def to_metadata(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "message": str(self).split(": ", 1)[-1],
            "context": dict(self.context),
        }


def _reject(message: str) -> None:
    raise PublicSimulationCompilationError(message)


def _json_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    payload = json.dumps(
        dict(value),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return MappingProxyType(json.loads(payload))


def _require_exact_keys(
    value: Mapping[str, object],
    expected: frozenset[str],
    description: str,
) -> dict[str, object]:
    observed = set(value)
    if observed != expected:
        missing = tuple(sorted(expected - observed))
        extra = tuple(sorted(observed - expected))
        _reject(
            f"{description} must declare every qualified option explicitly; "
            f"missing={missing!r}, extra={extra!r}"
        )
    return dict(value)


def _require_close(
    requested: float,
    effective: float,
    description: str,
) -> None:
    if not math.isclose(
        float(requested),
        float(effective),
        rel_tol=2.0e-14,
        abs_tol=2.0e-15,
    ):
        _reject(
            f"{description} changed during the qualified Plane translation: "
            f"requested={requested!r}, effective={effective!r}"
        )


def _refresh_arguments(refresh: Mapping[str, object]) -> dict[str, object]:
    values = dict(refresh)
    mode = values.pop("mode", None)
    if mode == "disabled" and not values:
        return {"disable_spectral_refresh": True}
    if mode == "steps" and set(values) == {"interval_steps"}:
        return {"spectral_refresh_steps": values["interval_steps"]}
    if mode == "physical_time" and set(values) == {"interval_time"}:
        return {"spectral_refresh_time": values["interval_time"]}
    _reject(
        "time refresh must be exactly one of "
        "{'mode': 'disabled'}, "
        "{'mode': 'steps', 'interval_steps': N}, or "
        "{'mode': 'physical_time', 'interval_time': T}"
    )


def _boundary_identity(specification: SimulationSpec) -> dict[str, object]:
    metadata = specification.boundaries.to_metadata()
    return {
        "ndim": metadata["ndim"],
        "components": metadata["components"],
    }


def _validate_translation(
    source: SimulationSpec,
    application: SimulationSpec,
    run_spec: PlaneBerisEdwardsRunSpec,
) -> None:
    if source.geometry != application.geometry:
        _reject("public geometry changed during Plane translation")
    if _boundary_identity(source) != _boundary_identity(application):
        _reject("public boundary laws changed during Plane translation")
    if source.numerics.to_metadata() != application.numerics.to_metadata():
        _reject("public numerical policy changed during Plane translation")
    if source.time_integration.integrator != application.time_integration.integrator:
        _reject("public integrator changed during Plane translation")
    if (
        source.initial_condition.to_metadata()
        != application.initial_condition.to_metadata()
    ):
        _reject("public initial-condition identity changed during Plane translation")
    if source.execution.backend != application.execution.backend:
        _reject("public backend changed during Plane translation")
    if source.execution.runtime_path != application.execution.runtime_path:
        _reject("public runtime path changed during Plane translation")
    source_execution = dict(source.execution.options)
    source_execution.pop("fallback_allowed")
    if source_execution != dict(application.execution.options):
        _reject("public execution controls changed during Plane translation")
    if source.workflow.to_metadata() != application.workflow.to_metadata():
        _reject("public workflow changed during Plane translation")

    parameters = source.equation_system.parameters
    material = parameters["material"]
    preset = run_spec.shendruk_preset
    for name in ("ldg_a", "ldg_b", "ldg_c", "gamma"):
        attribute = name if name != "gamma" else "rotational_viscosity"
        _require_close(material[name], getattr(preset, attribute), name)
    _require_close(parameters["ldg_l1"], preset.ldg_l1, "ldg_l1")
    _require_close(
        parameters["activity_amplitude"],
        preset.zeta,
        "activity amplitude",
    )


@dataclass(frozen=True, slots=True)
class PublicSimulationCompilation:
    """Internal typed product consumed by the public API runner."""

    application: str
    source_simulation_sha256: str
    application_simulation: SimulationSpec
    lowering_plan: object
    construction_plan: object
    run_spec: object
    normalization: Mapping[str, object]

    def __post_init__(self) -> None:
        if self.application not in {
            PUBLIC_PLANE_APPLICATION,
            PUBLIC_CHANNEL_APPLICATION,
            PUBLIC_PERIODIC_APPLICATION,
            PUBLIC_CHANNEL_COMPLETE_APPLICATION,
        }:
            raise ValueError("unsupported public application identity")
        if not callable(getattr(self.run_spec, "canonical_sha256", None)):
            raise TypeError("public application request must have stable identity")
        object.__setattr__(self, "normalization", _json_mapping(self.normalization))


def _components_from_run_spec(run_spec: PlaneBerisEdwardsRunSpec):
    """Construct canonical components without the private facade adapter."""

    return build_plane_beris_edwards_run_components(
        activity_number=run_spec.activity_number,
        output_dir=run_spec.output_dir,
        height=run_spec.height,
        parameterization=run_spec.parameterization,
        frank_k=run_spec.frank_k,
        coefficient_min=run_spec.coefficient_min,
        coefficient_max=run_spec.coefficient_max,
        lx=run_spec.lx,
        ly=run_spec.ly,
        nx=run_spec.nx,
        ny=run_spec.ny,
        nz=run_spec.nz,
        dt=run_spec.dt,
        steps=run_spec.steps,
        save_start_step=run_spec.save_start_step,
        save_interval=run_spec.save_interval,
        diagnostic_interval=run_spec.diagnostic_interval,
        seed=run_spec.seed,
        num_defect_pairs=run_spec.num_defect_pairs,
        defect_min_separation=run_spec.defect_min_separation,
        defect_core_radius=run_spec.defect_core_radius,
        background_angle=run_spec.background_angle,
        twist_amplitude=run_spec.twist_amplitude,
        twist_modes=run_spec.twist_modes,
        ldg_a=run_spec.ldg_a,
        ldg_b=run_spec.ldg_b,
        ldg_c=run_spec.ldg_c,
        gamma=run_spec.gamma,
        flow_alignment=run_spec.flow_alignment,
        eta=run_spec.eta,
        zero_mode_policy=run_spec.zero_mode_policy,
        friction_mode_fric=run_spec.friction_mode_fric,
        dealias_rule=run_spec.dealias_rule,
        projected_transform_execution=run_spec.projected_transform_execution,
        beta=run_spec.beta,
        initial_s=run_spec.initial_s,
        device=run_spec.device,
        dtype=run_spec.dtype,
        molecular_field_linear_space=run_spec.molecular_field_linear_space,
        stress_divergence_sum_space=run_spec.stress_divergence_sum_space,
        pointwise_execution=run_spec.pointwise_execution,
        transform_execution_order=run_spec.transform_execution_order,
        spectral_storage=run_spec.spectral_storage,
        tf32=run_spec.tf32,
        spectral_refresh=run_spec.spectral_refresh,
        diagnostics=run_spec.diagnostics,
        disable_q_gradient_reuse=run_spec.disable_q_gradient_reuse,
        save_hydrodynamics=run_spec.save_hydrodynamics,
        validation_config_sha256=run_spec.validation_config_sha256,
        dry_run=run_spec.dry_run,
        checkpoint_interval=run_spec.checkpoint_interval,
        restart_from=run_spec.restart_from,
        runtime_path=run_spec.runtime_path,
        boundaries=run_spec.boundaries,
        hermitian_axis=1,
    )


def _compile_plane_public_simulation(
    source: SimulationSpec,
) -> PublicSimulationCompilation:
    """Compile one public declaration into the qualified Plane application.

    No tensors, runtime objects, output directories, or application modules
    are constructed here.  Unsupported or incomplete declarations fail before
    allocation and are never mapped to a nearby runtime.
    """

    if (
        source.equation_system.variant,
        source.geometry.name,
    ) != _PLANE_COMPILER_KEY:
        raise AssertionError("Plane compiler received the wrong registry key")
    if source.execution.backend != "torch_spectral":
        _reject("only the torch_spectral backend is connected")
    if source.execution.runtime_path not in {"legacy_production", "compiled_v2"}:
        _reject("public Plane execution supports legacy_production or compiled_v2")
    if source.time_integration.integrator.scheme is not (
        IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
    ):
        _reject("the public Plane runner currently requires projected Euler")
    if source.initial_condition.source is not InitialConditionSource.GENERATED:
        _reject("the connected Plane application currently requires a generated initial condition")
    if source.initial_condition.family != "extruded_defect_gas":
        _reject("the connected Plane application requires extruded_defect_gas")
    if source.discretization_parameters:
        _reject("the connected Plane application accepts no public discretization overrides")

    initial = _require_exact_keys(
        source.initial_condition.parameters,
        _INITIAL_KEYS,
        "extruded_defect_gas parameters",
    )
    execution = _require_exact_keys(
        source.execution.options,
        _EXECUTION_KEYS,
        "torch_spectral execution options",
    )
    if execution["fallback_allowed"] is not False:
        _reject("runtime fallback must remain disabled")
    workflow = _require_exact_keys(
        source.workflow.options,
        _WORKFLOW_KEYS,
        "output workflow options",
    )
    invocation = dict(source.invocation.options)
    if not set(invocation) <= _INVOCATION_KEYS:
        _reject(
            "invocation options may contain only dry_run and "
            "validation_config_sha256"
        )

    parameters = source.equation_system.parameters
    material = dict(parameters["material"])
    stokes = dict(parameters["stokes"]["parameters"])
    if stokes["pressure_gauge"] != "zero_mean":
        _reject("the qualified Plane runner requires the zero-mean pressure gauge")
    if stokes["tangential_zero_mode_policy"] != "zero_mean":
        _reject("the public model currently connects only zero-mean tangential flow")
    if stokes["friction"] != 0.0:
        _reject("zero-mean public Plane execution requires zero friction")

    activity = float(parameters["activity_amplitude"])
    if not math.isfinite(activity) or activity <= 0.0:
        _reject("the qualified Plane translation requires positive activity")
    ldg_l1 = float(parameters["ldg_l1"])
    equilibrium_s = positive_equilibrium_S(
        material["ldg_a"],
        material["ldg_b"],
        material["ldg_c"],
    )
    equilibrium_q = 1.5 * equilibrium_s
    frank_k = 2.0 * equilibrium_q**2 * ldg_l1
    height = source.geometry.domain.lengths[2]
    activity_number = height * math.sqrt(activity / frank_k)
    refresh = _refresh_arguments(source.time_integration.refresh)
    domain = source.geometry.domain
    numerics = source.numerics

    run_spec = create_plane_beris_edwards_run_spec(
        activity_number=activity_number,
        output_dir=workflow["output_dir"],
        height=height,
        parameterization="fixed-k",
        frank_k=frank_k,
        lx=domain.lengths[0],
        ly=domain.lengths[1],
        nx=domain.shape[0],
        ny=domain.shape[1],
        nz=domain.shape[2],
        dt=source.time_integration.integrator.dt,
        steps=source.workflow.steps,
        save_start_step=workflow["save_start_step"],
        save_interval=workflow["save_interval"],
        diagnostic_interval=workflow["diagnostic_interval"],
        seed=initial["seed"],
        num_defect_pairs=initial["num_defect_pairs"],
        defect_min_separation=initial["defect_min_separation"],
        defect_core_radius=initial["defect_core_radius"],
        background_angle=initial["background_angle"],
        twist_amplitude=initial["twist_amplitude"],
        twist_modes=initial["twist_modes"],
        ldg_a=material["ldg_a"],
        ldg_b=material["ldg_b"],
        ldg_c=material["ldg_c"],
        gamma=material["gamma"],
        flow_alignment=material["flow_alignment"],
        eta=stokes["viscosity"],
        zero_mode_policy="zero_mean",
        dealias_rule=numerics.dealias_rule.value,
        projected_transform_execution=numerics.projected_transform_execution.value,
        beta=material["beta"],
        initial_s=initial["initial_s"],
        device=execution["device"],
        dtype=numerics.precision.value,
        molecular_field_linear_space=execution["molecular_field_linear_space"],
        stress_divergence_sum_space=execution["stress_divergence_sum_space"],
        pointwise_execution=execution["pointwise_execution"],
        transform_execution_order=numerics.transform_execution_order.value,
        spectral_storage=numerics.spectral_storage.value,
        tf32=execution["tf32"],
        diagnostics=workflow["diagnostics"],
        disable_q_gradient_reuse=execution["disable_q_gradient_reuse"],
        save_hydrodynamics=workflow["save_hydrodynamics"],
        validation_config_sha256=invocation.get("validation_config_sha256"),
        dry_run=invocation.get("dry_run", False),
        checkpoint_interval=workflow["checkpoint_interval"],
        restart_from=workflow["restart_from"],
        runtime_path=source.execution.runtime_path,
        **refresh,
    )
    application_simulation = compose_plane_beris_edwards_simulation(
        _components_from_run_spec(run_spec)
    )
    _validate_translation(source, application_simulation, run_spec)
    lowering_plan = lower_simulation_spec(application_simulation)
    construction_plan = plan_package_runtime_construction(application_simulation)
    return PublicSimulationCompilation(
        application=PUBLIC_PLANE_APPLICATION,
        source_simulation_sha256=source.canonical_sha256(),
        application_simulation=application_simulation,
        lowering_plan=lowering_plan,
        construction_plan=construction_plan,
        run_spec=run_spec,
        normalization={
            "schema_version": 1,
            "coefficient_parameterization": "fixed-k",
            "requested_activity": activity,
            "effective_activity": run_spec.shendruk_preset.zeta,
            "requested_ldg_l1": ldg_l1,
            "effective_ldg_l1": run_spec.shendruk_preset.ldg_l1,
            "derived_activity_number": activity_number,
            "derived_frank_k": frank_k,
            "runtime_fallback_allowed": False,
        },
    )


@dataclass(frozen=True, slots=True)
class PublicApplicationCompilerRegistration:
    """Tensor-free identity of one qualified public compiler adapter."""

    equation_variant: str
    geometry_name: str
    application: str
    adapter: str

    def to_metadata(self) -> dict[str, str]:
        return {
            "equation_variant": self.equation_variant,
            "geometry_name": self.geometry_name,
            "application": self.application,
            "adapter": self.adapter,
        }


_PUBLIC_COMPILER_REGISTRY = MappingProxyType(
    {
        _PLANE_COMPILER_KEY: PublicApplicationCompilerRegistration(
            *_PLANE_COMPILER_KEY,
            PUBLIC_PLANE_APPLICATION,
            (
                "pssolver.configuration.public_simulation_runner."
                "_compile_plane_public_simulation"
            ),
        ),
        _CHANNEL_COMPILER_KEY: PublicApplicationCompilerRegistration(
            *_CHANNEL_COMPILER_KEY,
            PUBLIC_CHANNEL_APPLICATION,
            (
                "pssolver.configuration.public_channel_simulation_compiler."
                "compile_channel_public_simulation"
            ),
        ),
        _PERIODIC_COMPILER_KEY: PublicApplicationCompilerRegistration(
            *_PERIODIC_COMPILER_KEY,
            PUBLIC_PERIODIC_APPLICATION,
            (
                "pssolver.configuration.public_periodic_simulation_compiler."
                "compile_periodic_public_simulation"
            ),
        ),
        _CHANNEL_COMPLETE_COMPILER_KEY: PublicApplicationCompilerRegistration(
            *_CHANNEL_COMPLETE_COMPILER_KEY,
            PUBLIC_CHANNEL_COMPLETE_APPLICATION,
            (
                "pssolver.configuration."
                "public_channel_beris_edwards_compiler."
                "compile_channel_beris_edwards_public_simulation"
            ),
        ),
    }
)


def public_compiler_capabilities() -> tuple[dict[str, str], ...]:
    """Return the immutable set of currently qualified compiler adapters."""

    return tuple(
        registration.to_metadata()
        for registration in _PUBLIC_COMPILER_REGISTRY.values()
    )


def compile_public_simulation(
    source: SimulationSpec,
) -> PublicSimulationCompilation:
    """Compile through a qualified model/geometry application adapter.

    The registry is deliberately finite because an executable combination
    needs a lowering, runtime binding, and scientific qualification.  Its key
    is not embedded in the public ``Simulation`` shape, so adding a qualified
    combination does not change that API or add dispatch to the timestep.
    """

    if not isinstance(source, SimulationSpec):
        raise TypeError("source must be a SimulationSpec")
    key = (source.equation_system.variant, source.geometry.name)
    registration = _PUBLIC_COMPILER_REGISTRY.get(key)
    if registration is None:
        raise PublicSimulationCompilationError(
            "the model/geometry pair has no qualified public compiler adapter",
            code=(
                PublicCompilationRejectionCode.UNREGISTERED_MODEL_GEOMETRY
            ),
            context={
                "equation_variant": key[0],
                "geometry_name": key[1],
                "required_capabilities": list(
                    source.equation_system.required_capabilities
                ),
                "registered_pairs": [
                    [candidate[0], candidate[1]]
                    for candidate in _PUBLIC_COMPILER_REGISTRY
                ],
            },
        )
    if registration.application == PUBLIC_PLANE_APPLICATION:
        return _compile_plane_public_simulation(source)
    if registration.application == PUBLIC_CHANNEL_APPLICATION:
        from .public_channel_simulation_compiler import (
            compile_channel_public_simulation,
        )

        return compile_channel_public_simulation(source)
    if registration.application == PUBLIC_PERIODIC_APPLICATION:
        from .public_periodic_simulation_compiler import (
            compile_periodic_public_simulation,
        )

        return compile_periodic_public_simulation(
            source,
            PublicSimulationCompilation,
        )
    if registration.application == PUBLIC_CHANNEL_COMPLETE_APPLICATION:
        from .public_channel_beris_edwards_compiler import (
            compile_channel_beris_edwards_public_simulation,
        )

        return compile_channel_beris_edwards_public_simulation(
            source,
            PublicSimulationCompilation,
        )
    raise AssertionError("registered public compiler was not dispatched")


__all__ = [
    "PUBLIC_CHANNEL_APPLICATION",
    "PUBLIC_PLANE_APPLICATION",
    "PUBLIC_PERIODIC_APPLICATION",
    "PUBLIC_CHANNEL_COMPLETE_APPLICATION",
    "PublicApplicationCompilerRegistration",
    "PublicCompilationRejectionCode",
    "PublicSimulationCompilation",
    "PublicSimulationCompilationError",
    "compile_public_simulation",
    "public_compiler_capabilities",
]
