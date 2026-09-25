"""Fail-closed P7.7.3 lowering of qualified simulation declarations.

Only the two model/geometry/boundary combinations already qualified by the
Plane and Channel migration are admitted.  Resolution happens entirely at
construction time and produces immutable requirements; this module imports no
Torch, backend, operator, concrete solver, runtime, workflow, or application.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import json
from typing import NoReturn

from pssolver.core.boundary import (
    BoundaryKind,
    BoundarySemantic,
    BoundarySide,
)
from pssolver.core.fields import FieldRole
from pssolver.core.geometry import AxisTopology
from pssolver.core.integrators import IntegratorScheme
from pssolver.core.numerics import SpectralStorage
from pssolver.planning.plan import TransformKind
from pssolver.planning.simulation import (
    AxisBasisRequirement,
    BasisProvenance,
    CapabilityImplementationRequirement,
    ComponentBasisRequirement,
    DerivativeMultiplier,
    DerivativeRequirement,
    GeometrySolverRequirement,
    NullspaceRequirement,
    SimulationLoweringPlan,
)
from pssolver.systems.stokes import (
    INCOMPRESSIBLE_STOKES_CAPABILITY,
    IncompressibleStokesSystemSpec,
    PressureGauge,
    TangentialZeroModePolicy,
)

from .simulation import SimulationSpec


_BASIS_BY_MODAL_KIND = {
    BoundaryKind.PERIODIC: TransformKind.FFT,
    BoundaryKind.NEUMANN: TransformKind.DCT,
    BoundaryKind.DIRICHLET: TransformKind.DST,
}
_MODAL_KIND_BY_BASIS = {
    value: key for key, value in _BASIS_BY_MODAL_KIND.items()
}
_DERIVATIVE_BY_BASIS = {
    TransformKind.FFT: (
        TransformKind.FFT,
        DerivativeMultiplier.FOURIER_IK,
    ),
    TransformKind.DCT: (
        TransformKind.DST,
        DerivativeMultiplier.COSINE_TO_NEGATIVE_SINE,
    ),
    TransformKind.DST: (
        TransformKind.DCT,
        DerivativeMultiplier.SINE_TO_POSITIVE_COSINE,
    ),
}

_PLANE_VARIANT = "complete_stress_beris_edwards"
_CHANNEL_VARIANT = "legacy_active_force_active_nematics"
_PLANE_GEOMETRY = "plane_slab"
_PERIODIC_GEOMETRY = "periodic_box"
_CHANNEL_GEOMETRY = "rectangular_channel"

_PLANE_CAPABILITIES = (
    "beris_edwards_q_evolution",
    "beris_edwards_q_gradient",
    "beris_edwards_velocity_gradient",
    "beris_edwards_molecular_field",
    "beris_edwards_stress",
    "beris_edwards_force",
    INCOMPRESSIBLE_STOKES_CAPABILITY,
)
_CHANNEL_CAPABILITIES = (
    "legacy_active_force_q_evolution",
    "beris_edwards_q_gradient",
    "beris_edwards_velocity_gradient",
    "legacy_active_force",
    INCOMPRESSIBLE_STOKES_CAPABILITY,
)


class LoweringRejectionCode(str, Enum):
    """Machine-readable reason that a simulation cannot be lowered."""

    UNSUPPORTED_MODEL_GEOMETRY = "unsupported_model_geometry"
    UNSUPPORTED_DIMENSION_OR_TOPOLOGY = "unsupported_dimension_or_topology"
    UNSUPPORTED_CAPABILITY_SET = "unsupported_capability_set"
    UNSUPPORTED_INTEGRATOR = "unsupported_integrator"
    UNSUPPORTED_BACKEND = "unsupported_backend"
    INVALID_HERMITIAN_AXIS = "invalid_hermitian_axis"
    ASYMMETRIC_FACE_PAIR = "asymmetric_face_pair"
    UNSUPPORTED_BOUNDARY_SIGNATURE = "unsupported_boundary_signature"
    UNSUPPORTED_BOUNDARY_SEMANTIC = "unsupported_boundary_semantic"
    INVALID_EQUATION_LAYOUT = "invalid_equation_layout"
    INVALID_STOKES_CONTRACT = "invalid_stokes_contract"
    INVALID_DERIVED_SPACE = "invalid_derived_space"
    UNSUPPORTED_NULLSPACE_POLICY = "unsupported_nullspace_policy"
    UNSUPPORTED_SOLVER_OPTIONS = "unsupported_solver_options"


@dataclass(frozen=True, slots=True)
class LoweringRejection:
    """Structured construction-time rejection with JSON-compatible context."""

    code: LoweringRejectionCode
    message: str
    context_json: str = "{}"

    def __post_init__(self) -> None:
        if not isinstance(self.code, LoweringRejectionCode):
            raise TypeError("rejection code must be a LoweringRejectionCode")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("rejection message must be a non-empty string")
        try:
            context = json.loads(self.context_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("rejection context must be valid JSON") from exc
        canonical = json.dumps(
            context,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        if canonical != self.context_json:
            raise ValueError("rejection context must use canonical JSON")

    def to_metadata(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "message": self.message,
            "context": json.loads(self.context_json),
        }


class SimulationLoweringError(ValueError):
    """Raised before allocation when no qualified lowering exists."""

    def __init__(self, rejection: LoweringRejection) -> None:
        if not isinstance(rejection, LoweringRejection):
            raise TypeError("rejection must be a LoweringRejection")
        self.rejection = rejection
        super().__init__(f"{rejection.code.value}: {rejection.message}")


def _reject(
    code: LoweringRejectionCode,
    message: str,
    **context: object,
) -> NoReturn:
    context_json = json.dumps(
        context,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    raise SimulationLoweringError(
        LoweringRejection(code, message, context_json)
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _field_lookup(simulation: SimulationSpec):
    return {
        component: (field_spec.name, field_spec.role)
        for field_spec in simulation.equation_system.fields
        for component in field_spec.components
    }


def _field_components(
    simulation: SimulationSpec,
    field_name: str,
) -> tuple[str, ...]:
    for field_spec in simulation.equation_system.fields:
        if field_spec.name == field_name:
            return field_spec.components
    _reject(
        LoweringRejectionCode.INVALID_EQUATION_LAYOUT,
        "required logical field is absent",
        field_name=field_name,
    )


def _term_by_capability(simulation: SimulationSpec, capability: str):
    terms = (
        *simulation.equation_system.evolution_laws,
        *simulation.equation_system.constitutive_laws,
        *simulation.equation_system.algebraic_systems,
    )
    matches = tuple(term for term in terms if term.capability == capability)
    if len(matches) != 1:
        _reject(
            LoweringRejectionCode.INVALID_EQUATION_LAYOUT,
            "a supported capability must have exactly one producer",
            capability=capability,
            producer_count=len(matches),
        )
    return matches[0]


def _validate_global_contract(
    simulation: SimulationSpec,
) -> tuple[str, tuple[str, ...]]:
    if not isinstance(simulation, SimulationSpec):
        raise TypeError("simulation must be a SimulationSpec")
    variant = simulation.equation_system.variant
    geometry = simulation.geometry.name
    pair = (variant, geometry)
    if pair == (_PLANE_VARIANT, _PLANE_GEOMETRY):
        kind = "plane"
        expected_topology = (
            AxisTopology.PERIODIC,
            AxisTopology.PERIODIC,
            AxisTopology.BOUNDED,
        )
        expected_capabilities = _PLANE_CAPABILITIES
    elif pair == (_PLANE_VARIANT, _PERIODIC_GEOMETRY):
        kind = "periodic"
        expected_topology = (
            AxisTopology.PERIODIC,
            AxisTopology.PERIODIC,
            AxisTopology.PERIODIC,
        )
        expected_capabilities = _PLANE_CAPABILITIES
    elif pair == (_CHANNEL_VARIANT, _CHANNEL_GEOMETRY):
        kind = "channel"
        expected_topology = (
            AxisTopology.PERIODIC,
            AxisTopology.BOUNDED,
            AxisTopology.BOUNDED,
        )
        expected_capabilities = _CHANNEL_CAPABILITIES
    else:
        _reject(
            LoweringRejectionCode.UNSUPPORTED_MODEL_GEOMETRY,
            "the model/geometry pair has no qualified implementation",
            equation_variant=variant,
            geometry_name=geometry,
            supported_pairs=[
                [_PLANE_VARIANT, _PLANE_GEOMETRY],
                [_PLANE_VARIANT, _PERIODIC_GEOMETRY],
                [_CHANNEL_VARIANT, _CHANNEL_GEOMETRY],
            ],
        )
    observed_topology = simulation.geometry.axis_topologies
    if observed_topology != expected_topology:
        _reject(
            LoweringRejectionCode.UNSUPPORTED_DIMENSION_OR_TOPOLOGY,
            "the qualified solver requires its frozen three-dimensional topology",
            expected=[value.value for value in expected_topology],
            observed=[value.value for value in observed_topology],
        )
    observed_capabilities = simulation.equation_system.required_capabilities
    if observed_capabilities != expected_capabilities:
        _reject(
            LoweringRejectionCode.UNSUPPORTED_CAPABILITY_SET,
            "the equation capability set or order is not qualified",
            expected=list(expected_capabilities),
            observed=list(observed_capabilities),
        )
    integrator = simulation.time_integration.integrator
    if integrator.scheme is not IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER:
        _reject(
            LoweringRejectionCode.UNSUPPORTED_INTEGRATOR,
            "P7.7.3 lowers only the currently composed Euler path",
            observed=integrator.scheme.value,
        )
    if simulation.execution.backend != "torch_spectral":
        _reject(
            LoweringRejectionCode.UNSUPPORTED_BACKEND,
            "the qualified implementations require the torch_spectral backend",
            observed=simulation.execution.backend,
        )
    numerics = simulation.numerics
    if numerics.spectral_storage is SpectralStorage.HERMITIAN_HALF:
        axis = numerics.hermitian_axis
        if (
            axis is None
            or axis >= simulation.geometry.domain.ndim
            or simulation.geometry.axis_topologies[axis]
            is not AxisTopology.PERIODIC
        ):
            _reject(
                LoweringRejectionCode.INVALID_HERMITIAN_AXIS,
                "Hermitian packing must select an existing periodic axis",
                hermitian_axis=axis,
                periodic_axes=list(simulation.geometry.periodic_axes),
            )
    return kind, expected_capabilities


def _component_modal_kinds(
    simulation: SimulationSpec,
    component: str,
) -> tuple[BoundaryKind, ...]:
    assignment = simulation.boundaries.for_component(component)
    kinds = []
    for axis in range(simulation.geometry.domain.ndim):
        faces = {
            face.side: face.condition.kind
            for face in assignment.faces
            if face.axis == axis
        }
        lower = faces[BoundarySide.LOWER]
        upper = faces[BoundarySide.UPPER]
        if lower is not upper:
            _reject(
                LoweringRejectionCode.ASYMMETRIC_FACE_PAIR,
                "the current eigenbasis lowering requires equal laws on both faces",
                component=component,
                axis=axis,
                lower=lower.value,
                upper=upper.value,
            )
        kinds.append(lower)
    return tuple(kinds)


def _axis_requirements(
    simulation: SimulationSpec,
    modal_kinds: tuple[BoundaryKind, ...],
) -> tuple[AxisBasisRequirement, ...]:
    domain = simulation.geometry.domain
    numerics = simulation.numerics
    values = []
    for axis, (axis_name, topology, size, modal_kind) in enumerate(
        zip(
            domain.axis_names,
            simulation.geometry.axis_topologies,
            domain.shape,
            modal_kinds,
            strict=True,
        )
    ):
        packed = (
            numerics.spectral_storage is SpectralStorage.HERMITIAN_HALF
            and numerics.hermitian_axis == axis
        )
        values.append(
            AxisBasisRequirement(
                axis=axis,
                axis_name=axis_name,
                topology=topology,
                modal_kind=modal_kind,
                transform_kind=_BASIS_BY_MODAL_KIND[modal_kind],
                physical_size=size,
                spectral_size=size // 2 + 1 if packed else size,
                hermitian_packed=packed,
            )
        )
    return tuple(values)


def _primary_bases(
    simulation: SimulationSpec,
) -> dict[str, ComponentBasisRequirement]:
    field_lookup = _field_lookup(simulation)
    values = {}
    for assignment in simulation.boundaries.components:
        field_name, role = field_lookup[assignment.component]
        if assignment.semantic is BoundarySemantic.PHYSICAL:
            provenance = BasisProvenance.PHYSICAL_BOUNDARY
        elif assignment.semantic is BoundarySemantic.ALGEBRAIC_COMPATIBILITY:
            provenance = BasisProvenance.ALGEBRAIC_COMPATIBILITY
        else:
            _reject(
                LoweringRejectionCode.UNSUPPORTED_BOUNDARY_SEMANTIC,
                "the boundary semantic has no lowering rule",
                component=assignment.component,
                semantic=assignment.semantic.value,
            )
        values[assignment.component] = ComponentBasisRequirement(
            field_name=field_name,
            component=assignment.component,
            role=role,
            axes=_axis_requirements(
                simulation,
                _component_modal_kinds(simulation, assignment.component),
            ),
            provenance=provenance,
        )
    return values


def _basis_signature(
    value: ComponentBasisRequirement,
) -> tuple[TransformKind, ...]:
    return value.transform_kinds


def _require_signature(
    bases: dict[str, ComponentBasisRequirement],
    components: tuple[str, ...],
    expected: tuple[TransformKind, ...],
) -> None:
    for component in components:
        observed = _basis_signature(bases[component])
        if observed != expected:
            _reject(
                LoweringRejectionCode.UNSUPPORTED_BOUNDARY_SIGNATURE,
                "component basis does not match the qualified solver",
                component=component,
                expected=[value.value for value in expected],
                observed=[value.value for value in observed],
            )


def _validate_boundary_signatures(
    simulation: SimulationSpec,
    kind: str,
    bases: dict[str, ComponentBasisRequirement],
) -> None:
    q = _field_components(simulation, "Q")
    velocity = _field_components(simulation, "velocity")
    pressure = _field_components(simulation, "pressure")
    if pressure != ("p",) or velocity != ("ux", "uy", "uz"):
        _reject(
            LoweringRejectionCode.INVALID_EQUATION_LAYOUT,
            "qualified Stokes output component names changed",
            pressure=list(pressure),
            velocity=list(velocity),
        )
    if kind == "plane":
        even = (TransformKind.FFT, TransformKind.FFT, TransformKind.DCT)
        odd = (TransformKind.FFT, TransformKind.FFT, TransformKind.DST)
        _require_signature(bases, q, even)
        _require_signature(bases, ("ux", "uy", "p"), even)
        _require_signature(bases, ("uz",), odd)
    elif kind == "periodic":
        periodic = (TransformKind.FFT,) * 3
        _require_signature(bases, q, periodic)
        _require_signature(bases, (*velocity, "p"), periodic)
    else:
        q_pressure = (
            TransformKind.FFT,
            TransformKind.DCT,
            TransformKind.DCT,
        )
        velocity_basis = (
            TransformKind.FFT,
            TransformKind.DST,
            TransformKind.DST,
        )
        _require_signature(bases, q, q_pressure)
        _require_signature(bases, ("p",), q_pressure)
        _require_signature(bases, velocity, velocity_basis)
    if bases["p"].provenance is not BasisProvenance.ALGEBRAIC_COMPATIBILITY:
        _reject(
            LoweringRejectionCode.UNSUPPORTED_BOUNDARY_SEMANTIC,
            "pressure basis must be labeled algebraic compatibility",
            observed=bases["p"].provenance.value,
        )


def _differentiate_basis(
    source: ComponentBasisRequirement,
    axis: int,
) -> tuple[
    tuple[AxisBasisRequirement, ...],
    DerivativeMultiplier,
]:
    transform, multiplier = _DERIVATIVE_BY_BASIS[
        source.axes[axis].transform_kind
    ]
    axes = []
    for current in source.axes:
        if current.axis != axis:
            axes.append(current)
            continue
        axes.append(
            AxisBasisRequirement(
                axis=current.axis,
                axis_name=current.axis_name,
                topology=current.topology,
                modal_kind=_MODAL_KIND_BY_BASIS[transform],
                transform_kind=transform,
                physical_size=current.physical_size,
                spectral_size=current.spectral_size,
                hermitian_packed=current.hermitian_packed,
            )
        )
    return tuple(axes), multiplier


def _lower_gradient_capability(
    simulation: SimulationSpec,
    capability: str,
    field_name: str,
    bases: dict[str, ComponentBasisRequirement],
) -> tuple[DerivativeRequirement, ...]:
    term = _term_by_capability(simulation, capability)
    sources = term.dependencies
    expected_outputs = tuple(
        f"d{component}_d{axis_name}"
        for axis_name in simulation.geometry.domain.axis_names
        for component in sources
    )
    if term.output_components != expected_outputs:
        _reject(
            LoweringRejectionCode.INVALID_EQUATION_LAYOUT,
            "gradient outputs do not follow the axis-major canonical layout",
            capability=capability,
            expected=list(expected_outputs),
            observed=list(term.output_components),
        )
    derivatives = []
    for axis, _axis_name in enumerate(simulation.geometry.domain.axis_names):
        for source_component in sources:
            output_component = f"d{source_component}_d{_axis_name}"
            source = bases[source_component]
            output_axes, multiplier = _differentiate_basis(source, axis)
            bases[output_component] = ComponentBasisRequirement(
                field_name=field_name,
                component=output_component,
                role=FieldRole.TRANSIENT,
                axes=output_axes,
                provenance=BasisProvenance.DERIVATIVE_PARITY,
                source_components=(source_component,),
            )
            derivatives.append(
                DerivativeRequirement(
                    capability=capability,
                    source_component=source_component,
                    output_component=output_component,
                    axis=axis,
                    multiplier=multiplier,
                    source_transform_kinds=source.transform_kinds,
                    output_transform_kinds=tuple(
                        value.transform_kind for value in output_axes
                    ),
                )
            )
    return tuple(derivatives)


def _copy_basis(
    bases: dict[str, ComponentBasisRequirement],
    *,
    field_name: str,
    component: str,
    source_component: str,
    provenance: BasisProvenance,
) -> None:
    source = bases[source_component]
    bases[component] = ComponentBasisRequirement(
        field_name=field_name,
        component=component,
        role=FieldRole.TRANSIENT,
        axes=source.axes,
        provenance=provenance,
        source_components=(source_component,),
    )


def _lower_transient_spaces(
    simulation: SimulationSpec,
    kind: str,
    bases: dict[str, ComponentBasisRequirement],
) -> tuple[DerivativeRequirement, ...]:
    derivatives = (
        *_lower_gradient_capability(
            simulation,
            "beris_edwards_q_gradient",
            "q_gradient",
            bases,
        ),
        *_lower_gradient_capability(
            simulation,
            "beris_edwards_velocity_gradient",
            "velocity_gradient",
            bases,
        ),
    )
    q_components = _field_components(simulation, "Q")
    velocity = _field_components(simulation, "velocity")
    if kind in {"plane", "periodic"}:
        for component in _field_components(simulation, "molecular_field"):
            q_source = f"Q{component[1:]}"
            _copy_basis(
                bases,
                field_name="molecular_field",
                component=component,
                source_component=q_source,
                provenance=BasisProvenance.CONSTITUTIVE_PARITY,
            )
        for component in _field_components(simulation, "algebraic_stress"):
            _copy_basis(
                bases,
                field_name="algebraic_stress",
                component=component,
                source_component=q_components[0],
                provenance=BasisProvenance.CONSTITUTIVE_PARITY,
            )
        odd_suffixes = {"xz", "yz", "zx", "zy"}
        for component in _field_components(simulation, "distortion_stress"):
            suffix = component.rsplit("_", 1)[-1]
            source = (
                "uz"
                if kind == "plane" and suffix in odd_suffixes
                else q_components[0]
            )
            _copy_basis(
                bases,
                field_name="distortion_stress",
                component=component,
                source_component=source,
                provenance=BasisProvenance.CONSTITUTIVE_PARITY,
            )
        if kind == "plane":
            _validate_plane_distortion_metadata(simulation, bases["uz"])
        force_field = "nematic_force"
    else:
        force_field = "active_force"
    for force, target in zip(
        _field_components(simulation, force_field),
        velocity,
        strict=True,
    ):
        _copy_basis(
            bases,
            field_name=force_field,
            component=force,
            source_component=target,
            provenance=BasisProvenance.PROJECTED_OUTPUT,
        )
    expected_components = set(simulation.equation_system.component_names)
    observed_components = set(bases)
    if observed_components != expected_components:
        _reject(
            LoweringRejectionCode.INVALID_EQUATION_LAYOUT,
            "lowering did not resolve exactly every declared component",
            missing=sorted(expected_components - observed_components),
            extra=sorted(observed_components - expected_components),
        )
    return derivatives


def _validate_plane_distortion_metadata(
    simulation: SimulationSpec,
    odd_basis: ComponentBasisRequirement,
) -> None:
    try:
        observed = simulation.discretization_parameters[
            "legacy_derived_boundary_spaces"
        ]["distortion_odd_z"]
    except (KeyError, TypeError) as exc:
        _reject(
            LoweringRejectionCode.INVALID_DERIVED_SPACE,
            "Plane distortion odd-space evidence is missing",
        )
    expected = [
        {
            "kind": axis.modal_kind.value,
            "is_homogeneous": True,
        }
        for axis in odd_basis.axes
    ]
    if observed != expected:
        _reject(
            LoweringRejectionCode.INVALID_DERIVED_SPACE,
            "Plane distortion odd-space evidence disagrees with derived parity",
            expected=expected,
            observed=observed,
        )


def _capability_requirements(
    simulation: SimulationSpec,
    kind: str,
) -> tuple[CapabilityImplementationRequirement, ...]:
    if kind in {"plane", "periodic"}:
        implementations = {
            "beris_edwards_q_evolution": (
                "plane_complete_beris_edwards_q_evolution",
                "imex_pointwise_q_rhs",
            ),
            "beris_edwards_q_gradient": (
                "tensor_product_spectral_q_gradient",
                "axis_major_spectral_derivatives",
            ),
            "beris_edwards_velocity_gradient": (
                "tensor_product_spectral_velocity_gradient",
                "axis_major_spectral_derivatives",
            ),
            "beris_edwards_molecular_field": (
                "plane_one_constant_molecular_field",
                "spectral_linear_plus_pointwise_bulk",
            ),
            "beris_edwards_stress": (
                "plane_complete_one_constant_nematic_stress",
                "algebraic_even_plus_distortion_parity_split",
            ),
            "beris_edwards_force": (
                "plane_projected_complete_stress_divergence",
                "spectral_sum_then_velocity_space_projection",
            ),
            INCOMPRESSIBLE_STOKES_CAPABILITY: (
                (
                    "plane_free_slip_modal_stokes"
                    if kind == "plane"
                    else "periodic_modal_stokes"
                ),
                (
                    "geometry_specific_saddle_solve"
                    if kind == "plane"
                    else "fourier_helmholtz_projection"
                ),
            ),
        }
    else:
        implementations = {
            "legacy_active_force_q_evolution": (
                "channel_legacy_active_force_q_evolution",
                "imex_pointwise_q_rhs",
            ),
            "beris_edwards_q_gradient": (
                "tensor_product_spectral_q_gradient",
                "axis_major_spectral_derivatives",
            ),
            "beris_edwards_velocity_gradient": (
                "tensor_product_spectral_velocity_gradient",
                "axis_major_spectral_derivatives",
            ),
            "legacy_active_force": (
                "channel_legacy_active_force_divergence",
                "physical_mixed_parity_sum_then_velocity_space_projection",
            ),
            INCOMPRESSIBLE_STOKES_CAPABILITY: (
                "channel_no_slip_modal_stokes_pcg",
                "geometry_specific_pressure_schur_pcg",
            ),
        }
    requirements = []
    for capability in simulation.equation_system.required_capabilities:
        term = _term_by_capability(simulation, capability)
        implementation, strategy = implementations[capability]
        requirements.append(
            CapabilityImplementationRequirement(
                capability=capability,
                implementation_name=implementation,
                strategy=strategy,
                input_components=term.dependencies,
                output_components=term.output_components,
            )
        )
    return tuple(requirements)


def _stokes_requirements(
    simulation: SimulationSpec,
    kind: str,
) -> tuple[GeometrySolverRequirement, NullspaceRequirement]:
    algebraic = _term_by_capability(
        simulation,
        INCOMPRESSIBLE_STOKES_CAPABILITY,
    )
    try:
        stokes = IncompressibleStokesSystemSpec.from_algebraic_system_spec(
            algebraic
        )
    except (TypeError, ValueError) as exc:
        _reject(
            LoweringRejectionCode.INVALID_STOKES_CONTRACT,
            "the incompressible Stokes declaration is invalid",
            error=str(exc),
        )
    if stokes.pressure_gauge is not PressureGauge.ZERO_MEAN:
        _reject(
            LoweringRejectionCode.UNSUPPORTED_NULLSPACE_POLICY,
            "only the zero-mean pressure gauge is qualified",
            observed=stokes.pressure_gauge.value,
        )
    if kind in {"plane", "periodic"}:
        if stokes.tangential_zero_mode_policy not in {
            TangentialZeroModePolicy.ZERO_MEAN,
            TangentialZeroModePolicy.FRICTION,
        }:
            _reject(
                LoweringRejectionCode.UNSUPPORTED_NULLSPACE_POLICY,
                "Plane requires an explicit tangential zero-mode policy",
                observed=stokes.tangential_zero_mode_policy.value,
            )
        if kind == "plane":
            action = (
                "remove_uniform_tangential_velocity_and_force"
                if stokes.tangential_zero_mode_policy
                is TangentialZeroModePolicy.ZERO_MEAN
                else "retain_uniform_tangential_mode_resolved_by_friction"
            )
            tangential_components = stokes.velocity_components[:2]
            family = "free_slip_modal_stokes"
            implementation = (
                "pssolver.linear_solvers.stokes.plane_free_slip."
                "FreeSlipModalStokesSolver"
            )
            extra_options = {
                "wall_normal_axis": 2,
                "pressure_algorithm": "direct_modal_schur",
            }
        else:
            action = (
                "remove_all_uniform_velocity_and_force"
                if stokes.tangential_zero_mode_policy
                is TangentialZeroModePolicy.ZERO_MEAN
                else "retain_all_uniform_velocity_resolved_by_friction"
            )
            # This legacy field name is retained in the version-1 lowering
            # schema, but all three periodic velocity components are listed.
            tangential_components = stokes.velocity_components
            family = "periodic_modal_stokes"
            implementation = (
                "pssolver.linear_solvers.stokes.periodic."
                "PeriodicModalStokesSolver"
            )
            extra_options = {
                "pressure_algorithm": "direct_fourier_projection",
                "uniform_velocity_components": list(
                    stokes.velocity_components
                ),
            }
    else:
        if stokes.tangential_zero_mode_policy is not (
            TangentialZeroModePolicy.NOT_APPLICABLE
        ):
            _reject(
                LoweringRejectionCode.UNSUPPORTED_NULLSPACE_POLICY,
                "no-slip Channel has no uniform velocity null mode",
                observed=stokes.tangential_zero_mode_policy.value,
            )
        action = "not_applicable_due_to_bounded_no_slip_axes"
        tangential_components = ()
        family = "channel_no_slip_modal_stokes_pcg"
        implementation = (
            "pssolver.linear_solvers.stokes.channel_no_slip."
            "ChannelNoSlipModalStokesSolver"
        )
        try:
            pressure_solver = simulation.discretization_parameters[
                "pressure_solver"
            ]
        except KeyError:
            _reject(
                LoweringRejectionCode.UNSUPPORTED_SOLVER_OPTIONS,
                "Channel pressure-solver controls are missing",
            )
        if not isinstance(pressure_solver, dict) and not hasattr(
            pressure_solver,
            "items",
        ):
            _reject(
                LoweringRejectionCode.UNSUPPORTED_SOLVER_OPTIONS,
                "Channel pressure-solver controls must be a mapping",
            )
        pressure_solver = dict(pressure_solver)
        if pressure_solver.get("algorithm") != (
            "preconditioned_conjugate_gradient"
        ):
            _reject(
                LoweringRejectionCode.UNSUPPORTED_SOLVER_OPTIONS,
                "Channel requires the qualified pressure PCG algorithm",
                observed=pressure_solver.get("algorithm"),
            )
        extra_options = {
            "streamwise_axis": 0,
            "pressure_solver": pressure_solver,
        }
    options = {
        "viscosity": stokes.viscosity,
        "friction": stokes.friction,
        "pressure_gauge": stokes.pressure_gauge.value,
        "tangential_zero_mode_policy": (
            stokes.tangential_zero_mode_policy.value
        ),
        **extra_options,
    }
    solver = GeometrySolverRequirement(
        capability=INCOMPRESSIBLE_STOKES_CAPABILITY,
        family=family,
        implementation=implementation,
        force_components=stokes.force_components,
        velocity_components=stokes.velocity_components,
        pressure_component=stokes.pressure_component,
        force_projection_components=stokes.velocity_components,
        options_json=_canonical_json(options),
    )
    nullspace = NullspaceRequirement(
        pressure_component=stokes.pressure_component,
        pressure_gauge=stokes.pressure_gauge.value,
        tangential_velocity_components=tangential_components,
        tangential_policy=stokes.tangential_zero_mode_policy.value,
        friction=stokes.friction,
        uniform_mode_action=action,
    )
    return solver, nullspace


def lower_simulation_spec(simulation: SimulationSpec) -> SimulationLoweringPlan:
    """Lower one qualified request without importing or allocating a runtime.

    Every unsupported combination raises :class:`SimulationLoweringError`
    with a stable machine-readable rejection code.  There is no fallback to a
    nearby geometry, model, boundary signature, integrator, or solver.
    """

    kind, _expected_capabilities = _validate_global_contract(simulation)
    bases = _primary_bases(simulation)
    _validate_boundary_signatures(simulation, kind, bases)
    derivatives = _lower_transient_spaces(simulation, kind, bases)
    capabilities = _capability_requirements(simulation, kind)
    solver, nullspace = _stokes_requirements(simulation, kind)
    return SimulationLoweringPlan(
        source_simulation_sha256=simulation.canonical_sha256(),
        equation_variant=simulation.equation_system.variant,
        geometry_name=simulation.geometry.name,
        component_bases=tuple(bases.values()),
        derivatives=derivatives,
        capabilities=capabilities,
        solver=solver,
        nullspace=nullspace,
        numerics_json=_canonical_json(simulation.numerics.to_metadata()),
        time_integration_json=_canonical_json(
            simulation.time_integration.to_metadata()
        ),
    )


__all__ = [
    "LoweringRejection",
    "LoweringRejectionCode",
    "SimulationLoweringError",
    "lower_simulation_spec",
]
