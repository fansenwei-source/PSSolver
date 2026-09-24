"""Geometry-neutral active-nematic equation-system declarations.

This module records the scientific difference between the qualified Plane
complete-stress model and the qualified Channel active-force-only model.  It
does not attach boundary conditions, select a geometry, import Torch, choose a
transform, or construct a runtime.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from numbers import Real

from pssolver.core.fields import FieldRole
from pssolver.systems.equations import (
    EquationFieldSpec,
    EquationSystemSpec,
    EquationTermSpec,
)
from pssolver.systems.stokes import IncompressibleStokesSystemSpec

from .fields import Q_COMPONENTS
from .specifications import BerisEdwardsMaterialRequest


CARTESIAN_AXES = ("x", "y", "z")
H_COMPONENTS = ("Hxx", "Hxy", "Hxz", "Hyy", "Hyz")
STRESS_COMPONENTS = ("xx", "xy", "xz", "yx", "yy", "yz", "zx", "zy", "zz")
ALGEBRAIC_STRESS_COMPONENTS = tuple(
    f"stress_algebraic_{component}" for component in STRESS_COMPONENTS
)
DISTORTION_STRESS_COMPONENTS = tuple(
    f"stress_distortion_{component}" for component in STRESS_COMPONENTS
)
Q_GRADIENT_COMPONENTS = tuple(
    f"d{component}_d{axis}"
    for axis in CARTESIAN_AXES
    for component in Q_COMPONENTS
)
VELOCITY_COMPONENTS = ("ux", "uy", "uz")
VELOCITY_GRADIENT_COMPONENTS = tuple(
    f"d{component}_d{axis}"
    for axis in CARTESIAN_AXES
    for component in VELOCITY_COMPONENTS
)
COMPLETE_STRESS_FORCE_COMPONENTS = ("force_x", "force_y", "force_z")
ACTIVE_FORCE_COMPONENTS = ("fx", "fy", "fz")

BERIS_EDWARDS_Q_EVOLUTION_CAPABILITY = "beris_edwards_q_evolution"
BERIS_EDWARDS_MOLECULAR_FIELD_CAPABILITY = "beris_edwards_molecular_field"
BERIS_EDWARDS_Q_GRADIENT_CAPABILITY = "beris_edwards_q_gradient"
BERIS_EDWARDS_STRESS_CAPABILITY = "beris_edwards_stress"
BERIS_EDWARDS_FORCE_CAPABILITY = "beris_edwards_force"
BERIS_EDWARDS_VELOCITY_GRADIENT_CAPABILITY = (
    "beris_edwards_velocity_gradient"
)
LEGACY_ACTIVE_FORCE_Q_EVOLUTION_CAPABILITY = (
    "legacy_active_force_q_evolution"
)
LEGACY_ACTIVE_FORCE_CAPABILITY = "legacy_active_force"


def _finite(value: object, description: str, *, positive: bool = False) -> float:
    if (
        not isinstance(value, Real)
        or isinstance(value, bool)
        or not math.isfinite(float(value))
        or (positive and float(value) <= 0.0)
    ):
        qualifier = "positive and finite" if positive else "finite"
        raise ValueError(f"{description} must be {qualifier}")
    return float(value)


def _field(
    name: str,
    role: FieldRole,
    components: tuple[str, ...],
) -> EquationFieldSpec:
    return EquationFieldSpec(name, role, components)


@dataclass(frozen=True, slots=True)
class CompleteStressBerisEdwardsEquationRequest:
    """Physical request for the qualified complete-stress Q/Stokes model."""

    material: BerisEdwardsMaterialRequest
    ldg_l1: float
    activity_amplitude: float
    stokes_system: IncompressibleStokesSystemSpec

    def __post_init__(self) -> None:
        if not isinstance(self.material, BerisEdwardsMaterialRequest):
            raise TypeError("material must be a BerisEdwardsMaterialRequest")
        object.__setattr__(
            self,
            "ldg_l1",
            _finite(self.ldg_l1, "ldg_l1", positive=True),
        )
        object.__setattr__(
            self,
            "activity_amplitude",
            _finite(self.activity_amplitude, "activity_amplitude"),
        )
        if not isinstance(self.stokes_system, IncompressibleStokesSystemSpec):
            raise TypeError(
                "stokes_system must be an IncompressibleStokesSystemSpec"
            )
        if (
            self.stokes_system.force_components
            != COMPLETE_STRESS_FORCE_COMPONENTS
        ):
            raise ValueError(
                "complete-stress Stokes force components must be "
                "('force_x', 'force_y', 'force_z')"
            )
        _validate_stokes_outputs(self.stokes_system)

    def to_equation_system_spec(self) -> EquationSystemSpec:
        material = self.material
        active_prefactor = material.beta * self.activity_amplitude
        fields = _common_q_flow_fields(
            force_components=COMPLETE_STRESS_FORCE_COMPONENTS,
            complete_stress=True,
        )
        return EquationSystemSpec(
            name="active_nematics",
            variant="complete_stress_beris_edwards",
            fields=fields,
            evolution_laws=(
                EquationTermSpec(
                    name="q_evolution",
                    capability=BERIS_EDWARDS_Q_EVOLUTION_CAPABILITY,
                    output_components=Q_COMPONENTS,
                    dependencies=(
                        *Q_COMPONENTS,
                        *VELOCITY_COMPONENTS,
                        *Q_GRADIENT_COMPONENTS,
                        *VELOCITY_GRADIENT_COMPONENTS,
                    ),
                    parameters={
                        "explicit_terms": [
                            "bulk_b_c",
                            "material_advection",
                            "flow_alignment",
                            "co_rotation",
                        ],
                        "implicit_terms": [
                            "bulk_a",
                            "one_constant_l1_laplacian",
                        ],
                        "ldg_a": material.ldg_a,
                        "ldg_b": material.ldg_b,
                        "ldg_c": material.ldg_c,
                        "ldg_l1": self.ldg_l1,
                        "rotational_viscosity": material.gamma,
                        "flow_alignment": material.flow_alignment,
                    },
                ),
            ),
            constitutive_laws=(
                EquationTermSpec(
                    name="q_gradient",
                    capability=BERIS_EDWARDS_Q_GRADIENT_CAPABILITY,
                    output_components=Q_GRADIENT_COMPONENTS,
                    dependencies=Q_COMPONENTS,
                ),
                EquationTermSpec(
                    name="velocity_gradient",
                    capability=BERIS_EDWARDS_VELOCITY_GRADIENT_CAPABILITY,
                    output_components=VELOCITY_GRADIENT_COMPONENTS,
                    dependencies=VELOCITY_COMPONENTS,
                ),
                EquationTermSpec(
                    name="molecular_field",
                    capability=BERIS_EDWARDS_MOLECULAR_FIELD_CAPABILITY,
                    output_components=H_COMPONENTS,
                    dependencies=Q_COMPONENTS,
                    parameters={
                        "ldg_a": material.ldg_a,
                        "ldg_b": material.ldg_b,
                        "ldg_c": material.ldg_c,
                        "ldg_l1": self.ldg_l1,
                    },
                ),
                EquationTermSpec(
                    name="nematic_stress",
                    capability=BERIS_EDWARDS_STRESS_CAPABILITY,
                    output_components=(
                        *ALGEBRAIC_STRESS_COMPONENTS,
                        *DISTORTION_STRESS_COMPONENTS,
                    ),
                    dependencies=(
                        *Q_COMPONENTS,
                        *H_COMPONENTS,
                        *Q_GRADIENT_COMPONENTS,
                    ),
                    parameters={
                        "active_amplitude": self.activity_amplitude,
                        "active_prefactor": active_prefactor,
                        "beta": material.beta,
                        "flow_alignment": material.flow_alignment,
                        "ldg_l1": self.ldg_l1,
                    },
                ),
                EquationTermSpec(
                    name="nematic_force",
                    capability=BERIS_EDWARDS_FORCE_CAPABILITY,
                    output_components=COMPLETE_STRESS_FORCE_COMPONENTS,
                    dependencies=(
                        *ALGEBRAIC_STRESS_COMPONENTS,
                        *DISTORTION_STRESS_COMPONENTS,
                    ),
                ),
            ),
            algebraic_systems=(
                self.stokes_system.to_algebraic_system_spec(),
            ),
            parameters={
                "force_law": "complete_one_constant_nematic_stress",
                "material": material.to_metadata(),
                "ldg_l1": self.ldg_l1,
                "activity_amplitude": self.activity_amplitude,
                "active_prefactor": active_prefactor,
                "q_convention": "Q=(3S/2)(nn-I/3)",
                "stokes": self.stokes_system.to_metadata(),
            },
            diagnostics=(
                "q_invariants",
                "stokes_incompressibility",
                "nematic_force_decomposition",
            ),
            initial_condition_families=("extruded_defect_gas",),
        )


@dataclass(frozen=True, slots=True)
class LegacyActiveForceEquationRequest:
    """Physical request for the qualified rho-parameterized Channel model."""

    rho: float
    elastic_constant: float
    activity: float
    beta: float
    flow_alignment: float
    stokes_system: IncompressibleStokesSystemSpec

    def __post_init__(self) -> None:
        object.__setattr__(self, "rho", _finite(self.rho, "rho", positive=True))
        object.__setattr__(
            self,
            "elastic_constant",
            _finite(
                self.elastic_constant,
                "elastic_constant",
                positive=True,
            ),
        )
        for name in ("activity", "beta", "flow_alignment"):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if not isinstance(self.stokes_system, IncompressibleStokesSystemSpec):
            raise TypeError(
                "stokes_system must be an IncompressibleStokesSystemSpec"
            )
        if self.stokes_system.force_components != ACTIVE_FORCE_COMPONENTS:
            raise ValueError(
                "legacy active-force Stokes force components must be "
                "('fx', 'fy', 'fz')"
            )
        _validate_stokes_outputs(self.stokes_system)

    @property
    def ldg_a(self) -> float:
        return 1.0 - self.rho / 3.0

    @property
    def ldg_b(self) -> float:
        return -self.rho

    @property
    def ldg_c(self) -> float:
        return self.rho

    def to_equation_system_spec(self) -> EquationSystemSpec:
        fields = _common_q_flow_fields(
            force_components=ACTIVE_FORCE_COMPONENTS,
            complete_stress=False,
        )
        return EquationSystemSpec(
            name="active_nematics",
            variant="legacy_active_force_active_nematics",
            fields=fields,
            evolution_laws=(
                EquationTermSpec(
                    name="q_evolution",
                    capability=LEGACY_ACTIVE_FORCE_Q_EVOLUTION_CAPABILITY,
                    output_components=Q_COMPONENTS,
                    dependencies=(
                        *Q_COMPONENTS,
                        *VELOCITY_COMPONENTS,
                        *Q_GRADIENT_COMPONENTS,
                        *VELOCITY_GRADIENT_COMPONENTS,
                    ),
                    parameters={
                        "explicit_terms": [
                            "bulk_b_c",
                            "material_advection",
                            "flow_alignment",
                            "co_rotation",
                        ],
                        "implicit_terms": [
                            "bulk_a",
                            "elastic_laplacian",
                        ],
                        "ldg_a": self.ldg_a,
                        "ldg_b": self.ldg_b,
                        "ldg_c": self.ldg_c,
                        "elastic_constant": self.elastic_constant,
                        "flow_alignment": self.flow_alignment,
                    },
                ),
            ),
            constitutive_laws=(
                EquationTermSpec(
                    name="q_gradient",
                    capability=BERIS_EDWARDS_Q_GRADIENT_CAPABILITY,
                    output_components=Q_GRADIENT_COMPONENTS,
                    dependencies=Q_COMPONENTS,
                ),
                EquationTermSpec(
                    name="velocity_gradient",
                    capability=BERIS_EDWARDS_VELOCITY_GRADIENT_CAPABILITY,
                    output_components=VELOCITY_GRADIENT_COMPONENTS,
                    dependencies=VELOCITY_COMPONENTS,
                ),
                EquationTermSpec(
                    name="active_force",
                    capability=LEGACY_ACTIVE_FORCE_CAPABILITY,
                    output_components=ACTIVE_FORCE_COMPONENTS,
                    dependencies=Q_COMPONENTS,
                    parameters={
                        "activity": self.activity,
                        "beta": self.beta,
                        "active_prefactor": self.beta * self.activity,
                    },
                ),
            ),
            algebraic_systems=(
                self.stokes_system.to_algebraic_system_spec(),
            ),
            parameters={
                "force_law": "active_force_divergence_only",
                "rho": self.rho,
                "ldg_a": self.ldg_a,
                "ldg_b": self.ldg_b,
                "ldg_c": self.ldg_c,
                "elastic_constant": self.elastic_constant,
                "activity": self.activity,
                "beta": self.beta,
                "flow_alignment": self.flow_alignment,
                "q_convention": "Q=(3S/2)(nn-I/3)",
                "stokes": self.stokes_system.to_metadata(),
            },
            diagnostics=(
                "q_invariants",
                "stokes_incompressibility",
                "pressure_pcg",
            ),
            initial_condition_families=("aligned_x_smooth_noise",),
        )


def _validate_stokes_outputs(system: IncompressibleStokesSystemSpec) -> None:
    if system.velocity_components != VELOCITY_COMPONENTS:
        raise ValueError(
            "Stokes velocity components must be ('ux', 'uy', 'uz')"
        )
    if system.pressure_component != "p":
        raise ValueError("Stokes pressure component must be 'p'")


def _common_q_flow_fields(
    *,
    force_components: tuple[str, str, str],
    complete_stress: bool,
) -> tuple[EquationFieldSpec, ...]:
    fields = [
        _field("Q", FieldRole.EVOLVED, Q_COMPONENTS),
        _field("q_gradient", FieldRole.TRANSIENT, Q_GRADIENT_COMPONENTS),
        _field(
            "velocity_gradient",
            FieldRole.TRANSIENT,
            VELOCITY_GRADIENT_COMPONENTS,
        ),
    ]
    if complete_stress:
        fields.extend(
            (
                _field("molecular_field", FieldRole.TRANSIENT, H_COMPONENTS),
                _field(
                    "algebraic_stress",
                    FieldRole.TRANSIENT,
                    ALGEBRAIC_STRESS_COMPONENTS,
                ),
                _field(
                    "distortion_stress",
                    FieldRole.TRANSIENT,
                    DISTORTION_STRESS_COMPONENTS,
                ),
                _field(
                    "nematic_force",
                    FieldRole.TRANSIENT,
                    force_components,
                ),
            )
        )
    else:
        fields.append(
            _field("active_force", FieldRole.TRANSIENT, force_components)
        )
    fields.extend(
        (
            _field("velocity", FieldRole.ALGEBRAIC, VELOCITY_COMPONENTS),
            _field("pressure", FieldRole.ALGEBRAIC, ("p",)),
        )
    )
    return tuple(fields)


__all__ = [
    "ACTIVE_FORCE_COMPONENTS",
    "ALGEBRAIC_STRESS_COMPONENTS",
    "COMPLETE_STRESS_FORCE_COMPONENTS",
    "CompleteStressBerisEdwardsEquationRequest",
    "DISTORTION_STRESS_COMPONENTS",
    "H_COMPONENTS",
    "LegacyActiveForceEquationRequest",
    "Q_GRADIENT_COMPONENTS",
    "VELOCITY_COMPONENTS",
    "VELOCITY_GRADIENT_COMPONENTS",
]
