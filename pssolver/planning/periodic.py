"""Hand-written Periodic plan; assembly replaces this wiring in S-26."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pssolver.core.spectral_identity import periodic_hermitian_state_projection
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


PERIODIC_BOUNDARIES = ("periodic", "periodic", "periodic")
Q_COMPONENTS = ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")


@dataclass(frozen=True, slots=True)
class PeriodicPlanInputs:
    shape: tuple[int, int, int]
    lengths: tuple[float, float, float]
    dt: float
    spectral_storage: str
    hermitian_axis: int | None
    transform_execution_order: str
    dealias_rule: str
    projected_transform_execution: str
    material: Mapping[str, float]
    ldg_l1: float
    stokes: Mapping[str, object]
    activity_amplitude: float
    pointwise_execution: str
    q_gradient_reuse: bool
    molecular_field_linear_space: str
    stress_divergence_sum_space: str
    flavour: Literal["production", "functional_1_1"]


def periodic_plan(inputs: PeriodicPlanInputs) -> Plan:
    if inputs.flavour not in {"production", "functional_1_1"}:
        raise ValueError("unsupported periodic plan flavour")
    functional = inputs.flavour == "functional_1_1"
    material = inputs.material
    bc = PERIODIC_BOUNDARIES
    return build_plan(
        schema_version=1,
        geometry=GeometryPlan(
            "periodic_box",
            inputs.shape,
            inputs.lengths,
            bc,
            inputs.spectral_storage,
            inputs.hermitian_axis,
            periodic_hermitian_state_projection(inputs.spectral_storage),
            inputs.transform_execution_order,
            "multidim",
        ),
        fields=(
            FieldPlan("Q", "evolved", Q_COMPONENTS, bc),
            FieldPlan("velocity", "constrained", ("ux", "uy", "uz"), bc),
            FieldPlan("pressure", "constrained", ("p",), bc),
        ),
        groups=(
            TransformGroup(Q_COMPONENTS, bc, "contiguous_slice"),
            TransformGroup(("ux", "uy", "uz", "p"), bc, "contiguous_slice"),
        ),
        linear=(
            LinearTermPlan(
                "Q",
                "diagonal",
                "pssolver.models.active_nematics:beris_edwards_linear_operator",
                {
                    "ldg_a": material["ldg_a"],
                    "ldg_l1": inputs.ldg_l1,
                    "rotational_viscosity": material["gamma"],
                },
            ),
        ),
        explicit=(
            ExplicitTermPlan(
                "Q",
                "pssolver.models.active_nematics.engine_terms:periodic_explicit_term",
                {
                    "ldg_b": material["ldg_b"],
                    "ldg_c": material["ldg_c"],
                    "rotational_viscosity": material["gamma"],
                    "flow_alignment": material["flow_alignment"],
                },
                "eager" if functional else inputs.pointwise_execution,
                False if functional else inputs.q_gradient_reuse,
            ),
        ),
        constraints=(
            ConstraintPlan(
                ("stokes", "periodic^3"),
                "pssolver.models.active_nematics.engine_terms:periodic_stokes_term",
                ("Q",),
                ("velocity", "pressure"),
                {
                    "beta_value": material["beta"],
                    "friction": inputs.stokes["friction"],
                    "viscosity": inputs.stokes["viscosity"],
                    "ldg_a": material["ldg_a"],
                    "ldg_b": material["ldg_b"],
                    "ldg_c": material["ldg_c"],
                    "ldg_l1": inputs.ldg_l1,
                    "flow_alignment": material["flow_alignment"],
                    "molecular_field_linear_space": inputs.molecular_field_linear_space,
                    "stress_divergence_sum_space": inputs.stress_divergence_sum_space,
                    "zero_mode_policy": inputs.stokes["tangential_zero_mode_policy"],
                },
                "stateless",
                "autograd",
                not functional,
            ),
        ),
        dealias=DealiasPlan(inputs.dealias_rule, inputs.projected_transform_execution),
        integrator=IntegratorPlan("semi_implicit_euler", inputs.dt, 0, None),
        parameters=(
            ParameterPlan(
                "alpha",
                "field" if functional else "scalar",
                functional,
                None if functional else inputs.activity_amplitude,
            ),
        ),
        boundary_data=(),
        outputs=OutputPlan(
            ("Q", "velocity", "pressure"),
            "after_constraints",
            "pssolver.runtime.observations:capture_periodic_observation",
        ),
    )
