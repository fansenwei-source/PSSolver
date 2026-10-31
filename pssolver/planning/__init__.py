"""Read-only spectral planning contracts and assembly."""

from pssolver.core._compatibility import (
    facade_dir,
    resolve_deprecated_facade_name,
)

from .assembler import assemble_spectral_plan
from .plan import (
    ComponentTransformPlan,
    DomainAxisPlan,
    FieldPlan,
    SPECTRAL_PLAN_SCHEMA_VERSION,
    SpectralPlan,
    TransformKind,
)


_PILOT_EXPORTS = {
    name: ("pssolver.experimental.boundary_pilots", name)
    for name in (
        "MAX_QUALIFIED_DIMENSIONLESS_ROBIN_IMPEDANCE",
        "PLANE_ROBIN_SCALAR_LOWERING_SCHEMA_VERSION",
        "ROBIN_EIGENBASIS_PLAN_SCHEMA_VERSION",
        "STATIC_LIFTING_PLAN_SCHEMA_VERSION",
        "CellCenteredRobinEigenbasisPlan",
        "PlaneRobinScalarLoweringPlan",
        "StaticLiftExtension",
        "StaticLiftingComponentPlan",
        "StaticLiftingPlan",
        "build_cell_centered_robin_eigenbasis_plan",
        "validate_dimensionless_robin_impedance",
    )
}


def __getattr__(name: str) -> object:
    return resolve_deprecated_facade_name(
        name,
        facade=__name__,
        replacement="pssolver.experimental.boundary_pilots",
        exports=_PILOT_EXPORTS,
        namespace=globals(),
    )


def __dir__() -> list[str]:
    return facade_dir(globals(), _PILOT_EXPORTS)

__all__ = [
    "ComponentTransformPlan",
    "DomainAxisPlan",
    "FieldPlan",
    "SPECTRAL_PLAN_SCHEMA_VERSION",
    "ROBIN_EIGENBASIS_PLAN_SCHEMA_VERSION",
    "MAX_QUALIFIED_DIMENSIONLESS_ROBIN_IMPEDANCE",
    "PLANE_ROBIN_SCALAR_LOWERING_SCHEMA_VERSION",
    "STATIC_LIFTING_PLAN_SCHEMA_VERSION",
    "SpectralPlan",
    "StaticLiftExtension",
    "StaticLiftingComponentPlan",
    "StaticLiftingPlan",
    "TransformKind",
    "assemble_spectral_plan",
    "CellCenteredRobinEigenbasisPlan",
    "PlaneRobinScalarLoweringPlan",
    "build_cell_centered_robin_eigenbasis_plan",
    "validate_dimensionless_robin_impedance",
]
