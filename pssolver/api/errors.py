"""Structured public compilation failures, independent of execution."""

from enum import Enum


class PublicCompilationRejectionCode(str, Enum):
    """Codes for unsupported mathematics, scope and invalid declarations."""

    BOUNDARY_DATA_RESERVED = "boundary_data_reserved"
    DECLARATION_CONTRACT = "declaration_contract"
    GEOMETRY_NOT_MIGRATED = "geometry_not_migrated"
    INCOMPATIBLE_BOUNDARIES = "incompatible_boundaries"
    INITIAL_CONDITION_CONTRACT = "initial_condition_contract"
    INTEGRATOR_NOT_AVAILABLE = "integrator_not_available"
    INVALID_FIELD_REFERENCE = "invalid_field_reference"
    INVALID_LINEAR_EXPRESSION = "invalid_linear_expression"
    MISSING_CONSTRAINT_SOLVER = "missing_constraint_solver"
    MISSING_EXPLICIT_IMPLEMENTATION = "missing_explicit_implementation"
    NONZERO_DIRICHLET_LIFTING = "nonzero_dirichlet_lifting"
    OUTPUT_CONTRACT = "output_contract"
    PRESSURE_TOLERANCE_BELOW_FLOOR = "pressure_tolerance_below_floor"
    REFRESH_NOT_SUPPORTED = "refresh_not_supported"
    ROBIN_OR_ANCHORING = "robin_or_anchoring"
    UNUSED_OR_MISSING_EXECUTION_OPTION = "unused_or_missing_execution_option"


class PublicSimulationCompilationError(ValueError):
    """A failed assembly with a public machine-readable explanation."""

    def __init__(self, code, reason: str, *, kind="mathematical", source=None, opens_in=None):
        self.code = PublicCompilationRejectionCode(code)
        self.kind = kind
        self.reason = reason
        self.source = source
        self.opens_in = opens_in
        super().__init__(f"{self.code.value}: {reason}")


__all__ = ["PublicCompilationRejectionCode", "PublicSimulationCompilationError"]
