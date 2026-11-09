"""Compatibility adapters to P7.7.1 active-nematic equation declarations.

The adapters read already validated Plane and Channel component graphs.  They
do not alter those graphs, attach boundaries to the equation declaration, or
connect the declarations to a runtime.
"""

from __future__ import annotations

from pssolver.models.active_nematics.equation_systems import (
    CompleteStressBerisEdwardsEquationRequest,
)
from pssolver.systems.equations import EquationSystemSpec

from .plane_beris_edwards_component_graph import (
    PlaneBerisEdwardsRunComponents,
)


def declare_plane_complete_stress_equation_system(
    components: PlaneBerisEdwardsRunComponents,
) -> EquationSystemSpec:
    """Adapt the qualified Plane physical request without geometry leakage."""

    if not isinstance(components, PlaneBerisEdwardsRunComponents):
        raise TypeError(
            "components must be a PlaneBerisEdwardsRunComponents"
        )
    return CompleteStressBerisEdwardsEquationRequest(
        material=components.physics.material,
        ldg_l1=components.preset.ldg_l1,
        activity_amplitude=components.preset.zeta,
        stokes_system=components.physics.stokes,
    ).to_equation_system_spec()


__all__ = [
    "declare_plane_complete_stress_equation_system",
]
