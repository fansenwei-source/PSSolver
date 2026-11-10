"""Final P7.7.9 compiler registry and two-combination contract."""

from __future__ import annotations

from pssolver import Simulation
from pssolver.configuration.public_simulation_runner import PUBLIC_PLANE_APPLICATION, public_compiler_capabilities


def _replace_channel_declarations(source, *, execution=None, discretization=None):
    return Simulation(
        model=source.model,
        geometry=source.geometry,
        boundaries=source.boundaries,
        numerics=source.numerics,
        time=source.time,
        discretization=(
            source.discretization
            if discretization is None
            else discretization
        ),
        initial_condition=source.initial_condition,
        execution=source.execution if execution is None else execution,
        output=source.output,
        invocation=source.invocation,
    )


def test_registry_is_capability_catalog_not_a_public_api_whitelist():
    capabilities = public_compiler_capabilities()
    assert capabilities[:1] == (
        {
            "equation_variant": "complete_stress_beris_edwards",
            "geometry_name": "plane_slab",
            "application": PUBLIC_PLANE_APPLICATION,
            "adapter": (
                "pssolver.configuration.public_simulation_runner."
                "_compile_plane_public_simulation"
            ),
        },
    )
    assert capabilities[1] == {
        "equation_variant": "complete_stress_beris_edwards",
        "geometry_name": "periodic_box",
        "application": "periodic_complete_stress_beris_edwards",
        "adapter": (
            "pssolver.configuration.legacy_periodic."
            "compile_periodic_public_simulation"
        ),
    }
    capabilities[0]["application"] = "local-copy-only"
    assert public_compiler_capabilities()[0]["application"] == (
        PUBLIC_PLANE_APPLICATION
    )
