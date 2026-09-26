"""Final P7.7.9 compiler registry and two-combination contract."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from pssolver import Simulation, compile_simulation
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_channel_active_nematics_simulation,
)
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.public_simulation_runner import (
    PUBLIC_CHANNEL_APPLICATION,
    PUBLIC_PLANE_APPLICATION,
    PublicCompilationRejectionCode,
    PublicSimulationCompilationError,
    public_compiler_capabilities,
)
from pssolver.core import DomainSpec
from pssolver.geometries import PlaneSlab
from pssolver.planning.construction import RuntimeConstructionKind


def _channel_simulation(runtime_path: str) -> Simulation:
    request = ChannelActiveNematicRunSpec(
        shape=(16, 8, 8),
        lengths=(16.0, 8.0, 8.0),
        steps=2,
        save_interval=1,
        diagnostic_interval=1,
        device="cpu",
        dtype="float32",
        runtime_path=runtime_path,
    )
    source = compose_channel_active_nematics_simulation(request.components)
    return Simulation(
        model=source.equation_system,
        geometry=source.geometry,
        boundaries=source.boundaries,
        numerics=source.numerics,
        time=source.time_integration,
        discretization=source.discretization_parameters,
        initial_condition=source.initial_condition,
        execution=source.execution,
        output=source.workflow,
        invocation=source.invocation,
    )


@pytest.mark.parametrize(
    ("runtime_path", "kind"),
    (
        ("legacy_channel", RuntimeConstructionKind.CHANNEL_LEGACY),
        (
            "compiled_channel_v2",
            RuntimeConstructionKind.CHANNEL_COMPILED_V2,
        ),
    ),
)
def test_channel_public_compiler_connects_both_qualified_runtimes(
    runtime_path,
    kind,
):
    source = _channel_simulation(runtime_path)
    compiled = compile_simulation(source)

    assert compiled.application == PUBLIC_CHANNEL_APPLICATION
    assert compiled.construction_plan.kind is kind
    assert compiled.application_request.runtime_path.value == runtime_path
    assert compiled.application_request.canonical_sha256() == (
        ChannelActiveNematicRunSpec(
            shape=(16, 8, 8),
            lengths=(16.0, 8.0, 8.0),
            steps=2,
            save_interval=1,
            diagnostic_interval=1,
            device="cpu",
            dtype="float32",
            runtime_path=runtime_path,
        ).canonical_sha256()
    )
    assert compiled.normalization["physical_values_changed"] is False
    assert compiled.normalization["runtime_fallback_allowed"] is False
    assert compiled.lowering_plan.source_simulation_sha256 == (
        compiled.application_specification.canonical_sha256()
    )
    json.dumps(compiled.to_metadata(), allow_nan=False, sort_keys=True)


def test_registry_is_capability_catalog_not_a_public_api_whitelist():
    capabilities = public_compiler_capabilities()
    assert capabilities[:2] == (
        {
            "equation_variant": "complete_stress_beris_edwards",
            "geometry_name": "plane_slab",
            "application": PUBLIC_PLANE_APPLICATION,
            "adapter": (
                "pssolver.configuration.public_simulation_runner."
                "_compile_plane_public_simulation"
            ),
        },
        {
            "equation_variant": "legacy_active_force_active_nematics",
            "geometry_name": "rectangular_channel",
            "application": PUBLIC_CHANNEL_APPLICATION,
            "adapter": (
                "pssolver.configuration.public_channel_simulation_compiler."
                "compile_channel_public_simulation"
            ),
        },
    )
    assert capabilities[2] == {
        "equation_variant": "complete_stress_beris_edwards",
        "geometry_name": "periodic_box",
        "application": "periodic_complete_stress_beris_edwards",
        "adapter": (
            "pssolver.configuration.public_periodic_simulation_compiler."
            "compile_periodic_public_simulation"
        ),
    }
    capabilities[0]["application"] = "local-copy-only"
    assert public_compiler_capabilities()[0]["application"] == (
        PUBLIC_PLANE_APPLICATION
    )


def test_unregistered_pair_reports_structured_capability_gap():
    channel = _channel_simulation("compiled_channel_v2")
    plane = PlaneSlab(
        DomainSpec(
            shape=(16, 16, 8),
            lengths=(16.0, 16.0, 8.0),
        )
    )
    boundaries = assign_boundaries(
        model=channel.model,
        geometry=plane,
        policies={
            "Q": neumann_q(),
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    source = Simulation(
        model=channel.model,
        geometry=plane,
        boundaries=boundaries,
        numerics=channel.numerics,
        time=channel.time,
        discretization=channel.discretization,
        initial_condition=channel.initial_condition,
        execution=channel.execution,
        output=channel.output,
        invocation=channel.invocation,
    )

    with pytest.raises(PublicSimulationCompilationError) as caught:
        compile_simulation(source)

    error = caught.value
    assert error.code is (
        PublicCompilationRejectionCode.UNREGISTERED_MODEL_GEOMETRY
    )
    metadata = error.to_metadata()
    assert metadata["context"]["equation_variant"] == (
        "legacy_active_force_active_nematics"
    )
    assert metadata["context"]["geometry_name"] == "plane_slab"
    assert len(metadata["context"]["required_capabilities"]) == 5
    assert metadata["context"]["registered_pairs"] == [
        ["complete_stress_beris_edwards", "plane_slab"],
        ["legacy_active_force_active_nematics", "rectangular_channel"],
        ["complete_stress_beris_edwards", "periodic_box"],
        ["complete_stress_beris_edwards", "rectangular_channel"],
    ]


def test_compiled_channel_is_immutable():
    compiled = compile_simulation(_channel_simulation("compiled_channel_v2"))
    with pytest.raises(FrozenInstanceError):
        compiled.application = "other"


def test_channel_compiler_preserves_snapshot_mode_without_guessing():
    request = ChannelActiveNematicRunSpec(
        shape=(16, 8, 8),
        lengths=(16.0, 8.0, 8.0),
        steps=2,
        save_interval=1,
        diagnostic_interval=1,
        device="cpu",
        dtype="float32",
        runtime_path="compiled_channel_v2",
        initialization_mode="snapshot",
        snapshot_mode="branch",
        snapshot_directory="data/source",
        snapshot_step=50,
        snapshot_output_directory="data/branch",
    )
    specification = compose_channel_active_nematics_simulation(
        request.components
    )
    source = Simulation(
        model=specification.equation_system,
        geometry=specification.geometry,
        boundaries=specification.boundaries,
        numerics=specification.numerics,
        time=specification.time_integration,
        discretization=specification.discretization_parameters,
        initial_condition=specification.initial_condition,
        execution=specification.execution,
        output=specification.workflow,
        invocation=specification.invocation,
    )

    compiled = compile_simulation(source)

    assert compiled.application_request.initialization_mode == "snapshot"
    assert compiled.application_request.snapshot_mode == "branch"
    assert compiled.application_request.snapshot_step == 50
    assert str(compiled.application_request.snapshot_directory) == "data/source"
    assert str(compiled.application_request.snapshot_output_directory) == (
        "data/branch"
    )
