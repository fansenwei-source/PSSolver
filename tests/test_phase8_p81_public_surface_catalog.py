"""P8.1 public facades for existing qualified capabilities only."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path
import pickle

import pytest

import pssolver
from pssolver import Simulation, compile_simulation
from pssolver.boundaries import (
    assign_boundaries,
    neumann_pressure_compatibility,
    neumann_q,
    no_slip_velocity,
)
from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_channel_active_nematics_simulation,
)
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.public_simulation_runner import (
    PUBLIC_CHANNEL_APPLICATION,
    PublicCompilationRejectionCode,
    PublicSimulationCompilationError,
    public_compiler_capabilities,
)
from pssolver.core.boundary import BoundaryKind
from pssolver.core.domain import DomainSpec
from pssolver.geometries import PeriodicBox, RectangularChannel
from pssolver.geometries.tensor_product import (
    PeriodicBox as CanonicalPeriodicBox,
)
from pssolver.geometries.tensor_product import (
    RectangularChannel as CanonicalRectangularChannel,
)
from pssolver.models.active_nematics import (
    LegacyActiveForceActiveNematics,
)


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / (
    "notes/architecture_v0_2/phase_8_p81_public_surface_catalog.json"
)


def _channel_source(runtime_path: str = "compiled_channel_v2"):
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
    return compose_channel_active_nematics_simulation(request.components)


def _manual_channel_simulation() -> Simulation:
    source = _channel_source()
    model = LegacyActiveForceActiveNematics(
        rho=6.0,
        elastic_constant=1.0,
        activity=5.0,
        beta=-1.0,
        flow_alignment=1.0,
        friction=0.0,
        viscosity=1.0,
    )
    geometry = RectangularChannel(
        shape=(16, 8, 8),
        lengths=(16.0, 8.0, 8.0),
    )
    boundaries = assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": no_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=boundaries,
        numerics=source.numerics,
        time=source.time_integration,
        discretization=source.discretization_parameters,
        initial_condition=source.initial_condition,
        execution=source.execution,
        output=source.workflow,
        invocation=source.invocation,
    )


def _boundary_identity(value) -> dict[str, object]:
    metadata = value.to_metadata()
    return {
        "ndim": metadata["ndim"],
        "components": metadata["components"],
    }


def test_legacy_model_facade_is_exact_existing_channel_declaration():
    source = _channel_source()
    manual = _manual_channel_simulation()

    assert manual.model == source.equation_system
    assert manual.model.variant == "legacy_active_force_active_nematics"
    assert manual.model.parameters["force_law"] == (
        "active_force_divergence_only"
    )
    assert "geometry" not in manual.model.to_metadata()
    assert "boundaries" not in manual.model.to_metadata()


def test_geometry_facades_preserve_canonical_identity_and_legacy_form():
    periodic = PeriodicBox(shape=(8, 6), lengths=(4.0, 3.0))
    periodic_legacy = PeriodicBox(
        DomainSpec(shape=(8, 6), lengths=(4.0, 3.0))
    )
    channel = RectangularChannel(
        shape=(16, 8, 6),
        lengths=(8.0, 4.0, 3.0),
        streamwise_axis=1,
    )

    assert type(periodic) is CanonicalPeriodicBox
    assert type(channel) is CanonicalRectangularChannel
    assert periodic == periodic_legacy
    assert pickle.loads(pickle.dumps(periodic)) == periodic
    assert pickle.loads(pickle.dumps(channel)) == channel
    assert channel.periodic_axes == (1,)
    assert channel.bounded_axes == (0, 2)
    with pytest.raises(ValueError, match="cannot be combined"):
        PeriodicBox(
            periodic.domain,
            shape=(8, 6),
            lengths=(4.0, 3.0),
        )


def test_no_slip_policy_matches_qualified_channel_wall_law():
    source = _channel_source()
    manual = _manual_channel_simulation()

    assert _boundary_identity(manual.boundaries) == _boundary_identity(
        source.boundaries
    )
    for component in ("ux", "uy", "uz"):
        assignment = manual.boundaries.for_component(component)
        for axis in (1, 2):
            assert {
                face.condition.kind
                for face in assignment.faces
                if face.axis == axis
            } == {BoundaryKind.DIRICHLET}
        assert {
            face.condition.kind
            for face in assignment.faces
            if face.axis == 0
        } == {BoundaryKind.PERIODIC}


def test_manual_public_channel_declarations_compile_without_hidden_names():
    compiled = compile_simulation(_manual_channel_simulation())

    assert compiled.application == PUBLIC_CHANNEL_APPLICATION
    assert compiled.application_specification == _channel_source()
    assert compiled.normalization["physical_values_changed"] is False


def test_discovery_is_frozen_and_matches_live_compiler_registry():
    catalog = pssolver.capability_catalog()
    combinations = pssolver.available_combinations()

    assert catalog.models is pssolver.available_models()
    assert catalog.geometries is pssolver.available_geometries()
    assert catalog.boundary_policies is pssolver.available_boundary_policies()
    assert tuple(
        (
            value.equation_variant,
            value.geometry_name,
            value.application,
            value.adapter,
        )
        for value in combinations
    ) == tuple(
        (
            value["equation_variant"],
            value["geometry_name"],
            value["application"],
            value["adapter"],
        )
        for value in public_compiler_capabilities()
    )
    assert [value.key for value in pssolver.available_geometries()] == [
        "periodic_box",
        "plane_slab",
        "rectangular_channel",
    ]
    # P8.1's archived result records the historical non-executable state;
    # later Phase 8 slices may qualify a PeriodicBox application.
    assert pssolver.available_geometries()[0].executable is True
    assert {value.key for value in pssolver.available_boundary_policies()} == {
        "free_slip_velocity",
        "neumann_pressure_compatibility",
        "neumann_q",
        "no_slip_velocity",
    }
    with pytest.raises(FrozenInstanceError):
        catalog.models[0].key = "changed"
    json.dumps(catalog.to_metadata(), allow_nan=False, sort_keys=True)


def test_declarable_periodic_pair_remains_unqualified_before_allocation():
    channel = _manual_channel_simulation()
    geometry = PeriodicBox(
        shape=(16, 8, 8),
        lengths=(16.0, 8.0, 8.0),
    )
    boundaries = assign_boundaries(
        model=channel.model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": no_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )
    simulation = Simulation(
        model=channel.model,
        geometry=geometry,
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
        compile_simulation(simulation)

    assert caught.value.code is (
        PublicCompilationRejectionCode.UNREGISTERED_MODEL_GEOMETRY
    )


def test_p81_machine_record_closes_only_existing_public_capabilities():
    record = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert record["classification"] == (
        "PASS_P8_1_EXISTING_CAPABILITY_PUBLIC_SURFACE_AND_CATALOG"
    )
    assert record["scope"]["new_equation_variant"] is False
    assert record["scope"]["new_compiler_registration"] is False
    assert record["scope"]["new_boundary_physics"] is False
    assert record["scope"]["runtime_or_timestep_changed"] is False
    assert record["scope"]["production_default_changed"] is False
    assert record["eligibility"]["p8_1_complete"] is True
    assert record["eligibility"]["p8_2_planning"] is True
    assert record["eligibility"]["p8_2_execution"] is False
    assert record["verification"]["h100_required"] is False
    assert all(len(value) == 64 for value in record["source_sha256"].values())
