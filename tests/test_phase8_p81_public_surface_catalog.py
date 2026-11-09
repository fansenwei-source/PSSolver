"""P8.1 public facades for existing qualified capabilities only."""

from __future__ import annotations

import json
import pickle
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

import pssolver
from pssolver.configuration.public_simulation_runner import (
    public_compiler_capabilities,
)
from pssolver.core.domain import DomainSpec
from pssolver.geometries import PeriodicBox, RectangularChannel
from pssolver.geometries.tensor_product import (
    PeriodicBox as CanonicalPeriodicBox,
)
from pssolver.geometries.tensor_product import (
    RectangularChannel as CanonicalRectangularChannel,
)


ROOT = Path(__file__).resolve().parents[1]


RESULT_PATH = ROOT / ("notes/architecture_v0_2/phase_8_p81_public_surface_catalog.json")


def _boundary_identity(value) -> dict[str, object]:
    metadata = value.to_metadata()
    return {
        "ndim": metadata["ndim"],
        "components": metadata["components"],
    }


def test_geometry_facades_preserve_canonical_identity_and_legacy_form():
    periodic = PeriodicBox(shape=(8, 6), lengths=(4.0, 3.0))
    periodic_legacy = PeriodicBox(DomainSpec(shape=(8, 6), lengths=(4.0, 3.0)))
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
        "prescribed_q",
        "robin",
        "strong_homeotropic_q",
        "strong_planar_q",
    }
    p843_policies = {
        value.key: value
        for value in pssolver.available_boundary_policies()
        if value.key
        in {
            "prescribed_q",
            "strong_homeotropic_q",
            "strong_planar_q",
        }
    }
    # P8.4.4 historically connected these declarations. RC4.3.1 found that
    # the lifted wall molecular field does not converge in the current DST
    # remainder representation, so RC4.3.3 keeps them declarable but removes
    # the public execution claim until a new representation is qualified.
    assert all(not value.executable for value in p843_policies.values())
    assert all(value.qualified_applications == () for value in p843_policies.values())
    no_slip = next(
        value for value in pssolver.available_boundary_policies() if value.key == "no_slip_velocity"
    )
    assert "periodic_complete_stress_beris_edwards" in (no_slip.qualified_applications)
    assert catalog.to_metadata()["qualified_policy_equivalences"] == [
        {
            "requested_policy": "no_slip_velocity",
            "geometry_name": "periodic_box",
            "application": "periodic_complete_stress_beris_edwards",
            "effective_policy": "periodic_continuation",
            "reason": "periodic_geometry_has_no_boundary_faces",
        }
    ]
    with pytest.raises(FrozenInstanceError):
        catalog.models[0].key = "changed"
    json.dumps(catalog.to_metadata(), allow_nan=False, sort_keys=True)


def test_p81_machine_record_closes_only_existing_public_capabilities():
    record = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert record["classification"] == ("PASS_P8_1_EXISTING_CAPABILITY_PUBLIC_SURFACE_AND_CATALOG")
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
