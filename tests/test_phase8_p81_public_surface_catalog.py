"""P8.1 public facades for existing qualified capabilities only."""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import pytest

from pssolver.core.domain import DomainSpec
from pssolver.geometries import PeriodicBox, RectangularChannel
from pssolver.geometries.tensor_product import (
    PeriodicBox as CanonicalPeriodicBox,
)
from pssolver.geometries.tensor_product import (
    RectangularChannel as CanonicalRectangularChannel,
)


ROOT = Path(__file__).resolve().parents[1]


RESULT_PATH = ROOT / (
    "notes/architecture_v0_2/phase_8_p81_public_surface_catalog.json"
)


def _boundary_identity(value) -> dict[str, object]:
    metadata = value.to_metadata()
    return {
        "ndim": metadata["ndim"],
        "components": metadata["components"],
    }


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
