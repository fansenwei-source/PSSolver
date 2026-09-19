"""P2.3 pre-extraction oracles for shared Plane declarations and views."""

from __future__ import annotations

import base64
from dataclasses import MISSING, fields, replace
from enum import Enum
import hashlib
import inspect
import json
from pathlib import Path
import pickle
from typing import get_type_hints

import pytest

import pssolver
import pssolver.configuration as configuration
from pssolver.configuration import (
    PLANE_FREE_SLIP_BOUNDARIES,
    PlaneBerisEdwardsRunSpec,
    PlaneFreeSlipBoundaryConditions,
    PlaneRuntimePath,
    SpectralRefreshSpec,
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration import plane_beris_edwards as facade
from pssolver.core import (
    BoundarySet,
    DomainSpec,
    HomogeneousDirichletBC,
    NumericsConfig,
    PeriodicBC,
)
from pssolver.geometries import PlaneSlab
from pssolver.presets import ShendrukPlanePreset


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "configuration"
SHARED_ORACLE = json.loads(
    (FIXTURE_ROOT / "plane_shared_declarations_api_v1.json").read_text(
        encoding="utf-8"
    )
)
RUN_CASES = json.loads(
    (FIXTURE_ROOT / "plane_run_spec_v1_cases.json").read_text(
        encoding="utf-8"
    )
)


def _spec(**overrides: object) -> PlaneBerisEdwardsRunSpec:
    values = dict(RUN_CASES["base_factory_kwargs"])
    values.update(overrides)
    return create_plane_beris_edwards_run_spec(**values)


def _field_surface(value_type: type) -> list[list[str]]:
    surface = []
    for item in fields(value_type):
        if item.default_factory is not MISSING:
            default = "factory"
        elif item.default is not MISSING:
            default = "value"
        else:
            default = "required"
        surface.append(
            [item.name, inspect.formatannotation(item.type), default]
        )
    return surface


def _alternate_boundaries() -> PlaneFreeSlipBoundaryConditions:
    return replace(
        PLANE_FREE_SLIP_BOUNDARIES,
        q=BoundarySet(
            (PeriodicBC(), PeriodicBC(), HomogeneousDirichletBC())
        ),
    )


def _pickle_objects() -> dict[str, object]:
    return {
        "PlaneRuntimePath_class": PlaneRuntimePath,
        "PlaneFreeSlipBoundaryConditions_class": (
            PlaneFreeSlipBoundaryConditions
        ),
        "SpectralRefreshSpec_class": SpectralRefreshSpec,
        "runtime_legacy_member": PlaneRuntimePath.LEGACY_PRODUCTION,
        "runtime_canary_member": PlaneRuntimePath.SEPARATED_CANARY,
        "default_boundaries": PLANE_FREE_SLIP_BOUNDARIES,
        "alternate_boundaries": _alternate_boundaries(),
        "refresh_physical_time": SpectralRefreshSpec(
            "physical_time", 0.2, None, 20, 0.2
        ),
        "refresh_steps": SpectralRefreshSpec(
            "steps", None, 7, 7, 0.07
        ),
        "refresh_disabled": SpectralRefreshSpec(
            "disabled", None, None, None, None
        ),
        "default_run_spec": _spec(),
    }


def test_shared_declaration_surfaces_match_the_pre_extraction_oracle():
    runtime = SHARED_ORACLE["runtime_path"]
    assert PlaneRuntimePath.__module__ == runtime["module"]
    assert PlaneRuntimePath.__qualname__ == runtime["qualname"]
    assert issubclass(PlaneRuntimePath, str)
    assert issubclass(PlaneRuntimePath, Enum)
    assert [[item.name, item.value] for item in PlaneRuntimePath] == (
        runtime["members"]
    )

    for value_type, key in (
        (PlaneFreeSlipBoundaryConditions, "boundaries"),
        (SpectralRefreshSpec, "spectral_refresh"),
    ):
        expected = SHARED_ORACLE[key]
        assert value_type.__module__ == expected["module"]
        assert value_type.__qualname__ == expected["qualname"]
        assert str(inspect.signature(value_type)) == expected["signature"]
        assert _field_surface(value_type) == expected["fields"]
        names = [item.name for item in fields(value_type)]
        assert list(value_type.__slots__) == names
        assert list(value_type.__match_args__) == names
        parameters = value_type.__dataclass_params__
        assert parameters.frozen is True
        assert parameters.eq is True
        assert parameters.order is False
        assert parameters.unsafe_hash is False


def test_legacy_facade_and_stable_package_share_exact_objects():
    for name in (
        "PlaneRuntimePath",
        "PlaneFreeSlipBoundaryConditions",
        "SpectralRefreshSpec",
        "PLANE_FREE_SLIP_BOUNDARIES",
    ):
        assert getattr(facade, name) is getattr(configuration, name)
        assert not hasattr(pssolver, name)

    spec = _spec()
    assert spec.boundaries is PLANE_FREE_SLIP_BOUNDARIES
    assert spec.runtime_path is PlaneRuntimePath.LEGACY_PRODUCTION


def test_legacy_nominal_module_resolves_postponed_boundary_annotations():
    hints = get_type_hints(PlaneFreeSlipBoundaryConditions)
    assert hints == {
        "q": BoundarySet,
        "tangential_velocity": BoundarySet,
        "normal_velocity": BoundarySet,
        "pressure_modal": BoundarySet,
        "distortion_odd_z": BoundarySet,
    }


def test_protocol4_pickle_bytes_and_legacy_globals_are_frozen():
    objects = _pickle_objects()
    oracle = SHARED_ORACLE["protocol4_pickles"]
    assert set(objects) == set(oracle)

    identity_names = {
        "PlaneRuntimePath_class",
        "PlaneFreeSlipBoundaryConditions_class",
        "SpectralRefreshSpec_class",
        "runtime_legacy_member",
        "runtime_canary_member",
    }
    for name, value in objects.items():
        expected = oracle[name]
        payload = base64.b64decode(expected["base64"])
        assert hashlib.sha256(payload).hexdigest() == expected["sha256"]
        assert pickle.dumps(value, protocol=4) == payload
        restored = pickle.loads(payload)
        if name in identity_names:
            assert restored is value
        else:
            assert restored == value
            assert type(restored) is type(value)


@pytest.mark.parametrize("protocol", (0, 2, 4, pickle.HIGHEST_PROTOCOL))
def test_shared_declarations_round_trip_under_supported_pickle_protocols(
    protocol: int,
):
    for name, value in _pickle_objects().items():
        restored = pickle.loads(pickle.dumps(value, protocol=protocol))
        if name.endswith("_class") or name.startswith("runtime_"):
            assert restored is value
        else:
            assert restored == value
            assert type(restored) is type(value)


def test_shared_leaf_validation_timing_is_frozen():
    with pytest.raises(
        TypeError,
        match="Plane boundary entries must be BoundarySet objects",
    ):
        PlaneFreeSlipBoundaryConditions(q=object())
    with pytest.raises(
        ValueError,
        match="Plane boundary entries must be three-dimensional",
    ):
        PlaneFreeSlipBoundaryConditions(
            q=BoundarySet((PeriodicBC(), PeriodicBC()))
        )
    with pytest.raises(ValueError, match="invalid spectral refresh mode"):
        SpectralRefreshSpec("invalid", None, None, None, None)

    # Cross-field consistency belongs to the composition root, not this leaf.
    inconsistent = SpectralRefreshSpec(
        "disabled", 0.2, 9, 11, 0.3
    )
    assert inconsistent.requested_interval_time == 0.2
    assert inconsistent.requested_interval_steps == 9


@pytest.mark.parametrize("case", RUN_CASES["cases"], ids=lambda x: x["name"])
def test_four_derived_views_match_all_schema_v1_golden_cases(case):
    spec = _spec(**case["factory_overrides"])
    metadata = case["expected_metadata"]

    assert isinstance(spec.domain, DomainSpec)
    assert isinstance(spec.geometry, PlaneSlab)
    assert isinstance(spec.numerics, NumericsConfig)
    assert isinstance(spec.shendruk_preset, ShendrukPlanePreset)
    assert spec.domain.to_metadata() == metadata["geometry"]["domain"]
    assert spec.geometry.to_metadata() == metadata["geometry"]
    assert spec.numerics.to_metadata() == metadata["numerics"]
    assert spec.shendruk_preset.to_metadata() == metadata["preset"]


def test_derived_views_remain_fresh_uncached_values():
    spec = _spec()
    for name in ("domain", "geometry", "numerics", "shendruk_preset"):
        first = getattr(spec, name)
        second = getattr(spec, name)
        assert first == second
        assert first is not second
    assert spec.geometry.domain == spec.domain
    assert spec.geometry.domain is not spec.domain


def test_derived_view_validation_remains_narrow_and_field_local():
    baseline = _spec()
    unrelated = replace(
        baseline,
        dt=-0.01,
        steps=-1,
        initial_s=-1.0,
        spectral_refresh=SpectralRefreshSpec(
            "disabled", 0.2, 9, 11, 0.3
        ),
    )
    for name in ("domain", "geometry", "numerics", "shendruk_preset"):
        assert getattr(unrelated, name) == getattr(baseline, name)

    invalid_dtype = replace(baseline, dtype="invalid")
    assert invalid_dtype.domain == baseline.domain
    assert invalid_dtype.geometry == baseline.geometry
    assert invalid_dtype.shendruk_preset == baseline.shendruk_preset
    with pytest.raises(ValueError, match="not a valid Precision"):
        _ = invalid_dtype.numerics

    invalid_shape = replace(baseline, nx=0)
    with pytest.raises(ValueError, match="positive integer"):
        _ = invalid_shape.domain
    with pytest.raises(ValueError, match="positive integer"):
        _ = invalid_shape.geometry
    assert invalid_shape.numerics == baseline.numerics
    assert invalid_shape.shendruk_preset == baseline.shendruk_preset

    invalid_activity = replace(baseline, activity_number=-1.0)
    assert invalid_activity.domain == baseline.domain
    assert invalid_activity.geometry == baseline.geometry
    assert invalid_activity.numerics == baseline.numerics
    with pytest.raises(
        ValueError,
        match="activity number, height, and Frank K must be positive",
    ):
        _ = invalid_activity.shendruk_preset
