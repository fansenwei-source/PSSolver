"""Local qualification for the disconnected Phase 2.1 components.

These provisional values make ownership explicit without connecting the
supported flat RunSpec to a new composition path.  The aggregate validates
its own component graph but is not constructed by the facade or consumed by a
runtime.
"""

from __future__ import annotations

import ast
from dataclasses import (
    FrozenInstanceError,
    MISSING,
    fields,
    is_dataclass,
    replace,
)
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import pickle

import pytest

import pssolver
import pssolver.configuration as configuration_package
import pssolver.models.active_nematics as active_nematics_package
import pssolver.presets as presets_package
from pssolver.configuration import (
    PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES,
    PLANE_FREE_SLIP_BOUNDARIES,
    PlaneFreeSlipBoundaryConditions,
    PlaneRuntimePath,
    SpectralRefreshSpec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    PlaneBerisEdwardsExecutionSpec,
    PlaneBerisEdwardsPhysicsSpec,
    PlaneBerisEdwardsRunComponents,
    PlaneInvocationSpec,
    PlaneTimeSteppingSpec,
    PlaneWorkflowSpec,
)
from pssolver.core import (
    BoundarySet,
    DealiasRule,
    DomainSpec,
    HomogeneousDirichletBC,
    HomogeneousNeumannBC,
    NumericsConfig,
    PeriodicBC,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.geometries import PlaneSlab
from pssolver.models.active_nematics.specifications import (
    BerisEdwardsMaterialRequest,
    ExtrudedDefectGasInitialConditionSpec,
)
from pssolver.presets.shendruk import (
    ShendrukPlaneParameterRequest,
    resolve_shendruk_plane_preset,
)
from pssolver.systems.stokes import (
    IncompressibleStokesSystemSpec,
    PressureGauge,
    TangentialZeroModePolicy,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_SPECIFICATIONS = (
    PROJECT_ROOT / "pssolver" / "models" / "active_nematics" / "specifications.py"
)
PLANE_COMPONENTS = (
    PROJECT_ROOT
    / "pssolver"
    / "configuration"
    / "plane_beris_edwards_components.py"
)
PLANE_COMPONENT_GRAPH = (
    PROJECT_ROOT
    / "pssolver"
    / "configuration"
    / "plane_beris_edwards_component_graph.py"
)


def _refresh() -> SpectralRefreshSpec:
    return SpectralRefreshSpec(
        mode="physical_time",
        requested_interval_time=0.2,
        requested_interval_steps=None,
        effective_interval_steps=20,
        effective_interval_time=0.2,
    )


def _material(**overrides: object) -> BerisEdwardsMaterialRequest:
    values = {
        "ldg_a": 0.0,
        "ldg_b": -0.3,
        "ldg_c": 0.3,
        "gamma": 2.94,
        "flow_alignment": 0.3,
        "beta": -1.0,
    }
    values.update(overrides)
    return BerisEdwardsMaterialRequest(**values)


def _initial(
    **overrides: object,
) -> ExtrudedDefectGasInitialConditionSpec:
    values = {
        "seed": 24,
        "num_defect_pairs": 6,
        "defect_min_separation": 10.0,
        "defect_core_radius": 1.5,
        "background_angle": 0.0,
        "twist_amplitude": 0.01,
        "twist_modes": (1, 2, 3),
        "initial_s": 1.0 / 3.0,
    }
    values.update(overrides)
    return ExtrudedDefectGasInitialConditionSpec(**values)


def _shendruk(**overrides: object) -> ShendrukPlaneParameterRequest:
    values = {
        "activity_number": 18.0,
        "parameterization": "paper-window",
        "frank_k": 0.01,
        "coefficient_min": 0.01,
        "coefficient_max": 0.05,
    }
    values.update(overrides)
    return ShendrukPlaneParameterRequest(**values)


def _time(**overrides: object) -> PlaneTimeSteppingSpec:
    values = {"dt": 0.01, "spectral_refresh": _refresh()}
    values.update(overrides)
    return PlaneTimeSteppingSpec(**values)


def _execution(**overrides: object) -> PlaneBerisEdwardsExecutionSpec:
    values = {
        "device": "auto",
        "tf32": "off",
        "molecular_field_linear_space": "spectral",
        "stress_divergence_sum_space": "spectral",
        "pointwise_execution": "compile",
        "disable_q_gradient_reuse": False,
        "runtime_path": PlaneRuntimePath.LEGACY_PRODUCTION,
    }
    values.update(overrides)
    return PlaneBerisEdwardsExecutionSpec(**values)


def _workflow(**overrides: object) -> PlaneWorkflowSpec:
    values = {
        "output_dir": "oracle/run",
        "steps": 10_000,
        "save_start_step": 5_000,
        "save_interval": 500,
        "diagnostic_interval": 100,
        "diagnostics": False,
        "save_hydrodynamics": False,
        "checkpoint_interval": None,
        "restart_from": None,
    }
    values.update(overrides)
    return PlaneWorkflowSpec(**values)


def _invocation(**overrides: object) -> PlaneInvocationSpec:
    values = {"validation_config_sha256": None, "dry_run": False}
    values.update(overrides)
    return PlaneInvocationSpec(**values)


def _geometry(
    *,
    shape: tuple[int, ...] = (256, 256, 64),
    lengths: tuple[float, ...] = (100.0, 100.0, 20.0),
    wall_normal_axis: int = 2,
) -> PlaneSlab:
    return PlaneSlab(
        DomainSpec(shape, lengths),
        wall_normal_axis=wall_normal_axis,
    )


def _numerics(**overrides: object) -> NumericsConfig:
    values = {
        "precision": Precision.FLOAT32,
        "dealias_rule": DealiasRule.CUBIC_HALF,
        "transform_execution_order": TransformExecutionOrder.REAL_FIRST,
        "projected_transform_execution": (
            ProjectedTransformExecution.TRUNCATED
        ),
        "spectral_storage": SpectralStorage.HERMITIAN_HALF,
        "hermitian_axis": 1,
    }
    values.update(overrides)
    return NumericsConfig(**values)


def _stokes(
    *,
    friction: float = 0.0,
    policy: TangentialZeroModePolicy = TangentialZeroModePolicy.ZERO_MEAN,
) -> IncompressibleStokesSystemSpec:
    return IncompressibleStokesSystemSpec(
        name="flow",
        force_components=("force_x", "force_y", "force_z"),
        velocity_components=("ux", "uy", "uz"),
        pressure_component="p",
        viscosity=2.0 / 3.0,
        friction=friction,
        pressure_gauge=PressureGauge.ZERO_MEAN,
        tangential_zero_mode_policy=policy,
    )


def _physics(**overrides: object) -> PlaneBerisEdwardsPhysicsSpec:
    values = {
        "material": _material(),
        "shendruk_request": _shendruk(),
        "stokes": _stokes(),
        "requested_friction_mode_fric": 0.1,
    }
    values.update(overrides)
    return PlaneBerisEdwardsPhysicsSpec(**values)


def _preset(
    *,
    geometry: PlaneSlab | None = None,
    physics: PlaneBerisEdwardsPhysicsSpec | None = None,
):
    geometry = _geometry() if geometry is None else geometry
    physics = _physics() if physics is None else physics
    raw = physics.shendruk_request
    material = physics.material
    return resolve_shendruk_plane_preset(
        activity_number=raw.activity_number,
        height=geometry.domain.lengths[2],
        parameterization=raw.parameterization,
        frank_k=raw.frank_k,
        coefficient_min=raw.coefficient_min,
        coefficient_max=raw.coefficient_max,
        ldg_a=material.ldg_a,
        ldg_b=material.ldg_b,
        ldg_c=material.ldg_c,
        gamma=material.gamma,
    )


def _components(**overrides: object) -> PlaneBerisEdwardsRunComponents:
    geometry = overrides.pop("geometry", _geometry())
    physics = overrides.pop("physics", _physics())
    values = {
        "geometry": geometry,
        "boundaries": PLANE_FREE_SLIP_BOUNDARIES,
        "effective_boundaries": PLANE_FREE_SLIP_BOUNDARIES,
        "numerics": _numerics(),
        "physics": physics,
        "preset": _preset(geometry=geometry, physics=physics),
        "time_stepping": _time(),
        "initial_condition": _initial(),
        "execution": _execution(),
        "workflow": _workflow(),
        "invocation": _invocation(),
    }
    values.update(overrides)
    return PlaneBerisEdwardsRunComponents(**values)


def test_component_graph_extraction_preserves_protocol4_pickle_oracles():
    values = {
        "PlaneBerisEdwardsPhysicsSpec": PlaneBerisEdwardsPhysicsSpec,
        "PlaneTimeSteppingSpec": PlaneTimeSteppingSpec,
        "PlaneBerisEdwardsExecutionSpec": PlaneBerisEdwardsExecutionSpec,
        "PlaneWorkflowSpec": PlaneWorkflowSpec,
        "PlaneInvocationSpec": PlaneInvocationSpec,
        "PlaneBerisEdwardsRunComponents": PlaneBerisEdwardsRunComponents,
        "representative_physics": _physics(),
        "representative_components": _components(),
    }
    oracles = {
        "PlaneBerisEdwardsPhysicsSpec": (
            101,
            "9b249577e6c264d6fa5e93f46914e380ed8005aa49c7f1ed5e05418471df94c4",
        ),
        "PlaneTimeSteppingSpec": (
            94,
            "16aadde85f367f1ccf355d4d5b752edc4174cbdb54b23413d6bd1b5d17f70b43",
        ),
        "PlaneBerisEdwardsExecutionSpec": (
            103,
            "e76958558d254ab8ba50331ac23b5117e06e630e635cf49de05515eaccc23d26",
        ),
        "PlaneWorkflowSpec": (
            90,
            "c4c86b5a106976a5641234070ef89be82943af6b341ef2875f69a94d3f497bf7",
        ),
        "PlaneInvocationSpec": (
            92,
            "21460de5b3bff745342169653ef7d855dc9995095e62d5cf80369a1ff4843719",
        ),
        "PlaneBerisEdwardsRunComponents": (
            103,
            "f92fab39c5c5f07bd59966b5fa8176cd53813b2516ed1809b5a0bb1a4ceb8dad",
        ),
        "representative_physics": (
            602,
            "6e9855bc3a245a3b8b50a6519d3d3c4b7a8723865e38a29be31f2fb5c4c8ed4b",
        ),
        "representative_components": (
            2318,
            "8900133d1a932248ca599e0c0fc331d624674afc054dff2c520ee76393148cd4",
        ),
    }

    assert set(values) == set(oracles)
    for name, value in values.items():
        payload = pickle.dumps(value, protocol=4)
        expected_length, expected_sha256 = oracles[name]
        assert len(payload) == expected_length
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        restored = pickle.loads(payload)
        if isinstance(value, type):
            assert restored is value
        else:
            assert restored == value


def _representative_values() -> tuple[object, ...]:
    return (
        _material(),
        _initial(twist_modes=[1, 2, 3]),
        _shendruk(),
        _time(),
        _execution(),
        _workflow(),
        _invocation(),
        _physics(),
        _components(),
    )


@pytest.mark.parametrize(
    "value",
    _representative_values(),
    ids=lambda value: type(value).__name__,
)
def test_leaf_components_are_explicit_frozen_slot_value_objects(value):
    value_type = type(value)

    assert is_dataclass(value_type)
    assert value_type.__dataclass_params__.frozen is True
    assert tuple(value_type.__slots__) == tuple(item.name for item in fields(value))
    assert not hasattr(value, "__dict__")
    assert all(item.default is MISSING for item in fields(value))
    assert all(item.default_factory is MISSING for item in fields(value))
    assert value == value
    hash(value)
    with pytest.raises(FrozenInstanceError):
        setattr(value, fields(value)[0].name, None)


def test_leaf_metadata_is_explicit_json_compatible_and_lossless():
    material = _material()
    initial = _initial(twist_modes=[1, 3])
    raw = _shendruk(frank_k=0.02, coefficient_min=0.011)
    time = _time()
    execution = _execution()
    workflow = _workflow(
        output_dir="oracle/a/../run",
        checkpoint_interval=250,
        restart_from="oracle/c/../checkpoint",
    )
    invocation = _invocation(validation_config_sha256="a" * 64, dry_run=True)

    assert material.to_metadata() == {
        "ldg_a": 0.0,
        "ldg_b": -0.3,
        "ldg_c": 0.3,
        "gamma": 2.94,
        "flow_alignment": 0.3,
        "beta": -1.0,
    }
    assert initial.twist_modes == (1, 3)
    assert initial.to_metadata() == {
        "seed": 24,
        "num_defect_pairs": 6,
        "defect_min_separation": 10.0,
        "defect_core_radius": 1.5,
        "background_angle": 0.0,
        "twist_amplitude": 0.01,
        "twist_modes": [1, 3],
        "initial_s": 1.0 / 3.0,
    }
    assert raw.to_metadata() == {
        "activity_number": 18.0,
        "parameterization": "paper-window",
        "frank_k": 0.02,
        "coefficient_min": 0.011,
        "coefficient_max": 0.05,
    }
    assert time.to_metadata() == {
        "dt": 0.01,
        "spectral_refresh": _refresh().to_metadata(),
    }
    assert execution.to_metadata() == {
        "device": "auto",
        "tf32": "off",
        "molecular_field_linear_space": "spectral",
        "stress_divergence_sum_space": "spectral",
        "pointwise_execution": "compile",
        "disable_q_gradient_reuse": False,
        "runtime_path": "legacy_production",
    }
    assert workflow.output_dir == Path("oracle/a/../run")
    assert workflow.restart_from == Path("oracle/c/../checkpoint")
    assert workflow.to_metadata() == {
        "output_dir": "oracle/a/../run",
        "steps": 10_000,
        "save_start_step": 5_000,
        "save_interval": 500,
        "diagnostic_interval": 100,
        "diagnostics": False,
        "save_hydrodynamics": False,
        "checkpoint_interval": 250,
        "restart_from": "oracle/c/../checkpoint",
    }
    assert invocation.to_metadata() == {
        "validation_config_sha256": "a" * 64,
        "dry_run": True,
    }

    for value in (
        material,
        initial,
        raw,
        time,
        execution,
        workflow,
        invocation,
    ):
        json.dumps(value.to_metadata(), allow_nan=False, sort_keys=True)

    mutated = initial.to_metadata()
    mutated["twist_modes"].append(99)
    assert initial.twist_modes == (1, 3)


def test_aggregate_field_order_and_provisional_metadata_are_explicit():
    physics = _physics()
    components = _components(physics=physics)

    assert tuple(item.name for item in fields(PlaneBerisEdwardsPhysicsSpec)) == (
        "material",
        "shendruk_request",
        "stokes",
        "requested_friction_mode_fric",
    )
    assert tuple(item.name for item in fields(PlaneBerisEdwardsRunComponents)) == (
        "geometry",
        "boundaries",
        "effective_boundaries",
        "numerics",
        "physics",
        "preset",
        "time_stepping",
        "initial_condition",
        "execution",
        "workflow",
        "invocation",
    )
    assert components.physics is physics
    physics_metadata = physics.to_metadata()
    assert set(physics_metadata) == {
        "material",
        "shendruk_request",
        "stokes",
        "requested_friction_mode_fric",
    }
    assert physics_metadata["requested_friction_mode_fric"] == 0.1

    metadata = components.to_metadata()
    assert tuple(metadata) == tuple(
        item.name for item in fields(PlaneBerisEdwardsRunComponents)
    )
    assert "schema_version" not in metadata
    assert "authority" not in metadata
    json.dumps(metadata, allow_nan=False, sort_keys=True)

    metadata["physics"]["material"]["gamma"] = 99.0
    assert components.physics.material.gamma == 2.94


@pytest.mark.parametrize(
    ("field_name", "expected_type"),
    (
        ("material", "BerisEdwardsMaterialRequest"),
        ("shendruk_request", "ShendrukPlaneParameterRequest"),
        ("stokes", "IncompressibleStokesSystemSpec"),
    ),
)
def test_physics_aggregate_requires_canonical_declaration_types(
    field_name,
    expected_type,
):
    with pytest.raises(TypeError, match=f"must be a {expected_type}"):
        replace(_physics(), **{field_name: object()})


@pytest.mark.parametrize(
    "value",
    (True, "0.1", Fraction(1, 10), float("nan"), float("inf")),
)
def test_physics_aggregate_requires_finite_requested_friction(value):
    with pytest.raises(ValueError, match="must be a finite int or float"):
        _physics(requested_friction_mode_fric=value)


def test_physics_aggregate_preserves_requested_and_effective_friction():
    dormant = _physics(requested_friction_mode_fric=-0.25)
    friction = _physics(
        stokes=_stokes(
            friction=0.2,
            policy=TangentialZeroModePolicy.FRICTION,
        ),
        requested_friction_mode_fric=0.2,
    )

    assert dormant.requested_friction_mode_fric == -0.25
    assert dormant.stokes.friction == 0.0
    assert friction.requested_friction_mode_fric == 0.2
    assert friction.stokes.friction == 0.2


def test_physics_aggregate_rejects_non_plane_or_mismatched_friction_policy():
    with pytest.raises(ValueError, match="must be zero_mean or friction"):
        _physics(
            stokes=_stokes(policy=TangentialZeroModePolicy.NOT_APPLICABLE)
        )
    with pytest.raises(ValueError, match="must be positive in friction mode"):
        _physics(
            stokes=_stokes(
                friction=0.2,
                policy=TangentialZeroModePolicy.FRICTION,
            ),
            requested_friction_mode_fric=0.0,
        )
    with pytest.raises(ValueError, match="must equal the requested"):
        _physics(
            stokes=_stokes(
                friction=0.2,
                policy=TangentialZeroModePolicy.FRICTION,
            ),
            requested_friction_mode_fric=0.3,
        )


@pytest.mark.parametrize(
    ("field_name", "expected_type"),
    (
        ("geometry", "PlaneSlab"),
        ("boundaries", "PlaneFreeSlipBoundaryConditions"),
        ("effective_boundaries", "PlaneFreeSlipBoundaryConditions"),
        ("numerics", "NumericsConfig"),
        ("physics", "PlaneBerisEdwardsPhysicsSpec"),
        ("preset", "ShendrukPlanePreset"),
        ("time_stepping", "PlaneTimeSteppingSpec"),
        ("initial_condition", "ExtrudedDefectGasInitialConditionSpec"),
        ("execution", "PlaneBerisEdwardsExecutionSpec"),
        ("workflow", "PlaneWorkflowSpec"),
        ("invocation", "PlaneInvocationSpec"),
    ),
)
def test_run_aggregate_requires_canonical_component_types(
    field_name,
    expected_type,
):
    with pytest.raises(TypeError, match=f"must be a {expected_type}"):
        replace(_components(), **{field_name: object()})


def _alternate_boundaries() -> PlaneFreeSlipBoundaryConditions:
    periodic = PeriodicBC()
    return PlaneFreeSlipBoundaryConditions(
        q=BoundarySet(
            (periodic, periodic, HomogeneousDirichletBC())
        ),
        tangential_velocity=BoundarySet(
            (periodic, periodic, HomogeneousNeumannBC())
        ),
        normal_velocity=BoundarySet(
            (periodic, periodic, HomogeneousDirichletBC())
        ),
        pressure_modal=BoundarySet(
            (periodic, periodic, HomogeneousNeumannBC())
        ),
        distortion_odd_z=BoundarySet(
            (periodic, periodic, HomogeneousDirichletBC())
        ),
    )


def test_run_aggregate_keeps_requested_and_effective_boundaries_distinct():
    requested = _alternate_boundaries()
    value = _components(boundaries=requested)

    assert value.boundaries is requested
    assert value.effective_boundaries is PLANE_FREE_SLIP_BOUNDARIES

    with pytest.raises(ValueError, match="qualified Plane free-slip"):
        _components(effective_boundaries=requested)


@pytest.mark.parametrize(
    "geometry",
    (
        _geometry(
            shape=(32, 16),
            lengths=(10.0, 5.0),
            wall_normal_axis=1,
        ),
        _geometry(wall_normal_axis=1),
    ),
)
def test_run_aggregate_requires_the_qualified_plane_topology(geometry):
    with pytest.raises(ValueError, match="three-dimensional Plane slab"):
        replace(_components(), geometry=geometry)


def test_run_aggregate_binds_preset_to_geometry_material_and_raw_request():
    base = _components()
    mismatched_preset = replace(base.preset, zeta=base.preset.zeta * 1.01)
    changed_physics = _physics(
        shendruk_request=_shendruk(activity_number=19.0)
    )

    with pytest.raises(ValueError, match="preset must match"):
        replace(base, preset=mismatched_preset)
    with pytest.raises(ValueError, match="preset must match"):
        replace(base, physics=changed_physics)


def test_run_aggregate_validates_twist_modes_against_nz():
    with pytest.raises(ValueError, match="1 <= mode < nz"):
        _components(initial_condition=_initial(twist_modes=(1, 64)))


@pytest.mark.parametrize(
    ("refresh", "message"),
    (
        (
            SpectralRefreshSpec(
                mode="disabled",
                requested_interval_time=None,
                requested_interval_steps=None,
                effective_interval_steps=1,
                effective_interval_time=None,
            ),
            "must not define intervals",
        ),
        (
            SpectralRefreshSpec(
                mode="steps",
                requested_interval_time=None,
                requested_interval_steps=7,
                effective_interval_steps=8,
                effective_interval_time=0.08,
            ),
            "must preserve requested steps",
        ),
        (
            SpectralRefreshSpec(
                mode="physical_time",
                requested_interval_time=0.21,
                requested_interval_steps=None,
                effective_interval_steps=20,
                effective_interval_time=0.2,
            ),
            "integer multiple of dt",
        ),
    ),
)
def test_run_aggregate_rejects_incoherent_refresh_schedules(refresh, message):
    with pytest.raises(ValueError, match=message):
        _components(time_stepping=_time(spectral_refresh=refresh))


def test_run_aggregate_enforces_plane_hermitian_axis_and_canary_cache_rule():
    with pytest.raises(ValueError, match="qualified periodic Plane axis"):
        _components(numerics=_numerics(hermitian_axis=0))
    with pytest.raises(ValueError, match="does not accept legacy"):
        _components(
            execution=_execution(
                runtime_path=PlaneRuntimePath.SEPARATED_CANARY,
                disable_q_gradient_reuse=True,
            )
        )


def test_raw_shendruk_request_does_not_resolve_or_discard_inputs():
    raw = _shendruk(frank_k=0.02, coefficient_min=0.011, coefficient_max=0.06)
    preset = resolve_shendruk_plane_preset(
        height=20.0,
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        gamma=2.94,
        **raw.to_metadata(),
    )

    assert raw.frank_k == 0.02
    assert raw.coefficient_min == 0.011
    assert raw.coefficient_max == 0.06
    assert preset.frank_k != raw.frank_k
    assert "coefficient_min" not in preset.to_metadata()
    assert "coefficient_max" not in preset.to_metadata()


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"ldg_a": float("nan")}, "ldg_a must be a finite real number"),
        ({"ldg_b": float("inf")}, "ldg_b must be a finite real number"),
        ({"gamma": 0.0}, "gamma must be positive"),
        ({"flow_alignment": True}, "flow_alignment must be a finite"),
        ({"beta": "active"}, "beta must be a finite real number"),
    ),
)
def test_material_request_rejects_invalid_local_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        _material(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"seed": True}, "seed must be an integer"),
        ({"num_defect_pairs": 0}, "num_defect_pairs must be a positive"),
        ({"num_defect_pairs": 1.5}, "num_defect_pairs must be a positive"),
        ({"defect_min_separation": 0.0}, "must be positive"),
        ({"defect_core_radius": float("inf")}, "must be a finite"),
        ({"background_angle": float("nan")}, "must be a finite"),
        ({"twist_amplitude": -0.1}, "must be non-negative"),
        ({"twist_modes": "1 2"}, "must be an iterable"),
        ({"twist_modes": ()}, "must not be empty"),
        ({"twist_modes": (1, 1)}, "must be unique"),
        ({"twist_modes": (0, 1)}, "twist mode must be a positive"),
        ({"twist_modes": (True, 1)}, "twist mode must be a positive"),
        ({"initial_s": 0.0}, "initial_s must be positive"),
    ),
)
def test_initial_condition_request_rejects_invalid_local_values(
    overrides,
    message,
):
    with pytest.raises(ValueError, match=message):
        _initial(**overrides)


@pytest.mark.parametrize(
    ("overrides", "message"),
    (
        ({"parameterization": "automatic"}, "parameterization must be"),
        ({"activity_number": float("nan")}, "must be finite numbers"),
        ({"activity_number": True}, "must be finite numbers"),
        ({"activity_number": 0.0}, "activity number.*must be positive"),
        ({"frank_k": 0.0}, "raw Frank K must be positive"),
        ({"coefficient_min": 0.0}, "bounds must be positive"),
        (
            {"coefficient_min": 0.05, "coefficient_max": 0.05},
            "strictly increasing",
        ),
    ),
)
def test_raw_shendruk_request_rejects_invalid_local_values(overrides, message):
    with pytest.raises(ValueError, match=message):
        _shendruk(**overrides)


@pytest.mark.parametrize("value", (0.0, -0.01, float("nan"), True))
def test_time_spec_requires_positive_finite_dt(value):
    with pytest.raises(ValueError, match="dt must be positive and finite"):
        _time(dt=value)


def test_time_spec_requires_the_canonical_refresh_type():
    with pytest.raises(TypeError, match="must be a SpectralRefreshSpec"):
        _time(spectral_refresh={"mode": "disabled"})


def test_cross_component_refresh_consistency_is_deliberately_deferred():
    inconsistent = SpectralRefreshSpec(
        mode="steps",
        requested_interval_time=None,
        requested_interval_steps=7,
        effective_interval_steps=9,
        effective_interval_time=0.123,
    )

    value = _time(spectral_refresh=inconsistent)

    assert value.spectral_refresh is inconsistent


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    (
        ({"device": ""}, ValueError, "device must be a non-empty string"),
        ({"tf32": "auto"}, ValueError, "tf32 must be one of"),
        (
            {"molecular_field_linear_space": "mixed"},
            ValueError,
            "molecular_field_linear_space must be one of",
        ),
        (
            {"stress_divergence_sum_space": "mixed"},
            ValueError,
            "stress_divergence_sum_space must be one of",
        ),
        (
            {"pointwise_execution": "automatic"},
            ValueError,
            "pointwise_execution must be one of",
        ),
        (
            {"disable_q_gradient_reuse": 0},
            TypeError,
            "disable_q_gradient_reuse must be a bool",
        ),
        (
            {"runtime_path": "legacy_production"},
            TypeError,
            "runtime_path must be a PlaneRuntimePath",
        ),
    ),
)
def test_execution_request_rejects_invalid_local_values(
    overrides,
    error,
    message,
):
    with pytest.raises(error, match=message):
        _execution(**overrides)


def test_separated_canary_cache_restriction_is_deliberately_deferred():
    value = _execution(
        runtime_path=PlaneRuntimePath.SEPARATED_CANARY,
        disable_q_gradient_reuse=True,
    )

    assert value.runtime_path is PlaneRuntimePath.SEPARATED_CANARY
    assert value.disable_q_gradient_reuse is True


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    (
        ({"output_dir": None}, TypeError, "output_dir must be path-like"),
        ({"steps": 0}, ValueError, "steps must be a positive integer"),
        ({"steps": True}, ValueError, "steps must be a positive integer"),
        (
            {"save_start_step": -1},
            ValueError,
            "save_start_step must be a non-negative integer",
        ),
        (
            {"save_start_step": 10_001},
            ValueError,
            "save_start_step must not exceed steps",
        ),
        (
            {"save_interval": 0},
            ValueError,
            "save_interval must be a positive integer",
        ),
        (
            {"diagnostic_interval": 0},
            ValueError,
            "diagnostic_interval must be a positive integer",
        ),
        ({"diagnostics": 0}, TypeError, "diagnostics must be a bool"),
        (
            {"save_hydrodynamics": 1},
            TypeError,
            "save_hydrodynamics must be a bool",
        ),
        (
            {"checkpoint_interval": 0},
            ValueError,
            "checkpoint_interval must be a positive integer",
        ),
        (
            {"restart_from": object()},
            TypeError,
            "restart_from must be path-like",
        ),
    ),
)
def test_workflow_request_rejects_invalid_local_values(
    overrides,
    error,
    message,
):
    with pytest.raises(error, match=message):
        _workflow(**overrides)


@pytest.mark.parametrize(
    ("overrides", "error", "message"),
    (
        (
            {"validation_config_sha256": "A" * 64},
            ValueError,
            "64 lowercase hexadecimal",
        ),
        (
            {"validation_config_sha256": "a" * 63},
            ValueError,
            "64 lowercase hexadecimal",
        ),
        (
            {"validation_config_sha256": 7},
            ValueError,
            "64 lowercase hexadecimal",
        ),
        ({"dry_run": 0}, TypeError, "dry_run must be a bool"),
    ),
)
def test_invocation_request_rejects_invalid_local_values(
    overrides,
    error,
    message,
):
    with pytest.raises(error, match=message):
        _invocation(**overrides)


def _import_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                roots.add(f"relative:{node.module or ''}")
            elif node.module:
                roots.add(node.module.split(".", 1)[0])
    return roots


def test_new_leaf_modules_preserve_the_dependency_boundary():
    assert _import_roots(MODEL_SPECIFICATIONS) == {
        "__future__",
        "dataclasses",
        "math",
        "numbers",
    }
    assert _import_roots(PLANE_COMPONENTS) == {
        "__future__",
        "relative:plane_beris_edwards",
        "relative:plane_beris_edwards_component_graph",
        "relative:plane_beris_edwards_declarations",
    }
    assert _import_roots(PLANE_COMPONENT_GRAPH) == {
        "__future__",
        "dataclasses",
        "math",
        "numbers",
        "pathlib",
        "pssolver",
        "relative:plane_beris_edwards_builders",
        "relative:plane_beris_edwards_declarations",
    }

    forbidden_text = (
        "torch",
        "numpy",
        "pssolver.backends",
        "pssolver.execution",
        "pssolver.experimental",
        "pssolver.linear_solvers",
        "pssolver.operators",
        "pssolver.runtime",
        "pssolver.workflows",
        "pssolver.applications",
        "pssolver.plane",
        "pssolver.transforms",
    )
    for path in (
        MODEL_SPECIFICATIONS,
        PLANE_COMPONENTS,
        PLANE_COMPONENT_GRAPH,
    ):
        source = path.read_text(encoding="utf-8")
        assert all(name not in source for name in forbidden_text)
    graph_source = PLANE_COMPONENT_GRAPH.read_text(encoding="utf-8")
    assert "from pssolver.systems.stokes import" in graph_source
    assert "from pssolver.execution" not in graph_source


def test_phase2_components_and_adapter_remain_provisional_and_disconnected():
    provisional_names = {
        "BerisEdwardsMaterialRequest",
        "ExtrudedDefectGasInitialConditionSpec",
        "ShendrukPlaneParameterRequest",
        "PlaneTimeSteppingSpec",
        "PlaneBerisEdwardsExecutionSpec",
        "PlaneBerisEdwardsPhysicsSpec",
        "PlaneBerisEdwardsRunComponents",
        "PlaneWorkflowSpec",
        "PlaneInvocationSpec",
    }
    for exports in (
        pssolver.__all__,
        configuration_package.__all__,
        active_nematics_package.__all__,
        presets_package.__all__,
    ):
        assert provisional_names.isdisjoint(exports)

    import pssolver.configuration.plane_beris_edwards_components as components

    assert components.PlaneBerisEdwardsPhysicsSpec is (
        PlaneBerisEdwardsPhysicsSpec
    )
    assert components.PlaneBerisEdwardsRunComponents is (
        PlaneBerisEdwardsRunComponents
    )
    assert "PlaneBerisEdwardsPhysicsSpec" in components.__all__
    assert "PlaneBerisEdwardsRunComponents" in components.__all__
    assert hasattr(components, "decompose_plane_beris_edwards_run_spec")
    assert "decompose_plane_beris_edwards_run_spec" in components.__all__
    assert "pssolver/models/active_nematics/specifications.py" not in (
        PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES
    )
    assert "pssolver/configuration/plane_beris_edwards_components.py" not in (
        PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES
    )

    for relative in (
        "pssolver/configuration/plane_beris_edwards.py",
        "pssolver/applications/plane_beris_edwards.py",
        "pssolver/runtime/plane_beris_edwards.py",
        "pssolver/runtime/plane_legacy.py",
        "pssolver/workflows/plane_beris_edwards.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "plane_beris_edwards_components" not in source
        assert "active_nematics.specifications" not in source
