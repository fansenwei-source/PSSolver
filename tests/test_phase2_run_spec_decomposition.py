"""P2.2 parity gates for the provisional flat-facade decomposition adapter."""

from __future__ import annotations

import ast
from dataclasses import fields, replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import pssolver
import pssolver.configuration as configuration_package
from pssolver.configuration import (
    PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES,
    PLANE_FREE_SLIP_BOUNDARIES,
    PlaneBerisEdwardsRunSpec,
    PlaneFreeSlipBoundaryConditions,
    create_plane_beris_edwards_run_spec,
    parse_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    PlaneBerisEdwardsRunComponents,
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.core import (
    BoundarySet,
    HomogeneousDirichletBC,
    PeriodicBC,
)
from pssolver.systems.stokes import (
    PressureGauge,
    TangentialZeroModePolicy,
)
from pssolver.workflows import read_plane_checkpoint_header


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIXTURE_ROOT = PROJECT_ROOT / "tests" / "fixtures" / "configuration"
CASES_ORACLE = json.loads(
    (FIXTURE_ROOT / "plane_run_spec_v1_cases.json").read_text(
        encoding="utf-8"
    )
)
CHECKPOINT_FIXTURE = FIXTURE_ROOT / "plane_checkpoint_v1_header"
COMPONENT_MODULE = (
    PROJECT_ROOT
    / "pssolver"
    / "configuration"
    / "plane_beris_edwards_components.py"
)
COMPONENT_GRAPH_MODULE = (
    PROJECT_ROOT
    / "pssolver"
    / "configuration"
    / "plane_beris_edwards_component_graph.py"
)
FACADE_MODULE = (
    PROJECT_ROOT / "pssolver" / "configuration" / "plane_beris_edwards.py"
)


def _base_kwargs(**overrides: object) -> dict[str, object]:
    values = dict(CASES_ORACLE["base_factory_kwargs"])
    values.update(overrides)
    return values


def _spec(**overrides: object) -> PlaneBerisEdwardsRunSpec:
    return create_plane_beris_edwards_run_spec(**_base_kwargs(**overrides))


def _alternate_boundaries() -> PlaneFreeSlipBoundaryConditions:
    alternate_q = BoundarySet(
        (PeriodicBC(), PeriodicBC(), HomogeneousDirichletBC())
    )
    return replace(PLANE_FREE_SLIP_BOUNDARIES, q=alternate_q)


def _recover_flat_fields(
    value: PlaneBerisEdwardsRunComponents,
) -> dict[str, object]:
    """Test-only lossless projection back to the 54 facade fields."""

    domain = value.geometry.domain
    physics = value.physics
    material = physics.material
    shendruk = physics.shendruk_request
    stokes = physics.stokes
    numerics = value.numerics
    time = value.time_stepping
    initial = value.initial_condition
    execution = value.execution
    workflow = value.workflow
    invocation = value.invocation
    return {
        "activity_number": shendruk.activity_number,
        "output_dir": workflow.output_dir,
        "height": domain.lengths[2],
        "parameterization": shendruk.parameterization,
        "frank_k": shendruk.frank_k,
        "coefficient_min": shendruk.coefficient_min,
        "coefficient_max": shendruk.coefficient_max,
        "lx": domain.lengths[0],
        "ly": domain.lengths[1],
        "nx": domain.shape[0],
        "ny": domain.shape[1],
        "nz": domain.shape[2],
        "dt": time.dt,
        "steps": workflow.steps,
        "save_start_step": workflow.save_start_step,
        "save_interval": workflow.save_interval,
        "diagnostic_interval": workflow.diagnostic_interval,
        "seed": initial.seed,
        "num_defect_pairs": initial.num_defect_pairs,
        "defect_min_separation": initial.defect_min_separation,
        "defect_core_radius": initial.defect_core_radius,
        "background_angle": initial.background_angle,
        "twist_amplitude": initial.twist_amplitude,
        "twist_modes": initial.twist_modes,
        "ldg_a": material.ldg_a,
        "ldg_b": material.ldg_b,
        "ldg_c": material.ldg_c,
        "gamma": material.gamma,
        "flow_alignment": material.flow_alignment,
        "eta": stokes.viscosity,
        "zero_mode_policy": stokes.tangential_zero_mode_policy.value,
        "friction_mode_fric": physics.requested_friction_mode_fric,
        "dealias_rule": numerics.dealias_rule.value,
        "projected_transform_execution": (
            numerics.projected_transform_execution.value
        ),
        "beta": material.beta,
        "initial_s": initial.initial_s,
        "device": execution.device,
        "dtype": numerics.precision.value,
        "molecular_field_linear_space": (
            execution.molecular_field_linear_space
        ),
        "stress_divergence_sum_space": (
            execution.stress_divergence_sum_space
        ),
        "pointwise_execution": execution.pointwise_execution,
        "transform_execution_order": (
            numerics.transform_execution_order.value
        ),
        "spectral_storage": numerics.spectral_storage.value,
        "tf32": execution.tf32,
        "spectral_refresh": time.spectral_refresh,
        "diagnostics": workflow.diagnostics,
        "disable_q_gradient_reuse": execution.disable_q_gradient_reuse,
        "save_hydrodynamics": workflow.save_hydrodynamics,
        "validation_config_sha256": invocation.validation_config_sha256,
        "dry_run": invocation.dry_run,
        "checkpoint_interval": workflow.checkpoint_interval,
        "restart_from": workflow.restart_from,
        "runtime_path": execution.runtime_path,
        "boundaries": value.boundaries,
    }


def _canonical_json(value: object) -> str:
    encoding = CASES_ORACLE["canonical_json_encoding"]
    return json.dumps(
        value,
        allow_nan=encoding["allow_nan"],
        separators=tuple(encoding["separators"]),
        sort_keys=encoding["sort_keys"],
    )


def _round_trip(
    spec: PlaneBerisEdwardsRunSpec,
) -> tuple[PlaneBerisEdwardsRunComponents, PlaneBerisEdwardsRunSpec]:
    components = decompose_plane_beris_edwards_run_spec(spec)
    recovered = _recover_flat_fields(components)
    expected_names = tuple(item.name for item in fields(spec))

    assert len(expected_names) == len(set(expected_names)) == 54
    assert tuple(recovered) == expected_names
    assert recovered == {
        name: getattr(spec, name)
        for name in expected_names
    }
    for name in expected_names:
        assert type(recovered[name]) is type(getattr(spec, name))
    assert recovered["spectral_refresh"] is spec.spectral_refresh
    assert recovered["boundaries"] is spec.boundaries

    reconstructed = PlaneBerisEdwardsRunSpec(**recovered)
    assert reconstructed == spec
    assert reconstructed.to_metadata() == spec.to_metadata()
    assert _canonical_json(reconstructed.to_metadata()) == _canonical_json(
        spec.to_metadata()
    )
    assert reconstructed.canonical_sha256() == spec.canonical_sha256()
    assert (
        reconstructed.runtime_identity_metadata()
        == spec.runtime_identity_metadata()
    )
    assert (
        reconstructed.runtime_identity_sha256()
        == spec.runtime_identity_sha256()
    )
    assert reconstructed.identity_metadata() == spec.identity_metadata()
    assert (
        reconstructed.runtime_selection_metadata()
        == spec.runtime_selection_metadata()
    )
    return components, reconstructed


IDENTITY_MUTATIONS = {
    "activity_number": {"activity_number": 19.0},
    "output_dir": {"output_dir": "oracle/identity-output"},
    "height": {"height": 21.0},
    "parameterization": {"parameterization": "fixed-k"},
    "frank_k": {"frank_k": 0.02},
    "coefficient_min": {"coefficient_min": 0.011},
    "coefficient_max": {"coefficient_max": 0.06},
    "lx": {"lx": 101.0},
    "ly": {"ly": 102.0},
    "nx": {"nx": 258},
    "ny": {"ny": 260},
    "nz": {"nz": 80},
    "dt": {"dt": 0.005},
    "steps": {"steps": 12_000},
    "save_start_step": {"save_start_step": 4_000},
    "save_interval": {"save_interval": 750},
    "diagnostic_interval": {"diagnostic_interval": 125},
    "seed": {"seed": 25},
    "num_defect_pairs": {"num_defect_pairs": 7},
    "defect_min_separation": {"defect_min_separation": 11.0},
    "defect_core_radius": {"defect_core_radius": 1.6},
    "background_angle": {"background_angle": 0.1},
    "twist_amplitude": {"twist_amplitude": 0.02},
    "twist_modes": {"twist_modes": (1, 4)},
    "ldg_a": {"ldg_a": -0.01},
    "ldg_b": {"ldg_b": -0.31},
    "ldg_c": {"ldg_c": 0.31},
    "gamma": {"gamma": 3.0},
    "flow_alignment": {"flow_alignment": 0.31},
    "eta": {"eta": 0.7},
    "zero_mode_policy": {"zero_mode_policy": "friction"},
    "friction_mode_fric": {"friction_mode_fric": 0.2},
    "dealias_rule": {"dealias_rule": "two_thirds"},
    "projected_transform_execution": {
        "projected_transform_execution": "full"
    },
    "beta": {"beta": -0.9},
    "initial_s": {"initial_s": 0.25},
    "device": {"device": "cpu"},
    "dtype": {"dtype": "float64"},
    "molecular_field_linear_space": {
        "molecular_field_linear_space": "physical"
    },
    "stress_divergence_sum_space": {
        "stress_divergence_sum_space": "physical"
    },
    "pointwise_execution": {"pointwise_execution": "eager"},
    "transform_execution_order": {"transform_execution_order": "legacy"},
    "spectral_storage": {"spectral_storage": "full_complex"},
    "tf32": {"tf32": "on"},
    "spectral_refresh": {"spectral_refresh_steps": 7},
    "diagnostics": {"diagnostics": True},
    "disable_q_gradient_reuse": {"disable_q_gradient_reuse": True},
    "save_hydrodynamics": {"save_hydrodynamics": True},
    "validation_config_sha256": {"validation_config_sha256": "a" * 64},
    "dry_run": {"dry_run": True},
    "checkpoint_interval": {"checkpoint_interval": 250},
    "restart_from": {"restart_from": "oracle/checkpoint"},
    "runtime_path": {"runtime_path": "separated_canary"},
}


def _identity_variant(field_name: str) -> PlaneBerisEdwardsRunSpec:
    baseline_overrides = {}
    if field_name == "frank_k":
        baseline_overrides["parameterization"] = "fixed-k"
    elif field_name == "transform_execution_order":
        baseline_overrides["spectral_storage"] = "full_complex"

    baseline = _spec(**baseline_overrides)
    if field_name == "boundaries":
        return replace(baseline, boundaries=_alternate_boundaries())
    changed_overrides = dict(baseline_overrides)
    changed_overrides.update(IDENTITY_MUTATIONS[field_name])
    return _spec(**changed_overrides)


@pytest.mark.parametrize(
    "case",
    CASES_ORACLE["cases"],
    ids=lambda case: case["name"],
)
def test_all_schema_v1_golden_cases_round_trip_through_components(case):
    spec = _spec(**case["factory_overrides"])
    _, reconstructed = _round_trip(spec)

    for value in (spec, reconstructed):
        assert value.to_metadata() == case["expected_metadata"]
        assert _canonical_json(value.to_metadata()) == (
            case["expected_canonical_json"]
        )
        assert value.canonical_sha256() == case["canonical_sha256"]
        assert value.runtime_identity_metadata() == (
            case["expected_runtime_identity_metadata"]
        )
        assert _canonical_json(value.runtime_identity_metadata()) == (
            case["expected_runtime_identity_json"]
        )
        assert value.runtime_identity_sha256() == (
            case["runtime_identity_sha256"]
        )
        assert value.identity_metadata() == case["expected_identity_metadata"]
        assert value.runtime_selection_metadata() == (
            case["expected_runtime_selection_metadata"]
        )


IDENTITY_FIELDS = tuple(item.name for item in fields(PlaneBerisEdwardsRunSpec))


@pytest.mark.parametrize("field_name", IDENTITY_FIELDS)
def test_each_of_the_54_flat_fields_is_recoverable(field_name):
    assert set(IDENTITY_MUTATIONS) | {"boundaries"} == set(IDENTITY_FIELDS)
    _round_trip(_identity_variant(field_name))


@pytest.mark.parametrize("field_name", IDENTITY_FIELDS)
def test_each_field_keeps_delegated_views_equal_to_adapter_components(
    field_name,
):
    spec = _identity_variant(field_name)
    components, _ = _round_trip(spec)

    assert spec.domain == components.geometry.domain
    assert spec.geometry == components.geometry
    assert spec.numerics == components.numerics
    assert spec.shendruk_preset == components.preset


def test_asymmetric_values_land_in_their_independent_component_owners():
    spec = replace(
        _spec(
            activity_number=19.0,
            output_dir="oracle/asymmetric/a/../run",
            height=23.0,
            parameterization="fixed-k",
            frank_k=0.012,
            coefficient_min=0.011,
            coefficient_max=0.061,
            lx=101.0,
            ly=103.0,
            nx=258,
            ny=260,
            nz=82,
            dt=0.005,
            steps=12_003,
            save_start_step=4_001,
            save_interval=751,
            diagnostic_interval=127,
            seed=25,
            num_defect_pairs=7,
            defect_min_separation=11.0,
            defect_core_radius=1.6,
            background_angle=0.11,
            twist_amplitude=0.021,
            twist_modes=(1, 4, 7),
            ldg_a=-0.01,
            ldg_b=-0.31,
            ldg_c=0.32,
            gamma=3.01,
            flow_alignment=0.33,
            eta=0.71,
            zero_mode_policy="friction",
            friction_mode_fric=0.125,
            dealias_rule="two_thirds",
            projected_transform_execution="full",
            beta=-0.91,
            initial_s=0.25,
            device="cpu",
            dtype="float64",
            molecular_field_linear_space="physical",
            stress_divergence_sum_space="physical",
            pointwise_execution="eager",
            transform_execution_order="legacy",
            spectral_storage="full_complex",
            tf32="on",
            spectral_refresh_steps=7,
            diagnostics=True,
            disable_q_gradient_reuse=True,
            save_hydrodynamics=True,
            validation_config_sha256="b" * 64,
            dry_run=True,
            checkpoint_interval=251,
            restart_from="oracle/asymmetric/c/../checkpoint",
        ),
        boundaries=_alternate_boundaries(),
    )
    value = decompose_plane_beris_edwards_run_spec(spec)

    assert value.geometry.domain.shape == (258, 260, 82)
    assert value.geometry.domain.lengths == (101.0, 103.0, 23.0)
    assert tuple(value.physics.material.to_metadata().values()) == (
        -0.01,
        -0.31,
        0.32,
        3.01,
        0.33,
        -0.91,
    )
    assert tuple(value.physics.shendruk_request.to_metadata().values()) == (
        19.0,
        "fixed-k",
        0.012,
        0.011,
        0.061,
    )
    assert value.physics.stokes.viscosity == 0.71
    assert value.physics.stokes.friction == 0.125
    assert value.physics.requested_friction_mode_fric == 0.125
    assert tuple(value.initial_condition.to_metadata().values()) == (
        25,
        7,
        11.0,
        1.6,
        0.11,
        0.021,
        [1, 4, 7],
        0.25,
    )
    assert value.time_stepping.dt == 0.005
    assert value.time_stepping.spectral_refresh is spec.spectral_refresh
    assert tuple(value.execution.to_metadata().values()) == (
        "cpu",
        "on",
        "physical",
        "physical",
        "eager",
        True,
        "legacy_production",
    )
    assert tuple(value.workflow.to_metadata().values()) == (
        "oracle/asymmetric/a/../run",
        12_003,
        4_001,
        751,
        127,
        True,
        True,
        251,
        "oracle/asymmetric/c/../checkpoint",
    )
    assert value.invocation.to_metadata() == {
        "validation_config_sha256": "b" * 64,
        "dry_run": True,
    }
    assert value.boundaries is spec.boundaries
    assert value.effective_boundaries is PLANE_FREE_SLIP_BOUNDARIES


def test_raw_shendruk_inputs_survive_resolution_and_round_trip():
    cases = (
        _spec(frank_k=0.02),
        _spec(coefficient_max=0.06),
        _spec(parameterization="fixed-k", frank_k=0.02),
        _spec(
            parameterization="fixed-k",
            coefficient_min=0.02,
            coefficient_max=0.06,
        ),
    )

    for spec in cases:
        components, _ = _round_trip(spec)
        raw = components.physics.shendruk_request
        assert raw.frank_k == spec.frank_k
        assert raw.coefficient_min == spec.coefficient_min
        assert raw.coefficient_max == spec.coefficient_max

    paper_components = decompose_plane_beris_edwards_run_spec(cases[0])
    assert (
        paper_components.physics.shendruk_request.frank_k
        != paper_components.preset.frank_k
    )


@pytest.mark.parametrize(
    ("policy", "requested", "effective"),
    (
        ("zero_mean", 0.2, 0.0),
        ("zero_mean", -0.25, 0.0),
        ("friction", 0.125, 0.125),
    ),
)
def test_requested_and_effective_friction_remain_distinct(
    policy,
    requested,
    effective,
):
    spec = _spec(
        zero_mode_policy=policy,
        friction_mode_fric=requested,
    )
    components, _ = _round_trip(spec)

    assert components.physics.requested_friction_mode_fric == requested
    assert components.physics.stokes.friction == effective
    assert components.physics.stokes.pressure_gauge is PressureGauge.ZERO_MEAN
    assert components.physics.stokes.tangential_zero_mode_policy is (
        TangentialZeroModePolicy(policy)
    )
    assert components.physics.stokes.force_components == (
        "force_x",
        "force_y",
        "force_z",
    )
    assert components.physics.stokes.velocity_components == (
        "ux",
        "uy",
        "uz",
    )
    assert components.physics.stokes.pressure_component == "p"


def test_requested_boundaries_and_paths_are_recovered_without_substitution():
    requested = _alternate_boundaries()
    spec = replace(
        _spec(
            output_dir="oracle/a/../run",
            restart_from="oracle/c/../checkpoint",
        ),
        boundaries=requested,
    )
    components, reconstructed = _round_trip(spec)

    assert components.boundaries is requested
    assert components.effective_boundaries is PLANE_FREE_SLIP_BOUNDARIES
    assert reconstructed.boundaries is requested
    assert str(components.workflow.output_dir) == "oracle/a/../run"
    assert str(components.workflow.restart_from) == "oracle/c/../checkpoint"
    provisional = components.to_metadata()
    assert "schema_version" not in provisional
    assert "authority" not in provisional


def test_default_round_trip_preserves_format_v1_checkpoint_identity():
    expected = json.loads(
        (CHECKPOINT_FIXTURE / "checkpoint.json").read_text(encoding="utf-8")
    )
    _, reconstructed = _round_trip(_spec())
    header = read_plane_checkpoint_header(CHECKPOINT_FIXTURE)

    assert reconstructed.runtime_identity_sha256() == (
        expected["runtime_identity_sha256"]
    )
    assert reconstructed.runtime_identity_sha256() == (
        header.runtime_identity_sha256
    )


@pytest.mark.parametrize(
    "arguments",
    (
        (
            "--activity-number",
            "18",
            "--output-dir",
            "oracle/cli-minimal",
        ),
        (
            "--activity-number",
            "18",
            "--output-dir",
            "oracle/cli-rich",
            "--zero-mode-policy",
            "friction",
            "--friction-mode-fric",
            "0.125",
            "--dtype",
            "float64",
            "--pointwise-execution",
            "eager",
            "--spectral-storage",
            "full_complex",
            "--transform-execution-order",
            "legacy",
            "--projected-transform-execution",
            "full",
            "--dealias-rule",
            "two_thirds",
            "--molecular-field-linear-space",
            "physical",
            "--stress-divergence-sum-space",
            "physical",
            "--spectral-refresh-steps",
            "7",
            "--steps",
            "21",
            "--save-start-step",
            "7",
            "--save-interval",
            "7",
            "--diagnostic-interval",
            "3",
            "--diagnostics",
            "--save-hydrodynamics",
            "--checkpoint-interval",
            "7",
            "--restart-from",
            "oracle/checkpoint",
            "--tf32",
            "on",
        ),
    ),
)
def test_cli_specs_round_trip_without_changing_identity(arguments):
    _round_trip(parse_plane_beris_edwards_run_spec(arguments))


def test_explicit_decomposition_does_not_move_facade_validation_timing():
    invalid = replace(_spec(), dt=-0.01)

    assert invalid.dt == -0.01
    assert invalid.to_metadata()["workflow"]["dt"] == -0.01
    with pytest.raises(ValueError, match="dt must be positive and finite"):
        decompose_plane_beris_edwards_run_spec(invalid)


@pytest.mark.parametrize("value", (None, object(), _spec().to_metadata()))
def test_decomposition_rejects_non_facade_inputs(value):
    with pytest.raises(
        TypeError,
        match="legacy_spec must be a PlaneBerisEdwardsRunSpec",
    ):
        decompose_plane_beris_edwards_run_spec(value)


def test_adapter_stays_private_and_only_runtime_consumers_are_migrated():
    import pssolver.configuration.plane_beris_edwards_components as components

    name = "decompose_plane_beris_edwards_run_spec"
    assert getattr(components, name) is decompose_plane_beris_edwards_run_spec
    assert name in components.__all__
    assert name not in pssolver.__all__
    assert name not in configuration_package.__all__
    assert not hasattr(pssolver, name)
    assert not hasattr(configuration_package, name)
    assert "pssolver/configuration/plane_beris_edwards_components.py" in (
        PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES
    )
    assert (
        "pssolver/configuration/plane_beris_edwards_component_graph.py"
        in PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES
    )

    facade_source = FACADE_MODULE.read_text(encoding="utf-8")
    assert "from .plane_beris_edwards_components" not in facade_source
    assert name not in facade_source
    for relative in (
        "pssolver/applications/plane_beris_edwards.py",
        "pssolver/workflows/plane_beris_edwards.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "plane_beris_edwards_components" not in source
        assert name not in source

    for relative in (
        "pssolver/runtime/plane_legacy.py",
        "pssolver/runtime/plane_beris_edwards.py",
    ):
        runtime_source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "plane_beris_edwards_components" in runtime_source
        assert runtime_source.count(name) == 2

    tree = ast.parse(
        COMPONENT_MODULE.read_text(encoding="utf-8"),
        filename=str(COMPONENT_MODULE),
    )
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        for node in ast.walk(function)
    )
    function_source = ast.get_source_segment(
        COMPONENT_MODULE.read_text(encoding="utf-8"),
        function,
    )
    assert "__import__" not in function_source
    assert "import_module" not in function_source
    derived_facade_views = {
        node.attr
        for node in ast.walk(function)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "legacy_spec"
        and node.attr in {
            "domain",
            "geometry",
            "numerics",
            "shendruk_preset",
        }
    }
    assert derived_facade_views == set()

    for path in sorted((PROJECT_ROOT / "pssolver").rglob("*.py")):
        if path in {
            COMPONENT_MODULE,
            COMPONENT_GRAPH_MODULE,
            FACADE_MODULE,
            PROJECT_ROOT / "pssolver/runtime/plane_legacy.py",
            PROJECT_ROOT / "pssolver/runtime/plane_beris_edwards.py",
        }:
            continue
        source = path.read_text(encoding="utf-8")
        assert "plane_beris_edwards_components" not in source
        assert name not in source


@pytest.mark.parametrize(
    "imports",
    (
        """
from pssolver.configuration import plane_beris_edwards as facade
from pssolver.configuration import plane_beris_edwards_components as adapter
create_plane_beris_edwards_run_spec = facade.create_plane_beris_edwards_run_spec
decompose_plane_beris_edwards_run_spec = adapter.decompose_plane_beris_edwards_run_spec
""",
        """
from pssolver.configuration import plane_beris_edwards_components as adapter
from pssolver.configuration import plane_beris_edwards as facade
create_plane_beris_edwards_run_spec = facade.create_plane_beris_edwards_run_spec
decompose_plane_beris_edwards_run_spec = adapter.decompose_plane_beris_edwards_run_spec
""",
        """
import pssolver.configuration as configuration
from pssolver.configuration import plane_beris_edwards_components as adapter
create_plane_beris_edwards_run_spec = configuration.create_plane_beris_edwards_run_spec
decompose_plane_beris_edwards_run_spec = adapter.decompose_plane_beris_edwards_run_spec
assert not hasattr(configuration, 'decompose_plane_beris_edwards_run_spec')
""",
    ),
)
def test_facade_and_adapter_import_orders_are_cycle_free(imports):
    script = imports + """
spec = create_plane_beris_edwards_run_spec(
    activity_number=18.0,
    output_dir='oracle/import-order',
)
components = decompose_plane_beris_edwards_run_spec(spec)
assert components.physics.shendruk_request.activity_number == 18.0
"""
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
