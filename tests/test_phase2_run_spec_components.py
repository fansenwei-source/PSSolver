"""Local qualification for the disconnected Phase 2.1 leaf components.

These provisional values make ownership explicit without connecting the
supported flat RunSpec to a new composition path.  In particular, this stage
does not introduce a Plane physics aggregate.  The Stokes declaration-layer
prerequisite is now satisfied, but the disconnected aggregate remains the
next separately qualified P2.1 change.
"""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, MISSING, fields, is_dataclass
import json
from pathlib import Path

import pytest

import pssolver
import pssolver.configuration as configuration_package
import pssolver.models.active_nematics as active_nematics_package
import pssolver.presets as presets_package
from pssolver.configuration import (
    PLANE_BERIS_EDWARDS_IMPLEMENTATION_SOURCE_FILES,
    PlaneRuntimePath,
    SpectralRefreshSpec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    PlaneBerisEdwardsExecutionSpec,
    PlaneInvocationSpec,
    PlaneTimeSteppingSpec,
    PlaneWorkflowSpec,
)
from pssolver.models.active_nematics.specifications import (
    BerisEdwardsMaterialRequest,
    ExtrudedDefectGasInitialConditionSpec,
)
from pssolver.presets.shendruk import (
    ShendrukPlaneParameterRequest,
    resolve_shendruk_plane_preset,
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


def _representative_values() -> tuple[object, ...]:
    return (
        _material(),
        _initial(twist_modes=[1, 2, 3]),
        _shendruk(),
        _time(),
        _execution(),
        _workflow(),
        _invocation(),
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
        "dataclasses",
        "math",
        "numbers",
        "pathlib",
        "relative:plane_beris_edwards",
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
    for path in (MODEL_SPECIFICATIONS, PLANE_COMPONENTS):
        source = path.read_text(encoding="utf-8")
        assert all(name not in source for name in forbidden_text)


def test_p21_leaf_types_remain_provisional_and_disconnected():
    provisional_names = {
        "BerisEdwardsMaterialRequest",
        "ExtrudedDefectGasInitialConditionSpec",
        "ShendrukPlaneParameterRequest",
        "PlaneTimeSteppingSpec",
        "PlaneBerisEdwardsExecutionSpec",
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

    assert not hasattr(components, "PlaneBerisEdwardsPhysicsSpec")
    assert not hasattr(components, "PlaneBerisEdwardsRunComponents")
    assert not hasattr(components, "decompose_plane_beris_edwards_run_spec")
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
