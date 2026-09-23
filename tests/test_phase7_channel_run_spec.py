"""P7.1 contracts for the disconnected Channel run specification."""

from __future__ import annotations

from dataclasses import fields
import hashlib
import json
from pathlib import Path

import pytest

from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
    create_channel_active_nematic_run_spec,
)
from pssolver.configuration.channel_active_nematics_declarations import (
    CHANNEL_BOUNDARIES,
    ChannelBoundaryConditions,
    ChannelRuntimePath,
)
from pssolver.core import (
    DealiasRule,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.systems.stokes import (
    PressureGauge,
    TangentialZeroModePolicy,
)


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def _sha256(relative: str) -> str:
    return hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()


def test_default_flat_facade_matches_the_frozen_channel_oracle():
    spec = ChannelActiveNematicRunSpec()
    components = spec.components

    assert spec.shape == (512, 40, 40)
    assert spec.lengths == (128.0, 10.0, 10.0)
    assert spec.dt == 0.01
    assert spec.steps == 1000
    assert spec.save_interval == 10
    assert spec.diagnostic_interval == 10
    assert spec.seed == 24
    assert components.geometry.periodic_axes == (0,)
    assert components.geometry.bounded_axes == (1, 2)
    assert components.boundaries is CHANNEL_BOUNDARIES
    assert components.material.to_metadata() == {
        "rho": 6.0,
        "ldg_a": -1.0,
        "ldg_b": -6.0,
        "ldg_c": 6.0,
        "elastic_constant": 1.0,
        "activity": 5.0,
        "beta": -1.0,
        "friction": 0.0,
        "viscosity": 1.0,
        "flow_alignment": 1.0,
    }
    assert components.initial_condition.to_metadata()["generated"] == {
        "name": "aligned_x_smooth_noise",
        "seed": 24,
        "initial_s": 2.0 / 3.0,
        "noise_theta": 0.01,
        "noise_phi": 0.01,
        "smoothing_sigma": [1.0, 1.0, 1.0],
    }


def test_channel_numerics_and_stokes_are_explicit_geometry_specific_values():
    components = ChannelActiveNematicRunSpec().components
    numerics = components.numerics
    stokes = components.stokes

    assert numerics.precision is Precision.FLOAT32
    assert numerics.dealias_rule is DealiasRule.NONE
    assert numerics.projected_transform_execution is ProjectedTransformExecution.FULL
    assert numerics.transform_execution_order is TransformExecutionOrder.REAL_FIRST
    assert numerics.spectral_storage is SpectralStorage.FULL_COMPLEX
    assert numerics.hermitian_axis is None
    assert stokes.pressure_gauge is PressureGauge.ZERO_MEAN
    assert (
        stokes.tangential_zero_mode_policy
        is TangentialZeroModePolicy.NOT_APPLICABLE
    )
    assert components.pressure_solver.to_metadata() == {
        "algorithm": "preconditioned_conjugate_gradient",
        "relative_tolerance": 1.0e-6,
        "max_iterations": 80,
        "fixed_iterations": None,
        "warm_start": True,
    }


def test_factory_is_additive_and_runtime_remains_legacy_only():
    spec = create_channel_active_nematic_run_spec(
        steps=17,
        runtime_path="legacy_channel",
    )

    assert isinstance(spec, ChannelActiveNematicRunSpec)
    assert spec.steps == 17
    assert spec.runtime_path is ChannelRuntimePath.LEGACY_CHANNEL
    assert spec.components.execution.runtime_path is ChannelRuntimePath.LEGACY_CHANNEL
    with pytest.raises(ValueError, match="unsupported Channel runtime path"):
        create_channel_active_nematic_run_spec(runtime_path="compiled_channel_v2")


def test_metadata_and_canonical_identity_are_deterministic_and_disconnected():
    left = ChannelActiveNematicRunSpec()
    right = ChannelActiveNematicRunSpec()
    changed = ChannelActiveNematicRunSpec(steps=1001)

    assert left.to_metadata() == right.to_metadata()
    assert left.canonical_sha256() == right.canonical_sha256()
    assert left.canonical_sha256() != changed.canonical_sha256()
    assert len(left.canonical_sha256()) == 64
    assert left.to_metadata()["production_connection"] is False
    json.dumps(left.to_metadata(), allow_nan=False, sort_keys=True)


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"shape": (4, 4), "lengths": (1.0, 1.0)}, "three dimensions"),
        ({"dt": 0.0}, "dt must be positive"),
        ({"friction": -1.0}, "friction must be non-negative"),
        ({"pressure_max_iterations": 0}, "max_iterations must be positive"),
        ({"initialization_mode": "unknown"}, "generated or snapshot"),
        ({"snapshot_mode": "unknown"}, "resume or branch"),
        ({"dealias_rule": "cubic_half"}, "requires dealias_rule='none'"),
        ({"spectral_storage": "hermitian_half"}, "invalid Channel numerical policy"),
    ),
)
def test_invalid_cross_component_requests_fail_closed(overrides, match):
    with pytest.raises((TypeError, ValueError), match=match):
        ChannelActiveNematicRunSpec(**overrides)


def test_p71_declarations_are_tensor_free_and_do_not_connect_production():
    declarations = (
        ROOT
        / "pssolver"
        / "configuration"
        / "channel_active_nematics_declarations.py"
    ).read_text(encoding="utf-8")
    facade = (
        ROOT / "pssolver" / "configuration" / "channel_active_nematics.py"
    ).read_text(encoding="utf-8")
    production = (ROOT / "Channel.py").read_text(encoding="utf-8")
    reusable = (ROOT / "pssolver" / "channel.py").read_text(encoding="utf-8")

    assert "import torch" not in declarations
    assert "import torch" not in facade
    assert "channel_active_nematics" not in production
    assert "channel_active_nematics" not in reusable
    assert "SpectralSolver(" not in facade
    assert "build_active_nematic_channel" not in facade


def test_p71_keeps_the_frozen_configuration_package_root_surface():
    import pssolver.configuration as configuration

    assert "ChannelActiveNematicRunSpec" not in configuration.__all__
    assert not hasattr(configuration, "ChannelActiveNematicRunSpec")


def test_p71_preserves_the_historical_oracle_source_identities():
    inventory = json.loads(
        (NOTES / "phase_7_channel_inventory.json").read_text(encoding="utf-8")
    )

    assert _sha256("Channel.py") == inventory["legacy_oracle"][
        "entry_point_sha256"
    ]
    assert inventory["legacy_oracle"]["reusable_module_sha256"] == (
        "9be8c7fd97e6167c44df4f38f2481683d108ac4e191ef1e0562d616fb4406223"
    )


def test_flat_facade_field_order_is_characterized():
    names = tuple(item.name for item in fields(ChannelActiveNematicRunSpec))

    assert names[:8] == (
        "shape",
        "lengths",
        "dt",
        "steps",
        "save_interval",
        "diagnostics_enabled",
        "diagnostic_interval",
        "batch_size",
    )
    assert names[-2:] == ("boundaries", "_components")


def test_p71_result_record_keeps_the_next_implementation_unauthorized():
    value = json.loads(
        (NOTES / "phase_7_p71_run_spec_decomposition.json").read_text(
            encoding="utf-8"
        )
    )

    assert value["classification"] == (
        "PASS_P7_1_CHANNEL_RUN_SPEC_DECOMPOSITION"
    )
    assert value["facade"]["public_initialization_field_count"] == 37
    assert value["facade"]["production_connection"] is False
    assert value["source_isolation"]["channel_entry_point_modified"] is False
    assert value["source_isolation"]["pssolver_channel_modified"] is False
    assert value["authorization"] == {
        "p7_1_complete": True,
        "p7_2_planning_eligible": True,
        "p7_2_implementation_authorized": False,
        "phase_7_h100_authorized": False,
        "new_boundary_law_authorized": False,
        "production_default_changed": False,
    }
