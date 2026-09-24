from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import pytest

from pssolver.configuration.active_nematics_equation_adapters import (
    declare_channel_active_force_equation_system,
    declare_plane_complete_stress_equation_system,
)
from pssolver.configuration.channel_active_nematics import (
    ChannelActiveNematicRunSpec,
)
from pssolver.configuration.plane_beris_edwards import (
    create_plane_beris_edwards_run_spec,
)
from pssolver.configuration.plane_beris_edwards_components import (
    decompose_plane_beris_edwards_run_spec,
)
from pssolver.core import FieldRole
from pssolver.models.active_nematics.equation_systems import (
    ACTIVE_FORCE_COMPONENTS,
    COMPLETE_STRESS_FORCE_COMPONENTS,
    CompleteStressBerisEdwardsEquationRequest,
    LegacyActiveForceEquationRequest,
)
from pssolver.models.active_nematics.specifications import (
    BerisEdwardsMaterialRequest,
)
from pssolver.systems.equations import (
    EquationFieldSpec,
    EquationSystemSpec,
    EquationTermSpec,
)
from pssolver.systems.stokes import (
    IncompressibleStokesSystemSpec,
    PressureGauge,
    TangentialZeroModePolicy,
)


ROOT = Path(__file__).resolve().parents[1]
EQUATION_DECLARATIONS = (
    ROOT / "pssolver" / "systems" / "equations.py"
)
ACTIVE_NEMATIC_DECLARATIONS = (
    ROOT
    / "pssolver"
    / "models"
    / "active_nematics"
    / "equation_systems.py"
)
ADAPTERS = (
    ROOT
    / "pssolver"
    / "configuration"
    / "active_nematics_equation_adapters.py"
)
RESULT_PATH = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_7_p771_equation_system_declarations.json"
)


def _plane_spec(**overrides: object):
    values = {
        "activity_number": 18.0,
        "output_dir": ROOT / "unused_p771_plane_output",
        "nx": 16,
        "ny": 16,
        "nz": 8,
        "steps": 1,
        "save_start_step": 0,
        "save_interval": 1,
        "diagnostic_interval": 1,
        "defect_min_separation": 2.0,
        "defect_core_radius": 0.5,
        "twist_modes": (1, 2, 3),
    }
    values.update(overrides)
    return create_plane_beris_edwards_run_spec(**values)


def _plane_stokes() -> IncompressibleStokesSystemSpec:
    return IncompressibleStokesSystemSpec(
        name="flow",
        force_components=COMPLETE_STRESS_FORCE_COMPONENTS,
        velocity_components=("ux", "uy", "uz"),
        pressure_component="p",
        viscosity=2.0 / 3.0,
        pressure_gauge=PressureGauge.ZERO_MEAN,
        tangential_zero_mode_policy=TangentialZeroModePolicy.ZERO_MEAN,
    )


def _channel_stokes() -> IncompressibleStokesSystemSpec:
    return IncompressibleStokesSystemSpec(
        name="channel_stokes",
        force_components=ACTIVE_FORCE_COMPONENTS,
        velocity_components=("ux", "uy", "uz"),
        pressure_component="p",
        viscosity=1.0,
        pressure_gauge=PressureGauge.ZERO_MEAN,
        tangential_zero_mode_policy=(
            TangentialZeroModePolicy.NOT_APPLICABLE
        ),
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _complete_request() -> CompleteStressBerisEdwardsEquationRequest:
    return CompleteStressBerisEdwardsEquationRequest(
        material=BerisEdwardsMaterialRequest(
            ldg_a=0.0,
            ldg_b=-0.3,
            ldg_c=0.3,
            gamma=2.94,
            flow_alignment=0.3,
            beta=-1.0,
        ),
        ldg_l1=0.01,
        activity_amplitude=0.02,
        stokes_system=_plane_stokes(),
    )


def test_equation_declarations_are_boundary_geometry_and_runtime_free():
    forbidden_modules = {
        "torch",
        "pssolver.backends",
        "pssolver.execution",
        "pssolver.geometries",
        "pssolver.linear_solvers",
        "pssolver.operators",
        "pssolver.runtime",
        "pssolver.transforms",
        "pssolver.workflows",
    }
    for path in (EQUATION_DECLARATIONS, ACTIVE_NEMATIC_DECLARATIONS):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        assert not {
            name
            for name in imported
            if any(
                name == forbidden or name.startswith(f"{forbidden}.")
                for forbidden in forbidden_modules
            )
        }, path
        source = path.read_text(encoding="utf-8")
        assert "BoundarySet" not in source
        assert "GeometrySpec" not in source


def test_generic_equation_declarations_are_frozen_and_deterministic():
    value = _complete_request().to_equation_system_spec()
    equivalent = _complete_request().to_equation_system_spec()

    assert value == equivalent
    assert value.canonical_sha256() == equivalent.canonical_sha256()
    assert len(value.canonical_sha256()) == 64
    json.dumps(value.to_metadata(), allow_nan=False, sort_keys=True)
    with pytest.raises(FrozenInstanceError):
        value.name = "changed"


def test_complete_stress_declaration_records_the_full_constitutive_chain():
    value = _complete_request().to_equation_system_spec()
    fields = {field.name: field for field in value.fields}
    laws = {law.name: law for law in value.constitutive_laws}

    assert value.variant == "complete_stress_beris_edwards"
    assert len(value.component_names) == 59
    assert set(fields) == {
        "Q",
        "q_gradient",
        "velocity_gradient",
        "molecular_field",
        "algebraic_stress",
        "distortion_stress",
        "nematic_force",
        "velocity",
        "pressure",
    }
    assert fields["Q"].role is FieldRole.EVOLVED
    assert fields["velocity"].role is FieldRole.ALGEBRAIC
    assert fields["nematic_force"].components == (
        "force_x",
        "force_y",
        "force_z",
    )
    assert set(laws) == {
        "q_gradient",
        "velocity_gradient",
        "molecular_field",
        "nematic_stress",
        "nematic_force",
    }
    assert laws["nematic_stress"].parameters["active_prefactor"] == -0.02
    assert value.parameters["force_law"] == (
        "complete_one_constant_nematic_stress"
    )
    assert value.initial_condition_families == ("extruded_defect_gas",)


def test_legacy_active_force_declaration_remains_a_distinct_model():
    value = LegacyActiveForceEquationRequest(
        rho=6.0,
        elastic_constant=1.0,
        activity=5.0,
        beta=-1.0,
        flow_alignment=1.0,
        stokes_system=_channel_stokes(),
    ).to_equation_system_spec()
    fields = {field.name: field for field in value.fields}
    laws = {law.name: law for law in value.constitutive_laws}

    assert value.variant == "legacy_active_force_active_nematics"
    assert len(value.component_names) == 36
    assert "molecular_field" not in fields
    assert "algebraic_stress" not in fields
    assert "distortion_stress" not in fields
    assert fields["active_force"].components == ("fx", "fy", "fz")
    assert set(laws) == {
        "q_gradient",
        "velocity_gradient",
        "active_force",
    }
    assert value.parameters["force_law"] == "active_force_divergence_only"
    assert value.parameters["ldg_a"] == -1.0
    assert value.parameters["ldg_b"] == -6.0
    assert value.parameters["ldg_c"] == 6.0
    assert value.initial_condition_families == ("aligned_x_smooth_noise",)
    assert value.canonical_sha256() != (
        _complete_request().to_equation_system_spec().canonical_sha256()
    )


def test_plane_adapter_is_lossless_for_model_and_stokes_parameters():
    components = decompose_plane_beris_edwards_run_spec(_plane_spec())
    value = declare_plane_complete_stress_equation_system(components)
    parameters = value.parameters

    assert value.variant == "complete_stress_beris_edwards"
    assert parameters["material"] == components.physics.material.to_metadata()
    assert parameters["ldg_l1"] == components.preset.ldg_l1
    assert parameters["activity_amplitude"] == components.preset.zeta
    assert parameters["stokes"] == components.physics.stokes.to_metadata()
    assert "geometry" not in value.to_metadata()
    assert "boundaries" not in value.to_metadata()


def test_channel_adapter_is_lossless_for_model_and_stokes_parameters():
    components = ChannelActiveNematicRunSpec().components
    value = declare_channel_active_force_equation_system(components)
    parameters = value.parameters
    material = components.material

    assert value.variant == "legacy_active_force_active_nematics"
    assert parameters["rho"] == material.rho
    assert parameters["ldg_a"] == material.ldg_a
    assert parameters["ldg_b"] == material.ldg_b
    assert parameters["ldg_c"] == material.ldg_c
    assert parameters["elastic_constant"] == material.elastic_constant
    assert parameters["activity"] == material.activity
    assert parameters["beta"] == material.beta
    assert parameters["flow_alignment"] == material.flow_alignment
    assert parameters["stokes"] == components.stokes.to_metadata()
    assert "geometry" not in value.to_metadata()
    assert "boundaries" not in value.to_metadata()


def test_adapters_reject_the_wrong_component_graph_before_conversion():
    plane = decompose_plane_beris_edwards_run_spec(_plane_spec())
    channel = ChannelActiveNematicRunSpec().components

    with pytest.raises(TypeError, match="PlaneBerisEdwardsRunComponents"):
        declare_plane_complete_stress_equation_system(channel)
    with pytest.raises(TypeError, match="ChannelRunComponents"):
        declare_channel_active_force_equation_system(plane)


@pytest.mark.parametrize(
    ("mutation", "match"),
    (
        (
            lambda request: replace(request, ldg_l1=0.0),
            "ldg_l1 must be positive",
        ),
        (
            lambda request: replace(
                request,
                stokes_system=replace(
                    request.stokes_system,
                    force_components=("fx", "fy", "fz"),
                ),
            ),
            "complete-stress Stokes force components",
        ),
    ),
)
def test_complete_stress_request_fails_closed(mutation, match):
    with pytest.raises((TypeError, ValueError), match=match):
        mutation(_complete_request())


def test_equation_system_rejects_unknown_or_unproduced_components():
    fields = (
        EquationFieldSpec("q", FieldRole.EVOLVED, ("q",)),
        EquationFieldSpec("aux", FieldRole.TRANSIENT, ("aux",)),
        EquationFieldSpec("u", FieldRole.ALGEBRAIC, ("u",)),
        EquationFieldSpec("v", FieldRole.ALGEBRAIC, ("v",)),
    )
    evolution = EquationTermSpec("evolve", "evolve", ("q",), ("q",))
    auxiliary = EquationTermSpec("make_aux", "make_aux", ("aux",), ("q",))

    with pytest.raises(ValueError, match="unknown components"):
        EquationSystemSpec(
            "example",
            "example_variant",
            fields,
            (replace(evolution, dependencies=("missing",)),),
            (auxiliary,),
            (_channel_stokes().to_algebraic_system_spec(),),
            {},
        )
    with pytest.raises(ValueError, match="missing=.*u"):
        EquationSystemSpec(
            "example",
            "example_variant",
            fields,
            (evolution,),
            (auxiliary,),
            (
                replace(
                    _channel_stokes().to_algebraic_system_spec(),
                    output_components=("v",),
                    dependencies=("aux",),
                ),
            ),
            {},
        )


def test_p771_does_not_connect_existing_applications_or_runtimes():
    adapter_source = ADAPTERS.read_text(encoding="utf-8")
    plane_application = (
        ROOT / "pssolver" / "applications" / "plane_beris_edwards.py"
    ).read_text(encoding="utf-8")
    channel_application = (
        ROOT / "pssolver" / "applications" / "channel_active_nematics.py"
    ).read_text(encoding="utf-8")

    assert "build_plane_beris_edwards_runtime" not in adapter_source
    assert "build_channel_active_nematic_runtime" not in adapter_source
    assert "active_nematics_equation_adapters" not in plane_application
    assert "active_nematics_equation_adapters" not in channel_application


def test_p771_result_records_scope_identity_and_next_authorization():
    result = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert result["phase"] == "P7.7.1"
    assert result["status"] == "P7_7_1_COMPLETE_P7_7_2_NOT_AUTHORIZED"
    assert result["classification"] == (
        "PASS_P7_7_1_GEOMETRY_NEUTRAL_EQUATION_DECLARATIONS"
    )
    assert result["scope"] == {
        "tensor_free": True,
        "geometry_free": True,
        "boundary_assignment_free": True,
        "runtime_connected": False,
        "changes_equations": False,
        "changes_numerical_path": False,
        "changes_output_or_checkpoint_schema": False,
        "changes_production_defaults": False,
    }
    assert result["next_slice"]["id"] == "P7.7.2"
    assert result["next_slice"]["authorized"] is False
    assert result["public_api"]["package_root_exports_added"] == []


def test_p771_result_binds_the_reviewed_source_identities():
    identities = json.loads(
        RESULT_PATH.read_text(encoding="utf-8")
    )["source_identities"]

    assert len(identities) == 5
    for relative_path, expected in identities.items():
        path = ROOT / relative_path
        assert path.is_file(), relative_path
        assert _sha256(path) == expected, relative_path
