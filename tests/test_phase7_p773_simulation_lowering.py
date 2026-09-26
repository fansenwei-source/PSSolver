from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError, replace
import hashlib
import json
from pathlib import Path

import pytest

from pssolver.configuration.active_nematics_simulation_adapters import (
    compose_channel_active_nematics_simulation,
    compose_plane_beris_edwards_simulation,
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
from pssolver.configuration.simulation_lowering import (
    LoweringRejectionCode,
    SimulationLoweringError,
    lower_simulation_spec,
)
from pssolver.core.boundary import (
    BoundarySemantic,
    BoundarySide,
    HomogeneousDirichletBC,
)
from pssolver.core.geometry import GeometrySpec
from pssolver.core.integrators import IntegratorSpec
from pssolver.core.numerics import SpectralStorage
from pssolver.planning.plan import TransformKind
from pssolver.planning.simulation import (
    BasisProvenance,
    DerivativeMultiplier,
)


ROOT = Path(__file__).resolve().parents[1]
LOWERING_DECLARATIONS = ROOT / "pssolver" / "planning" / "simulation.py"
LOWERING_RESOLVER = (
    ROOT / "pssolver" / "configuration" / "simulation_lowering.py"
)
RESULT_PATH = (
    ROOT
    / "notes"
    / "architecture_v0_2"
    / "phase_7_p773_capability_lowering.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _plane_simulation(**overrides: object):
    values = {
        "activity_number": 18.0,
        "output_dir": ROOT / "unused_p773_plane_output",
        "nx": 16,
        "ny": 16,
        "nz": 8,
        "steps": 2,
        "save_start_step": 0,
        "save_interval": 1,
        "diagnostic_interval": 1,
        "defect_min_separation": 2.0,
        "defect_core_radius": 0.5,
        "twist_modes": (1, 2, 3),
    }
    values.update(overrides)
    components = decompose_plane_beris_edwards_run_spec(
        create_plane_beris_edwards_run_spec(**values)
    )
    return compose_plane_beris_edwards_simulation(components)


def _channel_simulation():
    return compose_channel_active_nematics_simulation(
        ChannelActiveNematicRunSpec().components
    )


def _replace_boundary_component(simulation, replacement):
    boundaries = replace(
        simulation.boundaries,
        components=tuple(
            replacement if value.component == replacement.component else value
            for value in simulation.boundaries.components
        ),
    )
    return replace(simulation, boundaries=boundaries)


def _error_code(callable_) -> LoweringRejectionCode:
    with pytest.raises(SimulationLoweringError) as caught:
        callable_()
    rejection = caught.value.rejection
    json.dumps(rejection.to_metadata(), allow_nan=False, sort_keys=True)
    return rejection.code


def test_p773_modules_are_tensor_runtime_and_solver_import_free():
    forbidden = {
        "numpy",
        "torch",
        "pssolver.backends",
        "pssolver.execution",
        "pssolver.integrators",
        "pssolver.linear_solvers",
        "pssolver.models",
        "pssolver.operators",
        "pssolver.runtime",
        "pssolver.transforms",
        "pssolver.workflows",
    }
    for path in (LOWERING_DECLARATIONS, LOWERING_RESOLVER):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        assert not {
            name
            for name in imports
            if any(
                name == blocked or name.startswith(f"{blocked}.")
                for blocked in forbidden
            )
        }, path


def test_plane_lowering_resolves_every_component_and_exact_parity_maps():
    simulation = _plane_simulation()
    plan = lower_simulation_spec(simulation)

    assert plan.source_simulation_sha256 == simulation.canonical_sha256()
    assert plan.equation_variant == "complete_stress_beris_edwards"
    assert plan.geometry_name == "plane_slab"
    assert len(plan.component_bases) == len(
        simulation.equation_system.component_names
    ) == 59
    assert len(plan.derivatives) == 24
    assert len(plan.capabilities) == 7
    assert plan.basis_for("Qxx").transform_kinds == (
        TransformKind.FFT,
        TransformKind.FFT,
        TransformKind.DCT,
    )
    assert plan.basis_for("ux").transform_kinds == (
        TransformKind.FFT,
        TransformKind.FFT,
        TransformKind.DCT,
    )
    assert plan.basis_for("uz").transform_kinds == (
        TransformKind.FFT,
        TransformKind.FFT,
        TransformKind.DST,
    )
    assert plan.basis_for("p").provenance is (
        BasisProvenance.ALGEBRAIC_COMPATIBILITY
    )

    dz_q = plan.derivative_for("dQxx_dz")
    assert dz_q.multiplier is DerivativeMultiplier.COSINE_TO_NEGATIVE_SINE
    assert dz_q.output_transform_kinds[-1] is TransformKind.DST
    dz_uz = plan.derivative_for("duz_dz")
    assert dz_uz.multiplier is DerivativeMultiplier.SINE_TO_POSITIVE_COSINE
    assert dz_uz.output_transform_kinds[-1] is TransformKind.DCT
    dx_q = plan.derivative_for("dQxx_dx")
    assert dx_q.multiplier is DerivativeMultiplier.FOURIER_IK
    assert dx_q.output_transform_kinds == dx_q.source_transform_kinds

    for suffix in ("xz", "yz", "zx", "zy"):
        value = plan.basis_for(f"stress_distortion_{suffix}")
        assert value.transform_kinds[-1] is TransformKind.DST
        assert value.provenance is BasisProvenance.CONSTITUTIVE_PARITY
    for suffix in ("xx", "xy", "yx", "yy", "zz"):
        assert plan.basis_for(
            f"stress_distortion_{suffix}"
        ).transform_kinds[-1] is TransformKind.DCT
    for component, target in zip(
        ("force_x", "force_y", "force_z"),
        ("ux", "uy", "uz"),
        strict=True,
    ):
        force = plan.basis_for(component)
        assert force.transform_kinds == plan.basis_for(target).transform_kinds
        assert force.provenance is BasisProvenance.PROJECTED_OUTPUT


def test_plane_lowering_selects_exact_solver_and_explicit_zero_mode_model():
    plan = lower_simulation_spec(_plane_simulation())

    assert plan.solver.family == "free_slip_modal_stokes"
    assert plan.solver.implementation == (
        "pssolver.linear_solvers.stokes.plane_free_slip."
        "FreeSlipModalStokesSolver"
    )
    assert plan.nullspace.pressure_gauge == "zero_mean"
    assert plan.nullspace.tangential_velocity_components == ("ux", "uy")
    assert plan.nullspace.tangential_policy == "zero_mean"
    assert plan.nullspace.uniform_mode_action == (
        "remove_uniform_tangential_velocity_and_force"
    )
    assert plan.nullspace.friction == 0.0

    friction = lower_simulation_spec(
        _plane_simulation(
            zero_mode_policy="friction",
            friction_mode_fric=0.25,
        )
    )
    assert friction.nullspace.tangential_policy == "friction"
    assert friction.nullspace.friction == 0.25
    assert friction.nullspace.uniform_mode_action == (
        "retain_uniform_tangential_mode_resolved_by_friction"
    )


def test_channel_lowering_resolves_no_slip_pcg_and_mixed_force_projection():
    simulation = _channel_simulation()
    plan = lower_simulation_spec(simulation)

    assert plan.equation_variant == "legacy_active_force_active_nematics"
    assert plan.geometry_name == "rectangular_channel"
    assert len(plan.component_bases) == len(
        simulation.equation_system.component_names
    ) == 36
    assert len(plan.derivatives) == 24
    assert len(plan.capabilities) == 5
    assert plan.basis_for("Qxx").transform_kinds == (
        TransformKind.FFT,
        TransformKind.DCT,
        TransformKind.DCT,
    )
    assert plan.basis_for("ux").transform_kinds == (
        TransformKind.FFT,
        TransformKind.DST,
        TransformKind.DST,
    )
    assert plan.derivative_for("dux_dy").output_transform_kinds == (
        TransformKind.FFT,
        TransformKind.DCT,
        TransformKind.DST,
    )
    for component, target in zip(
        ("fx", "fy", "fz"),
        ("ux", "uy", "uz"),
        strict=True,
    ):
        force = plan.basis_for(component)
        assert force.transform_kinds == plan.basis_for(target).transform_kinds
        assert force.provenance is BasisProvenance.PROJECTED_OUTPUT

    assert plan.solver.family == "channel_no_slip_modal_stokes_pcg"
    assert plan.solver.implementation == (
        "pssolver.linear_solvers.stokes.channel_no_slip."
        "ChannelNoSlipModalStokesSolver"
    )
    options = plan.solver.to_metadata()["options"]
    assert options["pressure_solver"] == (
        simulation.discretization_parameters["pressure_solver"]
    )
    assert plan.nullspace.tangential_policy == "not_applicable"
    assert plan.nullspace.tangential_velocity_components == ()
    assert plan.nullspace.uniform_mode_action == (
        "not_applicable_due_to_bounded_no_slip_axes"
    )


def test_capability_resolution_is_exact_and_contains_no_runtime_objects():
    for simulation, expected in (
        (_plane_simulation(), set((
            "plane_complete_beris_edwards_q_evolution",
            "tensor_product_spectral_q_gradient",
            "tensor_product_spectral_velocity_gradient",
            "plane_one_constant_molecular_field",
            "plane_complete_one_constant_nematic_stress",
            "plane_projected_complete_stress_divergence",
            "plane_free_slip_modal_stokes",
        ))),
        (_channel_simulation(), set((
            "channel_legacy_active_force_q_evolution",
            "tensor_product_spectral_q_gradient",
            "tensor_product_spectral_velocity_gradient",
            "channel_legacy_active_force_divergence",
            "channel_no_slip_modal_stokes_pcg",
        ))),
    ):
        plan = lower_simulation_spec(simulation)
        assert {
            value.capability for value in plan.capabilities
        } == set(simulation.equation_system.required_capabilities)
        assert {
            value.implementation_name for value in plan.capabilities
        } == expected
        metadata = plan.to_metadata()
        json.dumps(metadata, allow_nan=False, sort_keys=True)
        assert "tensor" not in {
            type(value).__module__.split(".", 1)[0]
            for value in plan.component_bases
        }


def test_lowering_is_frozen_deterministic_and_shape_sensitive():
    first = lower_simulation_spec(_plane_simulation())
    second = lower_simulation_spec(_plane_simulation())
    refined = lower_simulation_spec(_plane_simulation(nx=32))

    assert first == second
    assert first.canonical_sha256() == second.canonical_sha256()
    assert len(first.canonical_sha256()) == 64
    assert refined.canonical_sha256() != first.canonical_sha256()
    assert refined.basis_for("Qxx").axes[0].physical_size == 32
    with pytest.raises(FrozenInstanceError):
        first.geometry_name = "changed"


def test_lowering_rejects_channel_name_with_plane_topology_without_fallback():
    simulation = _plane_simulation()
    mislabeled = GeometrySpec(
        name="rectangular_channel",
        domain=simulation.geometry.domain,
        axis_topologies=simulation.geometry.axis_topologies,
    )
    invalid = replace(simulation, geometry=mislabeled)

    assert _error_code(lambda: lower_simulation_spec(invalid)) is (
        LoweringRejectionCode.UNSUPPORTED_DIMENSION_OR_TOPOLOGY
    )


def test_lowering_rejects_asymmetric_and_unqualified_symmetric_wall_laws():
    simulation = _plane_simulation()
    qxx = simulation.boundaries.for_component("Qxx")
    asymmetric = replace(
        qxx,
        faces=tuple(
            replace(face, condition=HomogeneousDirichletBC())
            if face.axis == 2 and face.side is BoundarySide.LOWER
            else face
            for face in qxx.faces
        ),
    )
    invalid = _replace_boundary_component(simulation, asymmetric)
    assert _error_code(lambda: lower_simulation_spec(invalid)) is (
        LoweringRejectionCode.ASYMMETRIC_FACE_PAIR
    )

    symmetric = replace(
        qxx,
        faces=tuple(
            replace(face, condition=HomogeneousDirichletBC())
            if face.axis == 2
            else face
            for face in qxx.faces
        ),
    )
    invalid = _replace_boundary_component(simulation, symmetric)
    assert _error_code(lambda: lower_simulation_spec(invalid)) is (
        LoweringRejectionCode.UNSUPPORTED_BOUNDARY_SIGNATURE
    )


def test_lowering_rejects_pressure_semantic_capability_and_integrator_drift():
    simulation = _plane_simulation()
    pressure = simulation.boundaries.for_component("p")
    invalid = _replace_boundary_component(
        simulation,
        replace(pressure, semantic=BoundarySemantic.PHYSICAL),
    )
    assert _error_code(lambda: lower_simulation_spec(invalid)) is (
        LoweringRejectionCode.UNSUPPORTED_BOUNDARY_SEMANTIC
    )

    evolution = simulation.equation_system.evolution_laws[0]
    equation_system = replace(
        simulation.equation_system,
        evolution_laws=(replace(evolution, capability="unknown_q_evolution"),),
    )
    invalid = replace(simulation, equation_system=equation_system)
    assert _error_code(lambda: lower_simulation_spec(invalid)) is (
        LoweringRejectionCode.UNSUPPORTED_CAPABILITY_SET
    )

    invalid = replace(
        simulation,
        time_integration=replace(
            simulation.time_integration,
            integrator=IntegratorSpec.sbdf2(
                dt=simulation.time_integration.integrator.dt
            ),
        ),
    )
    assert _error_code(lambda: lower_simulation_spec(invalid)) is (
        LoweringRejectionCode.UNSUPPORTED_INTEGRATOR
    )


def test_lowering_rejects_backend_and_bounded_hermitian_axis():
    simulation = _plane_simulation()
    invalid = replace(
        simulation,
        execution=replace(simulation.execution, backend="different_backend"),
    )
    assert _error_code(lambda: lower_simulation_spec(invalid)) is (
        LoweringRejectionCode.UNSUPPORTED_BACKEND
    )

    invalid = replace(
        simulation,
        numerics=replace(
            simulation.numerics,
            spectral_storage=SpectralStorage.HERMITIAN_HALF,
            hermitian_axis=2,
        ),
    )
    assert _error_code(lambda: lower_simulation_spec(invalid)) is (
        LoweringRejectionCode.INVALID_HERMITIAN_AXIS
    )


def test_p773_remains_disconnected_from_applications_runtimes_and_roots():
    paths = (
        ROOT / "pssolver" / "applications" / "plane_beris_edwards.py",
        ROOT / "pssolver" / "applications" / "channel_active_nematics.py",
        ROOT / "pssolver" / "runtime" / "plane_beris_edwards.py",
        ROOT / "pssolver" / "runtime" / "channel_application_bridge.py",
        ROOT / "pssolver" / "configuration" / "__init__.py",
        ROOT / "pssolver" / "planning" / "__init__.py",
    )
    for path in paths:
        source = path.read_text(encoding="utf-8")
        assert "simulation_lowering" not in source
        assert "from .simulation import" not in source


def test_p773_machine_record_matches_sources_and_scope():
    record = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    assert record["schema_version"] == 1
    assert record["classification"] == (
        "PASS_P7_7_3_CAPABILITY_RESOLUTION_TENSOR_FREE_LOWERING"
    )
    assert record["runtime_connection"] is False
    assert record["tensor_allocation"] is False
    assert record["production_default_changed"] is False
    assert record["phase_8_authorized"] is False
    assert record["supported_pairs"] == [
        ["complete_stress_beris_edwards", "plane_slab"],
        ["legacy_active_force_active_nematics", "rectangular_channel"],
    ]
    assert set(record["source_sha256"]) == {
        "pssolver/configuration/simulation_lowering.py",
        "pssolver/planning/simulation.py",
    }
    assert all(len(value) == 64 for value in record["source_sha256"].values())
