"""P7.7.8 typed public declaration conveniences."""

from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

import pytest

import pssolver
from pssolver import (
    GeneratedInitialCondition,
    Output,
    Simulation,
    SnapshotInitialCondition,
    SpectralNumerics,
    TimeStepping,
    TorchSpectralExecution,
)
from pssolver.boundaries import (
    assign_boundaries,
    free_slip_velocity,
    neumann_pressure_compatibility,
    neumann_q,
)
from pssolver.configuration.simulation import (
    ExecutionSpec,
    InitialConditionSource,
    InitialConditionSpec,
    TimeIntegrationSpec,
    WorkflowSpec,
)
from pssolver.core.boundary import BoundaryKind, BoundarySemantic
from pssolver.core.domain import DomainSpec
from pssolver.core.integrators import IntegratorScheme
from pssolver.core.numerics import NumericsConfig, Precision
from pssolver.geometries import PlaneSlab, RectangularChannel
from pssolver.models.active_nematics import CompleteStressBerisEdwards
from pssolver.systems.equations import EquationSystemSpec


ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = ROOT / (
    "notes/architecture_v0_2/phase_7_p778_typed_public_declarations.json"
)
REVIEWED_SOURCES = (
    "pssolver/api/declarations.py",
    "pssolver/boundaries/homogeneous.py",
    "pssolver/models/active_nematics/public.py",
    "pssolver/geometries/public.py",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _model() -> EquationSystemSpec:
    return CompleteStressBerisEdwards(
        ldg_a=0.0,
        ldg_b=-0.3,
        ldg_c=0.3,
        ldg_l1=1.0 / 81.0,
        gamma=2.94,
        flow_alignment=0.3,
        activity=0.01,
        beta=-1.0,
        viscosity=2.0 / 3.0,
    )


def _geometry() -> PlaneSlab:
    return PlaneSlab(
        shape=(16, 16, 8),
        lengths=(100.0, 100.0, 20.0),
    )


def _boundaries(model: EquationSystemSpec, geometry: PlaneSlab):
    return assign_boundaries(
        model=model,
        geometry=geometry,
        policies={
            "Q": neumann_q(),
            "velocity": free_slip_velocity(),
            "pressure": neumann_pressure_compatibility(),
        },
    )


def _simulation() -> Simulation:
    model = _model()
    geometry = _geometry()
    return Simulation(
        model=model,
        geometry=geometry,
        boundaries=_boundaries(model, geometry),
        numerics=SpectralNumerics(
            dtype="float64",
            dealias_rule="cubic_half",
        ),
        time=TimeStepping(dt=0.005),
        initial_condition=GeneratedInitialCondition(
            "extruded_defect_gas",
            parameters={"seed": 24},
        ),
        execution=TorchSpectralExecution(
            runtime_path="compiled_v2",
            device="cuda",
            options={"tf32": "off"},
        ),
        output=Output(
            directory="data/A18",
            steps=20_000,
            save_interval=1_000,
            diagnostic_interval=100,
        ),
    )


def test_public_conveniences_normalize_to_p777_canonical_types():
    value = _simulation()

    assert isinstance(value.model, EquationSystemSpec)
    assert isinstance(value.numerics, NumericsConfig)
    assert isinstance(value.time, TimeIntegrationSpec)
    assert isinstance(value.initial_condition, InitialConditionSpec)
    assert isinstance(value.execution, ExecutionSpec)
    assert isinstance(value.output, WorkflowSpec)
    assert value.specification.equation_system is value.model
    assert value.specification.boundaries is value.boundaries
    json.dumps(value.to_metadata(), allow_nan=False, sort_keys=True)


def test_complete_stress_model_is_geometry_and_boundary_free():
    model = _model()
    parameters = model.parameters

    assert type(model) is EquationSystemSpec
    assert model.variant == "complete_stress_beris_edwards"
    assert parameters["material"] == {
        "ldg_a": 0.0,
        "ldg_b": -0.3,
        "ldg_c": 0.3,
        "gamma": 2.94,
        "flow_alignment": 0.3,
        "beta": -1.0,
    }
    assert parameters["ldg_l1"] == 1.0 / 81.0
    assert parameters["activity_amplitude"] == 0.01
    assert parameters["stokes"]["parameters"]["viscosity"] == 2.0 / 3.0
    assert parameters["stokes"]["parameters"]["friction"] == 0.0
    assert parameters["stokes"]["parameters"][
        "tangential_zero_mode_policy"
    ] == "zero_mean"
    assert "geometry" not in model.to_metadata()
    assert "boundaries" not in model.to_metadata()


def test_plane_slab_accepts_ergonomic_and_legacy_domain_forms():
    ergonomic = _geometry()
    legacy = PlaneSlab(DomainSpec((16, 16, 8), (100.0, 100.0, 20.0)))

    assert ergonomic == legacy
    assert ergonomic.periodic_axes == (0, 1)
    assert ergonomic.bounded_axes == (2,)
    with pytest.raises(ValueError, match="cannot be combined"):
        PlaneSlab(
            legacy.domain,
            shape=(16, 16, 8),
            lengths=(100.0, 100.0, 20.0),
        )


def test_boundary_builder_preserves_physical_and_compatibility_semantics():
    model = _model()
    geometry = _geometry()
    boundaries = _boundaries(model, geometry)

    for component in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz"):
        value = boundaries.for_component(component)
        assert value.semantic is BoundarySemantic.PHYSICAL
        assert {
            face.condition.kind for face in value.faces if face.axis == 2
        } == {BoundaryKind.NEUMANN}
    for component in ("ux", "uy"):
        assert {
            face.condition.kind
            for face in boundaries.for_component(component).faces
            if face.axis == 2
        } == {BoundaryKind.NEUMANN}
    assert {
        face.condition.kind
        for face in boundaries.for_component("uz").faces
        if face.axis == 2
    } == {BoundaryKind.DIRICHLET}
    pressure = boundaries.for_component("p")
    assert pressure.semantic is BoundarySemantic.ALGEBRAIC_COMPATIBILITY
    assert {
        face.condition.kind for face in pressure.faces if face.axis == 2
    } == {BoundaryKind.NEUMANN}


def test_boundary_builder_requires_explicit_pressure_compatibility():
    model = _model()
    geometry = _geometry()
    with pytest.raises(ValueError, match="missing=.*pressure"):
        assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "Q": neumann_q(),
                "velocity": free_slip_velocity(),
            },
        )
    with pytest.raises(ValueError, match="declares 'Q'"):
        assign_boundaries(
            model=model,
            geometry=geometry,
            policies={
                "Q": neumann_q(),
                "velocity": neumann_q(),
                "pressure": neumann_pressure_compatibility(),
            },
        )


def test_free_slip_policy_is_geometric_not_plane_hard_coded():
    model = _model()
    geometry = RectangularChannel(
        DomainSpec((16, 8, 8), (100.0, 20.0, 20.0))
    )
    boundaries = _boundaries(model, geometry)

    ux = boundaries.for_component("ux")
    uy = boundaries.for_component("uy")
    uz = boundaries.for_component("uz")
    assert {
        face.condition.kind for face in ux.faces if face.axis in (1, 2)
    } == {BoundaryKind.NEUMANN}
    assert {
        face.condition.kind for face in uy.faces if face.axis == 1
    } == {BoundaryKind.DIRICHLET}
    assert {
        face.condition.kind for face in uy.faces if face.axis == 2
    } == {BoundaryKind.NEUMANN}
    assert {
        face.condition.kind for face in uz.faces if face.axis == 1
    } == {BoundaryKind.NEUMANN}
    assert {
        face.condition.kind for face in uz.faces if face.axis == 2
    } == {BoundaryKind.DIRICHLET}


def test_typed_run_declarations_validate_and_remain_immutable():
    value = _simulation()

    assert value.numerics.precision is Precision.FLOAT64
    assert value.time.integrator.scheme is (
        IntegratorScheme.PROJECTED_SEMI_IMPLICIT_EULER
    )
    assert value.initial_condition.source is InitialConditionSource.GENERATED
    assert value.execution.backend == "torch_spectral"
    assert value.execution.options["fallback_allowed"] is False
    assert value.output.steps == 20_000
    assert value.output.options["output_dir"] == "data/A18"
    with pytest.raises(FrozenInstanceError):
        value.numerics.precision = Precision.FLOAT32
    with pytest.raises(ValueError, match="repeat device"):
        TorchSpectralExecution(
            runtime_path="compiled_v2",
            device="cuda",
            options={"device": "cpu"},
        )


def test_snapshot_and_sbdf2_conveniences_preserve_identity_ownership():
    snapshot = SnapshotInitialCondition("data/source", step=200, mode="branch")
    sbdf2 = TimeStepping(dt=0.001, integrator="sbdf2")

    assert snapshot.source is InitialConditionSource.SNAPSHOT
    assert snapshot.parameters == {
        "directory": "data/source",
        "mode": "branch",
        "step": 200,
    }
    assert sbdf2.integrator.scheme is IntegratorScheme.SBDF2
    assert sbdf2.integrator.formal_order == 2


def test_new_public_declaration_modules_are_tensor_and_runtime_free():
    forbidden = {
        "numpy",
        "torch",
        "pssolver.applications",
        "pssolver.backends",
        "pssolver.execution",
        "pssolver.integrators",
        "pssolver.linear_solvers",
        "pssolver.operators",
        "pssolver.runtime",
        "pssolver.transforms",
        "pssolver.workflows",
    }
    for relative in REVIEWED_SOURCES:
        path = ROOT / relative
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
        }, relative


def test_public_exports_retain_p778_declarations_after_runner_connection():
    expected = {
        "GeneratedInitialCondition",
        "Output",
        "Simulation",
        "SnapshotInitialCondition",
        "SpectralNumerics",
        "TimeStepping",
        "TorchSpectralExecution",
    }
    assert expected <= set(pssolver.__all__)
    assert "run_simulation" in pssolver.__all__
    assert callable(pssolver.run_simulation)


def test_machine_record_binds_scope_and_reviewed_sources():
    record = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    assert record["classification"] == (
        "PASS_P7_7_8_TYPED_PUBLIC_DECLARATIONS"
    )
    assert record["run_simulation_connected"] is False
    assert record["runtime_construction_connected"] is False
    assert record["phase_8_authorized"] is False
    assert record["production_default_changed"] is False
    assert record["source_sha256"] == {
        relative: _sha256(ROOT / relative) for relative in REVIEWED_SOURCES
    }
