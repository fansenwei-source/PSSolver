"""Golden restart-provenance gate across the P2.1S algebraic extraction."""

from __future__ import annotations

import json
from pathlib import Path

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
from pssolver.execution import (
    AlgebraicSystemSpec as LegacyAlgebraicSystemSpec,
    IncompressibleStokesSystemSpec,
    PressureGauge,
    TangentialZeroModePolicy,
)
from pssolver.experimental import (
    build_experimental_model_runtime,
    create_stokes_geometry_solver_registry,
)
from pssolver.geometries import PlaneSlab
from pssolver.models.canary import BodyForceStokesCanaryModel
from pssolver.systems.algebraic import AlgebraicSystemSpec


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES = json.loads(
    (
        PROJECT_ROOT
        / "tests"
        / "fixtures"
        / "systems"
        / "algebraic_stokes_cases_v1.json"
    ).read_text(encoding="utf-8")
)


def _runtime():
    periodic = PeriodicBC()
    neumann = HomogeneousNeumannBC()
    dirichlet = HomogeneousDirichletBC()
    tangential = BoundarySet((periodic, periodic, neumann))
    normal = BoundarySet((periodic, periodic, dirichlet))
    stokes = IncompressibleStokesSystemSpec(
        name="flow",
        force_components=("force_x", "force_y", "force_z"),
        velocity_components=("ux", "uy", "uz"),
        pressure_component="p",
        viscosity=0.73,
        friction=0.0,
        pressure_gauge=PressureGauge.ZERO_MEAN,
        tangential_zero_mode_policy=TangentialZeroModePolicy.ZERO_MEAN,
    )
    model = BodyForceStokesCanaryModel(
        force_boundaries=(tangential, tangential, normal),
        velocity_boundaries=(tangential, tangential, normal),
        pressure_boundaries=tangential,
        stokes_system=stokes,
        force_diffusivity=0.025,
        initial_amplitudes=(0.7, -0.35, 0.2),
        initial_modes=((1, 1, 1), (2, 1, 2), (1, 2, 1)),
    )
    numerics = NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=DealiasRule.NONE,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=ProjectedTransformExecution.FULL,
        spectral_storage=SpectralStorage.FULL_COMPLEX,
    )
    return build_experimental_model_runtime(
        model,
        PlaneSlab(DomainSpec((7, 6, 5), (4.0, 3.0, 2.5))),
        numerics,
        dt=0.01,
        device="cpu",
        batch_size=2,
        geometry_solver_registry=create_stokes_geometry_solver_registry(),
    )


def test_plane_stokes_restart_provenance_is_preserved_after_algebraic_extraction():
    runtime = _runtime()
    resolved = runtime.resolved_algebraic_systems[0]
    restart = runtime.capture_algebraic_restart_state()

    assert AlgebraicSystemSpec is LegacyAlgebraicSystemSpec
    assert type(resolved.system) is AlgebraicSystemSpec
    assert IncompressibleStokesSystemSpec.from_algebraic_system_spec(
        resolved.system
    ) == runtime.problem.model.stokes_system
    assert restart.systems[0].provenance_sha256 == (
        CASES["plane_restart_provenance_sha256"]
    )
