"""P8.5.3 generic scalar Plane Robin lowering and runtime-state pilot."""

from __future__ import annotations

import ast
from dataclasses import replace
import json
import math
from pathlib import Path

import pytest
import torch

import pssolver
from pssolver.boundaries import (
    HomogeneousBoundaryPolicy,
    assign_boundaries,
    robin,
)
from pssolver.configuration.robin import lower_plane_robin_scalar_pilot
from pssolver.configuration.simulation import (
    ExecutionSpec,
    InitialConditionSource,
    InitialConditionSpec,
    InvocationSpec,
    SimulationSpec,
    TimeIntegrationSpec,
    WorkflowSpec,
)
from pssolver.configuration.simulation_lowering import (
    SimulationLoweringError,
    lower_simulation_spec,
)
from pssolver.core.boundary import BoundarySemantic
from pssolver.core.fields import FieldRole
from pssolver.core.integrators import IntegratorSpec
from pssolver.core.numerics import (
    DealiasRule,
    NumericsConfig,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.geometries import PeriodicBox, PlaneSlab
from pssolver.planning import PlaneRobinScalarLoweringPlan
from pssolver.runtime import (
    PlaneRobinOperatorCache,
    PlaneRobinScalarRuntime,
)
from pssolver.systems.algebraic import AlgebraicSystemSpec
from pssolver.systems.equations import (
    EquationFieldSpec,
    EquationSystemSpec,
    EquationTermSpec,
)


ROOT = Path(__file__).resolve().parents[1]
NOTES = ROOT / "notes" / "architecture_v0_2"


def _system() -> EquationSystemSpec:
    return EquationSystemSpec(
        name="generic_transport",
        variant="generic_scalar_robin_transport",
        fields=(
            EquationFieldSpec("concentration", FieldRole.EVOLVED, ("c",)),
            EquationFieldSpec("flux", FieldRole.TRANSIENT, ("j",)),
            EquationFieldSpec(
                "multiplier",
                FieldRole.ALGEBRAIC,
                ("lambda_value",),
            ),
        ),
        evolution_laws=(
            EquationTermSpec(
                "evolve_concentration",
                "generic_transport_rhs",
                ("c",),
                ("c", "j"),
            ),
        ),
        constitutive_laws=(
            EquationTermSpec(
                "construct_flux",
                "generic_flux",
                ("j",),
                ("c",),
            ),
        ),
        algebraic_systems=(
            AlgebraicSystemSpec(
                "solve_multiplier",
                "generic_constraint",
                ("lambda_value",),
                ("c", "j"),
            ),
        ),
        parameters={"diffusivity": 0.2},
        initial_condition_families=("uniform_scalar",),
    )


def _simulation(
    *,
    shape: tuple[int, int, int] = (6, 5, 12),
    lower: tuple[float, float, float] = (1.3, 0.8, 0.4),
    upper: tuple[float, float, float] = (0.7, 1.1, -0.2),
) -> SimulationSpec:
    system = _system()
    geometry = PlaneSlab(shape=shape, lengths=(3.0, 2.5, 2.3))
    boundaries = assign_boundaries(
        model=system,
        geometry=geometry,
        policies={
            "concentration": robin(
                "concentration",
                {
                    ("c", 2, "lower"): lower,
                    ("c", 2, "upper"): upper,
                },
            ),
            "multiplier": HomogeneousBoundaryPolicy(
                "multiplier",
                BoundarySemantic.ALGEBRAIC_COMPATIBILITY,
                "neumann",
            ),
        },
    )
    return SimulationSpec(
        equation_system=system,
        geometry=geometry,
        boundaries=boundaries,
        numerics=NumericsConfig(
            precision=Precision.FLOAT64,
            dealias_rule=DealiasRule.CUBIC_HALF,
            transform_execution_order=TransformExecutionOrder.REAL_FIRST,
            projected_transform_execution=(
                ProjectedTransformExecution.TRUNCATED
            ),
            spectral_storage=SpectralStorage.FULL_COMPLEX,
        ),
        time_integration=TimeIntegrationSpec(
            IntegratorSpec.projected_semi_implicit_euler(dt=0.01),
        ),
        discretization_parameters={"pilot": "bounded_axis_only"},
        initial_condition=InitialConditionSpec(
            "uniform_scalar",
            InitialConditionSource.GENERATED,
            {"value": 0.0},
        ),
        execution=ExecutionSpec(
            "torch_spectral",
            "robin_scalar_bounded_axis_pilot",
            {"fallback_allowed": False},
        ),
        workflow=WorkflowSpec(steps=1, options={"save": False}),
        invocation=InvocationSpec({"origin": "p8_5_3_test"}),
    )


def _plan(simulation: SimulationSpec | None = None) -> PlaneRobinScalarLoweringPlan:
    return lower_plane_robin_scalar_pilot(
        _simulation() if simulation is None else simulation,
        field_name="concentration",
    )


def _initial(plan: PlaneRobinScalarLoweringPlan) -> torch.Tensor:
    x = (
        torch.arange(plan.domain_shape[0], dtype=torch.float64) + 0.5
    ) / plan.domain_shape[0]
    y = (
        torch.arange(plan.domain_shape[1], dtype=torch.float64) + 0.5
    ) / plan.domain_shape[1]
    z = (
        torch.arange(plan.domain_shape[2], dtype=torch.float64) + 0.5
    ) * (plan.domain_lengths[2] / plan.domain_shape[2])
    return (
        0.2 * torch.sin(2.0 * math.pi * x)[:, None, None]
        + 0.1 * torch.cos(2.0 * math.pi * y)[None, :, None]
        + torch.exp(0.3 * z)[None, None, :]
    ).contiguous()


def test_generic_scalar_lowering_binds_registered_field_and_robin_plan():
    simulation = _simulation()
    plan = _plan(simulation)

    assert plan.field_name == "concentration"
    assert plan.component == "c"
    assert plan.geometry_name == "plane_slab"
    assert plan.periodic_axes == (0, 1)
    assert plan.wall_normal_axis == 2
    assert plan.domain_shape == (6, 5, 12)
    assert plan.robin_plan.size == 12
    assert plan.robin_plan.lower.alpha.value == 1.3
    assert plan.robin_plan.upper.gamma.value == -0.2
    assert plan.source_simulation_sha256 == simulation.canonical_sha256()
    metadata = plan.to_metadata()
    assert metadata["field_role"] == "evolved"
    assert metadata["periodic_axis_method"] == (
        "fourier_declared_not_materialized_p8_5_3"
    )
    assert metadata["model_specialization"] is None
    assert metadata["complete_timestep_connected"] is False
    json.dumps(metadata, allow_nan=False, sort_keys=True)


def test_lowering_is_fail_closed_outside_the_exact_scalar_plane_slice():
    simulation = _simulation()
    with pytest.raises(ValueError, match="registered exactly once"):
        lower_plane_robin_scalar_pilot(simulation, field_name="missing")
    with pytest.raises(ValueError, match="periodic axis 2"):
        lower_plane_robin_scalar_pilot(
            replace(
                simulation,
                geometry=PeriodicBox(
                    shape=(6, 5, 12),
                    lengths=(3.0, 2.5, 2.3),
                ),
            ),
            field_name="concentration",
        )
    with pytest.raises(ValueError, match="float64"):
        lower_plane_robin_scalar_pilot(
            replace(
                simulation,
                numerics=replace(
                    simulation.numerics,
                    precision=Precision.FLOAT32,
                ),
            ),
            field_name="concentration",
        )
    with pytest.raises(ValueError, match="torch_spectral"):
        lower_plane_robin_scalar_pilot(
            replace(
                simulation,
                execution=ExecutionSpec(
                    "other_backend",
                    "robin_scalar_bounded_axis_pilot",
                ),
            ),
            field_name="concentration",
        )


def test_production_lowering_and_public_runner_remain_disconnected():
    simulation = _simulation()
    with pytest.raises(SimulationLoweringError):
        lower_simulation_spec(simulation)
    assert not hasattr(pssolver, "PlaneRobinScalarRuntime")
    assert not hasattr(pssolver, "lower_plane_robin_scalar_pilot")


def test_runtime_reconstructs_physical_observation_and_wall_residual():
    plan = _plan()
    initial = _initial(plan)
    runtime = PlaneRobinScalarRuntime(plan, initial, dt=0.01)

    assert torch.max(
        torch.abs(runtime.physical_observation() - initial)
    ).item() < 2.0e-15
    assert runtime.state.component_names == ("c",)
    assert runtime.state.physical.shape == (1, *plan.domain_shape)
    assert runtime.state.spectral.shape == (1, *plan.domain_shape)
    lower, upper = runtime.boundary_residual()
    assert lower.shape == plan.domain_shape[:2]
    assert upper.shape == plan.domain_shape[:2]
    assert torch.max(torch.abs(lower)).item() < 2.0e-13
    assert torch.max(torch.abs(upper)).item() < 2.0e-12


def test_bounded_helmholtz_operates_independently_for_every_periodic_point():
    plan = _plan()
    runtime = PlaneRobinScalarRuntime(plan, _initial(plan), dt=0.01)
    mass = 1.9
    physical = runtime.physical_observation()
    forcing = runtime.apply_bounded_helmholtz(mass=mass)
    recovered = runtime.solve_bounded_helmholtz(forcing, mass=mass)
    assert torch.max(torch.abs(recovered - physical)).item() < 2.0e-13


def test_operator_cache_identity_reuses_only_the_exact_plan():
    cache = PlaneRobinOperatorCache()
    first_plan = _plan()
    first = PlaneRobinScalarRuntime(
        first_plan,
        _initial(first_plan),
        dt=0.01,
        cache=cache,
    )
    second = PlaneRobinScalarRuntime(
        first_plan,
        _initial(first_plan),
        dt=0.01,
        cache=cache,
    )
    changed_plan = _plan(_simulation(lower=(1.4, 0.8, 0.4)))
    changed = PlaneRobinScalarRuntime(
        changed_plan,
        _initial(changed_plan),
        dt=0.01,
        cache=cache,
    )

    assert first.operator is second.operator
    assert first.cache_key == second.cache_key
    assert changed.operator is not first.operator
    assert changed.cache_key != first.cache_key
    assert cache.entry_count == 2
    assert first.cache_key.to_metadata()["domain_shape"] == [6, 5, 12]
    assert len(first.cache_key.canonical_sha256()) == 64


def test_checkpoint_restore_is_exact_and_preserves_physical_observation():
    plan = _plan()
    initial = _initial(plan)
    runtime = PlaneRobinScalarRuntime(plan, initial, dt=0.01)
    checkpoint = runtime.capture_checkpoint()
    replacement = initial + 0.25 * torch.sin(
        torch.linspace(0.0, 1.0, plan.domain_shape[2], dtype=torch.float64)
    )
    runtime.replace_physical_observation(replacement)
    assert not torch.equal(runtime.physical_observation(), initial)

    runtime.restore_checkpoint(checkpoint)

    assert torch.equal(runtime.remainder, checkpoint.remainder)
    assert torch.equal(runtime.bounded_modal, checkpoint.bounded_modal)
    assert torch.max(
        torch.abs(runtime.physical_observation() - initial)
    ).item() < 2.0e-15
    assert runtime.state.representations.current.value == "synchronized"
    assert runtime.state.progress.completed_steps == 0


def test_checkpoint_identity_or_payload_tamper_rejects_before_target_mutation():
    first_plan = _plan()
    first = PlaneRobinScalarRuntime(
        first_plan,
        _initial(first_plan),
        dt=0.01,
    )
    checkpoint = first.capture_checkpoint()
    other_plan = _plan(_simulation(upper=(0.8, 1.1, -0.2)))
    target = PlaneRobinScalarRuntime(
        other_plan,
        _initial(other_plan) + 0.5,
        dt=0.01,
    )
    before_remainder = target.remainder.clone()
    before_modal = target.bounded_modal.clone()
    with pytest.raises(ValueError, match="identity does not match"):
        target.restore_checkpoint(checkpoint)
    assert torch.equal(target.remainder, before_remainder)
    assert torch.equal(target.bounded_modal, before_modal)

    same_target = PlaneRobinScalarRuntime(
        first_plan,
        _initial(first_plan) + 0.5,
        dt=0.01,
    )
    before_remainder = same_target.remainder.clone()
    before_modal = same_target.bounded_modal.clone()
    checkpoint.remainder[0, 0, 0] += 1.0
    with pytest.raises(ValueError, match="payload identity"):
        same_target.restore_checkpoint(checkpoint)
    assert torch.equal(same_target.remainder, before_remainder)
    assert torch.equal(same_target.bounded_modal, before_modal)


def test_runtime_metadata_records_plan_cache_state_and_disconnected_scope():
    plan = _plan()
    runtime = PlaneRobinScalarRuntime(plan, _initial(plan), dt=0.01)
    metadata = runtime.to_metadata()
    identity = runtime.checkpoint_identity_metadata()

    assert metadata["runtime_kind"] == "plane_robin_scalar_bounded_axis_pilot"
    assert metadata["operator_cache_entry_count"] == 1
    assert metadata["complete_timestep_connected"] is False
    assert metadata["public_runner_connected"] is False
    assert metadata["runtime_selector_connected"] is False
    assert identity["normal_derivative_convention"] == "outward_unit_normal"
    assert identity["model_specialization"] is None
    assert identity["lowering_plan_sha256"] == plan.canonical_sha256()
    assert len(identity["operator_materialized_sha256"]) == 64
    json.dumps(metadata, allow_nan=False, sort_keys=True)


def test_generic_lowering_and_runtime_do_not_import_active_nematics():
    for relative in (
        "pssolver/configuration/robin.py",
        "pssolver/planning/robin.py",
        "pssolver/operators/robin.py",
        "pssolver/runtime/robin_scalar.py",
    ):
        path = ROOT / relative
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(path))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module)
        assert not any(
            value == "pssolver.models"
            or value.startswith("pssolver.models.")
            for value in imports
        )
        lowered = source.lower()
        assert "qstar" not in lowered
        assert "director" not in lowered
        assert "scalar_order" not in lowered


def test_p853_record_and_future_archive_are_present():
    record = json.loads(
        (NOTES / "phase_8_p853_robin_scalar_runtime.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["phase"] == "P8.5.3"
    assert record["classification"] == (
        "PASS_P8_5_3_GENERIC_PLANE_ROBIN_SCALAR_RUNTIME_STATE"
    )
    assert record["runtime_scope"]["complete_timestep_connected"] is False
    assert record["finite_q_anchoring_implemented"] is False
    assert record["h100_used"] is False
    assert record["authorization"]["eligible_for_p8_5_4"] is True
    assert record["authorization"]["p8_5_4_implemented"] is False
    archive = (NOTES / "build_verbatim_archive_pdf.py").read_text(
        encoding="utf-8"
    )
    for name in (
        "phase_8_p853_robin_scalar_runtime.md",
        "phase_8_p853_robin_scalar_runtime.json",
    ):
        assert archive.count(f'"{name}"') == 1
