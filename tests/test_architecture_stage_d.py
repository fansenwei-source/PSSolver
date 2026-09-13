"""Characterization tests for opt-in Stage D legacy assembly."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
import json

import pytest
import torch

import pssolver
from pssolver import SpectralSolver
from pssolver.adapters import compare_spectral_plan_to_runtime
from pssolver.core import (
    BoundarySet,
    DealiasRule,
    DomainSpec,
    FieldRole,
    FieldSpec,
    HomogeneousDirichletBC,
    HomogeneousNeumannBC,
    NumericsConfig,
    PeriodicBC,
    Precision,
    ProblemSpec,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.experimental import (
    create_legacy_projector,
    create_legacy_solver,
    declare_legacy_fields,
    materialize_legacy_assembly,
)
from pssolver.geometries import PeriodicBox, PlaneSlab, RectangularChannel
from pssolver.planning import assemble_spectral_plan


@dataclass(frozen=True)
class ScalarRuntimeModel:
    specs: tuple[FieldSpec, ...]
    name: str = "scalar_runtime_model"

    def field_specs(self):
        return self.specs

    def parameter_metadata(self):
        return {"diffusivity": 0.2}


def _numerics(
    *,
    storage=SpectralStorage.FULL_COMPLEX,
    hermitian_axis=None,
    dealias_rule=DealiasRule.CUBIC_HALF,
    execution=ProjectedTransformExecution.TRUNCATED,
):
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=dealias_rule,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=execution,
        spectral_storage=storage,
        hermitian_axis=hermitian_axis,
    )


def _scalar_problem(
    geometry,
    boundaries,
    numerics,
    *,
    include_algebraic=True,
    include_diagnostic=False,
):
    specs = [FieldSpec.scalar("phi", FieldRole.EVOLVED, boundaries)]
    if include_algebraic:
        specs.append(
            FieldSpec.scalar("response", FieldRole.ALGEBRAIC, boundaries)
        )
    if include_diagnostic:
        specs.append(
            FieldSpec.scalar("energy", FieldRole.DIAGNOSTIC, boundaries)
        )
    return ProblemSpec(
        ScalarRuntimeModel(tuple(specs)),
        geometry,
        numerics,
    )


def _initial_values(shape):
    size = 1
    for extent in shape:
        size *= extent
    return torch.linspace(-0.3, 0.7, size, dtype=torch.float64).reshape(shape)


def _manual_solver(problem, boundary_conditions, *, dt, initial):
    dtype = {
        Precision.FLOAT32: torch.float32,
        Precision.FLOAT64: torch.float64,
    }[problem.numerics.precision]
    solver = SpectralSolver(
        shape=problem.geometry.domain.shape,
        L=problem.geometry.domain.lengths,
        dt=dt,
        device="cpu",
        dtype=dtype,
        transform_execution_order=(
            problem.numerics.transform_execution_order.value
        ),
        spectral_storage=problem.numerics.spectral_storage.value,
        hermitian_axis=problem.numerics.hermitian_axis,
    )
    linear = -0.2 * solver.get_q2(boundary_conditions)
    solver.model.add_dynamic_field(
        "phi",
        initial.clone(),
        linear,
        boundary_conditions=boundary_conditions,
    )
    for field in problem.field_specs:
        if field.role is not FieldRole.ALGEBRAIC:
            continue
        component = field.components[0]
        solver.model.add_static_field(
            component.name,
            boundary_conditions=boundary_conditions,
        )
    solver.build()
    return solver


def _plan_assembled_solver(assembly, *, dt, initial):
    solver = create_legacy_solver(assembly, dt=dt, device="cpu")
    evolved = assembly.evolved_fields[0]
    linear = -0.2 * solver.get_q2(evolved.boundary_conditions)
    declare_legacy_fields(
        solver,
        assembly,
        initial_values={evolved.name: initial.clone()},
        linear_operators={evolved.name: linear},
    )
    solver.build()
    return solver


def test_materialization_resolves_backend_projector_and_flattened_fields():
    domain = DomainSpec((8, 6, 4), (8.0, 6.0, 4.0))
    q_boundaries = BoundarySet(
        (PeriodicBC(), PeriodicBC(), HomogeneousNeumannBC())
    )
    model = ScalarRuntimeModel(
        (
            FieldSpec.scalar("pressure", FieldRole.ALGEBRAIC, q_boundaries),
            FieldSpec.scalar("phase", FieldRole.EVOLVED, q_boundaries),
        )
    )
    problem = ProblemSpec(
        model,
        PlaneSlab(domain),
        _numerics(
            storage=SpectralStorage.HERMITIAN_HALF,
            hermitian_axis=1,
        ),
    )
    assembly = materialize_legacy_assembly(
        assemble_spectral_plan(problem)
    )

    assert assembly.backend.shape == (8, 6, 4)
    assert assembly.backend.lengths == (8.0, 6.0, 4.0)
    assert assembly.backend.spectral_storage is SpectralStorage.HERMITIAN_HALF
    assert assembly.backend.hermitian_axis == 1
    assert assembly.projector.dealias_rule is DealiasRule.CUBIC_HALF
    assert assembly.projector.transform_execution is (
        ProjectedTransformExecution.TRUNCATED
    )
    assert tuple(field.name for field in assembly.fields) == (
        "phase",
        "pressure",
    )
    assert tuple(field.legacy_role for field in assembly.fields) == (
        "dynamic",
        "static",
    )
    assert assembly.fields[0].boundary_conditions == (
        "periodic",
        "periodic",
        "neumann",
    )
    json.dumps(assembly.to_metadata(), allow_nan=False)


def test_diagnostics_require_explicit_acknowledgement_and_are_recorded():
    boundaries = BoundarySet((PeriodicBC(),))
    problem = _scalar_problem(
        PeriodicBox(DomainSpec((8,), (8.0,))),
        boundaries,
        _numerics(
            dealias_rule=DealiasRule.NONE,
            execution=ProjectedTransformExecution.FULL,
        ),
        include_diagnostic=True,
    )
    plan = assemble_spectral_plan(problem)

    with pytest.raises(ValueError, match="cannot store diagnostic"):
        materialize_legacy_assembly(plan)

    assembly = materialize_legacy_assembly(
        plan,
        allow_unstored_diagnostics=True,
    )
    assert assembly.omitted_diagnostic_components == ("energy",)
    assert tuple(field.name for field in assembly.fields) == (
        "phi",
        "response",
    )


def test_materialization_rejects_a_plan_without_an_evolved_component():
    boundaries = BoundarySet((PeriodicBC(),))
    model = ScalarRuntimeModel(
        (FieldSpec.scalar("response", FieldRole.ALGEBRAIC, boundaries),)
    )
    problem = ProblemSpec(
        model,
        PeriodicBox(DomainSpec((8,), (8.0,))),
        _numerics(
            dealias_rule=DealiasRule.NONE,
            execution=ProjectedTransformExecution.FULL,
        ),
    )

    with pytest.raises(ValueError, match="requires an evolved"):
        materialize_legacy_assembly(assemble_spectral_plan(problem))


def test_created_solver_and_projector_match_the_source_plan():
    boundaries = BoundarySet(
        (PeriodicBC(), PeriodicBC(), HomogeneousNeumannBC())
    )
    problem = _scalar_problem(
        PlaneSlab(DomainSpec((8, 6, 4), (8.0, 6.0, 4.0))),
        boundaries,
        _numerics(
            storage=SpectralStorage.HERMITIAN_HALF,
            hermitian_axis=0,
        ),
    )
    plan = assemble_spectral_plan(problem)
    assembly = materialize_legacy_assembly(plan)

    solver = create_legacy_solver(assembly, dt=0.01, device="cpu")
    projector = create_legacy_projector(solver, assembly)

    comparison = compare_spectral_plan_to_runtime(
        plan,
        solver.transform_backend,
        projector=projector,
    )
    assert comparison.matches
    assert solver.dtype is torch.float64
    assert solver.spectral_shape == (5, 6, 4)
    assert projector.rule == "cubic_half"
    assert projector.transform_execution == "truncated"


@pytest.mark.parametrize("geometry_case", ("periodic", "plane", "channel"))
def test_plan_assembled_and_manual_solvers_have_identical_short_trajectories(
    geometry_case,
):
    if geometry_case == "periodic":
        shape = (9,)
        domain = DomainSpec(shape, (9.0,))
        geometry = PeriodicBox(domain)
        boundaries = BoundarySet((PeriodicBC(),))
        legacy_boundaries = ("periodic",)
        numerics = _numerics(
            dealias_rule=DealiasRule.NONE,
            execution=ProjectedTransformExecution.FULL,
        )
    elif geometry_case == "plane":
        shape = (8, 7)
        domain = DomainSpec(shape, (8.0, 7.0))
        geometry = PlaneSlab(domain)
        boundaries = BoundarySet(
            (PeriodicBC(), HomogeneousNeumannBC())
        )
        legacy_boundaries = ("periodic", "neumann")
        numerics = _numerics(
            storage=SpectralStorage.HERMITIAN_HALF,
            hermitian_axis=0,
        )
    else:
        shape = (6, 5, 4)
        domain = DomainSpec(shape, (6.0, 5.0, 4.0))
        geometry = RectangularChannel(domain)
        boundaries = BoundarySet(
            (
                PeriodicBC(),
                HomogeneousDirichletBC(),
                HomogeneousNeumannBC(),
            )
        )
        legacy_boundaries = ("periodic", "dirichlet", "neumann")
        numerics = _numerics()

    problem = _scalar_problem(geometry, boundaries, numerics)
    assembly = materialize_legacy_assembly(
        assemble_spectral_plan(problem)
    )
    initial = _initial_values(shape)
    manual = _manual_solver(
        problem,
        legacy_boundaries,
        dt=0.01,
        initial=initial,
    )
    planned = _plan_assembled_solver(assembly, dt=0.01, initial=initial)

    assert manual.fields.name_to_idx == planned.fields.name_to_idx
    assert manual.fields.boundary_conditions == planned.fields.boundary_conditions
    assert torch.equal(manual.fields.spatial, planned.fields.spatial)
    assert torch.equal(manual.fields.spectral, planned.fields.spectral)
    assert torch.equal(manual.fields.L_hat, planned.fields.L_hat)

    for _ in range(5):
        manual.integrator.step()
        planned.integrator.step()
        assert torch.equal(manual.fields.spatial, planned.fields.spatial)
        assert torch.equal(manual.fields.spectral, planned.fields.spectral)


def test_field_input_validation_is_atomic_for_missing_and_extra_keys():
    boundaries = BoundarySet((PeriodicBC(),))
    problem = _scalar_problem(
        PeriodicBox(DomainSpec((8,), (8.0,))),
        boundaries,
        _numerics(
            dealias_rule=DealiasRule.NONE,
            execution=ProjectedTransformExecution.FULL,
        ),
    )
    assembly = materialize_legacy_assembly(
        assemble_spectral_plan(problem)
    )
    solver = create_legacy_solver(assembly, dt=0.01)
    operator = solver.get_q2(("periodic",))

    with pytest.raises(ValueError, match="missing=.*phi"):
        declare_legacy_fields(
            solver,
            assembly,
            initial_values={},
            linear_operators={"phi": operator},
        )
    assert solver.model.dyn_fields == []
    assert solver.model.stat_fields == []

    with pytest.raises(TypeError, match="keys must be strings"):
        declare_legacy_fields(
            solver,
            assembly,
            initial_values={1: torch.zeros(8)},
            linear_operators={"phi": operator},
        )
    assert solver.model.dyn_fields == []
    assert solver.model.stat_fields == []

    with pytest.raises(ValueError, match="unexpected=.*extra"):
        declare_legacy_fields(
            solver,
            assembly,
            initial_values={
                "phi": torch.zeros(8),
                "extra": torch.zeros(8),
            },
            linear_operators={"phi": operator},
        )
    assert solver.model.dyn_fields == []
    assert solver.model.stat_fields == []


def test_field_shape_validation_is_atomic():
    boundaries = BoundarySet((PeriodicBC(),))
    problem = _scalar_problem(
        PeriodicBox(DomainSpec((8,), (8.0,))),
        boundaries,
        _numerics(
            dealias_rule=DealiasRule.NONE,
            execution=ProjectedTransformExecution.FULL,
        ),
    )
    assembly = materialize_legacy_assembly(
        assemble_spectral_plan(problem)
    )
    solver = create_legacy_solver(assembly, dt=0.01)
    operator = solver.get_q2(("periodic",))

    with pytest.raises(ValueError, match="initial value"):
        declare_legacy_fields(
            solver,
            assembly,
            initial_values={"phi": torch.zeros(7)},
            linear_operators={"phi": operator},
        )
    assert solver.model.dyn_fields == []
    assert solver.model.stat_fields == []


def test_declaration_rejects_a_nonempty_or_mismatched_solver():
    boundaries = BoundarySet((PeriodicBC(),))
    problem = _scalar_problem(
        PeriodicBox(DomainSpec((8,), (8.0,))),
        boundaries,
        _numerics(
            dealias_rule=DealiasRule.NONE,
            execution=ProjectedTransformExecution.FULL,
        ),
    )
    assembly = materialize_legacy_assembly(
        assemble_spectral_plan(problem)
    )
    solver = create_legacy_solver(assembly, dt=0.01)
    operator = solver.get_q2(("periodic",))
    solver.model.add_static_field("existing")

    with pytest.raises(RuntimeError, match="already contains"):
        declare_legacy_fields(
            solver,
            assembly,
            initial_values={"phi": torch.zeros(8)},
            linear_operators={"phi": operator},
        )

    wrong_solver = SpectralSolver((7,), L=(7.0,), device="cpu")
    with pytest.raises(RuntimeError, match="shadow mismatch"):
        declare_legacy_fields(
            wrong_solver,
            assembly,
            initial_values={"phi": torch.zeros(8)},
            linear_operators={"phi": operator},
        )


@pytest.mark.parametrize(
    "kwargs, error",
    (
        ({"dt": 0.0}, "dt"),
        ({"dt": 0.01, "batchsize": 0}, "batchsize"),
        ({"dt": float("nan")}, "dt"),
    ),
)
def test_solver_creation_rejects_invalid_runtime_values(kwargs, error):
    boundaries = BoundarySet((PeriodicBC(),))
    problem = _scalar_problem(
        PeriodicBox(DomainSpec((8,), (8.0,))),
        boundaries,
        _numerics(
            dealias_rule=DealiasRule.NONE,
            execution=ProjectedTransformExecution.FULL,
        ),
    )
    assembly = materialize_legacy_assembly(
        assemble_spectral_plan(problem)
    )

    with pytest.raises(ValueError, match=error):
        create_legacy_solver(assembly, **kwargs)


def test_legacy_assembly_is_explicit_and_not_exported_by_production_api():
    assert not hasattr(pssolver, "materialize_legacy_assembly")
    assert not hasattr(pssolver, "create_legacy_solver")


def test_legacy_assembly_spec_is_immutable():
    boundaries = BoundarySet((PeriodicBC(),))
    problem = _scalar_problem(
        PeriodicBox(DomainSpec((8,), (8.0,))),
        boundaries,
        _numerics(
            dealias_rule=DealiasRule.NONE,
            execution=ProjectedTransformExecution.FULL,
        ),
    )
    assembly = materialize_legacy_assembly(
        assemble_spectral_plan(problem)
    )

    with pytest.raises(FrozenInstanceError):
        assembly.fields = ()
