"""Characterization tests for additive Stage A architecture contracts."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass
import json

import pytest

from pssolver import DEFAULT_SPECTRAL_STORAGE, SpectralSolver
from pssolver.core import (
    AxisTopology,
    BoundaryCondition,
    BoundaryKind,
    BoundarySet,
    DealiasRule,
    DomainSpec,
    FieldComponentSpec,
    FieldRole,
    FieldSpec,
    GeometrySpec,
    GridPlacement,
    HomogeneousDirichletBC,
    HomogeneousNeumannBC,
    ModelProtocol,
    NumericsConfig,
    PeriodicBC,
    Precision,
    ProblemSpec,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.plane import (
    DEFAULT_PLANE_SPECTRAL_STORAGE,
    PLANE_HERMITIAN_AXIS,
)


def _plane_boundaries(normal_condition):
    return BoundarySet(
        (
            PeriodicBC(),
            PeriodicBC(),
            normal_condition,
        )
    )


def _plane_geometry():
    return GeometrySpec(
        name="plane_slab",
        domain=DomainSpec(
            shape=(12, 10, 8),
            lengths=(100.0, 100.0, 20.0),
        ),
        axis_topologies=(
            AxisTopology.PERIODIC,
            AxisTopology.PERIODIC,
            AxisTopology.BOUNDED,
        ),
    )


def _plane_numerics(*, hermitian_axis=1):
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=DealiasRule.CUBIC_HALF,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=(
            ProjectedTransformExecution.TRUNCATED
        ),
        spectral_storage=SpectralStorage.HERMITIAN_HALF,
        hermitian_axis=hermitian_axis,
    )


@dataclass(frozen=True)
class ExampleModel:
    specs: tuple[FieldSpec, ...]
    name: str = "example_model"

    def field_specs(self):
        return self.specs

    def parameter_metadata(self):
        return {"coefficient": 2.0}


def _example_plane_model():
    q_boundaries = _plane_boundaries(HomogeneousNeumannBC())
    normal_velocity_boundaries = _plane_boundaries(
        HomogeneousDirichletBC()
    )
    return ExampleModel(
        specs=(
            FieldSpec(
                name="Q",
                role=FieldRole.EVOLVED,
                components=tuple(
                    FieldComponentSpec(name, q_boundaries)
                    for name in ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
                ),
            ),
            FieldSpec(
                name="velocity",
                role=FieldRole.ALGEBRAIC,
                components=(
                    FieldComponentSpec("ux", q_boundaries),
                    FieldComponentSpec("uy", q_boundaries),
                    FieldComponentSpec("uz", normal_velocity_boundaries),
                ),
            ),
            FieldSpec.scalar(
                "p",
                FieldRole.ALGEBRAIC,
                q_boundaries,
            ),
        )
    )


def test_boundary_contracts_preserve_physical_semantics():
    conditions = (
        PeriodicBC(),
        HomogeneousDirichletBC(),
        HomogeneousNeumannBC(),
    )

    assert all(isinstance(condition, BoundaryCondition) for condition in conditions)
    assert tuple(condition.kind for condition in conditions) == (
        BoundaryKind.PERIODIC,
        BoundaryKind.DIRICHLET,
        BoundaryKind.NEUMANN,
    )
    assert BoundarySet(conditions).to_metadata() == [
        {"kind": "periodic", "is_homogeneous": True},
        {"kind": "dirichlet", "is_homogeneous": True},
        {"kind": "neumann", "is_homogeneous": True},
    ]
    assert all(condition.is_homogeneous for condition in conditions)


def test_boundary_and_domain_specs_are_immutable_and_canonical():
    boundaries = BoundarySet([PeriodicBC(), HomogeneousNeumannBC()])
    domain = DomainSpec([8, 6], [4, 3])

    assert boundaries.axes == (PeriodicBC(), HomogeneousNeumannBC())
    assert domain.shape == (8, 6)
    assert domain.lengths == (4.0, 3.0)
    assert domain.axis_names == ("x", "y")
    assert domain.grid_placement is GridPlacement.CELL_CENTERED
    with pytest.raises(FrozenInstanceError):
        domain.shape = (4, 4)


@pytest.mark.parametrize(
    "kwargs, error",
    (
        ({"shape": (), "lengths": ()}, "one, two, or three"),
        ({"shape": (4, 0), "lengths": (1, 1)}, "positive integer"),
        ({"shape": (4,), "lengths": (0,)}, "positive and finite"),
        ({"shape": (4, 5), "lengths": (1,)}, "same dimension"),
        (
            {
                "shape": (4, 5),
                "lengths": (1, 1),
                "axis_names": ("x", "x"),
            },
            "unique",
        ),
    ),
)
def test_domain_spec_rejects_invalid_geometry_data(kwargs, error):
    with pytest.raises(ValueError, match=error):
        DomainSpec(**kwargs)


def test_geometry_reports_periodic_and_bounded_axes():
    geometry = _plane_geometry()

    assert geometry.periodic_axes == (0, 1)
    assert geometry.bounded_axes == (2,)
    assert geometry.to_metadata()["axis_topologies"] == [
        "periodic",
        "periodic",
        "bounded",
    ]


def test_field_specs_keep_component_boundaries_and_roles_separate():
    model = _example_plane_model()
    q_field, velocity_field, pressure_field = model.field_specs()

    assert q_field.role is FieldRole.EVOLVED
    assert q_field.component_names == ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
    assert velocity_field.role is FieldRole.ALGEBRAIC
    assert velocity_field.components[0].boundaries.axes[-1].kind is BoundaryKind.NEUMANN
    assert velocity_field.components[2].boundaries.axes[-1].kind is BoundaryKind.DIRICHLET
    assert pressure_field.component_names == ("p",)


def test_model_protocol_and_problem_spec_form_an_auditable_plane_problem():
    model = _example_plane_model()

    assert isinstance(model, ModelProtocol)
    problem = ProblemSpec(
        model=model,
        geometry=_plane_geometry(),
        numerics=_plane_numerics(),
    )
    metadata = problem.to_metadata()

    assert tuple(spec.name for spec in problem.field_specs) == (
        "Q",
        "velocity",
        "p",
    )
    assert metadata["model"]["name"] == "example_model"
    assert metadata["geometry"]["periodic_axes"] == [0, 1]
    assert metadata["numerics"]["spectral_storage"] == "hermitian_half"
    assert metadata["numerics"]["hermitian_axis"] == 1
    json.dumps(metadata)


def test_problem_snapshots_parameter_metadata_at_construction():
    model = _example_plane_model()
    problem = ProblemSpec(model, _plane_geometry(), _plane_numerics())

    first = problem.to_metadata()
    first["model"]["parameters"]["coefficient"] = 99.0

    assert problem.to_metadata()["model"]["parameters"] == {
        "coefficient": 2.0
    }


def test_problem_rejects_nonfinite_parameter_metadata():
    @dataclass(frozen=True)
    class NonfiniteModel(ExampleModel):
        def parameter_metadata(self):
            return {"coefficient": float("nan")}

    with pytest.raises(ValueError, match="finite and JSON-compatible"):
        ProblemSpec(
            NonfiniteModel(_example_plane_model().field_specs()),
            _plane_geometry(),
            _plane_numerics(),
        )


@pytest.mark.parametrize(
    "kwargs, error",
    (
        (
            {
                "dealias_rule": DealiasRule.NONE,
                "projected_transform_execution": (
                    ProjectedTransformExecution.TRUNCATED
                ),
            },
            "require dealiasing",
        ),
        (
            {
                "transform_execution_order": TransformExecutionOrder.LEGACY,
                "spectral_storage": SpectralStorage.HERMITIAN_HALF,
                "hermitian_axis": 1,
            },
            "real-first",
        ),
        (
            {
                "spectral_storage": SpectralStorage.HERMITIAN_HALF,
                "hermitian_axis": None,
            },
            "non-negative integer axis",
        ),
        (
            {
                "spectral_storage": SpectralStorage.FULL_COMPLEX,
                "hermitian_axis": 1,
            },
            "must be None",
        ),
    ),
)
def test_numerics_config_rejects_incompatible_policies(kwargs, error):
    values = {
        "precision": Precision.FLOAT64,
        "dealias_rule": DealiasRule.CUBIC_HALF,
        "transform_execution_order": TransformExecutionOrder.REAL_FIRST,
        "projected_transform_execution": ProjectedTransformExecution.FULL,
        "spectral_storage": SpectralStorage.FULL_COMPLEX,
        "hermitian_axis": None,
    }
    values.update(kwargs)

    with pytest.raises(ValueError, match=error):
        NumericsConfig(**values)


def test_problem_rejects_field_boundary_and_geometry_topology_mismatch():
    periodic_everywhere = BoundarySet(
        (PeriodicBC(), PeriodicBC(), PeriodicBC())
    )
    model = ExampleModel(
        specs=(
            FieldSpec.scalar(
                "phi",
                FieldRole.EVOLVED,
                periodic_everywhere,
            ),
        )
    )

    with pytest.raises(ValueError, match="cannot be periodic"):
        ProblemSpec(model, _plane_geometry(), _plane_numerics())


def test_problem_rejects_hermitian_storage_on_a_bounded_axis():
    with pytest.raises(ValueError, match="axis must be periodic"):
        ProblemSpec(
            _example_plane_model(),
            _plane_geometry(),
            _plane_numerics(hermitian_axis=2),
        )


def test_stage_a_does_not_change_runtime_or_geometry_specific_defaults():
    solver = SpectralSolver((4,), device="cpu")

    assert DEFAULT_SPECTRAL_STORAGE == "full_complex"
    assert solver.transform_backend.spectral_storage == "full_complex"
    assert DEFAULT_PLANE_SPECTRAL_STORAGE == "hermitian_half"
    assert PLANE_HERMITIAN_AXIS == 1
