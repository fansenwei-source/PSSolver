"""Characterization tests for Stage B geometries and legacy adaptation."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, dataclass

import pytest

from pssolver.Field import Fields
from pssolver.adapters import (
    boundary_condition_to_legacy,
    boundary_set_to_legacy,
)
from pssolver.core import (
    AxisTopology,
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
from pssolver.geometries import (
    PeriodicBox,
    PlaneSlab,
    RectangularChannel,
)


def _domain(shape=(12, 10, 8)):
    return DomainSpec(shape, tuple(float(size) for size in shape))


def _reference_numerics():
    return NumericsConfig(
        precision=Precision.FLOAT64,
        dealias_rule=DealiasRule.CUBIC_HALF,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=ProjectedTransformExecution.FULL,
        spectral_storage=SpectralStorage.FULL_COMPLEX,
    )


@dataclass(frozen=True)
class ScalarModel:
    boundaries: BoundarySet
    name: str = "scalar_model"

    def field_specs(self):
        return (
            FieldSpec.scalar("phi", FieldRole.EVOLVED, self.boundaries),
        )

    def parameter_metadata(self):
        return {"coefficient": 1.0}


def test_periodic_box_marks_every_axis_periodic():
    geometry = PeriodicBox(_domain())

    assert geometry.name == "periodic_box"
    assert geometry.axis_topologies == (AxisTopology.PERIODIC,) * 3
    assert geometry.periodic_axes == (0, 1, 2)
    assert geometry.bounded_axes == ()


def test_plane_slab_defaults_to_last_wall_normal_axis():
    geometry = PlaneSlab(_domain())

    assert geometry.name == "plane_slab"
    assert geometry.axis_topologies == (
        AxisTopology.PERIODIC,
        AxisTopology.PERIODIC,
        AxisTopology.BOUNDED,
    )
    assert geometry.periodic_axes == (0, 1)
    assert geometry.bounded_axes == (2,)


def test_plane_slab_allows_an_explicit_wall_normal_axis():
    geometry = PlaneSlab(_domain(), wall_normal_axis=0)

    assert geometry.axis_topologies == (
        AxisTopology.BOUNDED,
        AxisTopology.PERIODIC,
        AxisTopology.PERIODIC,
    )


def test_rectangular_channel_defaults_to_first_streamwise_axis():
    geometry = RectangularChannel(_domain())

    assert geometry.name == "rectangular_channel"
    assert geometry.axis_topologies == (
        AxisTopology.PERIODIC,
        AxisTopology.BOUNDED,
        AxisTopology.BOUNDED,
    )
    assert geometry.periodic_axes == (0,)
    assert geometry.bounded_axes == (1, 2)


def test_rectangular_channel_allows_an_explicit_streamwise_axis():
    geometry = RectangularChannel(_domain(), streamwise_axis=2)

    assert geometry.axis_topologies == (
        AxisTopology.BOUNDED,
        AxisTopology.BOUNDED,
        AxisTopology.PERIODIC,
    )


@pytest.mark.parametrize("geometry_type", (PlaneSlab, RectangularChannel))
def test_wall_bounded_geometries_require_at_least_two_dimensions(geometry_type):
    with pytest.raises(ValueError, match="at least two dimensions"):
        geometry_type(_domain((8,)))


@pytest.mark.parametrize(
    "geometry_type, keyword",
    (
        (PlaneSlab, "wall_normal_axis"),
        (RectangularChannel, "streamwise_axis"),
    ),
)
@pytest.mark.parametrize("axis", (True, -1, 3))
def test_geometry_axis_selection_is_explicit_and_validated(
    geometry_type,
    keyword,
    axis,
):
    error = TypeError if axis is True else ValueError
    with pytest.raises(error):
        geometry_type(_domain(), **{keyword: axis})


def test_concrete_geometries_remain_immutable_geometry_specs():
    geometry = PlaneSlab(_domain())

    with pytest.raises(FrozenInstanceError):
        geometry.name = "changed"


@pytest.mark.parametrize(
    "condition, expected",
    (
        (PeriodicBC(), "periodic"),
        (HomogeneousDirichletBC(), "dirichlet"),
        (HomogeneousNeumannBC(), "neumann"),
    ),
)
def test_boundary_adapter_maps_physical_conditions_to_legacy_labels(
    condition,
    expected,
):
    assert boundary_condition_to_legacy(condition) == expected


def test_boundary_set_adapter_preserves_axis_order():
    boundaries = BoundarySet(
        (
            PeriodicBC(),
            HomogeneousNeumannBC(),
            HomogeneousDirichletBC(),
        )
    )

    assert boundary_set_to_legacy(boundaries) == (
        "periodic",
        "neumann",
        "dirichlet",
    )


def test_adapter_output_is_accepted_by_the_current_fields_interface():
    boundaries = BoundarySet(
        (PeriodicBC(), PeriodicBC(), HomogeneousNeumannBC())
    )
    legacy = boundary_set_to_legacy(boundaries)
    fields = Fields((4, 4, 4), device="cpu")

    assert fields._normalize_boundary_conditions(legacy) == legacy


@pytest.mark.parametrize(
    "function, value",
    (
        (boundary_condition_to_legacy, "neumann"),
        (boundary_set_to_legacy, ("periodic", "neumann")),
    ),
)
def test_adapter_rejects_legacy_or_untyped_inputs(function, value):
    with pytest.raises(TypeError):
        function(value)


@pytest.mark.parametrize(
    "normal_condition",
    (HomogeneousDirichletBC(), HomogeneousNeumannBC()),
)
def test_plane_geometry_does_not_choose_the_physical_wall_condition(
    normal_condition,
):
    boundaries = BoundarySet(
        (PeriodicBC(), PeriodicBC(), normal_condition)
    )

    problem = ProblemSpec(
        ScalarModel(boundaries),
        PlaneSlab(_domain()),
        _reference_numerics(),
    )

    assert problem.geometry.bounded_axes == (2,)
    assert problem.field_specs[0].components[0].boundaries == boundaries


def test_channel_topology_composes_with_current_periodic_neumann_layout():
    boundaries = BoundarySet(
        (
            PeriodicBC(),
            HomogeneousNeumannBC(),
            HomogeneousNeumannBC(),
        )
    )

    problem = ProblemSpec(
        ScalarModel(boundaries),
        RectangularChannel(_domain()),
        _reference_numerics(),
    )

    assert boundary_set_to_legacy(
        problem.field_specs[0].components[0].boundaries
    ) == ("periodic", "neumann", "neumann")


def test_core_contracts_do_not_expose_legacy_conversion_methods():
    condition = HomogeneousNeumannBC()
    boundaries = BoundarySet((condition,))

    assert not hasattr(condition, "legacy_label")
    assert not hasattr(boundaries, "legacy_labels")
