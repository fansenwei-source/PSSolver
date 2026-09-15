"""CPU contracts for Stage O.4.3.2 producer-owned packed values."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

import pssolver
from pssolver.experimental import (
    BoundarySignatureTransformScheduler,
    ProducerPackedComponentValues,
    ProjectedBatchAssemblyPolicy,
    ProjectedTransformDirection,
)


PROJECT_ROOT = Path(__file__).parents[1]


class _TransformContext:
    batch_size = 2
    physical_shape = (3, 4)
    spectral_shape = (3, 4)
    real_dtype = torch.float64
    spectral_dtype = torch.complex128
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.received: list[torch.Tensor] = []

    def boundary_conditions(self, name: str) -> tuple[str, ...]:
        del name
        return ("periodic", "neumann")

    def forward_projected(
        self,
        name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        del name
        return torch.complex(value, torch.zeros_like(value))

    def inverse_projected(
        self,
        name: str,
        value: torch.Tensor,
    ) -> torch.Tensor:
        del name
        return value.real

    def transform_projected_packed(
        self,
        boundaries: tuple[str, ...],
        value: torch.Tensor,
        *,
        direction: ProjectedTransformDirection,
    ) -> torch.Tensor:
        assert boundaries == ("periodic", "neumann")
        self.received.append(value)
        if direction is ProjectedTransformDirection.FORWARD:
            return torch.complex(value, torch.zeros_like(value))
        return value.real


def test_producer_packed_values_are_read_only_zero_copy_component_views():
    packed = torch.arange(72, dtype=torch.float64).reshape(3, 2, 3, 4)
    values = ProducerPackedComponentValues(("a", "b", "c"), packed)

    assert tuple(values) == ("a", "b", "c")
    assert values.packed is packed
    assert values.batch_size == 2
    assert values.component_shape == (2, 3, 4)
    for index, name in enumerate(values):
        component = values[name]
        assert torch.equal(component, packed[index])
        assert component.untyped_storage().data_ptr() == (
            packed.untyped_storage().data_ptr()
        )
    metadata = values.to_metadata()
    assert metadata["ownership"] == "producer"
    assert metadata["copies_on_construction"] == 0
    assert metadata["owns_packed_tensor"] is True
    assert metadata["retained_packed_tensor_references"] == 1
    assert metadata["retained_component_view_references"] == 0


def test_producer_packed_values_supply_one_direct_transform_view():
    packed = torch.arange(72, dtype=torch.float64).reshape(3, 2, 3, 4)
    values = ProducerPackedComponentValues(("a", "b", "c"), packed)
    context = _TransformContext()
    scheduler = BoundarySignatureTransformScheduler(
        context,
        batch_assembly_policy=(
            ProjectedBatchAssemblyPolicy.contiguous_storage_view()
        ),
        enable_batch_assembly_diagnostics=True,
    )

    transformed = scheduler.forward_values_many(
        values.component_names,
        tuple(values.values()),
    )

    assert len(context.received) == 1
    assert context.received[0].untyped_storage().data_ptr() == (
        packed.untyped_storage().data_ptr()
    )
    assert context.received[0].shape == (6, 3, 4)
    diagnostics = scheduler.batch_assembly_diagnostics()
    assert diagnostics["contiguous_view_batches"] == 1
    assert diagnostics["copy_cat_batches"] == 0
    for actual, expected in zip(
        transformed,
        values.values(),
        strict=True,
    ):
        assert torch.equal(actual.real, expected)


def test_contiguous_transform_view_rejects_nonadjacent_component_requests():
    packed = torch.zeros((3, 2, 3, 4), dtype=torch.float64)
    values = ProducerPackedComponentValues(("a", "b", "c"), packed)

    direct = values.contiguous_transform_view(("a", "b"))
    assert direct.shape == (4, 3, 4)
    assert direct.untyped_storage().data_ptr() == (
        packed.untyped_storage().data_ptr()
    )
    with pytest.raises(ValueError, match="not adjacent"):
        values.contiguous_transform_view(("a", "c"))


@pytest.mark.parametrize(
    ("names", "packed", "error"),
    (
        (("a", "a"), torch.zeros((2, 1, 2)), "unique identifiers"),
        (("not-valid",), torch.zeros((1, 1, 2)), "unique identifiers"),
        (("a", "b"), torch.zeros((1, 1, 2)), "must have shape"),
        (("a",), torch.zeros((1, 0, 2)), "must be nonempty"),
        (("a",), torch.zeros((1, 2, 3)).transpose(1, 2), "contiguous"),
    ),
)
def test_producer_packed_values_fail_closed_on_invalid_storage(
    names,
    packed,
    error,
):
    with pytest.raises(ValueError, match=error):
        ProducerPackedComponentValues(names, packed)


def test_stage_o432_contract_remains_outside_production_paths():
    assert not hasattr(pssolver, "ProducerPackedComponentValues")
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "ProducerPackedComponentValues" not in source
