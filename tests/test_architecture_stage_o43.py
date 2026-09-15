"""CPU contracts for Stage O.4.3 zero-copy projected batch assembly."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from pssolver.experimental import (
    BoundarySignatureTransformScheduler,
    ProjectedBatchAssemblyPolicy,
    ProjectedTransformDirection,
)


PROJECT_ROOT = Path(__file__).parents[1]


class _BatchContext:
    batch_size = 1
    physical_shape = (2, 3)
    spectral_shape = (2, 3)
    real_dtype = torch.float64
    spectral_dtype = torch.complex128
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.packed_inputs: list[torch.Tensor] = []

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
        self.packed_inputs.append(value)
        if direction is ProjectedTransformDirection.FORWARD:
            return torch.complex(value, torch.zeros_like(value))
        return value.real


def test_contiguous_component_views_form_one_zero_copy_transform_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    context = _BatchContext()
    scheduler = BoundarySignatureTransformScheduler(
        context,
        batch_assembly_policy=(
            ProjectedBatchAssemblyPolicy.contiguous_storage_view()
        ),
        enable_batch_assembly_diagnostics=True,
    )
    packed = torch.arange(18, dtype=torch.float64).reshape(3, 1, 2, 3)
    values = tuple(packed[index] for index in range(3))

    def forbidden_cat(*args, **kwargs):
        del args, kwargs
        raise AssertionError("the contiguous zero-copy path called torch.cat")

    monkeypatch.setattr(torch, "cat", forbidden_cat)
    transformed = scheduler.forward_values_many(("a", "b", "c"), values)

    assert len(context.packed_inputs) == 1
    batch = context.packed_inputs[0]
    assert batch.untyped_storage().data_ptr() == packed.untyped_storage().data_ptr()
    assert batch.storage_offset() == packed.storage_offset()
    assert batch.shape == (3, 2, 3)
    for actual, expected in zip(transformed, values, strict=True):
        assert torch.equal(actual.real, expected)
    diagnostics = scheduler.batch_assembly_diagnostics()
    assert diagnostics["contiguous_view_batches"] == 1
    assert diagnostics["contiguous_view_components"] == 3
    assert diagnostics["copy_cat_batches"] == 0
    assert diagnostics["fallback_reasons"] == {}
    assert diagnostics["retained_tensor_references"] == 0


def test_distinct_storages_fall_back_to_historical_copy_cat_path():
    context = _BatchContext()
    scheduler = BoundarySignatureTransformScheduler(
        context,
        batch_assembly_policy=(
            ProjectedBatchAssemblyPolicy.contiguous_storage_view()
        ),
        enable_batch_assembly_diagnostics=True,
    )
    values = tuple(
        torch.full((1, 2, 3), float(index), dtype=torch.float64)
        for index in range(3)
    )
    transformed = scheduler.forward_values_many(("a", "b", "c"), values)

    assert len(context.packed_inputs) == 1
    for actual, expected in zip(transformed, values, strict=True):
        assert torch.equal(actual.real, expected)
    diagnostics = scheduler.batch_assembly_diagnostics()
    assert diagnostics["contiguous_view_batches"] == 0
    assert diagnostics["copy_cat_batches"] == 1
    assert diagnostics["copy_cat_components"] == 3
    assert diagnostics["fallback_reasons"] == {"distinct_storage": 1}


def test_nonadjacent_views_do_not_span_unrequested_storage():
    context = _BatchContext()
    scheduler = BoundarySignatureTransformScheduler(
        context,
        batch_assembly_policy=(
            ProjectedBatchAssemblyPolicy.contiguous_storage_view()
        ),
        enable_batch_assembly_diagnostics=True,
    )
    packed = torch.arange(18, dtype=torch.float64).reshape(3, 1, 2, 3)
    values = (packed[0], packed[2])
    transformed = scheduler.forward_values_many(("a", "c"), values)

    assert torch.equal(transformed[0].real, values[0])
    assert torch.equal(transformed[1].real, values[1])
    diagnostics = scheduler.batch_assembly_diagnostics()
    assert diagnostics["contiguous_view_batches"] == 0
    assert diagnostics["fallback_reasons"] == {"non_adjacent_storage": 1}


def test_copy_cat_remains_the_default_and_explicit_fallback():
    context = _BatchContext()
    scheduler = BoundarySignatureTransformScheduler(
        context,
        enable_batch_assembly_diagnostics=True,
    )
    packed = torch.arange(12, dtype=torch.float64).reshape(2, 1, 2, 3)
    values = (packed[0], packed[1])
    transformed = scheduler.forward_values_many(("a", "b"), values)

    assert torch.equal(transformed[0].real, values[0])
    assert torch.equal(transformed[1].real, values[1])
    diagnostics = scheduler.batch_assembly_diagnostics()
    assert diagnostics["policy"]["mode"] == "copy_cat"
    assert diagnostics["copy_cat_batches"] == 1
    assert diagnostics["fallback_reasons"] == {"policy_copy_cat": 1}


def test_storage_order_is_owned_by_the_boundary_scheduler():
    class MixedBoundaryContext(_BatchContext):
        def boundary_conditions(self, name: str) -> tuple[str, ...]:
            return (
                ("periodic", "neumann")
                if name != "b"
                else ("periodic", "dirichlet")
            )

    scheduler = BoundarySignatureTransformScheduler(MixedBoundaryContext())
    order = scheduler.storage_compatible_order(
        ("a", "b", "c"),
        direction=ProjectedTransformDirection.INVERSE,
    )
    assert order == ("a", "c", "b")


def test_stage_o43_remains_outside_production_and_channel_paths():
    for relative in (
        "Plane_beris_edwards_stokes.py",
        "pssolver/solver.py",
        "pssolver/plane.py",
        "pssolver/channel.py",
    ):
        source = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert "AlgebraicOutputPublicationPolicy" not in source
        assert "ProjectedBatchAssemblyPolicy" not in source
