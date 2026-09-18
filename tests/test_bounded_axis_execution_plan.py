"""Contracts for device-bound dense DCT/DST execution plans."""

from __future__ import annotations

import math

import pytest
import torch

from pssolver.backends.bounded import (
    BoundedAxisPlanKey,
    DenseBoundedAxisExecutionPlan,
    build_dense_orthonormal_matrix,
)
from pssolver.transforms import TensorProductTransformBackend


def _frozen_v011_matrix(kind, size, *, dtype=torch.float64):
    n = torch.arange(size, dtype=dtype)
    k = n.unsqueeze(1)
    phase = math.pi * (n + 0.5) / size
    if kind == "dct":
        matrix = torch.cos(k * phase)
        matrix[0] *= math.sqrt(1.0 / size)
        if size > 1:
            matrix[1:] *= math.sqrt(2.0 / size)
        return matrix
    matrix = math.sqrt(2.0 / size) * torch.sin((k + 1.0) * phase)
    matrix[-1] *= math.sqrt(0.5)
    return matrix


@pytest.mark.parametrize("kind", ("dct", "dst"))
@pytest.mark.parametrize("size", (3, 4, 7, 8))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_dense_matrix_builder_is_bitwise_identical_to_v011(kind, size, dtype):
    expected = _frozen_v011_matrix(kind, size, dtype=dtype)
    observed = build_dense_orthonormal_matrix(
        kind,
        size,
        device="cpu",
        dtype=dtype,
    )

    assert torch.equal(observed, expected)


@pytest.mark.parametrize("kind", ("dct", "dst"))
@pytest.mark.parametrize("size,retained", ((3, 1), (7, 3), (8, 8)))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("value_type", ("real", "complex"))
def test_dense_plan_matches_frozen_matmul_order(
    kind,
    size,
    retained,
    dtype,
    value_type,
):
    generator = torch.Generator().manual_seed(20260917 + size + retained)
    physical = torch.randn(2, 3, size, generator=generator, dtype=dtype)
    coefficients = torch.randn(
        2,
        3,
        retained,
        generator=generator,
        dtype=dtype,
    )
    if value_type == "complex":
        physical = torch.complex(
            physical,
            torch.randn(2, 3, size, generator=generator, dtype=dtype),
        )
        coefficients = torch.complex(
            coefficients,
            torch.randn(
                2,
                3,
                retained,
                generator=generator,
                dtype=dtype,
            ),
        )

    full_matrix = _frozen_v011_matrix(kind, size, dtype=dtype)
    matrix = full_matrix[:retained]
    key = BoundedAxisPlanKey(
        kind=kind,
        physical_size=size,
        retained_count=retained,
        device=torch.device("cpu"),
        real_dtype=dtype,
        value_type=value_type,
    )
    plan = DenseBoundedAxisExecutionPlan(key=key, matrix=matrix)
    expected_forward = physical @ matrix.to(physical.dtype).transpose(-1, -2)
    expected_inverse = coefficients @ matrix.to(coefficients.dtype)

    assert torch.equal(plan.forward_last_axis(physical), expected_forward)
    assert torch.equal(plan.inverse_last_axis(coefficients), expected_inverse)


def test_backend_reuses_static_plan_and_frozen_matrix_storage():
    backend = TensorProductTransformBackend(
        (7, 6, 5),
        (3.0, 4.0, 2.5),
        device="cpu",
        dtype=torch.float64,
    )
    values = torch.randn(2, 7, 6, 5, dtype=torch.float64)

    first = backend._get_bounded_axis_execution_plan(
        values,
        "dct",
        5,
        3,
    )
    second = backend._get_bounded_axis_execution_plan(
        values,
        "dct",
        5,
        3,
    )

    assert first is second
    assert len(backend._bounded_axis_plan_cache) == 1
    assert len(backend._matrix_cache) == 1
    assert first.matrix.untyped_storage().data_ptr() == (
        backend._get_matrix("dct", 5).untyped_storage().data_ptr()
    )

    complex_values = torch.complex(values, torch.zeros_like(values))
    complex_plan = backend._get_bounded_axis_execution_plan(
        complex_values,
        "dct",
        5,
        3,
    )
    full_plan = backend._get_bounded_axis_execution_plan(
        values,
        "dct",
        5,
        5,
    )

    assert complex_plan is not first
    assert full_plan is not first
    assert len(backend._bounded_axis_plan_cache) == 3
    assert len(backend._matrix_cache) == 1


def test_plan_cache_hit_does_not_repeat_matrix_lookup(monkeypatch):
    backend = TensorProductTransformBackend(
        (7, 6, 5),
        (3.0, 4.0, 2.5),
        device="cpu",
        dtype=torch.float64,
    )
    values = torch.randn(2, 7, 6, 5, dtype=torch.float64)
    expected = backend._get_bounded_axis_execution_plan(
        values,
        "dct",
        5,
        3,
    )

    def fail_matrix_lookup(*_args, **_kwargs):
        raise AssertionError("a plan cache hit must not look up its matrix")

    monkeypatch.setattr(backend, "_get_matrix", fail_matrix_lookup)

    assert (
        backend._get_bounded_axis_execution_plan(values, "dct", 5, 3)
        is expected
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_cuda_plan_key_uses_the_allocated_device_identity():
    backend = TensorProductTransformBackend(
        (7, 6, 5),
        (3.0, 4.0, 2.5),
        device="cuda",
        dtype=torch.float64,
    )
    values = torch.randn(2, 7, 6, 5, device="cuda", dtype=torch.float64)

    plan = backend._get_bounded_axis_execution_plan(
        values,
        "dst",
        5,
        3,
    )

    assert plan.key.device == plan.matrix.device == values.device


@pytest.mark.parametrize(
    "changes,exception,message",
    (
        ({"kind": "fft"}, ValueError, "kind"),
        ({"physical_size": 0}, ValueError, "physical_size"),
        ({"retained_count": -1}, ValueError, "retained_count"),
        ({"retained_count": 9}, ValueError, "retained_count"),
        ({"device": "cpu"}, TypeError, "device"),
        ({"real_dtype": torch.int64}, ValueError, "real_dtype"),
        ({"value_type": "quaternion"}, ValueError, "value_type"),
    ),
)
def test_plan_key_rejects_invalid_static_facts(changes, exception, message):
    options = {
        "kind": "dct",
        "physical_size": 8,
        "retained_count": 4,
        "device": torch.device("cpu"),
        "real_dtype": torch.float64,
        "value_type": "real",
    }
    options.update(changes)

    with pytest.raises(exception, match=message):
        BoundedAxisPlanKey(**options)


def test_dense_plan_rejects_matrix_identity_mismatch():
    key = BoundedAxisPlanKey(
        kind="dst",
        physical_size=8,
        retained_count=4,
        device=torch.device("cpu"),
        real_dtype=torch.float64,
        value_type="real",
    )

    with pytest.raises(ValueError, match="matrix shape"):
        DenseBoundedAxisExecutionPlan(
            key=key,
            matrix=torch.empty(5, 8, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="matrix dtype"):
        DenseBoundedAxisExecutionPlan(
            key=key,
            matrix=torch.empty(4, 8, dtype=torch.float32),
        )
