"""Regression tests for opt-in periodic transform fast paths."""

from __future__ import annotations

import pytest
import torch

from pssolver import SpectralSolver
from pssolver.Field import Fields
from pssolver.transforms import TensorProductTransformBackend


def _backend(
    shape,
    dtype,
    periodic_transform_execution,
    *,
    execution_order="real_first",
):
    return TensorProductTransformBackend(
        shape,
        tuple(float(size) for size in shape),
        device="cpu",
        dtype=dtype,
        execution_order=execution_order,
        periodic_transform_execution=periodic_transform_execution,
    )


def _tolerances(dtype):
    if dtype == torch.float64:
        return {"rtol": 8.0e-12, "atol": 8.0e-12}
    return {"rtol": 4.0e-5, "atol": 4.0e-5}


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("shape", ((9,), (8, 7), (7, 6, 5)))
def test_multidim_periodic_transform_matches_axiswise(dtype, shape):
    generator = torch.Generator().manual_seed(1701 + len(shape))
    values = torch.randn(2, 3, *shape, generator=generator, dtype=dtype)
    boundary_conditions = ("periodic",) * len(shape)
    axiswise = _backend(shape, dtype, "axiswise")
    multidim = _backend(shape, dtype, "multidim")

    axiswise_hat = axiswise.forward(values, boundary_conditions)
    multidim_hat = multidim.forward(values, boundary_conditions)

    torch.testing.assert_close(
        multidim_hat,
        axiswise_hat,
        **_tolerances(dtype),
    )
    torch.testing.assert_close(
        multidim.inverse(multidim_hat, boundary_conditions),
        values,
        **_tolerances(dtype),
    )


@pytest.mark.parametrize(
    "boundary_conditions",
    (
        ("periodic", "periodic", "neumann"),
        ("periodic", "dirichlet", "periodic"),
    ),
)
def test_multidim_periodic_transform_matches_axiswise_for_mixed_real_first(
    boundary_conditions,
):
    shape = (7, 6, 5)
    generator = torch.Generator().manual_seed(1702)
    values = torch.randn(2, *shape, generator=generator, dtype=torch.float64)
    axiswise = _backend(shape, torch.float64, "axiswise")
    multidim = _backend(shape, torch.float64, "multidim")

    expected = axiswise.forward(values, boundary_conditions)
    observed = multidim.forward(values, boundary_conditions)
    torch.testing.assert_close(observed, expected, rtol=8.0e-12, atol=8.0e-12)
    torch.testing.assert_close(
        multidim.inverse(observed, boundary_conditions),
        values,
        rtol=8.0e-12,
        atol=8.0e-12,
    )


def test_mixed_legacy_multidim_request_falls_back_to_axiswise():
    shape = (7, 6, 5)
    boundary_conditions = ("periodic", "neumann", "periodic")
    backend = _backend(
        shape,
        torch.float64,
        "multidim",
        execution_order="legacy",
    )
    metadata = backend.periodic_transform_execution_metadata(
        boundary_conditions
    )

    assert metadata == {
        "requested": "multidim",
        "effective": "axiswise",
        "fallback_reason": "mixed_legacy_execution_order",
    }


def test_multidim_periodic_transform_preserves_autograd():
    shape = (6, 5, 4)
    boundary_conditions = ("periodic",) * 3
    generator = torch.Generator().manual_seed(1703)
    initial = torch.randn(*shape, generator=generator, dtype=torch.float64)
    axiswise_input = initial.clone().requires_grad_(True)
    multidim_input = initial.clone().requires_grad_(True)
    axiswise = _backend(shape, torch.float64, "axiswise")
    multidim = _backend(shape, torch.float64, "multidim")

    axiswise_hat = axiswise.forward(axiswise_input, boundary_conditions)
    multidim_hat = multidim.forward(multidim_input, boundary_conditions)
    axiswise_hat.abs().square().sum().backward()
    multidim_hat.abs().square().sum().backward()

    torch.testing.assert_close(
        multidim_input.grad,
        axiswise_input.grad,
        rtol=8.0e-12,
        atol=8.0e-12,
    )


def _fields(indexing):
    fields = Fields(
        (4, 3),
        device="cpu",
        dtype=torch.float64,
        transform_group_indexing=indexing,
    )
    fields.boundary_conditions = [
        ("periodic", "periodic"),
        ("periodic", "periodic"),
        ("neumann", "periodic"),
        ("periodic", "periodic"),
    ]
    fields.spatial = torch.arange(
        4 * 2 * 4 * 3,
        dtype=torch.float64,
    ).reshape(4, 2, 4, 3)
    fields.spectral = torch.complex(fields.spatial.clone(), fields.spatial + 1.0)
    return fields


def test_contiguous_transform_group_uses_a_storage_view():
    fields = _fields("contiguous_slice")

    selected = fields.select_spatial_group([0, 1])

    assert selected.untyped_storage().data_ptr() == (
        fields.spatial.untyped_storage().data_ptr()
    )
    assert fields.transform_group_indexing_metadata([0, 1]) == {
        "requested": "contiguous_slice",
        "effective": "contiguous_slice",
        "fallback_reason": None,
        "indices": [0, 1],
    }


def test_noncontiguous_transform_group_falls_back_to_advanced_indexing():
    fields = _fields("contiguous_slice")

    selected = fields.select_spatial_group([0, 3])

    assert selected.untyped_storage().data_ptr() != (
        fields.spatial.untyped_storage().data_ptr()
    )
    assert fields.transform_group_indexing_metadata([0, 3]) == {
        "requested": "contiguous_slice",
        "effective": "advanced",
        "fallback_reason": "noncontiguous_field_indices",
        "indices": [0, 3],
    }


@pytest.mark.parametrize("indexing", ("advanced", "contiguous_slice"))
def test_transform_group_store_updates_original_tensor(indexing):
    fields = _fields(indexing)
    spatial = torch.full_like(fields.spatial[:2], 17.0)
    spectral = torch.full_like(fields.spectral[:2], 4.0 + 5.0j)

    fields.store_spatial_group([0, 1], spatial)
    fields.store_spectral_group([0, 1], spectral)

    assert torch.equal(fields.spatial[:2], spatial)
    assert torch.equal(fields.spectral[:2], spectral)


class _ZeroNonlinear(torch.nn.Module):
    def forward(self, fields, parameters):
        del parameters
        return torch.zeros_like(fields.spectral[: fields.dyn_count])


def _diffusion_solver(periodic_execution, group_indexing):
    shape = (8, 7)
    solver = SpectralSolver(
        shape,
        L=(4.0, 3.5),
        dt=0.01,
        device="cpu",
        dtype=torch.float64,
        periodic_transform_execution=periodic_execution,
        transform_group_indexing=group_indexing,
    )
    generator = torch.Generator().manual_seed(1704)
    for index in range(3):
        initial = torch.randn(
            shape,
            generator=generator,
            dtype=torch.float64,
        )
        solver.model.add_dynamic_field(
            f"q{index}",
            initial,
            -0.1 * solver.q2_raw,
        )
    solver.model.set_nonlinear_model(_ZeroNonlinear())
    solver.build()
    return solver


def test_combined_fast_paths_preserve_complete_timesteps():
    baseline = _diffusion_solver("axiswise", "advanced")
    candidate = _diffusion_solver("multidim", "contiguous_slice")

    baseline.run(4)
    candidate.run(4)

    torch.testing.assert_close(
        candidate.fields.spectral,
        baseline.fields.spectral,
        rtol=8.0e-12,
        atol=8.0e-12,
    )
    torch.testing.assert_close(
        candidate.fields.spatial,
        baseline.fields.spatial,
        rtol=8.0e-12,
        atol=8.0e-12,
    )


def test_fast_path_selectors_validate_and_default_to_legacy_behavior():
    solver = SpectralSolver((4, 3), device="cpu", dtype=torch.float64)

    assert solver.optimization_metadata() == {
        "periodic_transform_execution": {
            "requested": "axiswise",
            "effective": "axiswise",
            "fallback_reason": None,
        },
        "transform_group_indexing": {"requested": "advanced"},
    }
    with pytest.raises(ValueError, match="periodic_transform_execution"):
        _backend((4, 3), torch.float64, "unknown")
    with pytest.raises(ValueError, match="transform_group_indexing"):
        Fields((4, 3), device="cpu", transform_group_indexing="unknown")
