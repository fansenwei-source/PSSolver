"""Regression tests for bounded-transform-local data movement paths."""

from __future__ import annotations

import pytest
import torch

from pssolver import BasisAwareSpectralProjector, SpectralSolver


SHAPE = (8, 7, 6)
LENGTHS = (4.0, 3.5, 3.0)
BOUNDARY_CONDITIONS = ("periodic", "periodic", "neumann")


def _solver(*, indexing="contiguous_slice"):
    return SpectralSolver(
        SHAPE,
        L=LENGTHS,
        device="cpu",
        dtype=torch.float64,
        spectral_storage="hermitian_half",
        hermitian_axis=1,
        transform_execution_order="real_first",
        transform_group_indexing=indexing,
    )


def _built_dynamic_solver(*, indexing):
    solver = _solver(indexing=indexing)
    generator = torch.Generator().manual_seed(20260918)
    for index in range(2):
        initial = torch.randn(
            SHAPE,
            generator=generator,
            dtype=torch.float64,
        )
        solver.model.add_dynamic_field(
            f"q{index}",
            initial,
            -0.1 * solver.q2_raw,
            boundary_conditions=BOUNDARY_CONDITIONS,
        )
    solver.build()
    return solver


def test_periodic_gradient_factors_are_cached_without_mutating_modes():
    solver = _solver()
    backend = solver.transform_backend
    generator = torch.Generator().manual_seed(1701)
    values = torch.randn(
        2,
        *SHAPE,
        generator=generator,
        dtype=torch.float64,
    )
    spectral = backend.forward(values, BOUNDARY_CONDITIONS)
    modes_before = backend.get_metadata(
        BOUNDARY_CONDITIONS
    ).axis_modes[0].clone()

    first, first_bcs = backend.gradient_hat(
        spectral,
        BOUNDARY_CONDITIONS,
        axis=0,
    )
    cached = tuple(backend._periodic_gradient_factor_cache.values())
    second, second_bcs = backend.gradient_hat(
        spectral,
        BOUNDARY_CONDITIONS,
        axis=0,
    )

    assert first_bcs == second_bcs == BOUNDARY_CONDITIONS
    assert torch.equal(first, second)
    assert len(cached) == len(backend._periodic_gradient_factor_cache) == 1
    assert cached[0] is tuple(backend._periodic_gradient_factor_cache.values())[0]
    assert torch.equal(
        backend.get_metadata(BOUNDARY_CONDITIONS).axis_modes[0],
        modes_before,
    )


@pytest.mark.parametrize("bounded_bc", ("dirichlet", "neumann"))
def test_bounded_gradient_direct_write_is_bitwise_equivalent(bounded_bc):
    solver = _solver()
    backend = solver.transform_backend
    boundary_conditions = ("periodic", "periodic", bounded_bc)
    generator = torch.Generator().manual_seed(1705)
    values = torch.randn(
        2,
        *SHAPE,
        generator=generator,
        dtype=torch.float64,
    )
    spectral = backend.forward(values, boundary_conditions)
    moved = spectral.movedim(-1, -1)
    modes = backend.get_metadata(boundary_conditions).axis_modes[2]
    expected = torch.zeros_like(moved)
    if bounded_bc == "dirichlet":
        expected[..., 1:] = moved[..., :-1] * modes[:-1]
    else:
        expected[..., :-1] = -moved[..., 1:] * modes[1:]

    observed, observed_bcs = backend.gradient_hat(
        spectral,
        boundary_conditions,
        axis=2,
    )

    assert observed_bcs == backend.get_gradient_boundary_conditions(
        boundary_conditions,
        2,
    )
    assert torch.equal(observed, expected)


def test_bounded_gradient_keeps_autograd_compatible_path():
    solver = _solver()
    backend = solver.transform_backend
    boundary_conditions = ("periodic", "periodic", "neumann")
    values = torch.randn(
        2,
        *SHAPE,
        dtype=torch.float64,
        requires_grad=True,
    )
    spectral = backend.forward(values, boundary_conditions)

    derivative, _ = backend.gradient_hat(
        spectral,
        boundary_conditions,
        axis=2,
    )
    derivative.abs().square().sum().backward()

    assert values.grad is not None
    assert torch.isfinite(values.grad).all()


def test_in_place_projection_matches_allocating_projection_and_reuses_storage():
    solver = _solver()
    projector = BasisAwareSpectralProjector(
        solver,
        rule="cubic_half",
        transform_execution="full",
    )
    generator = torch.Generator().manual_seed(1702)
    values = torch.randn(
        2,
        *SHAPE,
        generator=generator,
        dtype=torch.float64,
    )
    spectral = solver.transform_backend.forward(
        values,
        BOUNDARY_CONDITIONS,
    )
    expected = projector.project(spectral, BOUNDARY_CONDITIONS)
    observed = spectral.clone()
    data_ptr = observed.untyped_storage().data_ptr()

    returned = projector.project_(observed, BOUNDARY_CONDITIONS)

    assert returned is observed
    assert returned.untyped_storage().data_ptr() == data_ptr
    assert torch.equal(observed, expected)


@pytest.mark.parametrize(
    "indexing,expected_store_calls",
    (("contiguous_slice", 0), ("advanced", 1)),
)
def test_dynamic_projection_updates_owned_storage_without_extra_contiguous_store(
    monkeypatch,
    indexing,
    expected_store_calls,
):
    solver = _built_dynamic_solver(indexing=indexing)
    fields = solver.fields
    projector = BasisAwareSpectralProjector(
        solver,
        rule="cubic_half",
        transform_execution="full",
    )
    initial = fields.spectral.clone()
    expected = projector.project(initial, BOUNDARY_CONDITIONS)
    store_calls = []
    original_store = fields.store_spectral_group

    def recording_store(group, values):
        store_calls.append(tuple(group))
        original_store(group, values)

    monkeypatch.setattr(fields, "store_spectral_group", recording_store)

    projector.project_dynamic_fields(fields, sync_spatial=False)

    assert store_calls == [(0, 1)] * expected_store_calls
    assert torch.equal(fields.spectral, expected)


def test_in_place_projection_preserves_disabled_projector_contract():
    projector = BasisAwareSpectralProjector(_solver(), rule="none")
    values = torch.randn(2, *SHAPE, dtype=torch.complex128)
    original = values.clone()

    returned = projector.project_(values, BOUNDARY_CONDITIONS)

    assert returned is values
    assert torch.equal(values, original)
