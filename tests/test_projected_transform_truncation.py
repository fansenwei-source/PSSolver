"""Equivalence tests for retained-mode DCT/DST transform execution."""

from __future__ import annotations

import pytest
import torch

from pssolver import SpectralSolver
from pssolver.transforms import (
    BasisAwareSpectralProjector,
    DEFAULT_PROJECTED_TRANSFORM_EXECUTION,
)


SHAPE = (7, 6, 5)
LENGTHS = (3.0, 4.0, 2.5)


def _solver(dtype=torch.float64, execution_order="real_first"):
    return SpectralSolver(
        SHAPE,
        L=LENGTHS,
        device="cpu",
        dtype=dtype,
        transform_execution_order=execution_order,
    )


def _projectors(
    dtype=torch.float64,
    rule="cubic_half",
    execution_order="real_first",
):
    solver = _solver(dtype, execution_order)
    return (
        solver,
        BasisAwareSpectralProjector(
            solver,
            rule=rule,
            transform_execution="full",
        ),
        BasisAwareSpectralProjector(
            solver,
            rule=rule,
            transform_execution="truncated",
        ),
    )


def _tolerances(dtype):
    if dtype == torch.float64:
        return {"rtol": 5.0e-13, "atol": 5.0e-13}
    return {"rtol": 3.0e-5, "atol": 3.0e-5}


def test_projected_transform_execution_defaults_to_truncated():
    solver = _solver()
    projector = BasisAwareSpectralProjector(solver, rule="cubic_half")

    assert DEFAULT_PROJECTED_TRANSFORM_EXECUTION == "truncated"
    assert projector.transform_execution == "truncated"
    assert projector.execution_metadata() == {
        "requested": "truncated",
        "effective": "truncated",
        "fallback_allowed": False,
        "fallback_reason": None,
        "truncated_real_basis_axes": True,
        "spectral_storage": "full_complex",
        "backend_storage_shape_preserved": True,
        "full_spectral_storage_preserved": True,
    }


def test_disabled_projector_defaults_to_compatible_full_execution():
    projector = BasisAwareSpectralProjector(_solver(), rule="none")

    assert projector.transform_execution == "full"
    assert projector.execution_metadata()["effective"] == "full"


def test_default_projector_matches_explicit_truncated_execution():
    solver = _solver()
    default = BasisAwareSpectralProjector(solver, rule="cubic_half")
    explicit = BasisAwareSpectralProjector(
        solver,
        rule="cubic_half",
        transform_execution="truncated",
    )
    values = torch.randn(2, *SHAPE, dtype=torch.float64)
    boundary_conditions = ("periodic", "periodic", "neumann")

    torch.testing.assert_close(
        default.forward_transform(values, boundary_conditions),
        explicit.forward_transform(values, boundary_conditions),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("rule", ("two_thirds", "cubic_half"))
@pytest.mark.parametrize("execution_order", ("legacy", "real_first"))
@pytest.mark.parametrize(
    "boundary_conditions",
    (
        ("periodic", "periodic", "neumann"),
        ("periodic", "periodic", "dirichlet"),
        ("neumann", "periodic", "dirichlet"),
        ("dirichlet", "neumann", "periodic"),
        ("neumann", "dirichlet", "neumann"),
    ),
)
def test_truncated_forward_and_inverse_match_full_projected_path(
    dtype,
    rule,
    execution_order,
    boundary_conditions,
):
    generator = torch.Generator().manual_seed(20260910)
    values = torch.randn(2, 3, *SHAPE, generator=generator, dtype=dtype)
    _, full, truncated = _projectors(dtype, rule, execution_order)

    expected_hat = full.forward_transform(values, boundary_conditions)
    observed_hat = truncated.forward_transform(values, boundary_conditions)

    assert observed_hat.shape == expected_hat.shape == (2, 3, *SHAPE)
    torch.testing.assert_close(
        observed_hat,
        expected_hat,
        **_tolerances(dtype),
    )
    torch.testing.assert_close(
        truncated.inverse_transform(observed_hat, boundary_conditions),
        full.inverse_transform(expected_hat, boundary_conditions),
        **_tolerances(dtype),
    )


@pytest.mark.parametrize("source_bc", ("neumann", "dirichlet"))
@pytest.mark.parametrize("axis", (0, 1, 2))
def test_truncated_projected_gradient_matches_full_path(source_bc, axis):
    boundary_conditions = ["periodic", "periodic", "periodic"]
    boundary_conditions[axis] = source_bc
    boundary_conditions = tuple(boundary_conditions)
    generator = torch.Generator().manual_seed(7110 + axis)
    values = torch.randn(2, *SHAPE, generator=generator, dtype=torch.float64)
    solver, full, truncated = _projectors()

    full_hat = full.forward_transform(values, boundary_conditions)
    truncated_hat = truncated.forward_transform(values, boundary_conditions)
    full_gradient_hat, gradient_bcs = solver.transform_backend.gradient_hat(
        full_hat,
        boundary_conditions,
        axis,
    )
    truncated_gradient_hat, truncated_gradient_bcs = (
        solver.transform_backend.gradient_hat(
            truncated_hat,
            boundary_conditions,
            axis,
        )
    )

    assert truncated_gradient_bcs == gradient_bcs
    torch.testing.assert_close(
        truncated.inverse_transform(
            truncated_gradient_hat,
            truncated_gradient_bcs,
        ),
        full.inverse_transform(full_gradient_hat, gradient_bcs),
        rtol=6.0e-13,
        atol=6.0e-13,
    )


def test_truncated_path_uses_retained_real_basis_matrix(monkeypatch):
    solver, _, truncated = _projectors(rule="cubic_half")
    backend = solver.transform_backend
    calls = []
    original = backend._apply_retained_real_axis_transform

    def recording_transform(
        tensor,
        kind,
        local_axis,
        retained_count,
        *,
        inverse,
    ):
        calls.append((kind, local_axis, retained_count, inverse))
        return original(
            tensor,
            kind,
            local_axis,
            retained_count,
            inverse=inverse,
        )

    monkeypatch.setattr(
        backend,
        "_apply_retained_real_axis_transform",
        recording_transform,
    )
    boundary_conditions = ("periodic", "periodic", "neumann")
    values = torch.randn(2, *SHAPE, dtype=torch.float64)
    spectral = truncated.forward_transform(values, boundary_conditions)
    truncated.inverse_transform(spectral, boundary_conditions)

    assert truncated.retained_axis_counts(boundary_conditions) == (3, 3, 3)
    assert truncated.computed_axis_sizes(boundary_conditions) == (7, 6, 3)
    assert calls == [
        ("dct", 2, 3, False),
        ("dct", 2, 3, True),
    ]


def test_truncated_storage_zeros_all_discarded_real_modes():
    _, _, truncated = _projectors(rule="cubic_half")
    boundary_conditions = ("periodic", "periodic", "dirichlet")
    values = torch.randn(2, *SHAPE, dtype=torch.float64)

    spectral = truncated.forward_transform(values, boundary_conditions)
    retained_z = truncated.retained_axis_counts(boundary_conditions)[2]

    assert spectral.shape == (2, *SHAPE)
    assert torch.count_nonzero(spectral[..., retained_z:]) == 0


@pytest.mark.parametrize(
    "options, message",
    (
        ({"transform_execution": "unknown"}, "transform_execution"),
        (
            {"rule": "none", "transform_execution": "truncated"},
            "enabled dealiasing",
        ),
    ),
)
def test_projector_rejects_invalid_truncated_configuration(options, message):
    with pytest.raises(ValueError, match=message):
        BasisAwareSpectralProjector(_solver(), **options)


def test_backend_rejects_truncating_a_periodic_fft_axis():
    solver = _solver()
    values = torch.randn(2, *SHAPE, dtype=torch.float64)

    with pytest.raises(ValueError, match="Periodic FFT axes"):
        solver.transform_backend.forward(
            values,
            ("periodic", "periodic", "neumann"),
            retained_axis_counts=(3, 6, 3),
        )
