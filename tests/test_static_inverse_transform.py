"""Tests for the optional projected static-field inverse hook."""

from __future__ import annotations

import pytest
import torch

from pssolver import SpectralSolver


BOUNDARY_CONDITIONS = ("periodic", "periodic", "neumann")


class _ZeroNonlinear(torch.nn.Module):
    def forward(self, fields, parameters):
        del parameters
        return torch.zeros_like(fields.spectral[: fields.dyn_count])


class _StaticMultiple(torch.nn.Module):
    def forward(self, fields, parameters):
        del parameters
        return fields.transform_tensor(
            (2.0 * fields["q"]).unsqueeze(0),
            BOUNDARY_CONDITIONS,
        )


def _solver():
    shape = (6, 5, 4)
    solver = SpectralSolver(
        shape,
        L=(3.0, 2.5, 2.0),
        device="cpu",
        dtype=torch.float64,
    )
    initial = torch.randn(
        shape,
        generator=torch.Generator().manual_seed(71),
        dtype=torch.float64,
    )
    solver.model.add_dynamic_field(
        "q",
        init=initial,
        L_hat=torch.zeros_like(solver.get_q2(BOUNDARY_CONDITIONS)),
        boundary_conditions=BOUNDARY_CONDITIONS,
    )
    solver.model.add_static_field(
        "u",
        boundary_conditions=BOUNDARY_CONDITIONS,
    )
    solver.model.set_nonlinear_model(_ZeroNonlinear())
    solver.model.set_static_compute_model(_StaticMultiple())
    solver.model.build()
    return solver


def test_unrelated_spectral_projector_does_not_activate_static_inverse_hook():
    solver = _solver()
    solver.model.spectral_projector = object()

    solver.model.update_static_fields()

    torch.testing.assert_close(
        solver.model.fields["u"],
        2.0 * solver.model.fields["q"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_explicit_static_inverse_hook_is_used():
    solver = _solver()
    calls = []

    def inverse_transform(spectral, boundary_conditions):
        calls.append((spectral.shape, tuple(boundary_conditions)))
        return solver.transform_backend.inverse(
            spectral,
            boundary_conditions,
        )

    solver.model.set_static_inverse_transform(inverse_transform)
    solver.model.update_static_fields()

    assert calls == [((1, 1, 6, 5, 4), BOUNDARY_CONDITIONS)]
    torch.testing.assert_close(
        solver.model.fields["u"],
        2.0 * solver.model.fields["q"],
        rtol=1.0e-12,
        atol=1.0e-12,
    )


def test_static_inverse_hook_rejects_noncallable():
    solver = _solver()

    with pytest.raises(TypeError, match="callable or None"):
        solver.model.set_static_inverse_transform("not callable")
