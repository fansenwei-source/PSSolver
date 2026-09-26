"""P8.3 manufactured force-projection tests for two bounded axes."""

from __future__ import annotations

import math

import torch

from pssolver.models.active_nematics import (
    CHANNEL_DISTORTION_STRESS_BOUNDARY_CONDITIONS,
)
from pssolver.operators.tensor_divergence import (
    projected_component_basis_stress_divergence,
)
from pssolver.transforms import TensorProductTransformBackend


VELOCITY_BASIS = ("periodic", "dirichlet", "dirichlet")


def _mode(kind, coordinate, length, mode):
    wave = (
        2.0 * math.pi * mode / length
        if kind == "periodic"
        else math.pi * mode / length
    )
    if kind == "dirichlet":
        return torch.sin(wave * coordinate), wave * torch.cos(wave * coordinate)
    return torch.cos(wave * coordinate), -wave * torch.sin(wave * coordinate)


def test_component_basis_stress_divergence_matches_analytic_projection():
    backend = TensorProductTransformBackend(
        (12, 10, 8),
        (2.0 * math.pi, 2.5 * math.pi, 3.0 * math.pi),
        device="cpu",
        dtype=torch.float64,
        execution_order="real_first",
        spectral_storage="full_complex",
    )
    coordinates = backend.spatial_grids
    stress = []
    analytic_rows = [torch.zeros(backend.shape, dtype=torch.float64) for _ in range(3)]
    for index, basis in enumerate(
        CHANNEL_DISTORTION_STRESS_BOUNDARY_CONDITIONS
    ):
        values = []
        derivatives = []
        for axis, kind in enumerate(basis):
            value, derivative = _mode(
                kind,
                coordinates[axis],
                backend.lengths[axis],
                axis + 1,
            )
            values.append(value)
            derivatives.append(derivative)
        amplitude = 0.1 * (index + 1)
        stress.append(amplitude * values[0] * values[1] * values[2])
        row, derivative_axis = divmod(index, 3)
        differentiated = amplitude * derivatives[derivative_axis]
        for axis, value in enumerate(values):
            if axis != derivative_axis:
                differentiated = differentiated * value
        analytic_rows[row] = analytic_rows[row] + differentiated

    observed = projected_component_basis_stress_divergence(
        backend,
        tuple(stress),
        CHANNEL_DISTORTION_STRESS_BOUNDARY_CONDITIONS,
        output_boundary_conditions=VELOCITY_BASIS,
    )
    analytic = torch.stack(tuple(analytic_rows))
    expected = backend.inverse(
        backend.forward(analytic, VELOCITY_BASIS),
        VELOCITY_BASIS,
    )

    torch.testing.assert_close(observed, expected, rtol=2.0e-12, atol=2.0e-12)


def test_component_basis_divergence_rejects_incomplete_basis_manifest():
    backend = TensorProductTransformBackend(
        (4, 4, 4),
        (4.0, 4.0, 4.0),
        device="cpu",
        dtype=torch.float64,
    )
    stress = tuple(torch.zeros(backend.shape) for _ in range(9))
    try:
        projected_component_basis_stress_divergence(
            backend,
            stress,
            CHANNEL_DISTORTION_STRESS_BOUNDARY_CONDITIONS[:-1],
            output_boundary_conditions=VELOCITY_BASIS,
        )
    except ValueError as exc:
        assert "nine basis tuples" in str(exc)
    else:
        raise AssertionError("an incomplete component-basis manifest was accepted")
