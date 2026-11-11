"""S-23 independent primitive references; existing numerical gates are retained.

Float64 gates are no weaker than test_phase8_p82_periodic_stokes (4e-12),
test_channel_no_slip_modal_stokes (3e-12 pressure, 2e-12 residual) and the
existing transform/transpose tests. No cross-version tolerance is introduced.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy import fft

from pssolver.backends.tensor_product import TensorProductTransformBackend
from pssolver.linear_solvers.stokes import ChannelNoSlipModalStokesSolver, PeriodicModalStokesSolver
from pssolver.linear_solvers.stokes.plane_free_slip import FreeSlipModalStokesSolver
from tests._s28a_pressure import production_pressure
from tests.mms.derive_primitives import expressions, numerical_functions
from tests.mms.schur_reference import schur_blocks


def test_frozen_symbolic_expressions():
    import sympy as sp

    frozen = json.loads(Path(__file__).with_name("expressions.json").read_text())
    _, values = expressions()
    assert {
        key: hashlib.sha256(sp.srepr(value).encode()).hexdigest() for key, value in values.items()
    } == frozen["expression_sha256"]
    assert all(
        values[key] == 0
        for key in ("diffusion_source", "periodic_div", "free_slip_div", "no_slip_div")
    )


@pytest.mark.parametrize("kind", ["periodic", "neumann", "dirichlet"])
@pytest.mark.parametrize("size", [12, 24])
def test_cell_centered_transform_derivative_and_laplacian(kind, size):
    backend = TensorProductTransformBackend(
        (size,), (2 * np.pi if kind == "periodic" else np.pi,), device="cpu", dtype=torch.float64
    )
    x = backend.spatial_grids[0].numpy()
    key = {"periodic": "fft", "neumann": "dct", "dirichlet": "dst"}[kind]
    functions = numerical_functions()
    value = torch.tensor(functions[key](x, 0, 0, 0, 0), dtype=torch.float64)
    spectral = backend.forward(value, (kind,))
    actual_derivative, derivative_bc = backend.gradient_hat(spectral, (kind,), 0)
    for actual, expected in (
        (backend.inverse(spectral, (kind,)), value),
        (
            backend.inverse(actual_derivative, derivative_bc),
            torch.tensor(functions[key + "_d1"](x, 0, 0, 0, 0)),
        ),
        (
            backend.inverse(backend.laplacian_hat(spectral, (kind,)), (kind,)),
            torch.tensor(functions[key + "_laplacian"](x, 0, 0, 0, 0)),
        ),
    ):
        torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-12)


@pytest.mark.parametrize("kind", ["fft", "dct", "dst"])
def test_nodal_reference_transform_derivative_and_laplacian(kind):
    # Node-centered bounded transforms are references for the future engine;
    # today's production backend exposes only cell-centered bounded transforms.
    n = 24
    x = (
        np.arange(n) * 2 * np.pi / n
        if kind == "fft"
        else np.linspace(0, np.pi, n)
        if kind == "dct"
        else np.arange(1, n + 1) * np.pi / (n + 1)
    )
    functions = numerical_functions()
    value = functions[kind](x, 0, 0, 0, 0)
    if kind == "fft":
        coefficients = fft.fft(value)
        frequencies = fft.fftfreq(n, 1 / n)
        recovered = fft.ifft(coefficients).real
        derivative = fft.ifft(1j * frequencies * coefficients).real
        laplacian = fft.ifft(-(frequencies**2) * coefficients).real
    else:
        transform, inverse = (fft.dct, fft.idct) if kind == "dct" else (fft.dst, fft.idst)
        coefficients = transform(value, type=1)
        recovered = inverse(coefficients, type=1)
        modes = np.arange(n) if kind == "dct" else np.arange(1, n + 1)
        weights = coefficients / (n - 1 if kind == "dct" else n + 1)
        if kind == "dct":
            weights[[0, -1]] /= 2
            derivative = (-np.sin(x[:, None] * modes) * (weights * modes)).sum(axis=1)
        else:
            derivative = (np.cos(x[:, None] * modes) * (weights * modes)).sum(axis=1)
        laplacian = inverse(-(modes**2) * coefficients, type=1)
    for actual, expected in (
        (recovered, value),
        (derivative, functions[kind + "_d1"](x, 0, 0, 0, 0)),
        (laplacian, functions[kind + "_laplacian"](x, 0, 0, 0, 0)),
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-12)


def test_diagonal_implicit_euler_modal_decay_and_first_order():
    exact = np.exp(-4 / 3)
    errors = []
    for steps in (40, 80, 160):
        amplitude = torch.ones((), dtype=torch.float64)
        for _ in range(steps):
            amplitude = amplitude / (1 + (4 / 3) / steps)
        discrete = (1 + (4 / 3) / steps) ** (-steps)
        assert abs(amplitude.item() - discrete) < 2e-14
        errors.append(abs(amplitude.item() - exact))
    assert all(np.log2(a / b) > 0.95 for a, b in zip(errors[:-1], errors[1:], strict=True))


@pytest.mark.parametrize("family", ["periodic", "free_slip"])
def test_direct_stokes_matches_independent_symbolic_forcing(family):
    lengths = (2 * np.pi, 2 * np.pi, 2 * np.pi if family == "periodic" else np.pi)
    backend = TensorProductTransformBackend((12, 10, 8), lengths, device="cpu", dtype=torch.float64)
    periodic = ("periodic",) * 3
    tangential = periodic if family == "periodic" else ("periodic", "periodic", "neumann")
    normal = periodic if family == "periodic" else ("periodic", "periodic", "dirichlet")
    stokes = (
        PeriodicModalStokesSolver(
            backend, friction=0.2, viscosity=0.73, zero_mode_policy="friction"
        )
        if family == "periodic"
        else FreeSlipModalStokesSolver(
            backend,
            tangential_boundary_conditions=tangential,
            normal_boundary_conditions=normal,
            pressure_boundary_conditions=tangential,
            friction=0.2,
            viscosity=0.73,
            zero_mode_policy="friction",
        )
    )
    coordinates = [grid.numpy() for grid in backend.spatial_grids]
    functions = numerical_functions()

    def evaluate(name):
        return torch.tensor(
            np.broadcast_to(functions[name](*coordinates, 0, 0), backend.shape).copy(),
            dtype=torch.float64,
        ).unsqueeze(0)

    boundary = (tangential, tangential, normal)
    forces = [backend.forward(evaluate(f"{family}_f{i}"), bc) for i, bc in enumerate(boundary)]
    values = stokes.solve_force_hats(*forces)
    for i, bc in enumerate((*boundary, tangential)):
        name = f"{family}_u{i}" if i < 3 else family + "_p"
        torch.testing.assert_close(
            backend.inverse(values[i], bc), evaluate(name), rtol=4e-12, atol=4e-12
        )


def test_no_slip_schur_manufacture_and_transpose_dot_product():
    shape, lengths = (5, 4, 3), (5.0, 4.0, 3.0)
    backend = TensorProductTransformBackend(shape, lengths, device="cpu", dtype=torch.float64)
    stokes = ChannelNoSlipModalStokesSolver(
        backend,
        friction=0.2,
        viscosity=0.73,
        pressure_relative_tolerance=1e-13,
        pressure_max_iterations=300,
    )
    blocks = schur_blocks(shape, lengths)
    index = np.arange(np.prod(shape)).reshape(shape)
    exact = np.sin(index * 0.19) + 1j * np.cos(index * 0.23)
    exact[0, 0, 0] = 0
    rhs = np.einsum("kij,kj->ki", blocks, exact.reshape(shape[0], -1)).reshape(shape)
    expected = torch.tensor(exact[None])
    source = torch.tensor(rhs[None])
    actual, _, _ = production_pressure(stokes, source)

    def relative(x, y):
        return float(torch.linalg.vector_norm(x) / torch.linalg.vector_norm(y))

    assert relative(stokes._pressure_operator(expected) - source, source) < 2e-12
    assert relative(actual - expected, expected) < 3e-12
    other = torch.tensor((np.cos(index * 0.31) + 1j * np.sin(index * 0.17))[None])
    other[:, 0, 0, 0] = 0
    left = torch.sum(stokes._pressure_operator(expected).conj() * other)
    right = torch.sum(expected.conj() * stokes.apply_pressure_operator_transpose(other))
    assert float(abs(left - right) / max(abs(left), abs(right))) < 2e-12
