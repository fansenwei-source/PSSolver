"""Independent dense Schur matrix from analytic collocation basis values.

Uses NumPy only, no PSSolver transforms or stored solver factors. Deliberate
reference duplication is required by ARCHITECTURE section 7.
"""

from __future__ import annotations

import numpy as np


def bounded_matrices(size: int, length: float) -> tuple:
    points = (np.arange(size) + 0.5) * length / size
    neumann_modes = np.arange(size) * np.pi / length
    dirichlet_modes = np.arange(1, size + 1) * np.pi / length
    n_scale = np.full(size, np.sqrt(2 / size))
    n_scale[0] = np.sqrt(1 / size)
    d_scale = np.full(size, np.sqrt(2 / size))
    d_scale[-1] = np.sqrt(1 / size)
    cosine = np.cos(points[:, None] * neumann_modes) * n_scale
    sine = np.sin(points[:, None] * dirichlet_modes) * d_scale
    d_cosine = -np.sin(points[:, None] * neumann_modes) * neumann_modes * n_scale
    d_sine = np.cos(points[:, None] * dirichlet_modes) * dirichlet_modes * d_scale
    return sine.T @ cosine, sine.T @ d_cosine, cosine.T @ d_sine, dirichlet_modes


def schur_blocks(shape=(5, 4, 3), lengths=(5.0, 4.0, 3.0), friction=0.2, viscosity=0.73):
    cy, gy, dy, ky = bounded_matrices(shape[1], lengths[1])
    cz, gz, dz, kz = bounded_matrices(shape[2], lengths[2])
    blocks = []
    for index, kx in enumerate(2 * np.pi * np.fft.fftfreq(shape[0], lengths[0] / shape[0])):
        gradient = (1j * kx * np.kron(cy, cz), np.kron(gy, cz), np.kron(cy, gz))
        divergence = (1j * kx * np.kron(cy.T, cz.T), np.kron(dy, cz.T), np.kron(cy.T, dz))
        helmholtz = friction + viscosity * (kx**2 + ky[:, None] ** 2 + kz[None, :] ** 2).reshape(-1)
        block = -sum(
            d @ (g / helmholtz[:, None]) for d, g in zip(divergence, gradient, strict=True)
        )
        if index == 0:
            block[0, :] = 0
            block[:, 0] = 0
        blocks.append(block)
    return np.stack(blocks)
