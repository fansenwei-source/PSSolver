"""Independent solutions for the S-29 diffusion qualification."""

import hashlib
from math import pi

import numpy as np
import sympy as sp
from scipy.linalg import expm


def coefficients(n):
    diffusion = tuple(.2 + .03 * i for i in range(n))
    coupling = tuple(tuple(
        -.3 - .04 * i if i == j else .15 if j == i + 1 else -.07 if i == j + 1 else 0.
        for j in range(n)
    ) for i in range(n))
    return coupling, diffusion


def coupled_exact(coupling, diffusion, time):
    n = len(diffusion)
    c = np.array(coupling)
    d = np.diag(diffusion)
    first = expm((c - d) * time) @ np.array([1 / (i + 1) for i in range(n)])
    second = expm((c - 4 * d) * time) @ np.array([(-1)**i / (i + 2) for i in range(n)])
    x = np.arange(32) * (2 * pi / 32)
    return np.broadcast_to(
        (first[:, None] * np.cos(x) + second[:, None] * np.sin(2 * x))[:, :, None, None],
        (n, 32, 8, 8),
    ).copy()


def variable_diffusion_expressions():
    x, t = sp.symbols("x t", real=True)
    solution = sp.exp(-t) * sp.sin(x)
    a = 1 + sp.cos(x) / 4
    source = sp.trigsimp(sp.diff(solution, t) - sp.diff(a * sp.diff(solution, x), x))
    assert sp.simplify(source - sp.exp(-t) * sp.sin(x) * sp.cos(x) / 2) == 0
    return (
        sp.lambdify((x, t), solution, "numpy"),
        sp.lambdify((x, t), source, "numpy"),
        {"sympy": sp.__version__, **{
            name: hashlib.sha256(sp.srepr(value).encode()).hexdigest()
            for name, value in {"solution": solution, "coefficient": a, "source": source}.items()
        }},
    )
