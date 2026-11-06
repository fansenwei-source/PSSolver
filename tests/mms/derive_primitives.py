"""S-23 symbolic sources, independent of PSSolver and its numerical kernels.

Run with the SymPy skill interpreter. This derives references for future engine
primitives, not a second production implementation. Expressions have exact
rational coefficients; only trusted expressions created here reach lambdify.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import sympy as sp


def expressions() -> tuple[tuple, dict]:
    x, y, z, t = sp.symbols("x y z t", real=True)
    coordinates = (x, y, z)
    result = {}
    for kind, value in {
        "fft": sp.sin(2 * x) + sp.cos(3 * x),
        "dct": sp.cos(2 * x),
        "dst": sp.sin(2 * x),
    }.items():
        result[kind] = value
        result[kind + "_d1"] = sp.diff(value, x)
        result[kind + "_laplacian"] = sp.diff(value, x, 2)
    result["diffusion"] = sp.exp(-sp.Rational(1, 3) * 4 * t) * sp.cos(2 * x)
    result["diffusion_source"] = sp.simplify(
        sp.diff(result["diffusion"], t) - sp.diff(result["diffusion"], x, 2) / 3
    )
    h = sp.Symbol("h", positive=True)
    result["euler_modal_factor"] = 1 / (1 + sp.Rational(4, 3) * h)
    psi = sp.sin(x) * sp.cos(y) * sp.sin(z)
    no_slip_psi = sp.sin(x) * sp.sin(y) ** 2 * sp.sin(z) ** 2
    velocities = {
        "periodic": (sp.sin(x) * sp.cos(y), -sp.cos(x) * sp.sin(y), sp.Integer(0)),
        "free_slip": (sp.diff(psi, z), sp.Integer(0), -sp.diff(psi, x)),
        "no_slip": (sp.diff(no_slip_psi, y), -sp.diff(no_slip_psi, x), sp.Integer(0)),
    }
    pressure = sp.cos(x) * sp.cos(y) * sp.cos(z)
    for family, velocity in velocities.items():
        result[family + "_p"] = pressure
        result[family + "_div"] = sp.simplify(
            sum(sp.diff(value, axis) for value, axis in zip(velocity, coordinates, strict=True))
        )
        for index, (value, axis) in enumerate(zip(velocity, coordinates, strict=True)):
            result[f"{family}_u{index}"] = value
            result[f"{family}_f{index}"] = sp.simplify(
                value / 5
                - sp.Rational(73, 100) * sum(sp.diff(value, c, 2) for c in coordinates)
                + sp.diff(pressure, axis)
            )
    return (x, y, z, t, h), result


def numerical_functions() -> dict:
    variables, values = expressions()
    return {name: sp.lambdify(variables, value, modules="numpy") for name, value in values.items()}


def manifest() -> dict:
    variables, values = expressions()
    encoded = {key: sp.srepr(value) for key, value in values.items()}
    assert values["diffusion_source"] == 0
    assert all(values[family + "_div"] == 0 for family in ("periodic", "free_slip", "no_slip"))
    assert not any(name == "pssolver" or name.startswith("pssolver.") for name in sys.modules)
    return {
        "schema": 1,
        "generator_sympy_version": sp.__version__,
        "variables": [str(value) for value in variables],
        "expressions": encoded,
        "expression_sha256": {
            key: hashlib.sha256(value.encode()).hexdigest() for key, value in encoded.items()
        },
        "sampling": {
            "periodic": "node or cell-centered on [0, 2*pi)",
            "bounded_nodes": "DCT-I endpoints; DST-I interior; [0, pi]",
            "bounded_cells": "DCT-II/DST-II at (j+1/2)*pi/N",
        },
        "expected": {
            "modal_spatial_error": "roundoff (resolved trigonometric modes)",
            "diffusion_time_order": 1,
            "schur": "independent collocation-basis dense assembly; not production-generated RHS",
        },
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    with args.output.open("x") as stream:
        json.dump(manifest(), stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(hashlib.sha256(args.output.read_bytes()).hexdigest())
