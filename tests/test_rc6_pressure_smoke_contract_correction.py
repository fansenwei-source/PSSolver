"""Regression tests for the RC6 fixed-work smoke-contract correction."""

from __future__ import annotations

import inspect

import pytest

from benchmarks import run_rc6_contract_smokes as smoke
from pssolver import SpectralSolver
from pssolver.linear_solvers.stokes import ChannelNoSlipModalStokesSolver


def test_rc6_smoke_uses_the_authoritative_public_termination_labels():
    assert smoke.RELATIVE_TOLERANCE_TERMINATION_REASON == (
        "relative_tolerance_met"
    )
    assert smoke.FIXED_WORK_TERMINATION_REASON == "fixed_work_completed"
    source = inspect.getsource(smoke)
    assert "fixed_iterations_completed" not in source


@pytest.mark.parametrize("dtype_name", ["float32", "float64"])
def test_cpu_fixed_work_reaches_the_exact_label_and_iteration_count(dtype_name):
    import torch

    dtype = getattr(torch, dtype_name)
    shape = (8, 6, 4)
    spectral = SpectralSolver(
        shape,
        L=(8.0, 6.0, 4.0),
        device="cpu",
        dtype=dtype,
    )
    solver = ChannelNoSlipModalStokesSolver(
        spectral.transform_backend,
        friction=0.2,
        viscosity=0.73,
        pressure_relative_tolerance=torch.finfo(dtype).eps,
        pressure_max_iterations=4,
        pressure_fixed_iterations=4,
    )
    complex_dtype = torch.complex64 if dtype is torch.float32 else torch.complex128
    generator = torch.Generator(device="cpu").manual_seed(9)
    rhs = torch.complex(
        torch.randn((1, *shape), generator=generator, dtype=dtype),
        torch.randn((1, *shape), generator=generator, dtype=dtype),
    ).to(complex_dtype)
    pressure = solver._solve_pressure(rhs)
    assert solver.last_pressure_termination_reason == (
        smoke.FIXED_WORK_TERMINATION_REASON
    )
    assert solver.last_pressure_iterations == 4
    assert bool(torch.isfinite(pressure).all().item()) is True
