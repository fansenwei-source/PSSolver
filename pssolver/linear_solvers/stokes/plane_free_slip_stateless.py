"""Caller-owned diagnostics for the frozen mixed-basis Schur solve."""

import math

import torch


def solve_force_hats_stateless(solver, fx_hat, fy_hat, fz_hat, *, pressure_diagnostics):
    """Execute the frozen Plane operations without publishing solver attributes."""
    report = {}
    ux_hat_free = solver._tangential_helmholtz_inverse(fx_hat)
    uy_hat_free = solver._tangential_helmholtz_inverse(fy_hat)
    uz_hat_free = solver._normal_helmholtz_inverse(fz_hat)

    pressure_rhs = solver._project_pressure_gauge(
        -solver.divergence_hat(ux_hat_free, uy_hat_free, uz_hat_free)
    )
    rhs_hat = solver._project_pressure_gauge(pressure_rhs)
    if pressure_diagnostics:
        rhs_norm = torch.linalg.vector_norm(rhs_hat.reshape(-1)).item()
    if pressure_diagnostics and rhs_norm == 0.0:
        report["last_pressure_iterations"] = 0
        report["last_pressure_residual"] = 0.0
        report["last_pressure_relative_residual"] = 0.0
        pressure_hat = torch.zeros_like(rhs_hat)
    else:
        pressure_hat = solver._project_pressure_gauge(rhs_hat / solver.schur_diag_safe)
        report["last_pressure_iterations"] = 1
        if pressure_diagnostics:
            residual = rhs_hat - solver._pressure_operator(pressure_hat)
            residual_norm = torch.linalg.vector_norm(residual.reshape(-1)).item()
            report["last_pressure_residual"] = residual_norm
            report["last_pressure_relative_residual"] = residual_norm / rhs_norm
        else:
            report["last_pressure_residual"] = math.nan
            report["last_pressure_relative_residual"] = math.nan
    report["last_pressure_hat"] = pressure_hat.detach()

    ux_hat = ux_hat_free - solver._tangential_helmholtz_inverse(solver.ikx * pressure_hat)
    uy_hat = uy_hat_free - solver._tangential_helmholtz_inverse(solver.iky * pressure_hat)
    uz_hat = uz_hat_free - solver._normal_helmholtz_inverse(solver._pressure_grad_z(pressure_hat))
    return ux_hat, uy_hat, uz_hat, pressure_hat, report
