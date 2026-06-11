import math
import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pssolver import SpectralSolver


def max_abs_error(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    return (lhs - rhs).abs().max().item()


def relative_error(lhs: torch.Tensor, rhs: torch.Tensor) -> float:
    scale = rhs.abs().max().item()
    if scale == 0.0:
        scale = 1.0
    return max_abs_error(lhs, rhs) / scale


def run_validation():
    results = []

    solver_1d = SpectralSolver(shape=(32,), L=(2 * torch.pi,), dt=1e-3, device="cpu")
    x = solver_1d.spatial_grids[0]
    roundtrip_cases = [
        (("periodic",), torch.sin(3 * x) + 0.3 * torch.cos(5 * x), "1D periodic transform round-trip"),
        (("dirichlet",), torch.sin(4 * math.pi * x / (2 * torch.pi)), "1D Dirichlet transform round-trip"),
        (("neumann",), torch.cos(3 * math.pi * x / (2 * torch.pi)), "1D Neumann transform round-trip"),
    ]
    for boundary_conditions, field, label in roundtrip_cases:
        spectral = solver_1d.transform_tensor(field.unsqueeze(0), boundary_conditions)
        reconstructed = solver_1d.inverse_transform_tensor(spectral, boundary_conditions).squeeze(0)
        results.append((label, boundary_conditions, "absolute error", max_abs_error(reconstructed, field)))

    solver_2d = SpectralSolver(shape=(24, 20), L=(2 * torch.pi, 1.0), dt=1e-3, device="cpu")
    x2, y2 = solver_2d.spatial_grids
    mixed_bc_2d = ("periodic", "dirichlet")
    u2 = torch.sin(2 * x2) * torch.sin(3 * math.pi * y2)
    uhat2 = solver_2d.transform_tensor(u2.unsqueeze(0), mixed_bc_2d)
    recon2 = solver_2d.inverse_transform_tensor(uhat2, mixed_bc_2d).squeeze(0)
    results.append(("2D mixed transform round-trip", mixed_bc_2d, "absolute error", max_abs_error(recon2, u2)))

    lap_hat2 = -solver_2d.get_q2(mixed_bc_2d) * uhat2.squeeze(0)
    lap2 = solver_2d.inverse_transform_tensor(lap_hat2.unsqueeze(0), mixed_bc_2d).squeeze(0)
    exact_lap2 = -(2.0**2 + (3 * math.pi) ** 2) * u2
    results.append(("2D mixed Laplacian consistency", mixed_bc_2d, "relative error", relative_error(lap2, exact_lap2)))

    solver_d = SpectralSolver(shape=(32,), L=(1.0,), dt=1e-3, device="cpu")
    xd = solver_d.spatial_grids[0]
    init_d = torch.sin(3 * math.pi * xd)
    solver_d.model.add_dynamic_field(
        "u",
        init_d,
        solver_d.get_laplacian_eigs(("dirichlet",)),
        boundary_conditions=("dirichlet",),
    )
    solver_d.build()
    grad_d = solver_d.fields.gradient("u", axis=0).squeeze(0)
    exact_grad_d = 3 * math.pi * torch.cos(3 * math.pi * xd)
    results.append(("1D Dirichlet first derivative", ("dirichlet",), "relative error", relative_error(grad_d, exact_grad_d)))

    solver_n = SpectralSolver(shape=(32,), L=(1.0,), dt=1e-3, device="cpu")
    xn = solver_n.spatial_grids[0]
    init_n = torch.cos(4 * math.pi * xn)
    solver_n.model.add_dynamic_field(
        "u",
        init_n,
        solver_n.get_laplacian_eigs(("neumann",)),
        boundary_conditions=("neumann",),
    )
    solver_n.build()
    grad_n = solver_n.fields.gradient("u", axis=0).squeeze(0)
    exact_grad_n = -4 * math.pi * torch.sin(4 * math.pi * xn)
    results.append(("1D Neumann first derivative", ("neumann",), "relative error", relative_error(grad_n, exact_grad_n)))

    solver_step_2d = SpectralSolver(shape=(24, 20), L=(2 * torch.pi, 1.0), dt=1e-2, device="cpu")
    x3, y3 = solver_step_2d.spatial_grids
    mode_x = 2.0
    mode_y = 3.0 * math.pi
    diffusivity_2d = 0.1
    init_step_2d = torch.sin(mode_x * x3) * torch.sin(mode_y * y3)
    solver_step_2d.model.add_dynamic_field(
        "u",
        init_step_2d,
        -diffusivity_2d * solver_step_2d.get_q2(mixed_bc_2d),
        boundary_conditions=mixed_bc_2d,
    )
    solver_step_2d.build()
    uhat_before_2d = solver_step_2d.fields["u.hat"].clone()
    solver_step_2d.integrator.step()
    uhat_after_2d = solver_step_2d.fields["u.hat"]
    idx_2d = torch.argmax(uhat_before_2d.abs())
    observed_2d = (uhat_after_2d.flatten()[idx_2d] / uhat_before_2d.flatten()[idx_2d]).real.item()
    expected_2d = 1.0 / (1.0 + solver_step_2d.dt * diffusivity_2d * (mode_x**2 + mode_y**2))
    results.append(("2D mixed one-step diffusion update", mixed_bc_2d, "spectral decay error", abs(observed_2d - expected_2d)))

    solver_3d = SpectralSolver(shape=(18, 16, 14), L=(2 * torch.pi, 1.0, 1.0), dt=5e-3, device="cpu")
    x4, y4, z4 = solver_3d.spatial_grids
    mixed_bc_3d = ("periodic", "dirichlet", "neumann")
    mode_x_3d = 2.0
    mode_y_3d = 3.0 * math.pi
    mode_z_3d = 2.0 * math.pi
    u3 = torch.sin(mode_x_3d * x4) * torch.sin(mode_y_3d * y4) * torch.cos(mode_z_3d * z4)
    uhat3 = solver_3d.transform_tensor(u3.unsqueeze(0), mixed_bc_3d)
    recon3 = solver_3d.inverse_transform_tensor(uhat3, mixed_bc_3d).squeeze(0)
    results.append(("3D mixed transform round-trip", mixed_bc_3d, "absolute error", max_abs_error(recon3, u3)))

    lap_hat3 = -solver_3d.get_q2(mixed_bc_3d) * uhat3.squeeze(0)
    lap3 = solver_3d.inverse_transform_tensor(lap_hat3.unsqueeze(0), mixed_bc_3d).squeeze(0)
    exact_lap3 = -(mode_x_3d**2 + mode_y_3d**2 + mode_z_3d**2) * u3
    results.append(("3D mixed Laplacian consistency", mixed_bc_3d, "relative error", relative_error(lap3, exact_lap3)))

    diffusivity_3d = 0.05
    solver_step_3d = SpectralSolver(shape=(18, 16, 14), L=(2 * torch.pi, 1.0, 1.0), dt=5e-3, device="cpu")
    xs, ys, zs = solver_step_3d.spatial_grids
    init_step_3d = torch.sin(mode_x_3d * xs) * torch.sin(mode_y_3d * ys) * torch.cos(mode_z_3d * zs)
    solver_step_3d.model.add_dynamic_field(
        "u",
        init_step_3d,
        -diffusivity_3d * solver_step_3d.get_q2(mixed_bc_3d),
        boundary_conditions=mixed_bc_3d,
    )
    solver_step_3d.build()
    uhat_before_3d = solver_step_3d.fields["u.hat"].clone()
    solver_step_3d.integrator.step()
    uhat_after_3d = solver_step_3d.fields["u.hat"]
    idx_3d = torch.argmax(uhat_before_3d.abs())
    observed_3d = (uhat_after_3d.flatten()[idx_3d] / uhat_before_3d.flatten()[idx_3d]).real.item()
    expected_3d = 1.0 / (
        1.0 + solver_step_3d.dt * diffusivity_3d * (mode_x_3d**2 + mode_y_3d**2 + mode_z_3d**2)
    )
    results.append(("3D mixed one-step diffusion update", mixed_bc_3d, "spectral decay error", abs(observed_3d - expected_3d)))

    return results


if __name__ == "__main__":
    for label, boundary_conditions, metric, value in run_validation():
        print(f"{label} | {boundary_conditions} | {metric} | {value:.10e}")
