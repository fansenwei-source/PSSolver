import pathlib
import sys

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pssolver import SpectralSolver


def main():
    shape = (64, 32, 32)
    boundary_conditions = ("periodic", "dirichlet", "neumann")

    solver = SpectralSolver(
        shape=shape,
        L=(2 * torch.pi, 1.0, 1.0),
        dt=1e-3,
        device="cpu",
    )

    x, y, z = solver.spatial_grids
    init = torch.sin(2 * x) * torch.sin(3 * torch.pi * y) * torch.cos(2 * torch.pi * z)

    diffusivity = 0.1
    q2 = solver.get_q2(boundary_conditions)
    L_hat = -diffusivity * q2

    solver.model.add_dynamic_field(
        name="u",
        init=init,
        L_hat=L_hat,
        boundary_conditions=boundary_conditions,
    )

    solver.build()
    solver.run(steps=100)

    print("u.bc =", solver.fields["u.bc"])
    print("u shape =", solver.fields["u"].shape)
    print("u.max() =", solver.fields["u"].max().item())
    print("u.min() =", solver.fields["u"].min().item())


if __name__ == "__main__":
    main()
