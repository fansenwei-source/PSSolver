"""Stage E scalar-diffusion canary using the opt-in architecture path."""

from __future__ import annotations

import torch
from tqdm import trange

from pssolver.core import (
    BoundarySet,
    DealiasRule,
    DomainSpec,
    NumericsConfig,
    PeriodicBC,
    Precision,
    ProjectedTransformExecution,
    SpectralStorage,
    TransformExecutionOrder,
)
from pssolver.experimental import build_experimental_model_runtime
from pssolver.geometries import PeriodicBox
from pssolver.models.canary import ScalarDiffusionModel
from pssolver.utils import visualize1D


def main() -> None:
    size = 256
    length = 256.0
    timestep = 0.04
    steps = 10_000
    device = "cuda" if torch.cuda.is_available() else "cpu"

    boundaries = BoundarySet((PeriodicBC(),))
    model = ScalarDiffusionModel(
        boundaries=boundaries,
        diffusivity=1.0,
        initial_amplitude=100.0,
        initial_modes=(1,),
    )
    geometry = PeriodicBox(DomainSpec((size,), (length,)))
    numerics = NumericsConfig(
        precision=Precision.FLOAT32,
        dealias_rule=DealiasRule.NONE,
        transform_execution_order=TransformExecutionOrder.REAL_FIRST,
        projected_transform_execution=ProjectedTransformExecution.FULL,
        spectral_storage=SpectralStorage.FULL_COMPLEX,
    )
    runtime = build_experimental_model_runtime(
        model,
        geometry,
        numerics,
        dt=timestep,
        device=device,
    )

    snapshots = []
    for step in trange(steps):
        runtime.solver.run(1)
        if step % (steps // 100) == 0:
            snapshots.append(runtime.solver.fields["phi"].detach().clone())

    trajectory = torch.stack(snapshots).permute(1, 0, 2)
    print(f"Simulation finished. Data has shape: {trajectory.shape}")
    visualize1D(
        data=trajectory.cpu().numpy(),
        filename="diffusion1D.png",
    )


if __name__ == "__main__":
    main()
