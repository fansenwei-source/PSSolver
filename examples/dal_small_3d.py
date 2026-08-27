"""Small 3D smoke test for the checkpointed PSSolver DAL implementation."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pssolver import SpectralSolver
from pssolver.control import (
    DiscreteAdjointLoop,
    FunctionalSemiImplicitStep,
    QTrackingObjective,
    TemporalMaskControl,
    smooth_box_mask,
)


class StaticResponse(torch.nn.Module):
    def forward(self, fields, params):
        response = params["alpha"] * fields["q"]
        return fields.transform_tensor(response.unsqueeze(0), ("periodic",) * 3)


class ControlledDynamics(torch.nn.Module):
    def forward(self, fields, params):
        del params
        q = fields["q"]
        response = fields["response"]
        rhs = response - 0.1 * q.square() * q
        return fields.transform_tensor(rhs.unsqueeze(0), ("periodic",) * 3)


def main():
    torch.manual_seed(4)
    shape = (12, 8, 6)
    steps = 12
    solver = SpectralSolver(shape, L=shape, dt=0.02, device="cpu")
    q0 = 0.05 * torch.randn(shape)
    periodic = ("periodic",) * 3
    solver.model.add_dynamic_field(
        "q",
        q0,
        L_hat=-0.2 * solver.get_q2(periodic),
        boundary_conditions=periodic,
    )
    solver.model.add_static_field("response", boundary_conditions=periodic)
    solver.model.parameters.new_param("alpha", torch.zeros((1, *shape)))
    solver.model.set_static_compute_model(StaticResponse())
    solver.model.set_nonlinear_model(ControlledDynamics())
    solver.build()

    mask = smooth_box_mask(shape, ((2, 10), (2, 6), (1, 5)), transition_width=0.8)
    control = TemporalMaskControl(
        mask.unsqueeze(0),
        num_steps=steps,
        block_size=3,
        alpha_min=0.0,
        alpha_max=2.0,
        initial_alpha=0.5,
    )
    initial_q = solver.fields.spatial[:1].detach().clone()
    objective = QTrackingObjective(
        target_q=1.4 * initial_q,
        dt=solver.dt,
        spatial_mask=mask,
        running_weight=0.5,
        terminal_weight=1.0,
        control_weight=1e-4,
    )
    dal = DiscreteAdjointLoop(
        FunctionalSemiImplicitStep(solver),
        objective,
        control,
        num_steps=steps,
        checkpoint_stride=4,
        temporal_control_weight=1e-4,
    )

    check = dal.directional_derivative_check(initial_q, epsilon=2e-3)
    print(
        "gradient_check "
        f"adjoint={check['adjoint']:.6e} "
        f"finite_difference={check['finite_difference']:.6e} "
        f"relative_error={check['relative_error']:.3e}"
    )
    taylor = dal.taylor_test(initial_q, epsilons=(0.2, 0.1, 0.05))
    print(
        "taylor_second_order "
        + " ".join(
            f"eps={item['epsilon']:.3g}:r2={item['second_order_residual']:.3e}"
            for item in taylor
        )
    )
    for iteration in range(3):
        result = dal.armijo_step(initial_q, initial_step=5.0)
        print(
            f"iteration={iteration} cost={result.cost_before:.6e}->{result.cost_after:.6e} "
            f"grad={result.gradient_norm:.3e} step={result.step_size:.3e} "
            f"accepted={result.accepted}"
        )


if __name__ == "__main__":
    main()
