import torch

from pssolver import SpectralSolver
from pssolver.control import (
    DiscreteAdjointLoop,
    FunctionalSemiImplicitStep,
    QTrackingObjective,
    TemporalMaskControl,
    partition_mask_along_axis,
    partition_mask_rbf,
    smooth_box_mask,
)


class ToyStaticModel(torch.nn.Module):
    def forward(self, fields, params):
        forcing = params["alpha"] * fields["q"]
        return fields.transform_tensor(forcing.unsqueeze(0), ("periodic",) * 3)


class ToyNonlinearModel(torch.nn.Module):
    def forward(self, fields, params):
        del params
        q = fields["q"]
        velocity = fields["u"]
        nonlinear = velocity - 0.1 * q.square() * q
        return fields.transform_tensor(nonlinear.unsqueeze(0), ("periodic",) * 3)


def build_problem(num_steps=5):
    shape = (6, 5, 4)
    solver = SpectralSolver(shape=shape, L=(6.0, 5.0, 4.0), dt=0.02, device="cpu")
    generator = torch.Generator().manual_seed(7)
    q0 = 0.05 * torch.randn(shape, generator=generator)
    q2 = solver.get_q2(("periodic",) * 3)
    solver.model.add_dynamic_field(
        "q",
        init=q0,
        L_hat=-0.2 * q2,
        boundary_conditions=("periodic",) * 3,
    )
    solver.model.add_static_field("u", boundary_conditions=("periodic",) * 3)
    solver.model.parameters.new_param("alpha", torch.ones((1, *shape)))
    solver.model.set_static_compute_model(ToyStaticModel())
    solver.model.set_nonlinear_model(ToyNonlinearModel())
    solver.build()

    mask = smooth_box_mask(
        shape,
        bounds=((1, 5), (1, 4), (1, 3)),
        transition_width=0.6,
    )
    control = TemporalMaskControl(
        mask.unsqueeze(0),
        num_steps=num_steps,
        block_size=2,
        alpha_min=0.0,
        alpha_max=2.0,
        initial_alpha=0.7,
        baseline=0.0,
    )
    initial_q = solver.model.fields.spatial[:1].detach().clone()
    target_q = initial_q * 1.2
    objective = QTrackingObjective(
        target_q,
        dt=solver.dt,
        spatial_mask=mask,
        running_weight=0.5,
        terminal_weight=1.0,
        control_weight=1e-4,
    )
    stepper = FunctionalSemiImplicitStep(solver)
    dal = DiscreteAdjointLoop(
        stepper,
        objective,
        control,
        num_steps=num_steps,
        checkpoint_stride=2,
        temporal_control_weight=1e-4,
    )
    return solver, stepper, control, dal, initial_q


def test_functional_step_matches_integrator():
    solver, stepper, control, _, initial_q = build_problem()
    alpha = control.field_for_step(0).detach()
    expected = stepper(initial_q, alpha).detach()

    solver.model.fields.spatial[:1] = initial_q
    solver.model.fields.spectral[:1] = solver.model.fields.fftn()[:1]
    solver.model.parameters["alpha"] = alpha
    solver.run(1)
    actual = solver.model.fields.spatial[:1]
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)


def test_discrete_adjoint_matches_finite_difference():
    _, _, _, dal, initial_q = build_problem()
    direction = torch.linspace(-1.0, 1.0, dal.control.logits.numel()).reshape_as(
        dal.control.logits
    )
    check = dal.directional_derivative_check(
        initial_q,
        epsilon=2e-3,
        direction=direction,
    )
    assert check["relative_error"] < 2e-2, check


def test_gradient_check_suite_reuses_gradient_for_random_directions():
    _, _, _, dal, initial_q = build_problem()
    suite = dal.gradient_check_suite(
        initial_q,
        epsilons=(4e-3, 2e-3),
        num_random_directions=3,
        seed=19,
    )
    assert len(suite["directions"]) == 4
    assert all(len(result["checks"]) == 2 for result in suite["directions"])
    assert suite["max_relative_error"] < 3e-2, suite


def test_armijo_step_reduces_cost():
    _, _, _, dal, initial_q = build_problem()
    result = dal.armijo_step(initial_q, initial_step=5.0, max_trials=10)
    assert result.accepted
    assert result.cost_after < result.cost_before


def test_lbfgs_armijo_steps_reduce_cost_monotonically():
    _, _, _, dal, initial_q = build_problem()
    costs = []
    for _ in range(5):
        result = dal.armijo_step(
            initial_q,
            initial_step=1.0,
            max_trials=10,
            direction_method="lbfgs",
            lbfgs_history_size=4,
        )
        assert result.accepted
        costs.append(result.cost_after)
    assert all(right < left for left, right in zip(costs, costs[1:]))


def test_taylor_remainder_is_second_order():
    _, _, _, dal, initial_q = build_problem()
    direction = torch.linspace(-1.0, 1.0, dal.control.logits.numel()).reshape_as(
        dal.control.logits
    )
    results = dal.taylor_test(
        initial_q,
        epsilons=(0.2, 0.1, 0.05),
        direction=direction,
    )
    first_ratio = results[0]["second_order_residual"] / max(
        results[1]["second_order_residual"], 1e-30
    )
    second_ratio = results[1]["second_order_residual"] / max(
        results[2]["second_order_residual"], 1e-30
    )
    assert first_ratio > 2.5, results
    assert second_ratio > 2.5, results


def test_partitioned_masks_reconstruct_path_mask():
    mask = smooth_box_mask(
        (24, 10, 8),
        bounds=((4, 20), (1, 9), (1, 7)),
        transition_width=0.8,
    )
    masks = partition_mask_along_axis(
        mask,
        6,
        axis=0,
        bounds=(4, 20),
        overlap=0.75,
    )
    assert masks.shape == (6, 24, 10, 8)
    assert torch.all(masks >= 0)
    torch.testing.assert_close(masks.sum(dim=0), mask, rtol=1e-6, atol=1e-7)
    peak_indices = masks.amax(dim=(2, 3)).argmax(dim=1)
    assert torch.all(peak_indices[1:] > peak_indices[:-1])


def test_periodic_partitioned_masks_reconstruct_path_mask():
    mask = smooth_box_mask(
        (24, 6, 5),
        bounds=((0, 6), (1, 5), (1, 4)),
        transition_width=0.6,
    )
    masks = partition_mask_along_axis(
        mask,
        4,
        axis=0,
        bounds=(0, 6),
        overlap=0.75,
        periodic=True,
    )
    torch.testing.assert_close(masks.sum(dim=0), mask, rtol=1e-6, atol=1e-7)
    assert torch.all(masks >= 0)


def test_3d_rbf_masks_reconstruct_path_mask():
    mask = smooth_box_mask(
        (16, 10, 8),
        bounds=((2, 14), (2, 8), (1, 7)),
        transition_width=0.7,
    )
    masks = partition_mask_rbf(
        mask,
        (4, 3, 2),
        bounds=((2, 14), (2, 8), (1, 7)),
        overlap=0.75,
        periodic=(True, False, False),
    )
    assert masks.shape == (24, 16, 10, 8)
    assert torch.all(masks >= 0)
    torch.testing.assert_close(masks.sum(dim=0), mask, rtol=1e-6, atol=1e-7)


def test_3d_mask_grid_spatial_regularization_uses_axis_neighbors():
    control = TemporalMaskControl(
        torch.ones((8, 2, 2, 2)) / 8.0,
        num_steps=1,
        block_size=1,
        alpha_min=0.0,
        alpha_max=10.0,
        initial_alpha=4.5,
        mask_grid_shape=(2, 2, 2),
    )
    amplitudes = torch.arange(1.0, 9.0).reshape(1, 2, 2, 2)
    fractions = amplitudes / 10.0
    with torch.no_grad():
        control.logits.copy_(torch.log(fractions / (1.0 - fractions)).reshape(1, 8))
    axis_penalties = [
        torch.diff(amplitudes, dim=axis).square().mean()
        for axis in (1, 2, 3)
    ]
    expected = 0.5 * 0.2 * torch.stack(axis_penalties).mean()
    torch.testing.assert_close(control.spatial_regularization(0.2), expected)


def test_control_baseline_makes_uniform_initial_activity_in_perturbation_mode():
    mask = smooth_box_mask(
        (12, 8, 6),
        bounds=((2, 10), (1, 7), (1, 5)),
        transition_width=0.8,
    )
    masks = partition_mask_along_axis(mask, 4, axis=0, bounds=(2, 10))
    control = TemporalMaskControl(
        masks,
        num_steps=2,
        block_size=1,
        alpha_min=0.0,
        alpha_max=10.0,
        initial_alpha=4.5,
        baseline=4.5,
    )
    torch.testing.assert_close(
        control.field_for_step(0),
        torch.full((1, 12, 8, 6), 4.5),
        rtol=1e-6,
        atol=1e-6,
    )


def test_control_amplitude_warm_start_round_trips_all_blocks():
    control = TemporalMaskControl(
        torch.ones((3, 4, 3, 2)) / 3.0,
        num_steps=6,
        block_size=2,
        alpha_min=0.0,
        alpha_max=10.0,
        initial_alpha=4.5,
    )
    amplitudes = torch.linspace(1.0, 9.0, 9).reshape(3, 3)
    control.set_amplitudes(amplitudes)
    torch.testing.assert_close(control.amplitudes(), amplitudes)
