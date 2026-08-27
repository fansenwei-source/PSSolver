import numpy as np
import torch

from pssolver.control import (
    CoreAwareAlignedShapeObjective,
    CoreAwareComovingQTrajectoryObjective,
    CoreAwareMomentObjective,
    CoreTranslationNoMassObjective,
    CoreTranslationObjective,
    FreePathShapePreservingObjective,
    QTrackingObjective,
    available_loss_functions,
    build_loss_function,
    core_centerline_geometry,
    loop_core_metrics,
    periodic_profile_shift,
    soft_core_center,
    x_disturbance_profile,
)


def test_loss_function_registry_builds_current_objective():
    target_values = synthetic_S_ring(shape=(8, 6, 6), lengths=(8.0, 6.0, 6.0))
    target = torch.from_numpy(target_values).movedim(-1, 0).unsqueeze(1)
    assert available_loss_functions() == (
        "core_aware_aligned_shape_tracking",
        "core_aware_comoving_q_trajectory",
        "core_aware_moment_tracking",
        "core_aware_q_tracking",
        "core_translation",
        "core_translation_no_mass",
        "free_path_shape_preserving",
    )
    objective = build_loss_function(
        "core_aware_q_tracking",
        target_q=target,
        dt=1e-3,
    )
    assert isinstance(objective, QTrackingObjective)

    translation = build_loss_function(
        "core_translation",
        target_q=target,
        dt=1e-3,
        domain_lengths=(8.0, 6.0, 6.0),
        S_bulk=0.8,
    )
    no_mass = build_loss_function(
        "core_translation_no_mass",
        target_q=target,
        dt=1e-3,
        domain_lengths=(8.0, 6.0, 6.0),
        S_bulk=0.8,
    )
    assert isinstance(translation, CoreTranslationObjective)
    assert isinstance(no_mass, CoreTranslationNoMassObjective)

    moment = build_loss_function(
        "core_aware_moment_tracking",
        target_q=target,
        dt=1e-3,
        domain_lengths=(8.0, 6.0, 6.0),
        S_bulk=0.8,
    )
    assert isinstance(moment, CoreAwareMomentObjective)

    aligned = build_loss_function(
        "core_aware_aligned_shape_tracking",
        target_q=target,
        dt=1e-3,
        domain_lengths=(8.0, 6.0, 6.0),
        S_bulk=0.8,
    )
    assert isinstance(aligned, CoreAwareAlignedShapeObjective)

    comoving = build_loss_function(
        "core_aware_comoving_q_trajectory",
        target_q=target,
        initial_q=target,
        num_steps=10,
        dt=1e-3,
        domain_lengths=(8.0, 6.0, 6.0),
        S_bulk=0.8,
    )
    assert isinstance(comoving, CoreAwareComovingQTrajectoryObjective)

    free_path = build_loss_function(
        "free_path_shape_preserving",
        target_q=target,
        initial_q=target,
        dt=1e-3,
        domain_lengths=(8.0, 6.0, 6.0),
        S_bulk=0.8,
    )
    assert isinstance(free_path, FreePathShapePreservingObjective)


def test_loss_function_registry_rejects_unknown_name():
    try:
        build_loss_function("not_registered")
    except ValueError as error:
        assert "core_aware_q_tracking" in str(error)
    else:
        raise AssertionError("Unknown loss-function name was accepted.")


def synthetic_S_ring(
    shape=(64, 20, 20),
    lengths=(128.0, 10.0, 10.0),
    center=(65.0, 5.25, 5.25),
    radius=1.5,
    S_bulk=0.8,
):
    axes = [
        (np.arange(size, dtype=float) + 0.5) * length / size
        for size, length in zip(shape, lengths)
    ]
    x, y, z = np.meshgrid(*axes, indexing="ij")
    dx = (x - center[0] + 0.5 * lengths[0]) % lengths[0] - 0.5 * lengths[0]
    radial = np.sqrt((y - center[1]) ** 2 + (z - center[2]) ** 2)
    distance = np.sqrt(dx**2 + (radial - radius) ** 2)
    S = S_bulk * (1.0 - 0.9 * np.exp(-(distance / 0.35) ** 2))

    q5 = np.zeros((*shape, 5), dtype=np.float32)
    q5[..., 0] = S
    q5[..., 3] = -S / 2.0
    return q5


def test_loop_core_metrics_recovers_S_ring_geometry():
    q5 = synthetic_S_ring()
    metrics = loop_core_metrics(
        q5,
        (128.0, 10.0, 10.0),
        S_bulk=0.8,
        deficit_threshold=0.2,
    )
    assert metrics["detected"]
    assert abs(metrics["center_x"] - 65.0) < 1e-6
    assert abs(metrics["center_y"] - 5.25) < 1e-6
    assert abs(metrics["center_z"] - 5.25) < 1e-6
    assert abs(metrics["radius_rms"] - 1.5) < 0.15
    assert "yz_plane_topology" in metrics
    assert "pure_splay_topology" not in metrics


def test_soft_core_center_and_centerline_recover_translated_ring_geometry():
    lengths = (16.0, 8.0, 8.0)
    center = (6.375, 4.125, 3.875)
    radius = 1.5
    q5 = synthetic_S_ring(
        shape=(64, 32, 32),
        lengths=lengths,
        center=center,
        radius=radius,
        S_bulk=0.8,
    )
    soft = soft_core_center(q5, lengths, S_bulk=0.8)
    geometry = core_centerline_geometry(q5, lengths, S_bulk=0.8)

    np.testing.assert_allclose(soft["center"], center, atol=0.04)
    np.testing.assert_allclose(geometry["center"], center, atol=0.06)
    assert abs(geometry["radius_mean"] - radius) < 0.08
    assert abs(geometry["line_length"] - 2.0 * np.pi * radius) < 0.4
    assert abs(geometry["normal"][0]) > 0.99
    assert geometry["ellipticity"] < 0.05
    assert geometry["core_component_count"] == 1


def test_centerline_geometry_is_translation_invariant_and_shape_sensitive():
    shape = (64, 32, 32)
    lengths = (16.0, 8.0, 8.0)
    initial = synthetic_S_ring(
        shape=shape,
        lengths=lengths,
        center=(15.875, 4.125, 4.125),
        radius=1.2,
        S_bulk=0.8,
    )
    translated = np.roll(initial, 9, axis=0)
    deformed = synthetic_S_ring(
        shape=shape,
        lengths=lengths,
        center=(2.125, 4.125, 4.125),
        radius=1.5,
        S_bulk=0.8,
    )
    initial_geometry = core_centerline_geometry(initial, lengths, S_bulk=0.8)
    translated_geometry = core_centerline_geometry(
        translated, lengths, S_bulk=0.8
    )
    deformed_geometry = core_centerline_geometry(deformed, lengths, S_bulk=0.8)

    assert abs(translated_geometry["radius_mean"] - initial_geometry["radius_mean"]) < 1e-5
    assert abs(translated_geometry["line_length"] - initial_geometry["line_length"]) < 1e-5
    assert deformed_geometry["radius_mean"] > 1.2 * initial_geometry["radius_mean"]


def test_periodic_profile_shift_recovers_integer_roll():
    q5 = synthetic_S_ring()
    profile = x_disturbance_profile(q5, S_bulk=0.8)
    shifted = np.roll(profile, -7)
    result = periodic_profile_shift(profile, shifted, period=128.0)
    assert abs(result["shift_indices"] + 7.0) < 1e-10
    assert abs(result["shift_physical"] + 14.0) < 1e-10
    assert result["correlation"] > 1.0 - 1e-12


def test_core_aware_objective_penalizes_loop_annihilation():
    target_values = synthetic_S_ring()
    target = torch.from_numpy(target_values).movedim(-1, 0).unsqueeze(1)
    shifted = torch.roll(target, shifts=1, dims=2).clone().requires_grad_(True)
    uniform = torch.zeros_like(target)
    uniform[0] = 0.8
    uniform[3] = -0.8 / 2.0
    objective = QTrackingObjective(
        target,
        dt=1e-3,
        S_bulk=0.8,
        target_core_q_weight=4.0,
        terminal_core_weight=0.5,
        terminal_core_mass_weight=2.0,
    )

    target_cost = objective.terminal_cost(target)
    shifted_cost = objective.terminal_cost(shifted)
    uniform_cost = objective.terminal_cost(uniform)
    assert target_cost.item() < 1e-12
    assert uniform_cost.item() > shifted_cost.item()
    shifted_cost.backward()
    assert shifted.grad is not None
    assert torch.isfinite(shifted.grad).all()


def test_core_aware_objective_directional_derivative():
    target_values = synthetic_S_ring()
    target = (
        torch.from_numpy(target_values)
        .movedim(-1, 0)
        .unsqueeze(1)
        .to(torch.float64)
    )
    q = torch.roll(target, shifts=1, dims=2).clone().requires_grad_(True)
    objective = QTrackingObjective(
        target,
        dt=1e-3,
        S_bulk=0.8,
        target_core_q_weight=4.0,
        terminal_core_weight=0.5,
        terminal_core_mass_weight=2.0,
    )
    cost = objective.terminal_cost(q)
    gradient = torch.autograd.grad(cost, q)[0]
    direction = torch.linspace(-1.0, 1.0, q.numel(), dtype=q.dtype).reshape_as(q)
    direction = direction / torch.linalg.vector_norm(direction)
    adjoint = torch.sum(gradient * direction).item()
    epsilon = 1e-4
    finite_difference = (
        objective.terminal_cost(q.detach() + epsilon * direction)
        - objective.terminal_cost(q.detach() - epsilon * direction)
    ).item() / (2.0 * epsilon)
    scale = max(abs(adjoint), abs(finite_difference), 1e-14)
    assert abs(adjoint - finite_difference) / scale < 1e-7


def test_translation_objectives_track_periodic_core_center_without_shape_loss():
    lengths = (8.0, 6.0, 6.0)
    target_values = synthetic_S_ring(
        shape=(16, 12, 12),
        lengths=lengths,
        center=(0.75, 3.25, 3.25),
        radius=1.0,
        S_bulk=0.8,
    )
    target = (
        torch.from_numpy(target_values)
        .movedim(-1, 0)
        .unsqueeze(1)
        .to(torch.float64)
    )
    shifted = torch.roll(target, shifts=-2, dims=2).clone().requires_grad_(True)
    common = dict(
        target_q=target,
        dt=1e-3,
        domain_lengths=lengths,
        S_bulk=0.8,
        terminal_weight=1.0,
        terminal_core_mass_weight=10.0,
    )
    with_mass = CoreTranslationObjective(**common)
    without_mass = CoreTranslationNoMassObjective(**common)

    target_components = with_mass.components(target)
    shifted_components = with_mass.components(shifted)
    assert target_components["core_position_penalty"].item() < 1e-12
    assert abs(shifted_components["core_position_penalty"].item() - 1.0) < 2e-2
    assert "q_tracking_mse" not in shifted_components
    assert "core_tracking_mse" not in shifted_components
    torch.testing.assert_close(
        with_mass.terminal_cost(shifted),
        without_mass.terminal_cost(shifted),
    )

    with_mass.terminal_cost(shifted).backward()
    assert shifted.grad is not None
    assert torch.isfinite(shifted.grad).all()


def test_translation_no_mass_ignores_core_mass_weight():
    lengths = (8.0, 6.0, 6.0)
    target_values = synthetic_S_ring(
        shape=(16, 12, 12),
        lengths=lengths,
        center=(4.25, 3.25, 3.25),
        radius=1.0,
        S_bulk=0.8,
    )
    target = torch.from_numpy(target_values).movedim(-1, 0).unsqueeze(1)
    candidate = target * 0.9
    common = dict(
        target_q=target,
        dt=1e-3,
        domain_lengths=lengths,
        S_bulk=0.8,
        terminal_weight=1.0,
        terminal_core_mass_weight=10.0,
    )
    with_mass = CoreTranslationObjective(**common)
    without_mass = CoreTranslationNoMassObjective(**common)
    assert with_mass.components(candidate)["core_mass_penalty"].item() > 0.0
    assert with_mass.terminal_cost(candidate) > without_mass.terminal_cost(candidate)


def test_core_moment_penalty_is_translation_invariant_and_shape_sensitive():
    shape = (32, 16, 16)
    lengths = (8.0, 6.0, 6.0)
    target_values = synthetic_S_ring(
        shape=shape,
        lengths=lengths,
        center=(2.125, 3.1875, 3.1875),
        radius=1.0,
        S_bulk=0.8,
    )
    deformed_values = synthetic_S_ring(
        shape=shape,
        lengths=lengths,
        center=(2.125, 3.1875, 3.1875),
        radius=1.4,
        S_bulk=0.8,
    )
    target = torch.from_numpy(target_values).movedim(-1, 0).unsqueeze(1)
    translated = torch.roll(target, shifts=7, dims=2)
    deformed = torch.from_numpy(deformed_values).movedim(-1, 0).unsqueeze(1)
    objective = CoreAwareMomentObjective(
        target_q=target,
        dt=1e-3,
        domain_lengths=lengths,
        S_bulk=0.8,
        terminal_moment_weight=1.0,
    )
    translated_penalty = objective.components(translated)["core_moment_penalty"]
    deformed_penalty = objective.components(deformed)["core_moment_penalty"]
    assert translated_penalty.item() < 1e-10
    assert deformed_penalty.item() > 0.1


def test_core_moment_objective_directional_derivative():
    lengths = (8.0, 6.0, 6.0)
    target_values = synthetic_S_ring(
        shape=(16, 12, 12),
        lengths=lengths,
        center=(4.25, 3.25, 3.25),
        radius=1.0,
        S_bulk=0.8,
    )
    target = torch.from_numpy(target_values).movedim(-1, 0).unsqueeze(1).double()
    q = (target * 0.97).detach().requires_grad_(True)
    objective = CoreAwareMomentObjective(
        target_q=target,
        dt=1e-3,
        domain_lengths=lengths,
        S_bulk=0.8,
        terminal_moment_weight=1.0,
    )
    cost = objective.terminal_cost(q)
    gradient = torch.autograd.grad(cost, q)[0]
    direction = torch.linspace(-1.0, 1.0, q.numel(), dtype=q.dtype).reshape_as(q)
    direction = direction / torch.linalg.vector_norm(direction)
    adjoint = torch.sum(gradient * direction).item()
    epsilon = 1e-5
    finite_difference = (
        objective.terminal_cost(q.detach() + epsilon * direction)
        - objective.terminal_cost(q.detach() - epsilon * direction)
    ).item() / (2.0 * epsilon)
    scale = max(abs(adjoint), abs(finite_difference), 1e-14)
    assert abs(adjoint - finite_difference) / scale < 1e-3


def test_aligned_core_shape_penalty_removes_translation_but_detects_shape_change():
    shape = (32, 16, 16)
    lengths = (8.0, 6.0, 6.0)
    target_values = synthetic_S_ring(
        shape=shape,
        lengths=lengths,
        center=(2.125, 3.1875, 3.1875),
        radius=1.0,
        S_bulk=0.8,
    )
    deformed_values = synthetic_S_ring(
        shape=shape,
        lengths=lengths,
        center=(2.125, 3.1875, 3.1875),
        radius=1.4,
        S_bulk=0.8,
    )
    target = torch.from_numpy(target_values).movedim(-1, 0).unsqueeze(1)
    translated = torch.roll(target, shifts=7, dims=2)
    deformed = torch.from_numpy(deformed_values).movedim(-1, 0).unsqueeze(1)
    objective = CoreAwareAlignedShapeObjective(
        target_q=target,
        dt=1e-3,
        domain_lengths=lengths,
        S_bulk=0.8,
        terminal_aligned_shape_weight=1.0,
    )
    translated_penalty = objective.components(translated)[
        "aligned_core_shape_penalty"
    ]
    deformed_penalty = objective.components(deformed)["aligned_core_shape_penalty"]
    assert translated_penalty.item() < 1e-10
    assert deformed_penalty.item() > 0.1


def test_aligned_core_shape_objective_directional_derivative():
    lengths = (8.0, 6.0, 6.0)
    target_values = synthetic_S_ring(
        shape=(16, 12, 12),
        lengths=lengths,
        center=(4.25, 3.25, 3.25),
        radius=1.0,
        S_bulk=0.8,
    )
    target = torch.from_numpy(target_values).movedim(-1, 0).unsqueeze(1).double()
    q = (target * 0.97).detach().requires_grad_(True)
    objective = CoreAwareAlignedShapeObjective(
        target_q=target,
        dt=1e-3,
        domain_lengths=lengths,
        S_bulk=0.8,
        terminal_aligned_shape_weight=1.0,
    )
    cost = objective.terminal_cost(q)
    gradient = torch.autograd.grad(cost, q)[0]
    direction = torch.linspace(-1.0, 1.0, q.numel(), dtype=q.dtype).reshape_as(q)
    direction = direction / torch.linalg.vector_norm(direction)
    adjoint = torch.sum(gradient * direction).item()
    epsilon = 1e-5
    finite_difference = (
        objective.terminal_cost(q.detach() + epsilon * direction)
        - objective.terminal_cost(q.detach() - epsilon * direction)
    ).item() / (2.0 * epsilon)
    scale = max(abs(adjoint), abs(finite_difference), 1e-14)
    assert abs(adjoint - finite_difference) / scale < 2e-3


def test_comoving_q_trajectory_separates_translation_from_shape():
    shape = (32, 16, 16)
    lengths = (8.0, 6.0, 6.0)
    initial_values = synthetic_S_ring(
        shape=shape,
        lengths=lengths,
        center=(2.125, 3.1875, 3.1875),
        radius=1.0,
        S_bulk=0.8,
    )
    deformed_values = synthetic_S_ring(
        shape=shape,
        lengths=lengths,
        center=(3.875, 3.1875, 3.1875),
        radius=1.4,
        S_bulk=0.8,
    )
    initial = torch.from_numpy(initial_values).movedim(-1, 0).unsqueeze(1)
    target = torch.roll(initial, shifts=7, dims=2)
    deformed = torch.from_numpy(deformed_values).movedim(-1, 0).unsqueeze(1)
    objective = CoreAwareComovingQTrajectoryObjective(
        target_q=target,
        initial_q=initial,
        num_steps=10,
        dt=1e-3,
        domain_lengths=lengths,
        S_bulk=0.8,
        target_core_q_weight=4.0,
    )

    initial_components = objective.components(initial, step=0)
    target_components = objective.components(target, step=10)
    deformed_components = objective.components(deformed, step=10)
    assert initial_components["trajectory_position_penalty"].item() < 1e-10
    assert target_components["trajectory_position_penalty"].item() < 1e-10
    assert target_components["comoving_q_shape_penalty"].item() < 1e-9
    assert deformed_components["comoving_q_shape_penalty"].item() > 0.05


def test_comoving_q_trajectory_directional_derivative():
    lengths = (8.0, 6.0, 6.0)
    initial_values = synthetic_S_ring(
        shape=(16, 12, 12),
        lengths=lengths,
        center=(3.75, 3.25, 3.25),
        radius=1.0,
        S_bulk=0.8,
    )
    initial = (
        torch.from_numpy(initial_values).movedim(-1, 0).unsqueeze(1).double()
    )
    target = torch.roll(initial, shifts=2, dims=2)
    q = (torch.roll(initial, shifts=1, dims=2) * 0.99).requires_grad_(True)
    objective = CoreAwareComovingQTrajectoryObjective(
        target_q=target,
        initial_q=initial,
        num_steps=10,
        dt=1e-3,
        domain_lengths=lengths,
        S_bulk=0.8,
        target_core_q_weight=4.0,
        terminal_weight=1.0,
        terminal_comoving_q_weight=1.0,
    )
    cost = objective.terminal_cost(q)
    gradient = torch.autograd.grad(cost, q)[0]
    direction = torch.linspace(-1.0, 1.0, q.numel(), dtype=q.dtype).reshape_as(q)
    direction = direction / torch.linalg.vector_norm(direction)
    adjoint = torch.sum(gradient * direction).item()
    epsilon = 1e-5
    finite_difference = (
        objective.terminal_cost(q.detach() + epsilon * direction)
        - objective.terminal_cost(q.detach() - epsilon * direction)
    ).item() / (2.0 * epsilon)
    scale = max(abs(adjoint), abs(finite_difference), 1e-14)
    assert abs(adjoint - finite_difference) / scale < 3e-3


def test_free_path_loss_uses_terminal_endpoint_and_3d_translation_invariant_shape():
    shape = (32, 24, 24)
    lengths = (8.0, 6.0, 6.0)
    initial_values = synthetic_S_ring(
        shape=shape,
        lengths=lengths,
        center=(2.125, 3.125, 3.125),
        radius=1.0,
        S_bulk=0.8,
    )
    initial = torch.from_numpy(initial_values).movedim(-1, 0).unsqueeze(1)
    target = torch.roll(initial, shifts=(-4, 2, -2), dims=(2, 3, 4))
    objective = FreePathShapePreservingObjective(
        target_q=target,
        initial_q=initial,
        dt=1e-3,
        domain_lengths=lengths,
        S_bulk=0.8,
        target_core_q_weight=4.0,
    )
    initial_components = objective.components(initial)
    target_components = objective.components(target)

    assert initial_components["endpoint_position_penalty"].item() > 1.0
    assert target_components["endpoint_position_penalty"].item() < 1e-8
    assert target_components["core_mass_penalty"].item() < 1e-8
    assert target_components["core_moment_penalty"].item() < 1e-6
    assert target_components["comoving_q_shape_penalty"].item() < 1e-5
    assert objective.stage_cost(initial, torch.zeros(1, *shape), 0).item() < 1e-8


def test_free_path_loss_detects_shape_change_and_has_finite_gradient():
    shape = (24, 16, 16)
    lengths = (8.0, 6.0, 6.0)
    initial_values = synthetic_S_ring(
        shape=shape,
        lengths=lengths,
        center=(3.5, 3.1875, 3.1875),
        radius=1.0,
        S_bulk=0.8,
    )
    deformed_values = synthetic_S_ring(
        shape=shape,
        lengths=lengths,
        center=(3.5, 3.1875, 3.1875),
        radius=1.3,
        S_bulk=0.8,
    )
    initial = torch.from_numpy(initial_values).movedim(-1, 0).unsqueeze(1).double()
    target = torch.roll(initial, shifts=-3, dims=2)
    deformed = (
        torch.from_numpy(deformed_values)
        .movedim(-1, 0)
        .unsqueeze(1)
        .double()
        .requires_grad_(True)
    )
    objective = FreePathShapePreservingObjective(
        target_q=target,
        initial_q=initial,
        dt=1e-3,
        domain_lengths=lengths,
        S_bulk=0.8,
        target_core_q_weight=4.0,
    )
    components = objective.components(deformed)
    assert components["core_moment_penalty"].item() > 1.0
    assert components["comoving_q_shape_penalty"].item() > 1.0
    cost = objective.terminal_cost(deformed)
    cost.backward()
    assert torch.isfinite(deformed.grad).all()


def test_free_path_core_center_is_independent_of_asymmetric_control_mask():
    shape = (32, 16, 16)
    lengths = (8.0, 6.0, 6.0)
    initial_values = synthetic_S_ring(
        shape=shape,
        lengths=lengths,
        center=(3.125, 3.1875, 3.1875),
        radius=1.0,
        S_bulk=0.8,
    )
    initial = torch.from_numpy(initial_values).movedim(-1, 0).unsqueeze(1)
    target = torch.roll(initial, shifts=-4, dims=2)
    asymmetric_mask = torch.zeros(shape)
    asymmetric_mask[8:24, 2:14, 2:14] = 1.0
    objective = FreePathShapePreservingObjective(
        target_q=target,
        initial_q=initial,
        dt=1e-3,
        domain_lengths=lengths,
        spatial_mask=asymmetric_mask,
        S_bulk=0.8,
        target_core_q_weight=4.0,
    )
    target_components = objective.components(target)
    assert target_components["endpoint_position_penalty"].item() < 1e-8
    assert target_components["core_mass_penalty"].item() < 1e-8
    assert target_components["core_moment_penalty"].item() < 1e-6
