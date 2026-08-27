from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ForwardRecord:
    cost: float
    checkpoints: dict[int, torch.Tensor]
    boundaries: list[int]


@dataclass
class DALResult:
    cost_before: float
    cost_after: float
    gradient_norm: float
    step_size: float
    accepted: bool


class DiscreteAdjointLoop:
    """Checkpointed direct-adjoint loop for a functional PSSolver step."""

    def __init__(
        self,
        stepper,
        objective,
        control,
        num_steps,
        checkpoint_stride=20,
        checkpoint_device="cpu",
        temporal_control_weight=0.0,
        spatial_control_weight=0.0,
    ):
        if num_steps != control.num_steps:
            raise ValueError("num_steps must match the control horizon.")
        if checkpoint_stride <= 0:
            raise ValueError("checkpoint_stride must be positive.")
        self.stepper = stepper
        self.objective = objective
        self.control = control.to(stepper.device)
        self.num_steps = int(num_steps)
        self.checkpoint_stride = int(checkpoint_stride)
        self.checkpoint_device = torch.device(checkpoint_device)
        self.temporal_control_weight = float(temporal_control_weight)
        self.spatial_control_weight = float(spatial_control_weight)
        self.last_accepted_record = None
        self._lbfgs_s_history = []
        self._lbfgs_y_history = []
        self._lbfgs_previous_logits = None
        self._lbfgs_previous_gradient = None

    def _checkpoint(self, q):
        return q.detach().to(self.checkpoint_device).clone()

    def forward(self, initial_q):
        q = initial_q.detach().to(self.stepper.device)
        checkpoints = {0: self._checkpoint(q)}
        cost = 0.0

        with torch.no_grad():
            for step in range(self.num_steps):
                alpha = self.control.field_for_step(step).detach()
                cost += float(self.objective.stage_cost(q, alpha, step).item())
                q = self.stepper(q, alpha)
                index = step + 1
                if index % self.checkpoint_stride == 0 or index == self.num_steps:
                    checkpoints[index] = self._checkpoint(q)

            cost += float(self.objective.terminal_cost(q).item())
            cost += float(
                self.control.temporal_regularization(self.temporal_control_weight).item()
            )
            cost += float(
                self.control.spatial_regularization(self.spatial_control_weight).item()
            )

        boundaries = sorted(checkpoints)
        return ForwardRecord(cost=cost, checkpoints=checkpoints, boundaries=boundaries)

    def _recompute_segment(self, start, end, start_q):
        q = start_q.detach().to(self.stepper.device)
        states = [self._checkpoint(q)]
        with torch.no_grad():
            for step in range(start, end):
                alpha = self.control.field_for_step(step).detach()
                q = self.stepper(q, alpha)
                states.append(self._checkpoint(q))
        return states

    def gradient(self, initial_q, forward_record=None):
        if forward_record is None:
            forward_record = self.forward(initial_q)

        final_q = forward_record.checkpoints[self.num_steps].to(self.stepper.device)
        final_q = final_q.detach().requires_grad_(True)
        with torch.enable_grad():
            terminal_cost = self.objective.terminal_cost(final_q)
            adjoint = torch.autograd.grad(terminal_cost, final_q)[0].detach()

        gradient_logits = torch.zeros_like(self.control.logits)
        boundaries = forward_record.boundaries

        for boundary_index in range(len(boundaries) - 1, 0, -1):
            start = boundaries[boundary_index - 1]
            end = boundaries[boundary_index]
            segment_states = self._recompute_segment(
                start,
                end,
                forward_record.checkpoints[start],
            )

            for step in range(end - 1, start - 1, -1):
                local_index = step - start
                q = segment_states[local_index].to(self.stepper.device)
                q = q.detach().requires_grad_(True)
                alpha = self.control.field_for_step(step).detach().requires_grad_(True)

                with torch.enable_grad():
                    q_next = self.stepper(q, alpha)
                    stage_cost = self.objective.stage_cost(q, alpha, step)
                    reverse_scalar = stage_cost + torch.sum(q_next * adjoint)
                    adjoint, gradient_alpha = torch.autograd.grad(
                        reverse_scalar,
                        (q, alpha),
                        allow_unused=False,
                    )
                    adjoint = adjoint.detach()
                    gradient_alpha = gradient_alpha.detach()

                    alpha_from_control = self.control.field_for_step(step)
                    alpha_pullback = torch.sum(alpha_from_control * gradient_alpha)
                    step_gradient = torch.autograd.grad(
                        alpha_pullback,
                        self.control.logits,
                    )[0]
                gradient_logits.add_(step_gradient.detach())

        if self.temporal_control_weight or self.spatial_control_weight:
            with torch.enable_grad():
                regularization_cost = self.control.temporal_regularization(
                    self.temporal_control_weight
                )
                regularization_cost = (
                    regularization_cost
                    + self.control.spatial_regularization(self.spatial_control_weight)
                )
                regularization_gradient = torch.autograd.grad(
                    regularization_cost,
                    self.control.logits,
                )[0]
            gradient_logits.add_(regularization_gradient.detach())

        return gradient_logits

    def armijo_step(
        self,
        initial_q,
        initial_step=1.0,
        contraction=0.5,
        sufficient_decrease=1e-4,
        max_trials=8,
        normalize_direction=True,
        direction_method="steepest_descent",
        lbfgs_history_size=10,
    ):
        base_record = self.forward(initial_q)
        gradient = self.gradient(initial_q, base_record)
        gradient_norm = float(torch.linalg.vector_norm(gradient).item())
        if not torch.isfinite(gradient).all() or gradient_norm == 0.0:
            self.last_accepted_record = base_record
            return DALResult(
                cost_before=base_record.cost,
                cost_after=base_record.cost,
                gradient_norm=gradient_norm,
                step_size=0.0,
                accepted=False,
            )

        original = self.control.logits.detach().clone()
        if direction_method == "steepest_descent":
            direction = -gradient
        elif direction_method == "lbfgs":
            direction = self._lbfgs_direction(
                original, gradient, history_size=lbfgs_history_size
            )
        else:
            raise ValueError(
                "direction_method must be 'steepest_descent' or 'lbfgs'."
            )
        if normalize_direction:
            direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-30)
        directional_derivative = float(torch.sum(gradient * direction).item())
        step_size = float(initial_step)

        for _ in range(max_trials):
            with torch.no_grad():
                self.control.logits.copy_(original + step_size * direction)
            trial_record = self.forward(initial_q)
            threshold = (
                base_record.cost
                + sufficient_decrease * step_size * directional_derivative
            )
            if trial_record.cost <= threshold:
                self.last_accepted_record = trial_record
                return DALResult(
                    cost_before=base_record.cost,
                    cost_after=trial_record.cost,
                    gradient_norm=gradient_norm,
                    step_size=step_size,
                    accepted=True,
                )
            step_size *= contraction

        with torch.no_grad():
            self.control.logits.copy_(original)
        self.last_accepted_record = base_record
        return DALResult(
            cost_before=base_record.cost,
            cost_after=base_record.cost,
            gradient_norm=gradient_norm,
            step_size=0.0,
            accepted=False,
        )

    def _lbfgs_direction(self, logits, gradient, *, history_size):
        if history_size <= 0:
            raise ValueError("lbfgs_history_size must be positive.")
        if self._lbfgs_previous_logits is not None:
            s = logits - self._lbfgs_previous_logits
            y = gradient - self._lbfgs_previous_gradient
            curvature = float(torch.sum(s * y).item())
            scale = float(torch.linalg.vector_norm(s) * torch.linalg.vector_norm(y))
            if curvature > 1e-10 * max(scale, 1e-30):
                self._lbfgs_s_history.append(s.detach().clone())
                self._lbfgs_y_history.append(y.detach().clone())
                self._lbfgs_s_history = self._lbfgs_s_history[-history_size:]
                self._lbfgs_y_history = self._lbfgs_y_history[-history_size:]
        self._lbfgs_previous_logits = logits.detach().clone()
        self._lbfgs_previous_gradient = gradient.detach().clone()

        value = gradient.detach().clone()
        coefficients = []
        inverse_curvatures = []
        for s, y in zip(
            reversed(self._lbfgs_s_history), reversed(self._lbfgs_y_history)
        ):
            inverse_curvature = 1.0 / torch.sum(s * y).clamp_min(1e-30)
            coefficient = inverse_curvature * torch.sum(s * value)
            value = value - coefficient * y
            coefficients.append(coefficient)
            inverse_curvatures.append(inverse_curvature)
        if self._lbfgs_s_history:
            last_s = self._lbfgs_s_history[-1]
            last_y = self._lbfgs_y_history[-1]
            initial_scale = torch.sum(last_s * last_y) / torch.sum(last_y * last_y).clamp_min(
                1e-30
            )
            value = initial_scale * value
        for s, y, coefficient, inverse_curvature in zip(
            self._lbfgs_s_history,
            self._lbfgs_y_history,
            reversed(coefficients),
            reversed(inverse_curvatures),
        ):
            beta = inverse_curvature * torch.sum(y * value)
            value = value + s * (coefficient - beta)
        direction = -value
        if float(torch.sum(direction * gradient).item()) >= 0.0:
            self._lbfgs_s_history.clear()
            self._lbfgs_y_history.clear()
            direction = -gradient
        return direction

    def directional_derivative_check(self, initial_q, epsilon=1e-3, direction=None):
        record = self.forward(initial_q)
        gradient = self.gradient(initial_q, record)
        if isinstance(direction, str) and direction == "gradient":
            direction = gradient
        elif direction is None:
            direction = torch.randn_like(self.control.logits)
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-30)
        adjoint_derivative = float(torch.sum(gradient * direction).item())

        original = self.control.logits.detach().clone()
        with torch.no_grad():
            self.control.logits.copy_(original + epsilon * direction)
        cost_plus = self.forward(initial_q).cost
        with torch.no_grad():
            self.control.logits.copy_(original - epsilon * direction)
        cost_minus = self.forward(initial_q).cost
        with torch.no_grad():
            self.control.logits.copy_(original)

        finite_difference = (cost_plus - cost_minus) / (2.0 * epsilon)
        scale = max(abs(adjoint_derivative), abs(finite_difference), 1e-12)
        relative_error = abs(adjoint_derivative - finite_difference) / scale
        return {
            "cost": record.cost,
            "adjoint": adjoint_derivative,
            "finite_difference": finite_difference,
            "relative_error": relative_error,
        }

    def gradient_check_suite(
        self,
        initial_q,
        *,
        epsilons=(0.1, 0.05, 0.025),
        num_random_directions=5,
        seed=0,
        include_gradient_direction=True,
    ):
        """Check one adjoint gradient against central differences in many directions."""

        epsilons = tuple(float(value) for value in epsilons)
        if not epsilons or any(value <= 0 for value in epsilons):
            raise ValueError("epsilons must contain positive values.")
        if num_random_directions < 0:
            raise ValueError("num_random_directions must be non-negative.")
        record = self.forward(initial_q)
        gradient = self.gradient(initial_q, record)
        generator = torch.Generator(device=self.control.logits.device)
        generator.manual_seed(int(seed))
        directions = []
        if include_gradient_direction:
            directions.append(("gradient", gradient.detach().clone()))
        for index in range(num_random_directions):
            directions.append(
                (
                    f"random_{index}",
                    torch.randn(
                        self.control.logits.shape,
                        generator=generator,
                        device=self.control.logits.device,
                        dtype=self.control.logits.dtype,
                    ),
                )
            )

        original = self.control.logits.detach().clone()
        results = []
        try:
            for name, direction in directions:
                direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-30)
                adjoint = float(torch.sum(gradient * direction).item())
                checks = []
                for epsilon in epsilons:
                    with torch.no_grad():
                        self.control.logits.copy_(original + epsilon * direction)
                    cost_plus = self.forward(initial_q).cost
                    with torch.no_grad():
                        self.control.logits.copy_(original - epsilon * direction)
                    cost_minus = self.forward(initial_q).cost
                    finite_difference = (cost_plus - cost_minus) / (2.0 * epsilon)
                    scale = max(abs(adjoint), abs(finite_difference), 1e-12)
                    checks.append(
                        {
                            "epsilon": epsilon,
                            "adjoint": adjoint,
                            "finite_difference": finite_difference,
                            "absolute_error": abs(adjoint - finite_difference),
                            "relative_error": abs(adjoint - finite_difference) / scale,
                        }
                    )
                results.append({"direction": name, "checks": checks})
        finally:
            with torch.no_grad():
                self.control.logits.copy_(original)
        relative_errors = [
            check["relative_error"] for result in results for check in result["checks"]
        ]
        return {
            "cost": record.cost,
            "gradient_norm": float(torch.linalg.vector_norm(gradient).item()),
            "directions": results,
            "max_relative_error": max(relative_errors, default=0.0),
            "min_relative_error": min(relative_errors, default=0.0),
        }

    def taylor_test(self, initial_q, epsilons=(0.1, 0.05, 0.025), direction=None):
        """Return first- and second-order Taylor residuals for the DAL gradient."""

        record = self.forward(initial_q)
        gradient = self.gradient(initial_q, record)
        if isinstance(direction, str) and direction == "gradient":
            direction = gradient
        elif direction is None:
            direction = torch.randn_like(self.control.logits)
        direction = direction / torch.linalg.vector_norm(direction).clamp_min(1e-30)
        directional_derivative = float(torch.sum(gradient * direction).item())
        original = self.control.logits.detach().clone()
        results = []

        try:
            for epsilon in epsilons:
                epsilon = float(epsilon)
                with torch.no_grad():
                    self.control.logits.copy_(original + epsilon * direction)
                perturbed_cost = self.forward(initial_q).cost
                cost_change = perturbed_cost - record.cost
                results.append(
                    {
                        "epsilon": epsilon,
                        "first_order_residual": abs(cost_change),
                        "second_order_residual": abs(
                            cost_change - epsilon * directional_derivative
                        ),
                    }
                )
        finally:
            with torch.no_grad():
                self.control.logits.copy_(original)

        return results
