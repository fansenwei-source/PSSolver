from __future__ import annotations

import torch


class FunctionalSemiImplicitStep:
    """Differentiable one-step view of a built PSSolver model.

    The production integrator updates its field buffers in place. DAL needs a
    mathematical map ``q_next = F(q, alpha)`` so that one-step vector-Jacobian
    products can be evaluated without retaining a full time-history graph.
    This adapter reuses the model, transforms, and static solver while replacing
    the field buffers with tensors belonging to the current one-step graph.
    """

    def __init__(self, solver, control_parameter: str = "alpha", deterministic: bool = True):
        if not hasattr(solver, "integrator"):
            raise RuntimeError("Build the solver before constructing a functional step.")

        self.solver = solver
        self.model = solver.model
        self.fields = solver.model.fields
        self.control_parameter = control_parameter
        self.deterministic = deterministic
        self.dyn_count = self.fields.dyn_count
        self.stat_count = self.fields.stat_count
        self.dt = solver.dt
        self.denom = solver.integrator.denom

        if self.dyn_count == 0:
            raise ValueError("The solver has no dynamic fields.")
        if control_parameter not in self.model.parameters.keys():
            raise KeyError(f"Model parameter '{control_parameter}' does not exist.")

    @property
    def device(self):
        return self.fields.spatial.device

    @property
    def dynamic_shape(self):
        return (self.dyn_count, self.solver.batchsize, *self.solver.shape)

    def current_state(self, detach: bool = True):
        state = self.fields.spatial[: self.dyn_count]
        return state.detach().clone() if detach else state.clone()

    def _clear_static_warm_start(self):
        if not self.deterministic:
            return
        static_model = self.model.static_model
        if hasattr(static_model, "pressure_guess"):
            static_model.pressure_guess = None

    def _dynamic_forward_transform(self, q):
        return torch.stack(
            [self.fields.forward_transform(index, tensor=q[index]) for index in range(self.dyn_count)]
        )

    def _dynamic_inverse_transform(self, q_hat):
        return torch.stack(
            [self.fields.inverse_transform(index, spectral=q_hat[index]) for index in range(self.dyn_count)]
        )

    def _static_inverse_transform(self, static_hat):
        return torch.stack(
            [
                self.fields.inverse_transform(self.dyn_count + index, spectral=static_hat[index])
                for index in range(self.stat_count)
            ]
        )

    def __call__(self, q, alpha):
        if q.shape != self.dynamic_shape:
            raise ValueError(f"Expected q shape {self.dynamic_shape}, got {tuple(q.shape)}.")
        expected_alpha = (self.solver.batchsize, *self.solver.shape)
        if alpha.shape != expected_alpha:
            raise ValueError(f"Expected alpha shape {expected_alpha}, got {tuple(alpha.shape)}.")

        q = q.to(device=self.device)
        alpha = alpha.to(device=self.device, dtype=q.dtype)
        self.model.parameters[self.control_parameter] = alpha
        self._clear_static_warm_start()

        q_hat = self._dynamic_forward_transform(q)
        if self.stat_count:
            static_spatial_seed = torch.zeros(
                (self.stat_count, self.solver.batchsize, *self.solver.shape),
                device=q.device,
                dtype=q.dtype,
            )
            static_spectral_seed = torch.zeros(
                (self.stat_count, self.solver.batchsize, *self.solver.shape),
                device=q_hat.device,
                dtype=q_hat.dtype,
            )
            self.fields.spatial = torch.cat((q, static_spatial_seed), dim=0)
            self.fields.spectral = torch.cat((q_hat, static_spectral_seed), dim=0)
            static_hat = self.model.compute_static()
            static_spatial = self._static_inverse_transform(static_hat)
            self.fields.spatial = torch.cat((q, static_spatial), dim=0)
            self.fields.spectral = torch.cat((q_hat, static_hat), dim=0)
        else:
            self.fields.spatial = q
            self.fields.spectral = q_hat

        nonlinear_hat = self.model.compute_nonlinear()
        q_next_hat = (q_hat + self.dt * nonlinear_hat) / self.denom
        return self._dynamic_inverse_transform(q_next_hat)
