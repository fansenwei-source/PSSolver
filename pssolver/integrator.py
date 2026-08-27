class TimeIntegrator:
    def __init__(self, dt, qx, qy, q2):
        self.dt = dt
        self.qx = qx
        self.qy = qy
        self.q2 = q2

    def step(self, pre_update_callback=None):
        raise NotImplementedError("Implement in subclass")


class SemiImplicitEulerIntegrator(TimeIntegrator):
    spectral_refresh_interval = 20

    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(dt, qx, qy, q2)
        self.model = model
        self.denom = 1 - model.fields.L_hat *self.dt # cache for denominator in the update step = (1 - L_hat*dt)
        self.dyn_count = self.model.fields.dyn_count
        self.stat_count = self.model.fields.stat_count
        self.dynamic_transform_groups = self.model.fields.group_indices_by_boundary_conditions(
            range(self.dyn_count)
        )

        self.step_count = 0
        self._static_fields_are_current = False

    def restore_progress(self, completed_steps, *, static_fields_are_current=False):
        """Restore counters and optionally reuse restored static fields once."""
        if not isinstance(completed_steps, int) or isinstance(completed_steps, bool):
            raise TypeError("completed_steps must be an integer")
        if completed_steps < 0:
            raise ValueError("completed_steps must be non-negative")
        self.step_count = completed_steps % self.spectral_refresh_interval
        self._static_fields_are_current = bool(static_fields_are_current)

    def step(self, pre_update_callback=None):
        # 1. static(Q^n): u^n, E^n, Omega^n, gradQ^n
        if self._static_fields_are_current:
            self._static_fields_are_current = False
        else:
            self.model.update_static_fields()

        if pre_update_callback is not None:
            pre_update_callback()

        # 2. N(Q^n, u^n, E^n, Omega^n)
        N_hats = self.model.compute_nonlinear()

        # 3. IMEX 更新 Q 到 Q^{n+1}
        dyn_fields = self.model.fields.spectral[:self.dyn_count]
        dyn_fields.add_(self.dt * N_hats)
        dyn_fields.div_(self.denom)

        # 4. 只更新 dynamic fields 的实空间 Q^{n+1}
        for group in self.dynamic_transform_groups:
            self.model.fields.spatial[group] = self.model.fields.inverse_transform_group(group)

        self.step_count += 1

        # Periodically rebuild dynamic spectra from real fields to limit accumulated roundoff drift.
        if self.step_count % self.spectral_refresh_interval == 0:
            for group in self.dynamic_transform_groups:
                self.model.fields.spectral[group] = self.model.fields.forward_transform_group(group)
            self.step_count = 0
