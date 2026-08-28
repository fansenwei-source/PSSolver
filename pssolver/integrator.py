class TimeIntegrator:
    def __init__(self, dt, qx, qy, q2):
        self.dt = dt
        self.qx = qx
        self.qy = qy
        self.q2 = q2

    def step(self, pre_update_callback=None):
        raise NotImplementedError("Implement in subclass")


class SemiImplicitEulerIntegrator(TimeIntegrator):
    DEFAULT_SPECTRAL_REFRESH_INTERVAL = 20

    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(dt, qx, qy, q2)
        self.model = model
        self.denom = 1 - model.fields.L_hat *self.dt # cache for denominator in the update step = (1 - L_hat*dt)
        self.dyn_count = self.model.fields.dyn_count
        self.stat_count = self.model.fields.stat_count
        self.dynamic_transform_groups = self.model.fields.group_indices_by_boundary_conditions(
            range(self.dyn_count)
        )

        self._spectral_refresh_interval = self.DEFAULT_SPECTRAL_REFRESH_INTERVAL
        self.step_count = 0
        self.refresh_count = 0
        self._static_fields_are_current = False

    @property
    def spectral_refresh_interval(self):
        """Number of steps between spectral rebuilds, or ``None`` to disable them."""
        return self._spectral_refresh_interval

    @spectral_refresh_interval.setter
    def spectral_refresh_interval(self, interval):
        if interval is None:
            self._spectral_refresh_interval = None
            return
        if not isinstance(interval, int) or isinstance(interval, bool):
            raise TypeError("spectral_refresh_interval must be a positive integer or None")
        if interval <= 0:
            raise ValueError("spectral_refresh_interval must be positive")
        self._spectral_refresh_interval = interval

    def set_spectral_refresh_interval(self, interval):
        """Configure periodic spectral refreshes using a step interval or ``None``."""
        self.spectral_refresh_interval = interval

    def restore_progress(self, completed_steps, *, static_fields_are_current=False):
        """Restore counters and optionally reuse restored static fields once."""
        if not isinstance(completed_steps, int) or isinstance(completed_steps, bool):
            raise TypeError("completed_steps must be an integer")
        if completed_steps < 0:
            raise ValueError("completed_steps must be non-negative")
        if self.spectral_refresh_interval is None:
            self.step_count = completed_steps
            self.refresh_count = 0
        else:
            self.refresh_count, self.step_count = divmod(
                completed_steps,
                self.spectral_refresh_interval,
            )
        self._static_fields_are_current = bool(static_fields_are_current)

    def _refresh_dynamic_spectra(self):
        """Rebuild dynamic spectra from spatial fields.

        Subclasses may override this hook when a refresh also needs to enforce a
        constraint, for example a spectral projection.
        """
        for group in self.dynamic_transform_groups:
            self.model.fields.spectral[group] = (
                self.model.fields.forward_transform_group(group)
            )

    def _advance_spectral_refresh_clock(self):
        """Advance the refresh phase and perform a scheduled refresh if needed."""
        self.step_count += 1
        interval = self.spectral_refresh_interval
        if interval is None or self.step_count < interval:
            return False

        self._refresh_dynamic_spectra()
        self.step_count = 0
        self.refresh_count += 1
        return True

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

        # Periodically rebuild dynamic spectra from real fields to limit accumulated roundoff drift.
        self._advance_spectral_refresh_clock()
