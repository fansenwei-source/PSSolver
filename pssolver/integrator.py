
import torch


class TimeIntegrator:
    def __init__(self, dt, qx, qy, q2):
        self.dt = dt
        self.qx = qx
        self.qy = qy
        self.q2 = q2

    def step(self):
        raise NotImplementedError("Implement in subclass")


class SemiImplicitEulerIntegrator(TimeIntegrator):
    def __init__(self, model, dt, qx, qy, q2):
        super().__init__(dt, qx, qy, q2)
        self.model = model
        self.denom = 1 - model.fields.L_hat *self.dt # cache for denominator in the update step = (1 - L_hat*dt)
        self.dyn_count = self.model.fields.dyn_count
        self.stat_count = self.model.fields.stat_count

        self.step_count = 0

    def step(self):
        # 1. static(Q^n): u^n, E^n, Omega^n, gradQ^n
        if self.stat_count != 0:
            S_hats = self.model.compute_static()
            for offset in range(self.stat_count):
                field_idx = self.dyn_count + offset
                self.model.fields.spectral[field_idx] = S_hats[offset]
                self.model.fields.spatial[field_idx] = self.model.fields.inverse_transform(
                    field_idx,
                    spectral=S_hats[offset],
                )

        # 2. N(Q^n, u^n, E^n, Omega^n)
        N_hats = self.model.compute_nonlinear()

        # 3. IMEX 更新 Q 到 Q^{n+1}
        dyn_fields = self.model.fields.spectral[:self.dyn_count]
        dyn_fields.add_(self.dt * N_hats)
        dyn_fields.div_(self.denom)

        # 4. 只更新 dynamic fields 的实空间 Q^{n+1}
        for field_idx in range(self.dyn_count):
            self.model.fields.spatial[field_idx] = self.model.fields.inverse_transform(field_idx)

        self.step_count += 1

        # Periodically rebuild dynamic spectra from real fields to limit accumulated roundoff drift.
        if self.step_count % 20 == 0:
            for field_idx in range(self.dyn_count):
                self.model.fields.spectral[field_idx] = self.model.fields.forward_transform(field_idx)
            self.step_count = 0
