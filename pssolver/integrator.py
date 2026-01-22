
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
        self.dyn_names = self.model.fields.dyn_names
        self.stat_names = self.model.fields.stat_names
        if isinstance(model.fields.L_hat, dict):
            self.denom = {name: 1 - model.fields.L_hat[name] * self.dt for name in self.dyn_names}
        else:
            self.denom = 1 - model.fields.L_hat * self.dt

        self.step_count = 0

    def step(self):
        N_hats = self.model.compute_nonlinear() 

        # Use in-place operations to reduce memory allocations and improve speed
        if isinstance(self.model.fields.spectral, dict):
            if not isinstance(N_hats, dict):
                raise TypeError("Nonlinear model must return dict in dict-field mode")
            for name in self.dyn_names:
                self.model.fields.spectral[name].add_(self.dt * N_hats[name])
                self.model.fields.spectral[name].div_(self.denom[name])
        else:
            dyn_fields = self.model.fields.spectral[:self.model.fields.dyn_count]
            dyn_fields.add_(self.dt * N_hats)
            dyn_fields.div_(self.denom)
        
        if self.model.fields.stat_count != 0:
            stat_out = self.model.compute_static()
            if isinstance(self.model.fields.spectral, dict):
                if not isinstance(stat_out, dict):
                    raise TypeError("Static model must return dict in dict-field mode")
                for name in self.stat_names:
                    self.model.fields.spectral[name] = stat_out[name]
            else:
                self.model.fields.spectral[self.model.fields.dyn_count:] = stat_out
        
        # self.model.fields.spectral *= self.model.fields.dealiasing_mask

        self.model.fields.spatial = self.model.fields.ifftn() # calculate spatial from spectral

        self.step_count += 1

        # spectral cleanup. Taking fft after ifft is critical for stability. 
        if self.step_count % 20 == 0:
            self.model.fields.spectral = self.model.fields.fftn() 
            self.step_count = 0
