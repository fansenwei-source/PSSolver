import torch
import torch.fft
from .integrator import SemiImplicitEulerIntegrator
from .PDEmodel import PDEModel
from .transforms import TensorProductTransformBackend

class SpectralSolver:
    def __init__(self, shape, L=2 * torch.pi, dt=0.01, batchsize = 1, device='cuda'):

        self.shape = shape
        if not isinstance(L, tuple):
            L = tuple([L] * len(shape))
        elif len(L) != len(shape):
            raise ValueError(f"Length of L ({len(L)}) must match length of shape ({len(shape)})")
        self.L = L
        self.dt = dt
        self.device = device
        self.batchsize = batchsize

        self.transform_backend = TensorProductTransformBackend(
            shape=self.shape,
            lengths=self.L,
            device=self.device,
        )
        self._init_periodic_metadata()

        self.model = PDEModel(shape, device, batchsize=batchsize)
        self.parameters = self.model.parameters
        self.fields = self.model.fields
        self.fields.set_transform_backend(self.transform_backend)
        self.fields.set_wavenumbers(self.qx, self.qy, self.qz, self.q2)
        self.integrator_cl = SemiImplicitEulerIntegrator

    def _init_periodic_metadata(self):
        dims = len(self.shape)
        periodic_bcs = ("periodic",) * dims
        metadata = self.transform_backend.get_metadata(periodic_bcs)

        self.spatial_grids = self.transform_backend.spatial_grids
        self.q_grids = torch.meshgrid(*metadata.axis_modes, indexing='ij')

        self.qx = self.q_grids[0]
        self.qy = self.q_grids[1] if dims > 1 else None
        self.qz = self.q_grids[2] if dims > 2 else None
        self.q2_raw = metadata.q2
        self.q2 = metadata.q2_safe

    def get_q2(self, boundary_conditions, regularize=False):
        boundary_conditions = self.fields._normalize_boundary_conditions(boundary_conditions)
        return self.transform_backend.get_q2(boundary_conditions, regularize=regularize)

    def get_laplacian_eigs(self, boundary_conditions):
        boundary_conditions = self.fields._normalize_boundary_conditions(boundary_conditions)
        return self.transform_backend.get_laplacian_eigs(boundary_conditions)

    def get_gradient_boundary_conditions(self, boundary_conditions, axis):
        boundary_conditions = self.fields._normalize_boundary_conditions(boundary_conditions)
        return self.transform_backend.get_gradient_boundary_conditions(boundary_conditions, axis)

    def transform_tensor(self, tensor, boundary_conditions):
        boundary_conditions = self.fields._normalize_boundary_conditions(boundary_conditions)
        return self.transform_backend.forward(tensor, boundary_conditions)

    def inverse_transform_tensor(self, spectral, boundary_conditions):
        boundary_conditions = self.fields._normalize_boundary_conditions(boundary_conditions)
        return self.transform_backend.inverse(spectral, boundary_conditions)

    def build(self):
        self.model.build()
        self.integrator = self.integrator_cl(self.model, self.dt, self.qx, self.qy, self.q2)

    def reset(self, inits={}):
        self.model.build()
        for name, val in inits.items():
            self.model.fields[name] = val
        self.model.fields.spectral = self.model.fields.fftn()

    def refresh_static_fields(self):
        """Recompute static fields from the current dynamic field state."""
        fields = self.model.fields
        if fields.stat_count == 0:
            return None

        S_hats = self.model.compute_static()
        for offset in range(fields.stat_count):
            field_idx = fields.dyn_count + offset
            fields.spectral[field_idx] = S_hats[offset]
            fields.spatial[field_idx] = fields.inverse_transform(field_idx, spectral=S_hats[offset])
        return S_hats

    def run(self, steps, callback = None):
        for step in range(steps):
            self.integrator.step()
            
            if callback is not None:
                callback(self, step)
