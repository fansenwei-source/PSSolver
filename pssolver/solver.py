import torch
import torch.fft
from .integrator import SemiImplicitEulerIntegrator
from .PDEmodel import PDEModel

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

        self._init_q_space()

        self.model = PDEModel(shape, device, batchsize=batchsize)
        self.parameters = self.model.parameters
        self.fields = self.model.fields
        self.integrator_cl = SemiImplicitEulerIntegrator

    def _init_q_space(self):
        # Support 1D, 2D, or 3D grids depending on self.shape
        dims = len(self.shape)
        axes = []
        q_axes = []
        for i, N in enumerate(self.shape):
            x = torch.linspace(0, self.L[i] - self.L[i] / N, N, device=self.device)
            axes.append(x)
            q = torch.fft.fftfreq(N, d=self.L[i] / N).to(self.device) * 2 * torch.pi
            q_axes.append(q)

        # Create spatial grids
        self.spatial_grids = torch.meshgrid(*axes, indexing='ij')

        # Create wavenumber grids
        q_grids = torch.meshgrid(*q_axes, indexing='ij')
        self.q_grids = q_grids

        # Assign qx, qy, qz if present
        self.qx = q_grids[0]
        self.qy = q_grids[1] if dims > 1 else None
        self.qz = q_grids[2] if dims > 2 else None

        # Compute q^2
        self.q2 = sum(q**2 for q in q_grids)
        if dims == 1:
            self.q2[0] = 1e-10
        elif dims == 2:
            self.q2[0, 0] = 1e-10
        elif dims == 3:
            self.q2[0, 0, 0] = 1e-10

    def build(self):
        self.model.fields.set_wavenumbers(self.qx, self.qy, self.qz, self.q2)
        self.model.build()
        self.integrator = self.integrator_cl(self.model, self.dt, self.qx, self.qy, self.q2)

    def reset(self, inits={}):
        self.model.build()
        for name, val in inits.items():
            self.model.fields[name] = val
        self.model.fields.spectral = self.model.fields.fftn()

    def run(self, steps, callback = None):
        for step in range(steps):
            self.integrator.step()
            
            if callback is not None:
                callback(self, step)


