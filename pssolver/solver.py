import torch
import torch.fft
from .integrator import SemiImplicitEulerIntegrator
from .PDEmodel import PDEModel
from .boundary import BoundaryCondition, normalize_bcs, is_periodic_only
from .transforms import _k_axis_for_bc

class SpectralSolver:
    def __init__(
        self,
        shape,
        L=2 * torch.pi,
        dt=0.01,
        batchsize=1,
        device='cuda',
        bcs=None,
        zero_mode_policy="none",
        zero_mode_eps=1e-10,
        zero_mode_handler=None,
    ):
        """
        zero_mode_policy:
          - "none" (default): leave q2 unchanged
          - "mask_zero": replace zero entries in q2 with zero_mode_eps
          - dict: per-field overrides, e.g. {"default": "none", "u": "mask_zero"}

        zero_mode_handler:
          - optional callable q2 -> q2 replacement for custom handling
          - signature: handler(q2, bcs, field_name)
        """

        self.shape = shape
        if not isinstance(L, tuple):
            L = tuple([L] * len(shape))
        elif len(L) != len(shape):
            raise ValueError(f"Length of L ({len(L)}) must match length of shape ({len(shape)})")
        self.L = L
        self.dt = dt
        self.device = device
        self.batchsize = batchsize
        self.bcs_config = bcs
        if self._is_field_bcs_config(bcs):
            default_bcs = bcs.get("default", None)
            self.bcs = normalize_bcs(default_bcs, len(shape))
        else:
            self.bcs = normalize_bcs(bcs, len(shape))

        self.zero_mode_policy = zero_mode_policy
        self.zero_mode_eps = zero_mode_eps
        self.zero_mode_handler = zero_mode_handler

        self._init_q_space()

        self.model = PDEModel(shape, device, batchsize=batchsize, bcs=bcs)
        self.parameters = self.model.parameters
        self.fields = self.model.fields
        self.integrator_cl = SemiImplicitEulerIntegrator

    @staticmethod
    def _is_field_bcs_config(bcs):
        if not isinstance(bcs, dict):
            return False
        axis_keys = {"x", "y", "z", 0, 1, 2}
        return any(key not in axis_keys for key in bcs.keys())

    def _init_q_space(self):
        # Support 1D, 2D, or 3D grids depending on self.shape
        dims = len(self.shape)
        axes = []
        q_axes = []
        for i, N in enumerate(self.shape):
            x = torch.linspace(0, self.L[i] - self.L[i] / N, N, device=self.device)
            axes.append(x)
            q = self._k_axis_for_bc(self.bcs[i], N, self.L[i])
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
        self.q2 = self._apply_zero_mode_policy_for(self.q2, self.bcs, field_name=None)

    def _k_axis_for_bc(self, bc, N, L):
        if bc == BoundaryCondition.PERIODIC:
            return torch.fft.fftfreq(N, d=L / N).to(self.device) * 2 * torch.pi
        q, _, _ = _k_axis_for_bc(bc, N, L, self.device, torch.float32)
        return q

    def _resolve_zero_mode_policy(self, field_name):
        if isinstance(self.zero_mode_policy, dict):
            if field_name in self.zero_mode_policy:
                return self.zero_mode_policy[field_name]
            if "default" in self.zero_mode_policy:
                return self.zero_mode_policy["default"]
        return self.zero_mode_policy

    def _apply_zero_mode_policy_for(self, q2, bcs, field_name):
        if self.zero_mode_handler is not None:
            return self.zero_mode_handler(q2, bcs, field_name)

        policy = (self._resolve_zero_mode_policy(field_name) or "none").lower()
        if policy == "none":
            return q2
        if policy == "mask_zero":
            zero_mask = q2 == 0
            if torch.any(zero_mask):
                q2 = q2.masked_fill(zero_mask, self.zero_mode_eps)
            return q2
        raise ValueError(f"Unknown zero_mode_policy: {self.zero_mode_policy!r}")

    def build(self):
        self.model.build()
        self.model.fields.set_wavenumbers(self.qx, self.qy, self.qz, self.q2)
        self._init_field_wavenumbers()
        self.integrator = self.integrator_cl(self.model, self.dt, self.qx, self.qy, self.q2)

    def _init_field_wavenumbers(self):
        field_bcs = self.model.fields.field_bcs
        if not field_bcs:
            return
        field_wavenumbers = {}
        spectral_shapes = {}
        spectral_slices = {}
        for name, bcs in field_bcs.items():
            q_axes = []
            shape = []
            slicers = []
            for i, N in enumerate(self.shape):
                k, N_eff, slicer = _k_axis_for_bc(bcs[i], N, self.L[i], self.device, torch.float32)
                q_axes.append(k)
                shape.append(N_eff)
                slicers.append(slicer)
            q_grids = torch.meshgrid(*q_axes, indexing='ij')
            q2 = sum(q**2 for q in q_grids)
            q2 = self._apply_zero_mode_policy_for(q2, bcs, field_name=name)
            qx = q_grids[0]
            qy = q_grids[1] if len(self.shape) > 1 else None
            qz = q_grids[2] if len(self.shape) > 2 else None
            field_wavenumbers[name] = (qx, qy, qz, q2)
            spectral_shapes[name] = tuple(shape)
            spectral_slices[name] = tuple(slicers)
        self.model.fields.set_wavenumbers_for_fields(field_wavenumbers)
        self.model.fields.set_spectral_meta_for_fields(spectral_shapes, spectral_slices)

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
