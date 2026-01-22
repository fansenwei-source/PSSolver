import math
import torch

from .boundary import BoundaryCondition, normalize_bcs
from .transforms import dct1, idct1, dst1, idst1, _real_dtype


class SpectralAxis:
    """
    1D spectral axis definition:
    - periodic : FFT modes k = 2π n/L, n in [-N/2, N/2)
    - neumann  : cosine modes k = π m/L, m=0..N-1
    - dirichlet: sine modes k = π j/L, j=1..N-2, but we store length N with endpoints zero
    """
    def __init__(self, bc, N, L, dim, device, dtype=torch.float32):
        self.bc = normalize_bcs(bc, 1)[0]
        self.N = N
        self.L = L
        self.dim = dim
        self.device = device
        self.dtype = _real_dtype(dtype)

        if self.bc == BoundaryCondition.PERIODIC:
            n = torch.arange(N, device=device, dtype=self.dtype)
            n = torch.where(n <= N // 2, n, n - N)
            k = 2 * math.pi * n / L

        elif self.bc == BoundaryCondition.NEUMANN:
            m = torch.arange(N, device=device, dtype=self.dtype)
            k = math.pi * m / L

        elif self.bc == BoundaryCondition.DIRICHLET:
            # store full length N, but endpoints k=0; interior matches j=1..N-2
            k = torch.zeros((N,), device=device, dtype=self.dtype)
            if N > 2:
                j = torch.arange(1, N - 1, device=device, dtype=self.dtype)
                k[1:-1] = math.pi * j / L
        else:
            raise ValueError(f"Unknown BC: {self.bc}")

        shape = [1] * 4
        shape[self.dim] = N
        self.k = k.view(*shape)

    def forward_1d(self, x):
        if self.bc == BoundaryCondition.PERIODIC:
            return torch.fft.fft(x, dim=self.dim)

        N = x.size(self.dim)

        if self.bc == BoundaryCondition.NEUMANN:
            return dct1(x, dim=self.dim)

        if self.bc == BoundaryCondition.DIRICHLET:
            if N <= 2:
                return torch.zeros_like(x)
            M = N - 2
            x_in = x.narrow(self.dim, 1, M)
            Y_in = dst1(x_in, dim=self.dim)
            Y_full = torch.zeros_like(x, dtype=Y_in.dtype)
            idx = [slice(None)] * x.ndim
            idx[self.dim] = slice(1, N - 1)
            Y_full[tuple(idx)] = Y_in
            return Y_full

        raise RuntimeError("Unknown BC in forward_1d")

    def inverse_1d(self, X):
        if self.bc == BoundaryCondition.PERIODIC:
            return torch.fft.ifft(X, dim=self.dim)

        N = X.size(self.dim)

        if self.bc == BoundaryCondition.NEUMANN:
            return idct1(X, dim=self.dim)

        if self.bc == BoundaryCondition.DIRICHLET:
            if N <= 2:
                return torch.zeros_like(X)
            M = N - 2
            X_in = X.narrow(self.dim, 1, M)
            x_in = idst1(X_in, dim=self.dim)
            x_full = torch.zeros_like(X, dtype=x_in.dtype)
            idx = [slice(None)] * X.ndim
            idx[self.dim] = slice(1, N - 1)
            x_full[tuple(idx)] = x_in
            return x_full

        raise RuntimeError("Unknown BC in inverse_1d")

    def deriv_factor(self):
        # Only periodic axis uses spectral derivative by default
        if self.bc == BoundaryCondition.PERIODIC:
            return 1j * self.k
        return None


class SpectralTransform3D:
    """
    3D transform using SpectralAxis in each direction.
    """
    def __init__(self, shape, L, bcs, device, dtype=torch.float32):
        Nx, Ny, Nz = shape
        Lx, Ly, Lz = L
        self.device = device
        self.dtype = dtype

        self.ax_x = SpectralAxis(bcs["x"], Nx, Lx, dim=-3, device=device, dtype=dtype)
        self.ax_y = SpectralAxis(bcs["y"], Ny, Ly, dim=-2, device=device, dtype=dtype)
        self.ax_z = SpectralAxis(bcs["z"], Nz, Lz, dim=-1, device=device, dtype=dtype)

    def forward_scalar(self, F):
        F = self.ax_z.forward_1d(F)
        F = self.ax_x.forward_1d(F)
        F = self.ax_y.forward_1d(F)
        return F

    def inverse_scalar(self, Fh):
        Fh = self.ax_y.inverse_1d(Fh)
        Fh = self.ax_x.inverse_1d(Fh)
        Fh = self.ax_z.inverse_1d(Fh)
        return Fh

    def k2(self):
        kx = self.ax_x.k
        ky = self.ax_y.k
        kz = self.ax_z.k
        return kx * kx + ky * ky + kz * kz
