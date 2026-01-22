import math
import torch

from .boundary import BoundaryCondition, normalize_bcs


def _real_dtype(dtype):
    if dtype == torch.complex64:
        return torch.float32
    if dtype == torch.complex128:
        return torch.float64
    return dtype


def _as_bc(value):
    return normalize_bcs(value, 1)[0]


# ============================================================
# DCT-I / IDCT-I (half-scaled, invertible pair)
# ============================================================

def dct1(x, dim: int = -1):
    """
    DCT-I (half-scaled):

    For length N along dim:
        X_k = 1/2*(x_0 + (-1)^k x_{N-1}) + sum_{n=1}^{N-2} x_n cos(pi n k/(N-1))

    This definition satisfies:
        dct1(dct1(x)) = (N-1)/2 * x
    """
    N = x.size(dim)
    if N < 2:
        return x

    if torch.is_complex(x):
        return dct1(x.real, dim) + 1j * dct1(x.imag, dim)

    x_mid = x.narrow(dim, 1, N - 2)                            # x[1:N-1]
    x_ext = torch.cat([x, torch.flip(x_mid, [dim])], dim=dim)  # len = 2*(N-1)
    X = torch.fft.rfft(x_ext, dim=dim).real
    return X.narrow(dim, 0, N) * 0.5


def idct1(X, dim: int = -1):
    """
    Inverse of the half-scaled DCT-I:
        dct1(dct1(x)) = (N-1)/2 * x
        => x = (2/(N-1)) dct1(X)
    """
    N = X.size(dim)
    if N < 2:
        return X

    if torch.is_complex(X):
        return idct1(X.real, dim) + 1j * idct1(X.imag, dim)

    x = dct1(X, dim=dim)
    return x * (2.0 / (N - 1))


# ============================================================
# DST-I / IDST-I (Wikipedia unnormalized)
# ============================================================

def dst1(x, dim: int = -1):
    """
    DST-I (Wikipedia unnormalized):
        X_k = sum_{n=0}^{M-1} x_n sin(pi (n+1)(k+1)/(M+1))

    Implementation:
      - odd extension with zeros at both ends to length 2*(M+1)
      - FFT
      - take k=1..M imag part, multiply by -1/2
    """
    M = x.size(dim)
    if M < 1:
        return x

    if torch.is_complex(x):
        return dst1(x.real, dim) + 1j * dst1(x.imag, dim)

    zero = torch.zeros_like(x.narrow(dim, 0, 1))
    x_ext = torch.cat([zero, x, zero, -torch.flip(x, [dim])], dim=dim)
    X = torch.fft.fft(x_ext, dim=dim)
    return -0.5 * X.narrow(dim, 1, M).imag


def idst1(X, dim: int = -1):
    """
    Inverse DST-I under unnormalized convention:
        dst1(dst1(x)) = (M+1)/2 * x
        => x = (2/(M+1)) dst1(X)
    """
    M = X.size(dim)
    if M < 1:
        return X

    if torch.is_complex(X):
        return idst1(X.real, dim) + 1j * idst1(X.imag, dim)

    x = dst1(X, dim=dim)
    return x * (2.0 / (M + 1))


# ============================================================
# Low-level BC helpers for mixed transforms
# ============================================================

def _k_axis_for_bc(bc, N, L, device, dtype):
    """
    Return (k, N_eff, slicer):
      - periodic : N modes
      - neumann  : N modes (m=0..N-1)
      - dirichlet: N-2 interior modes (j=1..N-2)
    """
    bc = _as_bc(bc)
    dtype = _real_dtype(dtype)

    if bc == BoundaryCondition.PERIODIC:
        n = torch.arange(N, device=device, dtype=dtype)
        n = torch.where(n <= N // 2, n, n - N)
        k = 2 * math.pi * n / L
        return k, N, slice(None)

    if bc == BoundaryCondition.NEUMANN:
        m = torch.arange(N, device=device, dtype=dtype)
        k = math.pi * m / L
        return k, N, slice(None)

    if bc == BoundaryCondition.DIRICHLET:
        if N <= 2:
            k = torch.zeros(0, device=device, dtype=dtype)
            return k, 0, slice(1, 1)
        j = torch.arange(1, N - 1, device=device, dtype=dtype)
        k = math.pi * j / L
        return k, N - 2, slice(1, -1)

    raise ValueError(f"Unknown BC: {bc}")


def _forward_1d_bc(x, bc, dim):
    """
    Forward transform along one dim consistent with bc, possibly reducing size if Dirichlet.
    """
    bc = _as_bc(bc)
    if bc == BoundaryCondition.PERIODIC:
        return torch.fft.fft(x, dim=dim)
    if bc == BoundaryCondition.NEUMANN:
        return dct1(x, dim=dim)
    if bc == BoundaryCondition.DIRICHLET:
        N = x.size(dim)
        if N <= 2:
            shape = list(x.shape)
            shape[dim] = 0
            return x.new_zeros(shape)
        x_in = x.narrow(dim, 1, N - 2)
        return dst1(x_in, dim=dim)
    raise ValueError(f"Unknown BC: {bc}")


def _inverse_1d_bc(X, bc, dim, N_full):
    """
    Inverse transform along one dim, padding boundaries if Dirichlet.
    """
    bc = _as_bc(bc)
    d = dim % X.ndim

    if bc == BoundaryCondition.PERIODIC:
        return torch.fft.ifft(X, dim=dim).real
    if bc == BoundaryCondition.NEUMANN:
        return idct1(X, dim=dim).real
    if bc == BoundaryCondition.DIRICHLET:
        if N_full <= 2:
            out_shape = list(X.shape)
            out_shape[d] = N_full
            return X.new_zeros(out_shape)

        x_in = idst1(X, dim=dim).real  # length N_full-2 along dim
        out_shape = list(x_in.shape)
        out_shape[d] = N_full
        out = x_in.new_zeros(out_shape)
        out.narrow(d, 1, N_full - 2).copy_(x_in)
        return out

    raise ValueError(f"Unknown BC: {bc}")
