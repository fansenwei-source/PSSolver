"""Initial-condition factories for fields used with PSSolver."""

from collections.abc import Callable
import math

import numpy as np
import torch


Q_COMPONENTS = ("Qxx", "Qxy", "Qxz", "Qyy", "Qyz")
QInitialCondition = Callable[..., dict[str, torch.Tensor]]


def aligned_x_smooth_noise(
    shape: tuple[int, int, int],
    *,
    boundary_conditions: tuple[str, str, str],
    seed: int = 42,
    noise_theta: float = 0.01,
    noise_phi: float = 0.01,
    sigma_x: float = 1.0,
    sigma_y: float = 1.0,
    sigma_z: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Create a +x-aligned uniaxial Q tensor with BC-aware smooth noise.

    Noise is added to the director angles and smoothed independently along
    each spatial axis. Periodic and Neumann axes use periodic and even
    extensions, respectively. The returned tensors are float32 CPU tensors
    with the requested shape.
    """
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError(f"shape must contain three positive sizes, got {shape!r}")
    boundary_conditions = tuple(boundary_conditions)
    if len(boundary_conditions) != 3:
        raise ValueError(
            "boundary_conditions must contain one entry for each spatial axis, "
            f"got {boundary_conditions!r}"
        )
    supported_boundary_conditions = {"periodic", "neumann"}
    unsupported = set(boundary_conditions) - supported_boundary_conditions
    if unsupported:
        raise ValueError(
            f"unsupported boundary conditions {sorted(unsupported)}; supported: "
            f"{sorted(supported_boundary_conditions)}"
        )
    if noise_theta < 0 or noise_phi < 0:
        raise ValueError(
            "angular noise amplitudes must be non-negative, got "
            f"noise_theta={noise_theta}, noise_phi={noise_phi}"
        )
    sigmas = (sigma_x, sigma_y, sigma_z)
    if any(sigma <= 0 for sigma in sigmas):
        raise ValueError(
            f"smoothing sigmas must be positive, got "
            f"sigma_x={sigma_x}, sigma_y={sigma_y}, sigma_z={sigma_z}"
        )

    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError as error:
        raise ImportError(
            "aligned_x_smooth_noise requires scipy; install it with "
            "`python -m pip install scipy`"
        ) from error

    nx_size, ny_size, nz_size = shape
    rng = np.random.default_rng(seed)

    angle_theta_0 = math.pi / 2.0
    angle_phi_0 = 0.0

    delta_theta = rng.uniform(
        -1.0, 1.0, size=(nx_size, ny_size, nz_size)
    ).astype(np.float32)
    delta_phi = rng.uniform(
        -1.0, 1.0, size=(nx_size, ny_size, nz_size)
    ).astype(np.float32)

    def smooth_axis(
        values: np.ndarray,
        *,
        sigma: float,
        axis: int,
        boundary_condition: str,
    ) -> np.ndarray:
        if boundary_condition == "periodic":
            return gaussian_filter1d(values, sigma=sigma, axis=axis, mode="wrap")

        reversed_values = np.flip(values, axis=axis)
        extension = np.concatenate((values, reversed_values), axis=axis)

        smoothed_extension = gaussian_filter1d(
            extension,
            sigma=sigma,
            axis=axis,
            mode="wrap",
        )
        original_domain = [slice(None)] * values.ndim
        original_domain[axis] = slice(0, values.shape[axis])
        return smoothed_extension[tuple(original_domain)]

    for values in (delta_theta, delta_phi):
        for axis, (sigma, boundary_condition) in enumerate(
            zip(sigmas, boundary_conditions)
        ):
            values[:] = smooth_axis(
                values,
                sigma=sigma,
                axis=axis,
                boundary_condition=boundary_condition,
            )

    def rescale_to_amplitude(values: np.ndarray, amplitude: float) -> np.ndarray:
        standard_deviation = values.std()
        if standard_deviation > 1e-12:
            return values / standard_deviation * amplitude
        return values * 0.0

    angle_theta = angle_theta_0 + rescale_to_amplitude(
        delta_theta, noise_theta
    )
    angle_phi = angle_phi_0 + rescale_to_amplitude(delta_phi, noise_phi)

    sin_theta = np.sin(angle_theta)
    cos_theta = np.cos(angle_theta)
    cos_phi = np.cos(angle_phi)
    sin_phi = np.sin(angle_phi)

    director_x = sin_theta * cos_phi
    director_y = sin_theta * sin_phi
    director_z = cos_theta

    components = {
        "Qxx": director_x * director_x - 1.0 / 3.0,
        "Qxy": director_x * director_y,
        "Qxz": director_x * director_z,
        "Qyy": director_y * director_y - 1.0 / 3.0,
        "Qyz": director_y * director_z,
    }
    return {
        name: torch.from_numpy(values.astype(np.float32))
        for name, values in components.items()
    }


_Q_INITIAL_CONDITIONS: dict[str, QInitialCondition] = {
    "aligned_x_smooth_noise": aligned_x_smooth_noise,
}


def available_q_initial_conditions() -> tuple[str, ...]:
    """Return the registered Q-tensor initial-condition names."""
    return tuple(_Q_INITIAL_CONDITIONS)


def create_q_initial_condition(
    name: str,
    shape: tuple[int, int, int],
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Create a registered Q-tensor initial condition by name."""
    try:
        initializer = _Q_INITIAL_CONDITIONS[name]
    except KeyError as error:
        available = ", ".join(available_q_initial_conditions())
        raise ValueError(
            f"unknown Q initial condition {name!r}; available: {available}"
        ) from error

    fields = initializer(shape, **kwargs)
    missing = set(Q_COMPONENTS) - fields.keys()
    extra = fields.keys() - set(Q_COMPONENTS)
    if missing or extra:
        raise RuntimeError(
            f"Q initial condition {name!r} returned invalid fields; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return fields


__all__ = [
    "Q_COMPONENTS",
    "aligned_x_smooth_noise",
    "available_q_initial_conditions",
    "create_q_initial_condition",
]
