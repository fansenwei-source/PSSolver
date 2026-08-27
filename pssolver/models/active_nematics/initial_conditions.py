"""Initial conditions owned by the three-dimensional active-nematic model."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math

import numpy as np
import torch

from .fields import Q_COMPONENTS
from .q_tensor import Q_components, uniaxial_Q


InitialCondition = Callable[..., dict[str, torch.Tensor]]


def _validate_shape(shape: tuple[int, int, int]) -> tuple[int, int, int]:
    if len(shape) != 3 or any(int(size) != size or size <= 0 for size in shape):
        raise ValueError(f"shape must contain three positive integer sizes, got {shape!r}")
    return tuple(int(size) for size in shape)


def _as_Q_2d_components(
    Q_2d: Mapping[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor,
    expected_shape: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """Normalize supported 2D Q layouts to five float32 CPU tensors."""
    if isinstance(Q_2d, Mapping):
        input_names = set(Q_2d)
        missing = set(Q_COMPONENTS) - input_names
        extra = input_names - set(Q_COMPONENTS)
        if missing or extra:
            raise ValueError(
                "Q_2d mapping must contain exactly the five independent Q "
                f"components; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        components = {}
        for name in Q_COMPONENTS:
            values = torch.as_tensor(Q_2d[name], dtype=torch.float32, device="cpu")
            if values.shape == (*expected_shape, 1):
                values = values[..., 0]
            if values.shape != expected_shape:
                raise ValueError(
                    f"Q_2d[{name!r}] has shape {tuple(values.shape)}; "
                    f"expected {expected_shape} or {(*expected_shape, 1)}"
                )
            components[name] = values
        return components

    values = torch.as_tensor(Q_2d, dtype=torch.float32, device="cpu")
    if values.shape == (*expected_shape, 1, len(Q_COMPONENTS)):
        values = values[..., 0, :]
    expected_array_shape = (*expected_shape, len(Q_COMPONENTS))
    if values.shape != expected_array_shape:
        raise ValueError(
            f"Q_2d array has shape {tuple(values.shape)}; expected "
            f"{expected_array_shape} or {(*expected_shape, 1, len(Q_COMPONENTS))}"
        )
    return {
        name: values[..., component]
        for component, name in enumerate(Q_COMPONENTS)
    }


def _validate_2d_geometry(
    shape: tuple[int, int],
    lengths: tuple[float, float],
) -> tuple[int, int, float, float]:
    if len(shape) != 2 or any(int(size) != size or size < 4 for size in shape):
        raise ValueError(
            f"shape must contain two integer sizes of at least four, got {shape!r}"
        )
    if len(lengths) != 2 or any(
        not np.isfinite(length) or length <= 0 for length in lengths
    ):
        raise ValueError(
            f"lengths must contain two positive finite values, got {lengths!r}"
        )
    return int(shape[0]), int(shape[1]), float(lengths[0]), float(lengths[1])


def _periodic_displacements(
    values: np.ndarray,
    centers: np.ndarray | float,
    period: float,
) -> np.ndarray:
    delta = values - centers
    return delta - period * np.round(delta / period)


def sample_periodic_neutral_defects_2d(
    *,
    lengths: tuple[float, float],
    num_defect_pairs: int,
    min_separation: float,
    seed: int = 42,
    max_placement_attempts: int = 100_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a reproducible, charge-neutral configuration on a 2D torus."""
    if len(lengths) != 2 or any(
        not np.isfinite(length) or length <= 0 for length in lengths
    ):
        raise ValueError(
            f"lengths must contain two positive finite values, got {lengths!r}"
        )
    lengths_array = np.asarray(lengths, dtype=float)
    if int(num_defect_pairs) != num_defect_pairs or num_defect_pairs <= 0:
        raise ValueError(
            f"num_defect_pairs must be a positive integer, got {num_defect_pairs!r}"
        )
    pair_count = int(num_defect_pairs)
    if not np.isfinite(min_separation) or min_separation <= 0:
        raise ValueError(
            f"min_separation must be positive and finite, got {min_separation!r}"
        )
    if min_separation > 0.5 * np.linalg.norm(lengths_array):
        raise ValueError(
            "min_separation exceeds the largest possible minimum-image "
            f"distance for lengths={tuple(lengths_array)}"
        )
    if (
        int(max_placement_attempts) != max_placement_attempts
        or max_placement_attempts <= 0
    ):
        raise ValueError(
            "max_placement_attempts must be a positive integer, got "
            f"{max_placement_attempts!r}"
        )

    rng = np.random.default_rng(seed)
    target_count = 2 * pair_count
    positions: list[np.ndarray] = []
    attempts = 0
    while len(positions) < target_count and attempts < int(max_placement_attempts):
        attempts += 1
        candidate = rng.uniform(0.0, lengths_array)
        is_far_enough = True
        for existing in positions:
            delta = candidate - existing
            delta -= lengths_array * np.round(delta / lengths_array)
            if np.linalg.norm(delta) < min_separation:
                is_far_enough = False
                break
        if is_far_enough:
            positions.append(candidate)

    if len(positions) != target_count:
        raise RuntimeError(
            f"could place only {len(positions)}/{target_count} periodic defects "
            f"with min_separation={min_separation} after {attempts} attempts; "
            "reduce the defect count or minimum separation"
        )

    charges = np.concatenate(
        (
            np.full(pair_count, 0.5, dtype=float),
            np.full(pair_count, -0.5, dtype=float),
        )
    )
    rng.shuffle(charges)
    return np.asarray(positions, dtype=float), charges


def _jacobi_theta1_centered(
    real_argument: np.ndarray,
    imaginary_argument: np.ndarray,
    *,
    nome: float,
    aspect_ratio: float,
    tolerance: float = 1e-14,
) -> np.ndarray:
    """Evaluate theta_1 after reducing both arguments to a centered cell."""
    if not (0.0 < nome <= math.exp(-math.pi) * (1.0 + 1e-12)):
        raise ValueError(f"unexpected theta nome {nome}")
    argument = math.pi * (
        real_argument + 1j * aspect_ratio * imaginary_argument
    )
    result = np.zeros(argument.shape, dtype=np.complex128)
    leading_coefficient = nome**0.25
    for mode in range(64):
        coefficient = 2.0 * ((-1.0) ** mode) * nome ** ((mode + 0.5) ** 2)
        result += coefficient * np.sin((2 * mode + 1) * argument)
        if mode > 0 and abs(coefficient) <= tolerance * leading_coefficient:
            break
    else:
        raise RuntimeError("Jacobi theta_1 series did not converge")
    return result


def _periodic_nematic_phase_2d(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lengths: tuple[float, float],
    positions: np.ndarray,
    charges: np.ndarray,
    background_angle: float = 0.0,
) -> np.ndarray:
    """Return the seamless torus field ``exp(i*2*theta)``."""
    lx, ly = (float(lengths[0]), float(lengths[1]))
    positions = np.asarray(positions, dtype=float)
    charges = np.asarray(charges, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError(f"positions must have shape (N, 2), got {positions.shape}")
    if charges.shape != (len(positions),):
        raise ValueError(
            f"charges must have shape ({len(positions)},), got {charges.shape}"
        )
    doubled_charges = 2.0 * charges
    integer_charges = np.rint(doubled_charges).astype(int)
    if not np.allclose(doubled_charges, integer_charges) or not np.all(
        np.isin(integer_charges, (-1, 1))
    ):
        raise ValueError("charges must contain only nematic +/-1/2 defects")
    if integer_charges.sum() != 0:
        raise ValueError("periodic defect charges must sum to zero")

    # Choose the shorter direction as the real period so the Fourier series
    # converges rapidly. The coordinate rotation preserves orientation.
    if lx <= ly:
        primary = np.asarray(x, dtype=float)
        secondary = np.asarray(y, dtype=float)
        primary_positions = positions[:, 0]
        secondary_positions = positions[:, 1]
        primary_length, secondary_length = lx, ly
    else:
        primary = np.asarray(y, dtype=float)
        secondary = -np.asarray(x, dtype=float)
        primary_positions = positions[:, 1]
        secondary_positions = -positions[:, 0]
        primary_length, secondary_length = ly, lx

    aspect_ratio = secondary_length / primary_length
    nome = math.exp(-math.pi * aspect_ratio)
    phase_angle = np.full(
        np.broadcast_shapes(primary.shape, secondary.shape),
        2.0 * background_angle,
    )

    for position_primary, position_secondary, charge in zip(
        primary_positions,
        secondary_positions,
        integer_charges,
    ):
        delta_primary = primary - position_primary
        delta_secondary = secondary - position_secondary
        primary_shift = np.floor(delta_primary / primary_length + 0.5).astype(int)
        secondary_shift = np.floor(
            delta_secondary / secondary_length + 0.5
        ).astype(int)
        centered_primary = delta_primary / primary_length - primary_shift
        centered_secondary = delta_secondary / secondary_length - secondary_shift
        theta_value = _jacobi_theta1_centered(
            centered_primary,
            centered_secondary,
            nome=nome,
            aspect_ratio=aspect_ratio,
        )
        theta_phase = (
            np.angle(theta_value)
            + math.pi * (primary_shift + secondary_shift)
            - 2.0 * math.pi * secondary_shift * centered_primary
        )
        phase_angle += charge * theta_phase

    primary_dipole = float(np.sum(integer_charges * primary_positions))
    phase_angle -= (
        2.0
        * math.pi
        * secondary
        * primary_dipole
        / (primary_length * secondary_length)
    )
    return np.exp(1j * phase_angle)


def analytic_periodic_defect_gas_2d(
    shape: tuple[int, int],
    *,
    lengths: tuple[float, float],
    num_defect_pairs: int,
    min_separation: float,
    core_radius: float,
    S_initial: float,
    seed: int = 42,
    background_angle: float = 0.0,
    max_placement_attempts: int = 100_000,
) -> dict[str, torch.Tensor]:
    """Construct a seamless periodic gas of nematic ``+/-1/2`` defects.

    The initial far-field order ``S_initial`` is suppressed by a product of smooth
    ``tanh(distance/core_radius)`` profiles. Returned tensors use the sole
    active-nematic Q convention and have shape ``(Nx, Ny)``.
    """
    nx, ny, lx, ly = _validate_2d_geometry(shape, lengths)
    if not np.isfinite(core_radius) or core_radius <= 0:
        raise ValueError(
            f"core_radius must be positive and finite, got {core_radius!r}"
        )
    if not np.isfinite(S_initial) or S_initial <= 0:
        raise ValueError(
            f"S_initial must be positive and finite, got {S_initial!r}"
        )
    if not np.isfinite(background_angle):
        raise ValueError(
            f"background_angle must be finite, got {background_angle!r}"
        )

    positions, charges = sample_periodic_neutral_defects_2d(
        lengths=(lx, ly),
        num_defect_pairs=num_defect_pairs,
        min_separation=min_separation,
        seed=seed,
        max_placement_attempts=max_placement_attempts,
    )
    x = (np.arange(nx, dtype=float) + 0.5) * lx / nx
    y = (np.arange(ny, dtype=float) + 0.5) * ly / ny
    grid_x, grid_y = np.meshgrid(x, y, indexing="ij")
    unit_phase = _periodic_nematic_phase_2d(
        grid_x,
        grid_y,
        lengths=(lx, ly),
        positions=positions,
        charges=charges,
        background_angle=background_angle,
    )

    S_field = np.full((nx, ny), S_initial, dtype=float)
    for position in positions:
        delta_x = _periodic_displacements(grid_x, position[0], lx)
        delta_y = _periodic_displacements(grid_y, position[1], ly)
        distance = np.hypot(delta_x, delta_y)
        S_field *= np.tanh(distance / core_radius)

    theta = 0.5 * np.angle(unit_phase)
    director = np.stack(
        (np.cos(theta), np.sin(theta), np.zeros_like(theta)),
        axis=-1,
    )
    full_Q = uniaxial_Q(director, S_field)
    components = Q_components(full_Q)
    return {
        name: torch.from_numpy(np.asarray(values, dtype=np.float32))
        for name, values in components.items()
    }


def neumann_twist_profile(
    nz: int,
    *,
    amplitude: float = 0.01,
    modes: Sequence[int] = (1, 2, 3),
    seed: int = 42,
) -> torch.Tensor:
    """Create a wall-normal twist angle from Neumann cosine modes."""
    if int(nz) != nz or nz <= 1:
        raise ValueError(f"nz must be an integer greater than one, got {nz}")
    nz = int(nz)
    if not math.isfinite(amplitude) or amplitude < 0:
        raise ValueError(
            f"amplitude must be non-negative and finite, got {amplitude}"
        )
    modes = tuple(int(mode) for mode in modes)
    if not modes:
        raise ValueError("modes must contain at least one positive DCT mode")
    if len(set(modes)) != len(modes):
        raise ValueError(f"modes must be unique, got {modes!r}")
    invalid = [mode for mode in modes if mode <= 0 or mode >= nz]
    if invalid:
        raise ValueError(
            f"twist modes must satisfy 1 <= mode < nz={nz}; got {invalid}"
        )

    cell_center = (torch.arange(nz, dtype=torch.float64) + 0.5) / nz
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    coefficients = torch.randn(
        len(modes),
        generator=generator,
        dtype=torch.float64,
    )
    profile = torch.zeros(nz, dtype=torch.float64)
    for coefficient, mode in zip(coefficients, modes):
        profile += coefficient * torch.cos(math.pi * mode * cell_center)

    rms = torch.sqrt(torch.mean(profile.square()))
    if amplitude == 0:
        profile.zero_()
    elif rms <= 1e-14:
        raise RuntimeError("generated a degenerate twist profile")
    else:
        profile *= amplitude / rms
    return profile.to(dtype=torch.float32)


def extruded_2d_twist(
    shape: tuple[int, int, int],
    *,
    Q_2d: Mapping[str, np.ndarray | torch.Tensor] | np.ndarray | torch.Tensor,
    boundary_conditions: tuple[str, str, str],
    twist_amplitude: float = 0.01,
    twist_modes: Sequence[int] = (1, 2, 3),
    seed: int = 42,
) -> dict[str, torch.Tensor]:
    """Extrude a canonical 2D Q field and rotate each wall-normal plane.

    ``Q_2d`` may be a mapping with five independent components or an array
    whose final dimension follows ``Q_COMPONENTS``. Each plane is transformed
    as ``Q(x,y,z) = Rz(psi(z)) Q_2d(x,y) Rz(psi(z))^T``. Rotation preserves
    pointwise eigenvalues and core locations.
    """
    nx, ny, nz = _validate_shape(shape)
    boundary_conditions = tuple(boundary_conditions)
    if len(boundary_conditions) != 3:
        raise ValueError(
            "boundary_conditions must contain one entry for each spatial axis, "
            f"got {boundary_conditions!r}"
        )
    if boundary_conditions[2] != "neumann":
        raise ValueError(
            "extruded_2d_twist currently constructs a Neumann-compatible z "
            f"profile, got z boundary condition {boundary_conditions[2]!r}"
        )

    source = _as_Q_2d_components(Q_2d, (nx, ny))
    psi = neumann_twist_profile(
        nz,
        amplitude=twist_amplitude,
        modes=twist_modes,
        seed=seed,
    ).reshape(1, 1, nz)
    cosine = torch.cos(psi)
    sine = torch.sin(psi)
    cosine_squared = cosine.square()
    sine_squared = sine.square()
    sine_cosine = sine * cosine

    Qxx = source["Qxx"].unsqueeze(-1)
    Qxy = source["Qxy"].unsqueeze(-1)
    Qxz = source["Qxz"].unsqueeze(-1)
    Qyy = source["Qyy"].unsqueeze(-1)
    Qyz = source["Qyz"].unsqueeze(-1)

    fields = {
        "Qxx": cosine_squared * Qxx + sine_squared * Qyy - 2.0 * sine_cosine * Qxy,
        "Qxy": sine_cosine * (Qxx - Qyy) + (cosine_squared - sine_squared) * Qxy,
        "Qxz": cosine * Qxz - sine * Qyz,
        "Qyy": sine_squared * Qxx + cosine_squared * Qyy + 2.0 * sine_cosine * Qxy,
        "Qyz": sine * Qxz + cosine * Qyz,
    }
    return {name: values.contiguous() for name, values in fields.items()}


def aligned_x_smooth_noise(
    shape: tuple[int, int, int],
    *,
    boundary_conditions: tuple[str, str, str],
    S_initial: float,
    seed: int = 42,
    noise_theta: float = 0.01,
    noise_phi: float = 0.01,
    sigma_x: float = 1.0,
    sigma_y: float = 1.0,
    sigma_z: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Create a ``+x``-aligned canonical uniaxial Q field with smooth noise."""
    nx_size, ny_size, nz_size = _validate_shape(shape)
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
    if not math.isfinite(noise_theta) or not math.isfinite(noise_phi):
        raise ValueError("angular noise amplitudes must be finite")
    if noise_theta < 0 or noise_phi < 0:
        raise ValueError(
            "angular noise amplitudes must be non-negative, got "
            f"noise_theta={noise_theta}, noise_phi={noise_phi}"
        )
    if not np.isfinite(S_initial) or S_initial <= 0:
        raise ValueError(
            f"S_initial must be positive and finite, got {S_initial!r}"
        )
    sigmas = (sigma_x, sigma_y, sigma_z)
    if any(not math.isfinite(sigma) or sigma <= 0 for sigma in sigmas):
        raise ValueError(
            "smoothing sigmas must be positive and finite, got "
            f"sigma_x={sigma_x}, sigma_y={sigma_y}, sigma_z={sigma_z}"
        )

    try:
        from scipy.ndimage import gaussian_filter1d
    except ImportError as error:
        raise ImportError(
            "aligned_x_smooth_noise requires scipy; install it with "
            "`python -m pip install scipy`"
        ) from error

    rng = np.random.default_rng(seed)
    delta_theta = rng.uniform(
        -1.0,
        1.0,
        size=(nx_size, ny_size, nz_size),
    ).astype(np.float32)
    delta_phi = rng.uniform(
        -1.0,
        1.0,
        size=(nx_size, ny_size, nz_size),
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

    angle_theta = math.pi / 2.0 + rescale_to_amplitude(
        delta_theta,
        noise_theta,
    )
    angle_phi = rescale_to_amplitude(delta_phi, noise_phi)
    sin_theta = np.sin(angle_theta)
    director = np.stack(
        (
            sin_theta * np.cos(angle_phi),
            sin_theta * np.sin(angle_phi),
            np.cos(angle_theta),
        ),
        axis=-1,
    )
    full_Q = uniaxial_Q(director, S_initial)
    components = Q_components(full_Q)
    return {
        name: torch.from_numpy(np.asarray(values, dtype=np.float32))
        for name, values in components.items()
    }


_INITIAL_CONDITIONS: dict[str, InitialCondition] = {
    "analytic_periodic_defect_gas_2d": analytic_periodic_defect_gas_2d,
    "aligned_x_smooth_noise": aligned_x_smooth_noise,
    "extruded_2d_twist": extruded_2d_twist,
}


def available_initial_conditions() -> tuple[str, ...]:
    """Return the registered active-nematic initial-condition names."""
    return tuple(_INITIAL_CONDITIONS)


def create_initial_condition(
    name: str,
    shape: tuple[int, ...],
    **kwargs,
) -> dict[str, torch.Tensor]:
    """Create a registered active-nematic Q initial condition by name."""
    try:
        initializer = _INITIAL_CONDITIONS[name]
    except KeyError as error:
        available = ", ".join(available_initial_conditions())
        raise ValueError(
            f"unknown active-nematic initial condition {name!r}; "
            f"available: {available}"
        ) from error

    fields = initializer(shape, **kwargs)
    missing = set(Q_COMPONENTS) - fields.keys()
    extra = fields.keys() - set(Q_COMPONENTS)
    if missing or extra:
        raise RuntimeError(
            f"initial condition {name!r} returned invalid Q fields; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return fields


__all__ = [
    "aligned_x_smooth_noise",
    "analytic_periodic_defect_gas_2d",
    "available_initial_conditions",
    "create_initial_condition",
    "extruded_2d_twist",
    "neumann_twist_profile",
    "sample_periodic_neutral_defects_2d",
]
