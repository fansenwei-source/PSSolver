"""Lightweight geometric and topological diagnostics for a loop Q field."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from ..models.active_nematics import Q_magnitude


def q5_to_matrix(q5: np.ndarray) -> np.ndarray:
    q5 = np.asarray(q5)
    if q5.ndim != 4 or q5.shape[-1] != 5:
        raise ValueError(f"Expected Q shape (Nx, Ny, Nz, 5), got {q5.shape}.")
    matrix = np.empty(q5.shape[:-1] + (3, 3), dtype=q5.dtype)
    matrix[..., 0, 0] = q5[..., 0]
    matrix[..., 0, 1] = matrix[..., 1, 0] = q5[..., 1]
    matrix[..., 0, 2] = matrix[..., 2, 0] = q5[..., 2]
    matrix[..., 1, 1] = q5[..., 3]
    matrix[..., 1, 2] = matrix[..., 2, 1] = q5[..., 4]
    matrix[..., 2, 2] = -q5[..., 0] - q5[..., 3]
    return matrix


def principal_director_and_S(q5: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(q5_to_matrix(q5))
    return eigenvectors[..., :, -1], eigenvalues[..., -1]


def _defect_plaquette_mask(
    director: np.ndarray,
    axis_a: int,
    axis_b: int,
) -> np.ndarray:
    n00 = director
    n10 = np.roll(director, -1, axis=axis_a)
    n01 = np.roll(director, -1, axis=axis_b)
    n11 = np.roll(n10, -1, axis=axis_b)
    link_product = (
        np.sum(n00 * n10, axis=-1)
        * np.sum(n10 * n11, axis=-1)
        * np.sum(n11 * n01, axis=-1)
        * np.sum(n01 * n00, axis=-1)
    )
    valid = [slice(None), slice(None), slice(None)]
    for axis in (axis_a, axis_b):
        if axis != 0:
            valid[axis] = slice(0, -1)
    return link_product[tuple(valid)] < 0.0


def defect_plaquette_counts(director: np.ndarray) -> dict[str, int]:
    return {
        "xy": int(np.count_nonzero(_defect_plaquette_mask(director, 0, 1))),
        "xz": int(np.count_nonzero(_defect_plaquette_mask(director, 0, 2))),
        "yz": int(np.count_nonzero(_defect_plaquette_mask(director, 1, 2))),
    }


def _periodic_delta(values: np.ndarray, center: float, period: float) -> np.ndarray:
    return (values - center + 0.5 * period) % period - 0.5 * period


def _periodic_weighted_center(
    coordinates: np.ndarray,
    weights: np.ndarray,
    period: float,
) -> tuple[float, float]:
    angles = 2.0 * np.pi * coordinates / period
    cosine = float(np.sum(weights * np.cos(angles)))
    sine = float(np.sum(weights * np.sin(angles)))
    resultant = np.hypot(cosine, sine)
    center = (np.arctan2(sine, cosine) % (2.0 * np.pi)) * period / (2.0 * np.pi)
    confidence = resultant / max(float(np.sum(weights)), np.finfo(float).eps)
    return float(center), float(confidence)


def soft_core_density(
    q5: np.ndarray,
    *,
    S_bulk: float,
    threshold_fraction: float = 0.65,
    transition_fraction: float = 0.08,
) -> np.ndarray:
    """Return the smooth low-order density used by the differentiable objectives."""

    if S_bulk <= 0:
        raise ValueError("S_bulk must be positive.")
    if not 0 < threshold_fraction < 1:
        raise ValueError("threshold_fraction must lie in (0, 1).")
    if transition_fraction <= 0:
        raise ValueError("transition_fraction must be positive.")
    q5 = np.asarray(q5)
    if q5.ndim != 4 or q5.shape[-1] != 5:
        raise ValueError(f"Expected Q shape (Nx, Ny, Nz, 5), got {q5.shape}.")
    magnitude = Q_magnitude(q5)
    threshold = threshold_fraction * S_bulk
    transition = transition_fraction * S_bulk
    density = 1.0 / (1.0 + np.exp((magnitude - threshold) / transition))
    bulk_density = 1.0 / (
        1.0 + np.exp((S_bulk - threshold) / transition)
    )
    normalized = (density - bulk_density) / (1.0 - bulk_density)
    return np.maximum(normalized, 0.0) ** 2


def _weighted_core_center(
    weights: np.ndarray,
    lengths: tuple[float, float, float],
) -> tuple[np.ndarray, float]:
    mass = float(np.sum(weights))
    if not np.isfinite(mass) or mass <= np.finfo(float).eps:
        raise ValueError("Cannot locate an empty or non-finite soft core.")
    axes = [
        (np.arange(size, dtype=float) + 0.5) * length / size
        for size, length in zip(weights.shape, lengths)
    ]
    center_x, confidence = _periodic_weighted_center(
        axes[0], np.sum(weights, axis=(1, 2)), lengths[0]
    )
    center_y = float(np.sum(weights * axes[1][None, :, None]) / mass)
    center_z = float(np.sum(weights * axes[2][None, None, :]) / mass)
    return np.asarray((center_x, center_y, center_z)), confidence


def soft_core_center(
    q5: np.ndarray,
    lengths: tuple[float, float, float],
    *,
    S_bulk: float,
    threshold_fraction: float = 0.65,
    transition_fraction: float = 0.08,
) -> dict[str, object]:
    """Measure the differentiable soft-core surrogate with NumPy diagnostics."""

    density = soft_core_density(
        q5,
        S_bulk=S_bulk,
        threshold_fraction=threshold_fraction,
        transition_fraction=transition_fraction,
    )
    center, confidence = _weighted_core_center(density, lengths)
    return {
        "center": center,
        "mass": float(np.sum(density)),
        "periodic_center_confidence": confidence,
    }


def _periodic_unwrapped_coordinates(
    shape: tuple[int, int, int],
    lengths: tuple[float, float, float],
    center_x: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    axes = [
        (np.arange(size, dtype=float) + 0.5) * length / size
        for size, length in zip(shape, lengths)
    ]
    x = center_x + _periodic_delta(axes[0], center_x, lengths[0])
    return np.meshgrid(x, axes[1], axes[2], indexing="ij", sparse=True)


def _core_component_count(density: np.ndarray, threshold_fraction: float = 0.2) -> int:
    active = density >= threshold_fraction * float(np.max(density))
    labels, count = ndimage.label(active, structure=ndimage.generate_binary_structure(3, 3))
    if count <= 1 or not np.any(active[0]) or not np.any(active[-1]):
        return int(count)

    touching_start = set(np.unique(labels[0][active[0]])) - {0}
    touching_stop = set(np.unique(labels[-1][active[-1]])) - {0}
    parent = list(range(count + 1))

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for start in touching_start:
        for stop in touching_stop:
            parent[find(stop)] = find(start)
    return len({find(value) for value in range(1, count + 1)})


def core_centerline_geometry(
    q5: np.ndarray,
    lengths: tuple[float, float, float],
    *,
    S_bulk: float,
    threshold_fraction: float = 0.65,
    transition_fraction: float = 0.08,
    num_samples: int | None = None,
) -> dict[str, object]:
    """Reconstruct a sub-grid centerline for one approximately planar closed core.

    The estimator fits the core plane to the smooth low-order density and uses
    angular kernel centroids to collapse the finite-width core tube to an ordered
    line. It is intended for the single, nearly planar loop in the DAL problem.
    """

    q5 = np.asarray(q5)
    shape = q5.shape[:3]
    lengths = tuple(float(value) for value in lengths)
    spacing = np.asarray(lengths) / np.asarray(shape)
    density = soft_core_density(
        q5,
        S_bulk=S_bulk,
        threshold_fraction=threshold_fraction,
        transition_fraction=transition_fraction,
    )
    soft_center, confidence = _weighted_core_center(density, lengths)
    x, y, z = _periodic_unwrapped_coordinates(shape, lengths, soft_center[0])
    coordinates = np.stack(np.broadcast_arrays(x, y, z), axis=-1)
    centered = coordinates - soft_center
    mass = float(np.sum(density))
    covariance = np.einsum(
        "...,...i,...j->ij", density, centered, centered, optimize=True
    ) / mass
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    normal = eigenvectors[:, 0]
    axis_u = eigenvectors[:, 2]
    axis_v = np.cross(normal, axis_u)
    axis_v /= np.linalg.norm(axis_v)
    projected_u = centered @ axis_u
    projected_v = centered @ axis_v
    angle = np.arctan2(projected_v, projected_u)
    radial_rms = float(
        np.sqrt(np.sum(density * (projected_u**2 + projected_v**2)) / mass)
    )
    if num_samples is None:
        in_plane_spacing = max(float(np.min(spacing)), np.finfo(float).eps)
        num_samples = int(np.clip(np.ceil(2.0 * np.pi * radial_rms / in_plane_spacing), 16, 64))
    if num_samples < 8:
        raise ValueError("num_samples must be at least 8.")

    sample_angles = np.linspace(-np.pi, np.pi, num_samples, endpoint=False)
    bandwidth = 1.25 * 2.0 * np.pi / num_samples
    line_points = []
    angular_masses = []
    for sample_angle in sample_angles:
        delta = np.angle(np.exp(1j * (angle - sample_angle)))
        local_weights = density * np.exp(-0.5 * (delta / bandwidth) ** 2)
        local_mass = float(np.sum(local_weights))
        if local_mass <= 1e-8 * mass:
            raise ValueError("Soft core does not support a complete closed centerline.")
        line_points.append(
            np.sum(local_weights[..., None] * coordinates, axis=(0, 1, 2)) / local_mass
        )
        angular_masses.append(local_mass)
    line_points = np.asarray(line_points)
    line_points[:, 0] = np.mod(line_points[:, 0], lengths[0])

    unwrapped = line_points.copy()
    unwrapped[:, 0] = soft_center[0] + _periodic_delta(
        line_points[:, 0], soft_center[0], lengths[0]
    )
    segment_vectors = np.roll(unwrapped, -1, axis=0) - unwrapped
    segment_lengths = np.linalg.norm(segment_vectors, axis=1)
    line_length = float(np.sum(segment_lengths))
    segment_midpoints = unwrapped + 0.5 * segment_vectors
    arc_center_x, _ = _periodic_weighted_center(
        np.mod(segment_midpoints[:, 0], lengths[0]), segment_lengths, lengths[0]
    )
    arc_center = np.asarray(
        (
            arc_center_x,
            np.average(segment_midpoints[:, 1], weights=segment_lengths),
            np.average(segment_midpoints[:, 2], weights=segment_lengths),
        )
    )
    arc_unwrapped = unwrapped - arc_center
    arc_unwrapped[:, 0] = _periodic_delta(
        unwrapped[:, 0], arc_center[0], lengths[0]
    )
    line_covariance = arc_unwrapped.T @ arc_unwrapped / num_samples
    line_values, line_vectors = np.linalg.eigh(line_covariance)
    line_normal = line_vectors[:, 0]
    semiaxes = np.sqrt(np.maximum(2.0 * line_values[1:], 0.0))[::-1]
    planarity_rms = float(np.sqrt(max(line_values[0], 0.0)))
    radii = np.linalg.norm(
        arc_unwrapped - (arc_unwrapped @ line_normal)[:, None] * line_normal,
        axis=1,
    )

    previous = unwrapped - np.roll(unwrapped, 1, axis=0)
    following = np.roll(unwrapped, -1, axis=0) - unwrapped
    chord = np.roll(unwrapped, -1, axis=0) - np.roll(unwrapped, 1, axis=0)
    curvature_denominator = (
        np.linalg.norm(previous, axis=1)
        * np.linalg.norm(following, axis=1)
        * np.linalg.norm(chord, axis=1)
    )
    curvature = 2.0 * np.linalg.norm(np.cross(previous, following), axis=1) / np.maximum(
        curvature_denominator, np.finfo(float).eps
    )
    if np.dot(line_normal, normal) < 0:
        line_normal = -line_normal
    return {
        "detected": True,
        "method": "soft-core angular-kernel centerline",
        "center": arc_center,
        "soft_volume_center": soft_center,
        "soft_volume_center_offset": float(
            np.linalg.norm(
                np.asarray(
                    (
                        _periodic_delta(soft_center[0], arc_center[0], lengths[0]),
                        soft_center[1] - arc_center[1],
                        soft_center[2] - arc_center[2],
                    )
                )
            )
        ),
        "periodic_center_confidence": confidence,
        "line_length": line_length,
        "normal": line_normal,
        "planarity_rms": planarity_rms,
        "radius_mean": float(np.mean(radii)),
        "radius_rms": float(np.sqrt(np.mean(radii**2))),
        "radius_std": float(np.std(radii)),
        "semiaxis_major": float(semiaxes[0]),
        "semiaxis_minor": float(semiaxes[1]),
        "ellipticity": float(
            (semiaxes[0] - semiaxes[1]) / max(np.mean(semiaxes), np.finfo(float).eps)
        ),
        "curvature_mean": float(np.mean(curvature)),
        "curvature_max": float(np.max(curvature)),
        "core_component_count": _core_component_count(density),
        "soft_core_mass": mass,
        "num_centerline_samples": int(num_samples),
        "centerline": line_points,
        "angular_mass_min_fraction": float(min(angular_masses) / max(angular_masses)),
    }


def loop_core_metrics(
    q5: np.ndarray,
    lengths: tuple[float, float, float],
    *,
    S_bulk: float,
    deficit_threshold: float = 0.2,
) -> dict[str, float | bool | dict[str, int]]:
    """Estimate loop geometry from the low-scalar-order defect core."""

    if S_bulk <= 0:
        raise ValueError("S_bulk must be positive.")
    if not 0 <= deficit_threshold < 1:
        raise ValueError("deficit_threshold must lie in [0, 1).")
    q5 = np.asarray(q5)
    director, S = principal_director_and_S(q5)
    deficit = np.maximum((S_bulk - S) / S_bulk, 0.0)
    weights = np.maximum(deficit - deficit_threshold, 0.0)
    mass = float(np.sum(weights))
    counts = defect_plaquette_counts(director)
    finite = bool(np.isfinite(q5).all() and np.isfinite(S).all())

    result: dict[str, float | bool | dict[str, int]] = {
        "detected": bool(finite and mass > np.finfo(float).eps),
        "finite": finite,
        "core_mass": mass,
        "S_min": float(np.min(S)),
        "S_median": float(np.median(S)),
        "defect_plaquettes": counts,
        # Plaquette orientation describes line geometry only. It cannot
        # distinguish Yingyou's splay, bend, and twist director textures.
        "yz_plane_topology": counts["xy"] > 0 and counts["xz"] > 0 and counts["yz"] == 0,
    }
    if not result["detected"]:
        result.update(
            {
                "center_x": float("nan"),
                "center_y": float("nan"),
                "center_z": float("nan"),
                "periodic_center_confidence": 0.0,
                "radius_rms": float("nan"),
                "radius_y_rms": float("nan"),
                "radius_z_rms": float("nan"),
                "x_thickness_rms": float("nan"),
            }
        )
        return result

    shape = q5.shape[:3]
    axes = [
        (np.arange(size, dtype=float) + 0.5) * length / size
        for size, length in zip(shape, lengths)
    ]
    x, y, z = axes
    weight_x = np.sum(weights, axis=(1, 2))
    center_x, confidence = _periodic_weighted_center(x, weight_x, lengths[0])
    center_y = float(np.sum(weights * y[None, :, None]) / mass)
    center_z = float(np.sum(weights * z[None, None, :]) / mass)
    dx = _periodic_delta(x, center_x, lengths[0])
    dy = y - center_y
    dz = z - center_z
    y_variance = float(np.sum(weights * dy[None, :, None] ** 2) / mass)
    z_variance = float(np.sum(weights * dz[None, None, :] ** 2) / mass)
    x_variance = float(np.sum(weights * dx[:, None, None] ** 2) / mass)
    result.update(
        {
            "center_x": center_x,
            "center_y": center_y,
            "center_z": center_z,
            "periodic_center_confidence": confidence,
            "radius_rms": float(np.sqrt(y_variance + z_variance)),
            "radius_y_rms": float(np.sqrt(y_variance)),
            "radius_z_rms": float(np.sqrt(z_variance)),
            "x_thickness_rms": float(np.sqrt(x_variance)),
        }
    )
    return result


def x_disturbance_profile(
    q5: np.ndarray,
    *,
    S_bulk: float,
) -> np.ndarray:
    background = np.zeros(5, dtype=np.asarray(q5).dtype)
    background[0] = S_bulk
    background[3] = -S_bulk / 2.0
    return np.sum((np.asarray(q5) - background) ** 2, axis=(1, 2, 3))


def periodic_profile_shift(
    reference: np.ndarray,
    current: np.ndarray,
    period: float,
) -> dict[str, float]:
    """Return the sub-grid circular shift that aligns reference with current."""

    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    if reference.ndim != 1 or current.shape != reference.shape:
        raise ValueError("reference and current must be one-dimensional arrays of equal shape.")
    if period <= 0:
        raise ValueError("period must be positive.")
    reference = reference - np.mean(reference)
    current = current - np.mean(current)
    reference_norm = float(np.linalg.norm(reference))
    current_norm = float(np.linalg.norm(current))
    if reference_norm == 0 or current_norm == 0:
        return {"shift_indices": float("nan"), "shift_physical": float("nan"), "correlation": 0.0}

    correlation = np.fft.ifft(
        np.conj(np.fft.fft(reference)) * np.fft.fft(current)
    ).real
    peak = int(np.argmax(correlation))
    previous_value = float(correlation[(peak - 1) % correlation.size])
    peak_value = float(correlation[peak])
    next_value = float(correlation[(peak + 1) % correlation.size])
    denominator = previous_value - 2.0 * peak_value + next_value
    subgrid = (
        0.5 * (previous_value - next_value) / denominator
        if abs(denominator) > np.finfo(float).eps
        else 0.0
    )
    shift = float(peak) + float(np.clip(subgrid, -0.5, 0.5))
    if shift > 0.5 * correlation.size:
        shift -= correlation.size
    return {
        "shift_indices": shift,
        "shift_physical": shift * period / correlation.size,
        "correlation": peak_value / (reference_norm * current_norm),
    }


__all__ = [
    "core_centerline_geometry",
    "defect_plaquette_counts",
    "loop_core_metrics",
    "periodic_profile_shift",
    "principal_director_and_S",
    "q5_to_matrix",
    "soft_core_center",
    "soft_core_density",
    "x_disturbance_profile",
]
