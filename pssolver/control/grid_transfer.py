"""Coordinate-consistent transfer operations for cell-centered Q fields."""

from __future__ import annotations

import numpy as np


def conservative_block_average(values: np.ndarray, factors: tuple[int, int, int]) -> np.ndarray:
    """Restrict the first three spatial axes by equal-volume block averaging."""

    values = np.asarray(values)
    if values.ndim < 3:
        raise ValueError("values must have at least three spatial dimensions.")
    factors = tuple(int(value) for value in factors)
    if len(factors) != 3 or any(value <= 0 for value in factors):
        raise ValueError("factors must contain three positive integers.")
    spatial_shape = values.shape[:3]
    if any(size % factor for size, factor in zip(spatial_shape, factors)):
        raise ValueError(
            "Block averaging requires each spatial dimension to be divisible by "
            f"its factor; got shape={spatial_shape}, factors={factors}."
        )

    reshaped_shape = []
    for size, factor in zip(spatial_shape, factors):
        reshaped_shape.extend((size // factor, factor))
    reshaped_shape.extend(values.shape[3:])
    restricted = values.reshape(reshaped_shape).mean(axis=(1, 3, 5))
    return restricted.astype(values.dtype, copy=False)


def periodic_translate_axis(
    values: np.ndarray,
    displacement: float,
    period: float,
    *,
    axis: int = 0,
) -> np.ndarray:
    """Translate a periodic cell-centered field by a physical displacement."""

    values = np.asarray(values)
    if period <= 0:
        raise ValueError("period must be positive.")
    axis = int(axis)
    if not -values.ndim <= axis < values.ndim:
        raise ValueError(f"axis={axis} is outside an array with ndim={values.ndim}.")
    axis %= values.ndim
    frequencies = np.fft.fftfreq(values.shape[axis], d=period / values.shape[axis])
    phase_shape = [1] * values.ndim
    phase_shape[axis] = values.shape[axis]
    phase = np.exp(-2j * np.pi * frequencies * float(displacement)).reshape(
        phase_shape
    )
    translated = np.fft.ifft(np.fft.fft(values, axis=axis) * phase, axis=axis)
    if np.isrealobj(values):
        translated = translated.real
    return translated.astype(values.dtype, copy=False)


__all__ = ["conservative_block_average", "periodic_translate_axis"]
