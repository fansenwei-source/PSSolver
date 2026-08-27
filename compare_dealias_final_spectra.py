#!/usr/bin/env python3
"""Measure final-frame lateral high-wavenumber energy for each dealiasing run."""

from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
PATHS = {
    "two_thirds": ROOT / "data_free_slip_dealias_two_thirds_dt0p001_100000_save1000",
    "cubic_half": ROOT / "data_free_slip_dealias_cubic_half_dt0p001_100000_save1000",
    "none": ROOT / "data_free_slip_dealias_control_none_dt0p001_resume96000_to100000",
}
STEPS = {"two_thirds": 100000, "cubic_half": 100000, "none": 4000}


def fractions(path):
    array = np.load(path, mmap_mode="r")
    nx, ny = array.shape[:2]
    kx = np.fft.fftfreq(nx) * nx
    ky = np.fft.rfftfreq(ny) * ny
    kx, ky = np.meshgrid(kx, ky, indexing="ij")
    outside_half = (np.abs(kx) > nx / 4) | (np.abs(ky) > ny / 4)
    outside_two_thirds = (np.abs(kx) > nx / 3) | (np.abs(ky) > ny / 3)
    total = high_half = high_two_thirds = 0.0
    for z0 in range(0, array.shape[2], 4):
        chunk = np.asarray(array[:, :, z0:z0 + 4, :], dtype=np.float64)
        energy = np.abs(np.fft.rfft2(chunk, axes=(0, 1), norm="ortho")) ** 2
        total += float(energy.sum())
        high_half += float(energy[outside_half, ...].sum())
        high_two_thirds += float(energy[outside_two_thirds, ...].sum())
    return high_half / total, high_two_thirds / total


for mode, directory in PATHS.items():
    step = STEPS[mode]
    for field in ("Q", "u"):
        half, two_thirds = fractions(directory / f"{field}_{step}.npy")
        print(mode, field, half, two_thirds)
