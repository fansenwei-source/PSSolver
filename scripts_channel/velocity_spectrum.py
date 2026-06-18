#!/usr/bin/env python3
"""Plot horizontal velocity spectra from saved PSSolver velocity snapshots."""

from __future__ import annotations

import argparse
import csv
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from scipy import fft as scipy_fft
except ImportError:  # pragma: no cover - scipy is an optional accelerator
    scipy_fft = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit("matplotlib is required to write velocity-spectrum PNG files") from exc


ROOT = Path(__file__).resolve().parents[1]
U_PATTERN = re.compile(r"^u_(\d+)\.npy$")
J0_FIRST_MINIMUM = 3.8317059702075125


@dataclass(frozen=True)
class Grid:
    nx: int
    ny: int
    nz: int
    dx: float
    dy: float
    dz: float

    @property
    def lx(self) -> float:
        return self.nx * self.dx

    @property
    def ly(self) -> float:
        return self.ny * self.dy

    @property
    def lz(self) -> float:
        return self.nz * self.dz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute radial horizontal velocity spectra E(k_parallel) from "
            "data/u_<step>.npy snapshots."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Default: <data-dir>/velocity_spectrum.png",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=None,
        help="Output CSV path. Default: same basename as --out with .csv suffix.",
    )
    parser.add_argument(
        "--npz",
        type=Path,
        default=None,
        help="Output NPZ path. Default: same basename as --out with .npz suffix.",
    )
    parser.add_argument("--dx", type=float, default=0.25)
    parser.add_argument("--dy", type=float, default=0.25)
    parser.add_argument("--dz", type=float, default=0.25)
    parser.add_argument(
        "--mode",
        choices=("zavg", "midplane", "both"),
        default="both",
        help="Use all z planes, the midplane only, or plot both spectra.",
    )
    parser.add_argument(
        "--components",
        choices=("xyz", "xy"),
        default="xyz",
        help="Velocity components included in spectral energy.",
    )
    parser.add_argument("--start-step", type=int, default=None)
    parser.add_argument("--end-step", type=int, default=None)
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--component-axis",
        type=int,
        default=-1,
        help="Axis containing velocity components. Default -1 for (Nx, Ny, Nz, 3).",
    )
    parser.add_argument(
        "--k-bin-width",
        type=float,
        default=None,
        help="Radial wavenumber bin width. Default: min(2*pi/Lx, 2*pi/Ly).",
    )
    parser.add_argument(
        "--k-max",
        type=float,
        default=None,
        help="Maximum plotted radial wavenumber. Default: Nyquist radial diagonal.",
    )
    parser.add_argument(
        "--midplane-index",
        type=int,
        default=None,
        help="z index for midplane mode. Default: Nz//2.",
    )
    parser.add_argument(
        "--logy",
        action="store_true",
        help="Use log scale for spectral energy.",
    )
    parser.add_argument(
        "--fft-workers",
        type=int,
        default=-1,
        help=(
            "Number of scipy FFT worker threads. Default -1 uses all available "
            "workers when scipy is installed; NumPy fallback ignores this option."
        ),
    )
    return parser.parse_args()


def velocity_steps(data_dir: Path) -> list[int]:
    steps = []
    for path in data_dir.glob("u_*.npy"):
        match = U_PATTERN.match(path.name)
        if match:
            steps.append(int(match.group(1)))
    return sorted(steps)


def select_steps(steps: list[int], args: argparse.Namespace) -> list[int]:
    if args.steps is not None:
        requested = set(args.steps)
        steps = [step for step in steps if step in requested]
    if args.start_step is not None:
        steps = [step for step in steps if step >= args.start_step]
    if args.end_step is not None:
        steps = [step for step in steps if step <= args.end_step]
    if args.limit is not None:
        steps = steps[: args.limit]
    return steps


def load_velocity(path: Path, component_axis: int) -> np.ndarray:
    velocity = np.load(path, mmap_mode="r")
    axis = component_axis if component_axis >= 0 else velocity.ndim + component_axis
    if velocity.ndim != 4:
        raise ValueError(f"{path} must have 4 dimensions, got shape {velocity.shape}")
    if axis < 0 or axis >= velocity.ndim:
        raise ValueError(f"{path} has invalid component axis {component_axis}")
    if velocity.shape[axis] != 3:
        raise ValueError(
            f"{path} component axis {component_axis} must have length 3, "
            f"got shape {velocity.shape}"
        )
    if axis != velocity.ndim - 1:
        velocity = np.moveaxis(velocity, axis, -1)
    return np.asarray(velocity, dtype=np.float64)


def infer_grid(sample: np.ndarray, dx: float, dy: float, dz: float) -> Grid:
    nx, ny, nz = sample.shape[:3]
    return Grid(nx=nx, ny=ny, nz=nz, dx=dx, dy=dy, dz=dz)


def component_indices(components: str) -> tuple[int, ...]:
    return (0, 1, 2) if components == "xyz" else (0, 1)


def k_bins(grid: Grid, k_bin_width: float | None, k_max: float | None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    kx = 2.0 * np.pi * np.fft.fftfreq(grid.nx, d=grid.dx)
    ky = 2.0 * np.pi * np.fft.fftfreq(grid.ny, d=grid.dy)
    k_radius = np.sqrt(kx[:, None] ** 2 + ky[None, :] ** 2)
    dk = k_bin_width if k_bin_width is not None else min(2.0 * np.pi / grid.lx, 2.0 * np.pi / grid.ly)
    if dk <= 0.0:
        raise ValueError("--k-bin-width must be positive")
    k_limit = k_max if k_max is not None else float(np.max(k_radius))
    if k_limit <= 0.0:
        raise ValueError("--k-max must be positive")
    bin_index = np.floor(k_radius / dk).astype(np.int64)
    max_bin = int(np.floor(k_limit / dk))
    valid = (k_radius <= k_limit + 1e-12) & (bin_index <= max_bin)
    centers = (np.arange(max_bin + 1, dtype=np.float64) + 0.5) * dk
    centers[0] = 0.0
    counts = np.bincount(bin_index[valid].ravel(), minlength=max_bin + 1).astype(np.int64)
    return centers, bin_index, valid, counts, k_radius


def radial_sum(power_xy: np.ndarray, bin_index: np.ndarray, valid: np.ndarray, n_bins: int) -> np.ndarray:
    return np.bincount(
        bin_index[valid].ravel(),
        weights=power_xy[valid].ravel(),
        minlength=n_bins,
    )


def fftn_xy(field: np.ndarray, workers: int) -> np.ndarray:
    if scipy_fft is not None:
        return scipy_fft.fftn(field, axes=(0, 1), workers=workers)
    return np.fft.fftn(field, axes=(0, 1))


def spectrum_zavg_frame(
    velocity: np.ndarray,
    comps: tuple[int, ...],
    bin_index: np.ndarray,
    valid: np.ndarray,
    n_bins: int,
    fft_workers: int,
) -> tuple[np.ndarray, float]:
    power_xy = np.zeros(velocity.shape[:2], dtype=np.float64)
    energy = 0.0
    nxy = velocity.shape[0] * velocity.shape[1]
    for comp in comps:
        field = velocity[..., comp]
        spectrum = fftn_xy(field, fft_workers)
        power_xy += np.sum(np.abs(spectrum) ** 2, axis=2) / nxy
        energy += float(np.sum(field * field))
    if energy <= 0.0:
        raise ValueError("Velocity energy is zero; cannot normalize spectrum")
    return radial_sum(power_xy / energy, bin_index, valid, n_bins), energy


def spectrum_midplane_frame(
    velocity: np.ndarray,
    comps: tuple[int, ...],
    midplane_index: int,
    bin_index: np.ndarray,
    valid: np.ndarray,
    n_bins: int,
    fft_workers: int,
) -> tuple[np.ndarray, float]:
    power_xy = np.zeros(velocity.shape[:2], dtype=np.float64)
    energy = 0.0
    nxy = velocity.shape[0] * velocity.shape[1]
    for comp in comps:
        field = velocity[:, :, midplane_index, comp]
        spectrum = fftn_xy(field, fft_workers)
        power_xy += np.abs(spectrum) ** 2 / nxy
        energy += float(np.sum(field * field))
    if energy <= 0.0:
        raise ValueError("Midplane velocity energy is zero; cannot normalize spectrum")
    return radial_sum(power_xy / energy, bin_index, valid, n_bins), energy


def summarize_spectra(per_frame: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    array = np.asarray(per_frame, dtype=np.float64)
    return np.nanmean(array, axis=0), np.nanstd(array, axis=0)


def peak_summary(k: np.ndarray, spectrum: np.ndarray) -> tuple[float | None, float | None, float | None, int | None]:
    valid = np.isfinite(spectrum) & (k > 0.0)
    if not np.any(valid):
        return None, None, None, None
    indices = np.flatnonzero(valid)
    peak_index = indices[np.nanargmax(spectrum[indices])]
    k_peak = float(k[peak_index])
    wavelength = 2.0 * np.pi / k_peak
    j0_min_distance = J0_FIRST_MINIMUM / k_peak
    return k_peak, wavelength, j0_min_distance, int(peak_index)


def write_csv(
    path: Path,
    k: np.ndarray,
    counts: np.ndarray,
    zavg_mean: np.ndarray | None,
    zavg_std: np.ndarray | None,
    mid_mean: np.ndarray | None,
    mid_std: np.ndarray | None,
) -> None:
    fieldnames = ["k", "mode_count"]
    if zavg_mean is not None:
        fieldnames += ["E_zavg_mean", "E_zavg_std"]
    if mid_mean is not None:
        fieldnames += ["E_mid_mean", "E_mid_std"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, kval in enumerate(k):
            row: dict[str, float | int] = {"k": float(kval), "mode_count": int(counts[i])}
            if zavg_mean is not None and zavg_std is not None:
                row["E_zavg_mean"] = float(zavg_mean[i])
                row["E_zavg_std"] = float(zavg_std[i])
            if mid_mean is not None and mid_std is not None:
                row["E_mid_mean"] = float(mid_mean[i])
                row["E_mid_std"] = float(mid_std[i])
            writer.writerow(row)


def write_npz(
    path: Path,
    k: np.ndarray,
    counts: np.ndarray,
    steps: list[int],
    grid: Grid,
    zavg_per_frame: np.ndarray | None,
    zavg_mean: np.ndarray | None,
    zavg_std: np.ndarray | None,
    mid_per_frame: np.ndarray | None,
    mid_mean: np.ndarray | None,
    mid_std: np.ndarray | None,
) -> None:
    payload: dict[str, object] = {
        "k": k,
        "mode_count": counts,
        "steps": np.asarray(steps, dtype=np.int64),
        "dx": grid.dx,
        "dy": grid.dy,
        "dz": grid.dz,
        "Lx": grid.lx,
        "Ly": grid.ly,
        "Lz": grid.lz,
    }
    if zavg_mean is not None and zavg_std is not None and zavg_per_frame is not None:
        k_peak, wavelength, j0_min_distance, peak_index = peak_summary(k, zavg_mean)
        payload.update(
            E_zavg_mean=zavg_mean,
            E_zavg_std=zavg_std,
            E_zavg_per_frame=zavg_per_frame,
            k_peak_zavg=np.nan if k_peak is None else k_peak,
            wavelength_peak_zavg=np.nan if wavelength is None else wavelength,
            j0_min_distance_zavg=np.nan if j0_min_distance is None else j0_min_distance,
            peak_index_zavg=-1 if peak_index is None else peak_index,
        )
    if mid_mean is not None and mid_std is not None and mid_per_frame is not None:
        k_peak, wavelength, j0_min_distance, peak_index = peak_summary(k, mid_mean)
        payload.update(
            E_mid_mean=mid_mean,
            E_mid_std=mid_std,
            E_mid_per_frame=mid_per_frame,
            k_peak_mid=np.nan if k_peak is None else k_peak,
            wavelength_peak_mid=np.nan if wavelength is None else wavelength,
            j0_min_distance_mid=np.nan if j0_min_distance is None else j0_min_distance,
            peak_index_mid=-1 if peak_index is None else peak_index,
        )
    np.savez_compressed(path, **payload)


def plot_spectrum(
    path: Path,
    k: np.ndarray,
    zavg_mean: np.ndarray | None,
    zavg_std: np.ndarray | None,
    mid_mean: np.ndarray | None,
    mid_std: np.ndarray | None,
    grid: Grid,
    logy: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=180)
    if zavg_mean is not None and zavg_std is not None:
        ax.plot(k, zavg_mean, label="z-averaged", linewidth=1.9)
        ax.fill_between(k, zavg_mean - zavg_std, zavg_mean + zavg_std, alpha=0.15)
        k_peak, wavelength, j0_min_distance, _ = peak_summary(k, zavg_mean)
        if k_peak is not None:
            ax.axvline(k_peak, color="C0", linestyle="--", linewidth=1.3)
            ax.annotate(
                f"$k_{{peak,zavg}}$={k_peak:.3g}\n$2\\pi/k$={wavelength:.3g}\n$3.83/k$={j0_min_distance:.3g}",
                xy=(k_peak, np.nanmax(zavg_mean)),
                xytext=(8, -34),
                textcoords="offset points",
                color="C0",
            )
    if mid_mean is not None and mid_std is not None:
        ax.plot(k, mid_mean, label="midplane", linewidth=1.9)
        ax.fill_between(k, mid_mean - mid_std, mid_mean + mid_std, alpha=0.15)
        k_peak, wavelength, j0_min_distance, _ = peak_summary(k, mid_mean)
        if k_peak is not None:
            ax.axvline(k_peak, color="C1", linestyle="--", linewidth=1.3)
            ax.annotate(
                f"$k_{{peak,mid}}$={k_peak:.3g}\n$2\\pi/k$={wavelength:.3g}\n$3.83/k$={j0_min_distance:.3g}",
                xy=(k_peak, np.nanmax(mid_mean)),
                xytext=(8, 10),
                textcoords="offset points",
                color="C1",
            )
    kmin = min(2.0 * np.pi / grid.lx, 2.0 * np.pi / grid.ly)
    ax.axvline(kmin, color="black", linestyle=":", linewidth=1.1, label=r"$k_{min}$")
    ax.set_xlabel(r"horizontal wavenumber $k_\parallel$")
    ax.set_ylabel(r"normalized shell energy $E(k_\parallel)$")
    ax.set_title("Horizontal Velocity Spectrum")
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def print_peak(name: str, k: np.ndarray, spectrum: np.ndarray | None) -> None:
    if spectrum is None:
        return
    k_peak, wavelength, j0_min_distance, _ = peak_summary(k, spectrum)
    if k_peak is None:
        print(f"{name}: no positive-k peak")
    else:
        print(
            f"{name}: k_peak={k_peak:.6g} "
            f"lambda=2pi/k={wavelength:.6g} "
            f"J0_first_min_distance=3.8317/k={j0_min_distance:.6g}"
        )


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    steps = select_steps(velocity_steps(data_dir), args)
    if not steps:
        raise RuntimeError(f"No selected u_<step>.npy snapshots found in {data_dir}")

    sample = load_velocity(data_dir / f"u_{steps[0]}.npy", args.component_axis)
    grid = infer_grid(sample, args.dx, args.dy, args.dz)
    midplane_index = args.midplane_index if args.midplane_index is not None else grid.nz // 2
    if midplane_index < 0 or midplane_index >= grid.nz:
        raise ValueError(f"midplane index {midplane_index} is outside [0, {grid.nz})")

    k, bin_index, valid, counts, _ = k_bins(grid, args.k_bin_width, args.k_max)
    n_bins = len(k)
    comps = component_indices(args.components)
    want_zavg = args.mode in ("zavg", "both")
    want_mid = args.mode in ("midplane", "both")

    zavg_frames: list[np.ndarray] = []
    mid_frames: list[np.ndarray] = []

    print(
        f"Grid: Nx={grid.nx} Ny={grid.ny} Nz={grid.nz} "
        f"Lx={grid.lx:.6g} Ly={grid.ly:.6g} Lz={grid.lz:.6g}"
    )
    print(f"Frames: {len(steps)}  components={args.components}  mode={args.mode}")
    if scipy_fft is not None:
        print(f"FFT backend: scipy workers={args.fft_workers}")
    else:
        print("FFT backend: numpy")
    print(f"k_min={min(2*np.pi/grid.lx, 2*np.pi/grid.ly):.6g}")

    for index, step in enumerate(steps, start=1):
        frame_t0 = time.perf_counter()
        print(f"[{index}/{len(steps)}] step={step} ...", flush=True)
        velocity = load_velocity(data_dir / f"u_{step}.npy", args.component_axis)
        if velocity.shape[:3] != (grid.nx, grid.ny, grid.nz):
            raise ValueError(
                f"u_{step}.npy shape {velocity.shape[:3]} does not match first snapshot "
                f"{(grid.nx, grid.ny, grid.nz)}"
            )
        if want_zavg:
            spectrum, _ = spectrum_zavg_frame(
                velocity, comps, bin_index, valid, n_bins, args.fft_workers
            )
            zavg_frames.append(spectrum)
        if want_mid:
            spectrum, _ = spectrum_midplane_frame(
                velocity, comps, midplane_index, bin_index, valid, n_bins, args.fft_workers
            )
            mid_frames.append(spectrum)
        print(f"[{index}/{len(steps)}] step={step} done in {time.perf_counter() - frame_t0:.2f}s", flush=True)

    zavg_per = np.asarray(zavg_frames, dtype=np.float64) if want_zavg else None
    mid_per = np.asarray(mid_frames, dtype=np.float64) if want_mid else None
    zavg_mean, zavg_std = summarize_spectra(zavg_frames) if want_zavg else (None, None)
    mid_mean, mid_std = summarize_spectra(mid_frames) if want_mid else (None, None)

    out_path = args.out.resolve() if args.out is not None else data_dir / "velocity_spectrum.png"
    csv_path = args.csv.resolve() if args.csv is not None else out_path.with_suffix(".csv")
    npz_path = args.npz.resolve() if args.npz is not None else out_path.with_suffix(".npz")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    plot_spectrum(out_path, k, zavg_mean, zavg_std, mid_mean, mid_std, grid, args.logy)
    write_csv(csv_path, k, counts, zavg_mean, zavg_std, mid_mean, mid_std)
    write_npz(npz_path, k, counts, steps, grid, zavg_per, zavg_mean, zavg_std, mid_per, mid_mean, mid_std)

    print_peak("zavg", k, zavg_mean)
    print_peak("midplane", k, mid_mean)
    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
