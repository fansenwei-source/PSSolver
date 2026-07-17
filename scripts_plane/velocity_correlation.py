#!/usr/bin/env python3
"""Compute planar velocity correlations and vortex size from saved snapshots."""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - depends on local environment
    raise SystemExit("matplotlib is required to write velocity-correlation PNG files") from exc


ROOT = Path(__file__).resolve().parents[1]
U_PATTERN = re.compile(r"^u_(\d+)\.npy$")
DEFAULT_DATA_DIR = ROOT / "data_plane_H=10"
DEFAULT_NX, DEFAULT_NY, DEFAULT_NZ = 512, 512, 40
DEFAULT_LX, DEFAULT_LY, DEFAULT_LZ = 128.0, 128.0, 10.0
DEFAULT_DX = DEFAULT_LX / DEFAULT_NX
DEFAULT_DY = DEFAULT_LY / DEFAULT_NY
DEFAULT_DZ = DEFAULT_LZ / DEFAULT_NZ


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


@dataclass
class CorrelationResult:
    mean: np.ndarray
    std: np.ndarray
    per_frame: np.ndarray
    vortex_size: float | None
    vortex_index: int | None
    vortex_value: float | None
    unresolved_reason: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute z-averaged and/or midplane planar velocity correlations "
            "from PSSolver data/u_<step>.npy snapshots and mark the vortex size."
        )
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PNG path. Default: <data-dir>/velocity_correlation.png",
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
    parser.add_argument("--dx", type=float, default=DEFAULT_DX)
    parser.add_argument("--dy", type=float, default=DEFAULT_DY)
    parser.add_argument("--dz", type=float, default=DEFAULT_DZ)
    parser.add_argument(
        "--bin-width",
        type=float,
        default=None,
        help="Radial bin width. Default: dx.",
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=None,
        help="Maximum reliable horizontal distance. Default: 0.5 * min(Lx, Ly).",
    )
    parser.add_argument(
        "--mode",
        choices=("zavg", "midplane", "both"),
        default="both",
        help="Correlation mode to compute.",
    )
    parser.add_argument(
        "--components",
        choices=("xyz", "xy"),
        default="xyz",
        help="Velocity components used in dot products.",
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
        "--min-radius",
        type=float,
        default=None,
        help="Smallest radius allowed when detecting vortex size. Default: 2 * dx.",
    )
    parser.add_argument(
        "--edge-fraction",
        type=float,
        default=0.05,
        help=(
            "If the minimum is in the last fraction of the reliable radial range, "
            "mark vortex size as unresolved. Default: 0.05."
        ),
    )
    parser.add_argument(
        "--midplane-index",
        type=int,
        default=None,
        help="z index for midplane mode. Default: Nz//2.",
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


def radial_bins(grid: Grid, bin_width: float, r_max: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dx_idx = np.minimum(np.arange(grid.nx), grid.nx - np.arange(grid.nx))
    dy_idx = np.minimum(np.arange(grid.ny), grid.ny - np.arange(grid.ny))
    xdist = dx_idx[:, None] * grid.dx
    ydist = dy_idx[None, :] * grid.dy
    radius = np.sqrt(xdist**2 + ydist**2)

    bin_index = np.floor(radius / bin_width).astype(np.int64)
    max_bin = int(np.floor(r_max / bin_width))
    valid = (radius <= r_max + 1e-12) & (bin_index <= max_bin)
    centers = (np.arange(max_bin + 1, dtype=np.float64) + 0.5) * bin_width
    centers[0] = 0.0
    counts = np.bincount(bin_index[valid].ravel(), minlength=max_bin + 1).astype(np.int64)
    populated = np.zeros_like(valid, dtype=bool)
    populated[valid] = counts[bin_index[valid]] > 0
    return centers, bin_index, valid & populated


def fft_autocorrelation_2d(field: np.ndarray) -> np.ndarray:
    spectrum = np.fft.rfftn(field, axes=(0, 1))
    corr = np.fft.irfftn(spectrum * np.conj(spectrum), s=field.shape[:2], axes=(0, 1))
    return np.asarray(corr.real, dtype=np.float64)


def radial_average(
    corr_xy: np.ndarray,
    bin_index: np.ndarray,
    valid: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    sums = np.bincount(
        bin_index[valid].ravel(),
        weights=corr_xy[valid].ravel(),
        minlength=n_bins,
    )
    counts = np.bincount(bin_index[valid].ravel(), minlength=n_bins)
    out = np.full(n_bins, np.nan, dtype=np.float64)
    np.divide(sums, counts, out=out, where=counts > 0)
    return out


def zavg_correlation_frame(
    velocity: np.ndarray,
    comps: tuple[int, ...],
    bin_index: np.ndarray,
    valid: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    corr_sum = np.zeros(velocity.shape[:2], dtype=np.float64)
    energy = 0.0
    for comp in comps:
        field = velocity[..., comp]
        corr_sum += np.sum(fft_autocorrelation_2d(field), axis=2)
        energy += float(np.sum(field * field))
    if energy <= 0.0:
        raise ValueError("Velocity energy is zero; cannot normalize correlation")
    return radial_average(corr_sum / energy, bin_index, valid, n_bins)


def midplane_correlation_frame(
    velocity: np.ndarray,
    comps: tuple[int, ...],
    midplane_index: int,
    bin_index: np.ndarray,
    valid: np.ndarray,
    n_bins: int,
) -> np.ndarray:
    corr_sum = np.zeros(velocity.shape[:2], dtype=np.float64)
    energy = 0.0
    for comp in comps:
        field = velocity[:, :, midplane_index, comp]
        corr_sum += fft_autocorrelation_2d(field)
        energy += float(np.sum(field * field))
    if energy <= 0.0:
        raise ValueError("Midplane velocity energy is zero; cannot normalize correlation")
    return radial_average(corr_sum / energy, bin_index, valid, n_bins)


def energy_diagnostics(velocity: np.ndarray) -> tuple[np.ndarray, float]:
    energy_z = np.sum(velocity * velocity, axis=(0, 1, 3))
    total = float(np.sum(energy_z))
    if total <= 0.0:
        return energy_z, np.nan
    uz_energy = float(np.sum(velocity[..., 2] * velocity[..., 2]))
    return energy_z, uz_energy / total


def summarize_correlation(
    per_frame: list[np.ndarray],
    radii: np.ndarray,
    min_radius: float,
    r_max: float,
    edge_fraction: float,
) -> CorrelationResult:
    array = np.asarray(per_frame, dtype=np.float64)
    mean = np.nanmean(array, axis=0)
    std = np.nanstd(array, axis=0)

    search = np.isfinite(mean) & (radii >= min_radius) & (radii <= r_max)
    if not np.any(search):
        return CorrelationResult(mean, std, array, None, None, None, "no valid radial bins")

    candidate_indices = np.flatnonzero(search)
    local = candidate_indices[np.nanargmin(mean[candidate_indices])]
    vortex_value = float(mean[local])
    vortex_size = float(radii[local])

    if vortex_value >= 0.0:
        return CorrelationResult(
            mean, std, array, None, None, vortex_value, "minimum correlation is non-negative"
        )
    edge_start = r_max * (1.0 - edge_fraction)
    if vortex_size >= edge_start:
        return CorrelationResult(
            mean,
            std,
            array,
            None,
            int(local),
            vortex_value,
            "minimum lies too close to r_max; domain may not resolve vortex size",
        )
    return CorrelationResult(mean, std, array, vortex_size, int(local), vortex_value, None)


def write_csv(
    path: Path,
    radii: np.ndarray,
    zavg: CorrelationResult | None,
    mid: CorrelationResult | None,
) -> None:
    fieldnames = ["r"]
    if zavg is not None:
        fieldnames += ["C_zavg_mean", "C_zavg_std"]
    if mid is not None:
        fieldnames += ["C_mid_mean", "C_mid_std"]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for i, radius in enumerate(radii):
            row: dict[str, float] = {"r": float(radius)}
            if zavg is not None:
                row["C_zavg_mean"] = float(zavg.mean[i])
                row["C_zavg_std"] = float(zavg.std[i])
            if mid is not None:
                row["C_mid_mean"] = float(mid.mean[i])
                row["C_mid_std"] = float(mid.std[i])
            writer.writerow(row)


def write_npz(
    path: Path,
    radii: np.ndarray,
    steps: list[int],
    grid: Grid,
    r_max: float,
    zavg: CorrelationResult | None,
    mid: CorrelationResult | None,
    energy_z_mean: np.ndarray,
    rz_mean: float,
) -> None:
    payload: dict[str, object] = {
        "r": radii,
        "steps": np.asarray(steps, dtype=np.int64),
        "dx": grid.dx,
        "dy": grid.dy,
        "dz": grid.dz,
        "Lx": grid.lx,
        "Ly": grid.ly,
        "Lz": grid.lz,
        "r_max": r_max,
        "energy_z_mean": energy_z_mean,
        "wall_normal_energy_fraction": rz_mean,
    }
    if zavg is not None:
        payload.update(
            C_zavg_mean=zavg.mean,
            C_zavg_std=zavg.std,
            C_zavg_per_frame=zavg.per_frame,
            vortex_size_zavg=np.nan if zavg.vortex_size is None else zavg.vortex_size,
            vortex_value_zavg=np.nan if zavg.vortex_value is None else zavg.vortex_value,
            unresolved_zavg="" if zavg.unresolved_reason is None else zavg.unresolved_reason,
        )
    if mid is not None:
        payload.update(
            C_mid_mean=mid.mean,
            C_mid_std=mid.std,
            C_mid_per_frame=mid.per_frame,
            vortex_size_mid=np.nan if mid.vortex_size is None else mid.vortex_size,
            vortex_value_mid=np.nan if mid.vortex_value is None else mid.vortex_value,
            unresolved_mid="" if mid.unresolved_reason is None else mid.unresolved_reason,
        )
    np.savez_compressed(path, **payload)


def plot_results(
    path: Path,
    radii: np.ndarray,
    zavg: CorrelationResult | None,
    mid: CorrelationResult | None,
    r_max: float,
) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2), dpi=180)
    if zavg is not None:
        ax.plot(radii, zavg.mean, label="z-averaged", linewidth=1.9)
        ax.fill_between(radii, zavg.mean - zavg.std, zavg.mean + zavg.std, alpha=0.15)
        if zavg.vortex_size is not None:
            ax.axvline(zavg.vortex_size, color="C0", linestyle="--", linewidth=1.4)
            ax.annotate(
                f"$\\ell_{{zavg}}$={zavg.vortex_size:.3g}",
                xy=(zavg.vortex_size, zavg.vortex_value),
                xytext=(8, 10),
                textcoords="offset points",
                color="C0",
            )
    if mid is not None:
        ax.plot(radii, mid.mean, label="midplane", linewidth=1.9)
        ax.fill_between(radii, mid.mean - mid.std, mid.mean + mid.std, alpha=0.15)
        if mid.vortex_size is not None:
            ax.axvline(mid.vortex_size, color="C1", linestyle="--", linewidth=1.4)
            ax.annotate(
                f"$\\ell_{{mid}}$={mid.vortex_size:.3g}",
                xy=(mid.vortex_size, mid.vortex_value),
                xytext=(8, -16),
                textcoords="offset points",
                color="C1",
            )
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.55)
    ax.set_xlim(0.0, r_max)
    ax.set_xlabel(r"horizontal distance $\Delta r_\parallel$")
    ax.set_ylabel(r"velocity correlation $C(\Delta r_\parallel)$")
    ax.set_title("Planar Velocity Correlation")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def print_result(name: str, result: CorrelationResult | None) -> None:
    if result is None:
        return
    if result.vortex_size is None:
        print(f"{name}: no well-defined vortex size ({result.unresolved_reason})")
    else:
        print(
            f"{name}: vortex_size={result.vortex_size:.6g} "
            f"C(vortex_size)={result.vortex_value:.6e}"
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
    bin_width = args.bin_width if args.bin_width is not None else args.dx
    if bin_width <= 0.0:
        raise ValueError("--bin-width must be positive")
    r_max = args.r_max if args.r_max is not None else 0.5 * min(grid.lx, grid.ly)
    if r_max <= 0.0:
        raise ValueError("--r-max must be positive")
    min_radius = args.min_radius if args.min_radius is not None else 2.0 * args.dx
    midplane_index = args.midplane_index if args.midplane_index is not None else grid.nz // 2
    if midplane_index < 0 or midplane_index >= grid.nz:
        raise ValueError(f"midplane index {midplane_index} is outside [0, {grid.nz})")

    radii, bin_index, valid = radial_bins(grid, bin_width, r_max)
    n_bins = len(radii)
    comps = component_indices(args.components)
    want_zavg = args.mode in ("zavg", "both")
    want_mid = args.mode in ("midplane", "both")

    zavg_frames: list[np.ndarray] = []
    mid_frames: list[np.ndarray] = []
    energy_z_frames: list[np.ndarray] = []
    rz_values: list[float] = []

    print(
        f"Grid: Nx={grid.nx} Ny={grid.ny} Nz={grid.nz} "
        f"Lx={grid.lx:.6g} Ly={grid.ly:.6g} Lz={grid.lz:.6g}"
    )
    print(
        f"Frames: {len(steps)}  r_max={r_max:.6g}  bin_width={bin_width:.6g} "
        f"components={args.components}  mode={args.mode}"
    )

    for index, step in enumerate(steps, start=1):
        velocity = load_velocity(data_dir / f"u_{step}.npy", args.component_axis)
        if velocity.shape[:3] != (grid.nx, grid.ny, grid.nz):
            raise ValueError(
                f"u_{step}.npy shape {velocity.shape[:3]} does not match first snapshot "
                f"{(grid.nx, grid.ny, grid.nz)}"
            )
        energy_z, rz = energy_diagnostics(velocity)
        energy_z_frames.append(energy_z)
        rz_values.append(rz)
        if want_zavg:
            zavg_frames.append(zavg_correlation_frame(velocity, comps, bin_index, valid, n_bins))
        if want_mid:
            mid_frames.append(
                midplane_correlation_frame(velocity, comps, midplane_index, bin_index, valid, n_bins)
            )
        print(f"[{index}/{len(steps)}] step={step} Rz={rz:.6e}", flush=True)

    zavg = (
        summarize_correlation(zavg_frames, radii, min_radius, r_max, args.edge_fraction)
        if want_zavg
        else None
    )
    mid = (
        summarize_correlation(mid_frames, radii, min_radius, r_max, args.edge_fraction)
        if want_mid
        else None
    )
    energy_z_mean = np.mean(np.asarray(energy_z_frames), axis=0)
    rz_mean = float(np.nanmean(np.asarray(rz_values)))

    out_path = (
        args.out.resolve()
        if args.out is not None
        else data_dir / "velocity_correlation.png"
    )
    csv_path = (
        args.csv.resolve()
        if args.csv is not None
        else out_path.with_suffix(".csv")
    )
    npz_path = (
        args.npz.resolve()
        if args.npz is not None
        else out_path.with_suffix(".npz")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    npz_path.parent.mkdir(parents=True, exist_ok=True)

    plot_results(out_path, radii, zavg, mid, r_max)
    write_csv(csv_path, radii, zavg, mid)
    write_npz(npz_path, radii, steps, grid, r_max, zavg, mid, energy_z_mean, rz_mean)

    print_result("zavg", zavg)
    print_result("midplane", mid)
    if zavg is not None and mid is not None and zavg.vortex_size and mid.vortex_size:
        rel = abs(mid.vortex_size - zavg.vortex_size) / max(abs(zavg.vortex_size), 1e-30)
        print(f"midplane_vs_zavg_relative_difference={rel:.6e}")
    print(f"wall_normal_energy_fraction_mean={rz_mean:.6e}")
    print(f"Wrote {out_path}")
    print(f"Wrote {csv_path}")
    print(f"Wrote {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
