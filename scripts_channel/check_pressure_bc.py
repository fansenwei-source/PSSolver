#!/usr/bin/env python3
"""Check pressure wall compatibility from saved Q/u/p snapshots.

The diagnostic is evaluated with the same FFT/DST/DCT modal derivatives used by
Plane.py. For the two near-wall collocation planes it checks

    dz(p) = fz + eta * laplacian(uz) - fric * uz

where fz = beta * alpha * (dx Qxz + dy Qyz - dz Qxx - dz Qyy).
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pssolver.transforms import TensorProductTransformBackend  # noqa: E402


Q_BC = ("periodic", "periodic", "neumann")
U_BC = ("periodic", "periodic", "dirichlet")
P_BC = ("periodic", "periodic", "neumann")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check saved pressure snapshots against the wall normal momentum condition.",
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--Lx", type=float, default=64.0)
    parser.add_argument("--Ly", type=float, default=64.0)
    parser.add_argument("--Lz", type=float, default=30.0)
    parser.add_argument("--alpha", type=float, default=5.0)
    parser.add_argument("--beta", type=float, default=-1.0)
    parser.add_argument("--fric", type=float, default=0.1)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--steps", type=int, nargs="*", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dtype", default="float32", choices=("float32", "float64"))
    return parser.parse_args()


def snapshot_steps(data_dir: Path) -> list[int]:
    pattern = re.compile(r"^([Qup])_(\d+)\.npy$")
    available: dict[str, set[int]] = {"Q": set(), "u": set(), "p": set()}
    for path in data_dir.glob("*.npy"):
        match = pattern.match(path.name)
        if match:
            available[match.group(1)].add(int(match.group(2)))
    return sorted(available["Q"] & available["u"] & available["p"])


def load_snapshot(path: Path, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    array = np.load(path)
    return torch.as_tensor(array, device=device, dtype=dtype)


def transform_tensor(
    backend: TensorProductTransformBackend,
    tensor: torch.Tensor,
    boundary_conditions: tuple[str, str, str],
) -> torch.Tensor:
    return backend.forward(tensor.unsqueeze(0), boundary_conditions)


def inverse_tensor(
    backend: TensorProductTransformBackend,
    spectral: torch.Tensor,
    boundary_conditions: tuple[str, str, str],
) -> torch.Tensor:
    return backend.inverse(spectral, boundary_conditions).squeeze(0)


def gradient(
    backend: TensorProductTransformBackend,
    tensor: torch.Tensor,
    boundary_conditions: tuple[str, str, str],
    axis: int,
) -> torch.Tensor:
    spectral = transform_tensor(backend, tensor, boundary_conditions)
    grad_hat, grad_bcs = backend.gradient_hat(spectral, boundary_conditions, axis)
    return inverse_tensor(backend, grad_hat, grad_bcs)


def laplacian(
    backend: TensorProductTransformBackend,
    tensor: torch.Tensor,
    boundary_conditions: tuple[str, str, str],
) -> torch.Tensor:
    spectral = transform_tensor(backend, tensor, boundary_conditions)
    lap_hat = backend.laplacian_hat(spectral, boundary_conditions)
    return inverse_tensor(backend, lap_hat, boundary_conditions)


@torch.no_grad()
def check_step(
    data_dir: Path,
    step: int,
    backend: TensorProductTransformBackend,
    device: torch.device,
    dtype: torch.dtype,
    alpha: float,
    beta: float,
    fric: float,
    eta: float,
) -> dict[str, float]:
    q = load_snapshot(data_dir / f"Q_{step}.npy", device, dtype)
    u = load_snapshot(data_dir / f"u_{step}.npy", device, dtype)
    p = load_snapshot(data_dir / f"p_{step}.npy", device, dtype)

    if q.ndim != 4 or q.shape[-1] != 5:
        raise ValueError(f"Q_{step}.npy must have shape (Nx, Ny, Nz, 5), got {tuple(q.shape)}")
    if u.ndim != 4 or u.shape[-1] != 3:
        raise ValueError(f"u_{step}.npy must have shape (Nx, Ny, Nz, 3), got {tuple(u.shape)}")
    if p.shape != q.shape[:3]:
        raise ValueError(f"p_{step}.npy shape {tuple(p.shape)} does not match Q grid {tuple(q.shape[:3])}")

    qxx = q[..., 0]
    qxz = q[..., 2]
    qyy = q[..., 3]
    qyz = q[..., 4]
    uz = u[..., 2]

    fz = beta * alpha * (
        gradient(backend, qxz, Q_BC, axis=0)
        + gradient(backend, qyz, Q_BC, axis=1)
        - gradient(backend, qxx, Q_BC, axis=2)
        - gradient(backend, qyy, Q_BC, axis=2)
    )
    dzp = gradient(backend, p, P_BC, axis=2)
    lap_uz = laplacian(backend, uz, U_BC)

    rhs = fz + eta * lap_uz - fric * uz
    residual = dzp - rhs
    wall = torch.stack((residual[..., 0], residual[..., -1]), dim=-1)
    wall_rhs = torch.stack((rhs[..., 0], rhs[..., -1]), dim=-1)
    wall_dzp = torch.stack((dzp[..., 0], dzp[..., -1]), dim=-1)

    wall_abs = wall.abs()
    rhs_rms = torch.sqrt(torch.mean(wall_rhs.abs().square())).item()
    dzp_rms = torch.sqrt(torch.mean(wall_dzp.abs().square())).item()
    residual_rms = torch.sqrt(torch.mean(wall_abs.square())).item()
    relative = residual_rms / max(rhs_rms, dzp_rms, 1e-30)

    return {
        "step": step,
        "wall_residual_max": wall_abs.max().item(),
        "wall_residual_rms": residual_rms,
        "wall_residual_relative": relative,
        "wall_dzp_rms": dzp_rms,
        "wall_rhs_rms": rhs_rms,
        "bottom_residual_rms": torch.sqrt(torch.mean(residual[..., 0].abs().square())).item(),
        "top_residual_rms": torch.sqrt(torch.mean(residual[..., -1].abs().square())).item(),
    }


def main() -> int:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    steps = snapshot_steps(data_dir)
    if args.steps is not None:
        requested = set(args.steps)
        steps = [step for step in steps if step in requested]
    if args.limit is not None:
        steps = steps[: args.limit]
    if not steps:
        raise RuntimeError(f"No matching Q/u/p snapshots found in {data_dir}")

    sample = np.load(data_dir / f"Q_{steps[0]}.npy", mmap_mode="r")
    shape = tuple(sample.shape[:3])

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    backend = TensorProductTransformBackend(
        shape=shape,
        lengths=(args.Lx, args.Ly, args.Lz),
        device=str(device),
        dtype=dtype,
    )

    rows = []
    for index, step in enumerate(steps, start=1):
        row = check_step(
            data_dir=data_dir,
            step=step,
            backend=backend,
            device=device,
            dtype=dtype,
            alpha=args.alpha,
            beta=args.beta,
            fric=args.fric,
            eta=args.eta,
        )
        rows.append(row)
        print(
            f"[{index}/{len(steps)}] step={step} "
            f"wall_rms={row['wall_residual_rms']:.6e} "
            f"wall_rel={row['wall_residual_relative']:.6e} "
            f"wall_max={row['wall_residual_max']:.6e}",
            flush=True,
        )

    out_base = args.out
    if out_base is None:
        out_base = data_dir / "pressure_bc_diagnostics"
    out_base.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(rows[0].keys())
    csv_path = out_base.with_suffix(".csv")
    npy_path = out_base.with_suffix(".npy")

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    dtype_fields = [("step", np.int64)] + [
        (name, np.float64) for name in fieldnames if name != "step"
    ]
    array = np.array([tuple(row[name] for name in fieldnames) for row in rows], dtype=dtype_fields)
    np.save(npy_path, array)

    worst = max(rows, key=lambda row: row["wall_residual_relative"])
    print(f"Saved CSV: {csv_path}")
    print(f"Saved NPY: {npy_path}")
    print(
        "Worst relative residual: "
        f"step={int(worst['step'])}, "
        f"wall_rel={worst['wall_residual_relative']:.6e}, "
        f"wall_rms={worst['wall_residual_rms']:.6e}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
