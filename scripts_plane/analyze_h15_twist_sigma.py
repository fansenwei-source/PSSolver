#!/usr/bin/env python3
"""Track wall-to-wall defect displacement and Q-tensor twist in one run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
for candidate in (
    PROJECT_ROOT.parent / "Nematics3D" / "src",
    PROJECT_ROOT / "Nematics3D" / "src",
):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

import nematics3d as n3d
from pssolver.models.active_nematics import Q_convention_metadata, Q_magnitude, S_from_Q
from pssolver.models.active_nematics.nematics3d_adapter import eigenframe_from_Q


Q_PATTERN = re.compile(r"Q_(\d+)\.npy$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step-stride", type=int, default=100)
    parser.add_argument("--start-step", type=int, default=0)
    parser.add_argument("--end-step", type=int, default=None)
    parser.add_argument("--output-name", default="h15_twist_sigma_growth.csv")
    args = parser.parse_args()
    if args.step_stride <= 0:
        parser.error("--step-stride must be positive")
    return args


def selected_steps(run_dir: Path, start: int, end: int | None, stride: int):
    steps = []
    for path in run_dir.glob("Q_*.npy"):
        match = Q_PATTERN.match(path.name)
        if match:
            step = int(match.group(1))
            if step >= start and (end is None or step <= end) and step % stride == 0:
                steps.append(step)
    return sorted(set(steps))


def q_matrix(q):
    matrix = np.empty(q.shape[:-1] + (3, 3), dtype=np.float32)
    matrix[..., 0, 0] = q[..., 0]
    matrix[..., 0, 1] = matrix[..., 1, 0] = q[..., 1]
    matrix[..., 0, 2] = matrix[..., 2, 0] = q[..., 2]
    matrix[..., 1, 1] = q[..., 3]
    matrix[..., 1, 2] = matrix[..., 2, 1] = q[..., 4]
    matrix[..., 2, 2] = -q[..., 0] - q[..., 3]
    return matrix


def derivative(values, axis, spacing):
    if axis in (0, 1):
        return (
            np.roll(values, -1, axis=axis) - np.roll(values, 1, axis=axis)
        ) / (2.0 * spacing)
    return np.gradient(values, spacing, axis=axis, edge_order=2)


def twist_metrics(q, spacings, S_reference):
    epsilon = np.zeros((3, 3, 3), dtype=np.float32)
    epsilon[0, 1, 2] = epsilon[1, 2, 0] = epsilon[2, 0, 1] = 1.0
    epsilon[0, 2, 1] = epsilon[2, 1, 0] = epsilon[1, 0, 2] = -1.0
    matrix = q_matrix(q)
    pseudoscalar = np.zeros(q.shape[:-1], dtype=np.float32)
    z_pseudoscalar = None
    for axis, spacing in enumerate(spacings):
        gradient = derivative(matrix, axis, spacing)
        contribution = np.einsum(
            "il,...ij,...lj->...",
            epsilon[:, axis, :],
            matrix,
            gradient,
            optimize=True,
        )
        pseudoscalar += contribution
        if axis == 2:
            z_pseudoscalar = contribution
    dz_q = derivative(q, 2, spacings[2])
    return {
        "mean_twist_qtensor": float(np.mean((pseudoscalar / S_reference) ** 2)),
        "mean_cross_channel_twist_qtensor": float(
            np.mean((z_pseudoscalar / S_reference) ** 2)
        ),
        "dz_q_rms": float(np.sqrt(np.mean(np.square(dz_q)))),
        "top_bottom_q_rms": float(
            np.sqrt(np.mean((q[:, :, -1] - q[:, :, 0]) ** 2))
        ),
    }


def wall_defects(q_wall, lengths):
    eigenvectors = eigenframe_from_Q(q_wall)
    director = eigenvectors[..., 0]
    indices = n3d.defect_detect(
        director[:, :, None, :],
        threshold=0.0,
        is_boundary_periodic=(True, True, False),
        planes=(False, False, True),
    )[:, :2]
    shape = np.asarray(q_wall.shape[:2], dtype=float)
    return (indices + 0.5) * np.asarray(lengths) / shape


def matched_wall_sigma(bottom, top, lengths):
    if len(bottom) == 0 or len(top) == 0:
        return np.empty(0, dtype=float)
    periods = np.asarray(lengths, dtype=float)
    delta = top[:, None, :] - bottom[None, :, :]
    delta -= periods * np.round(delta / periods)
    distance = np.linalg.norm(delta, axis=-1)
    top_index, bottom_index = linear_sum_assignment(distance)
    return distance[top_index, bottom_index]


def load_metadata(run_dir: Path) -> dict:
    path = run_dir / "metadata.json"
    if not path.is_file():
        raise FileNotFoundError(f"Missing H15 metadata: {path}")
    metadata = json.loads(path.read_text())
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError(f"{path} must contain schema_version=1 metadata")

    solver = metadata.get("solver")
    if not isinstance(solver, dict):
        raise ValueError(f"{path} is missing solver metadata")
    missing_solver = [
        name for name in ("shape", "lengths", "dt") if name not in solver
    ]
    if missing_solver:
        raise ValueError(f"{path} is missing solver keys {missing_solver}")
    shape = solver["shape"]
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 3
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        raise ValueError(f"{path} has invalid solver.shape={shape!r}")
    lengths = solver["lengths"]
    if (
        not isinstance(lengths, (list, tuple))
        or len(lengths) != 3
        or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
            for value in lengths
        )
    ):
        raise ValueError(f"{path} has invalid solver.lengths={lengths!r}")
    dt = solver["dt"]
    if (
        isinstance(dt, bool)
        or not isinstance(dt, (int, float))
        or not np.isfinite(dt)
        or dt <= 0
    ):
        raise ValueError(f"{path} has invalid solver.dt={dt!r}")

    model = metadata.get("model")
    if not isinstance(model, dict) or model.get("name") != "active_nematics":
        raise ValueError(f"{path} must declare model.name='active_nematics'")
    expected_convention = Q_convention_metadata()
    if model.get("Q_convention") != expected_convention:
        raise ValueError(
            f"{path} must declare the complete canonical Q convention "
            f"{expected_convention!r}"
        )
    parameters = model.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"{path} is missing model.parameters")
    for name in ("S_initial", "S_bulk"):
        value = parameters.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not np.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"{path} has invalid model.parameters.{name}={value!r}")
    return metadata


def load_Q(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    Q = np.load(path, mmap_mode="r")
    if Q.shape != (*shape, 5):
        raise ValueError(f"{path} has shape {Q.shape}; expected {(*shape, 5)}")
    if not np.isfinite(Q).all():
        raise ValueError(f"{path} contains non-finite Q values")
    return np.asarray(Q, dtype=np.float32)


def ordered_S_median(Q: np.ndarray) -> float:
    S = np.asarray(S_from_Q(Q))
    magnitude = np.asarray(Q_magnitude(Q))
    ordered_cutoff = float(np.quantile(magnitude, 0.75))
    return float(np.median(S[magnitude >= ordered_cutoff]))


def validate_initial_S(
    run_dir: Path,
    shape: tuple[int, int, int],
    S_initial: float,
) -> float | None:
    initial_path = run_dir / "Q_0.npy"
    if not initial_path.is_file():
        return None
    observed_S = ordered_S_median(load_Q(initial_path, shape))
    tolerance = max(5.0e-3, 0.05 * S_initial)
    if abs(observed_S - S_initial) > tolerance:
        raise ValueError(
            f"{initial_path} has ordered-region median S={observed_S:.8g}, "
            f"inconsistent with model.parameters.S_initial={S_initial:.8g}"
        )
    return observed_S


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    metadata = load_metadata(run_dir)
    solver = metadata["solver"]
    parameters = metadata["model"]["parameters"]
    lengths = tuple(float(value) for value in solver["lengths"])
    shape = tuple(int(value) for value in solver["shape"])
    spacings = tuple(length / count for length, count in zip(lengths, shape))
    S_reference = float(parameters["S_initial"])
    S_bulk = float(parameters["S_bulk"])
    validate_initial_S(run_dir, shape, S_reference)
    steps = selected_steps(
        run_dir, args.start_step, args.end_step, args.step_stride
    )
    if not steps:
        raise RuntimeError("no selected Q snapshots")

    rows = []
    for index, step in enumerate(steps, start=1):
        q = load_Q(run_dir / f"Q_{step}.npy", shape)
        bottom = wall_defects(q[:, :, 0, :], lengths[:2])
        top = wall_defects(q[:, :, -1, :], lengths[:2])
        sigma = matched_wall_sigma(bottom, top, lengths[:2])
        row = {
            "step": step,
            "time": step * float(solver["dt"]),
            "S_reference": S_reference,
            "S_bulk": S_bulk,
            "ordered_S_median": ordered_S_median(q),
            "bottom_defect_count": len(bottom),
            "top_defect_count": len(top),
            "matched_defect_count": len(sigma),
            "mean_sigma_over_h": float(np.mean(sigma) / lengths[2]) if len(sigma) else math.nan,
            "max_sigma_over_h": float(np.max(sigma) / lengths[2]) if len(sigma) else math.nan,
            **twist_metrics(q, spacings, S_reference),
        }
        rows.append(row)
        print(
            f"[{index}/{len(steps)}] step={step} defects={len(bottom)}/{len(top)} "
            f"mean_sigma/H={row['mean_sigma_over_h']:.6g} "
            f"Tz={row['mean_cross_channel_twist_qtensor']:.6g}"
        )

    output = run_dir / args.output_name
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
