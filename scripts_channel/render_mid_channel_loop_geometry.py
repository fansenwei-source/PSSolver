#!/usr/bin/env python3
"""Render the middle half of the channel geometry with one disclination loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_NEMATICS_SRC_CANDIDATES = [
    REPO_ROOT.parent / "Nematics3D" / "src",
    REPO_ROOT / "Nematics3D" / "src",
]
for candidate in LOCAL_NEMATICS_SRC_CANDIDATES:
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

import nematics3d as n3d


ALPHA_ON = 5.0
ACTIVITY_BOX_BOUNDS = ((246, 266), (4, 36), (4, 36))
OBSERVATION_BOX_BOUNDS = ((242, 270), (0, 40), (0, 40))
PERIODIC_BOUNDARY = (True, False, False)
DETECTION_PLANES = (True, True, True)
DEFAULT_STEP = 6450
DEFAULT_TARGET_CENTER = np.array([256.5, 26.4, 21.0], dtype=float)


def middle_half_bounds(
    shape: tuple[int, int, int],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    nx, ny, nz = shape
    return (
        (float(nx) * 0.25, float(nx) * 0.75),
        (0.0, float(ny)),
        (0.0, float(nz)),
    )


def add_box(
    fig: n3d.PlotFigure,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    *,
    radius: float,
    color: tuple[float, float, float],
    opacity: float,
    name: str,
) -> None:
    corners = n3d.get_box_corners(
        bounds[0][1] - bounds[0][0],
        bounds[1][1] - bounds[1][0],
        bounds[2][1] - bounds[2][0],
    )
    corners += np.array([bounds[0][0], bounds[1][0], bounds[2][0]], dtype=float)
    n3d.PlotExtent(
        corners,
        figure=fig,
        name=name,
        opts=n3d.OptsTube(
            radius=radius,
            color=color,
            opacity=opacity,
            sides=12,
            is_scalar_bar=False,
            is_reset_camera=False,
        ),
    )


def slice_points(
    alpha: np.ndarray,
    axis: int,
    index: int,
    threshold: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return nonzero alpha points on one coordinate slice."""
    if axis == 0:
        plane = alpha[index, :, :]
        yy, zz = np.meshgrid(np.arange(alpha.shape[1]), np.arange(alpha.shape[2]), indexing="ij")
        coords = np.column_stack([
            np.full(plane.size, float(index)),
            yy.ravel().astype(float),
            zz.ravel().astype(float),
        ])
    elif axis == 1:
        plane = alpha[:, index, :]
        xx, zz = np.meshgrid(np.arange(alpha.shape[0]), np.arange(alpha.shape[2]), indexing="ij")
        coords = np.column_stack([
            xx.ravel().astype(float),
            np.full(plane.size, float(index)),
            zz.ravel().astype(float),
        ])
    elif axis == 2:
        plane = alpha[:, :, index]
        xx, yy = np.meshgrid(np.arange(alpha.shape[0]), np.arange(alpha.shape[1]), indexing="ij")
        coords = np.column_stack([
            xx.ravel().astype(float),
            yy.ravel().astype(float),
            np.full(plane.size, float(index)),
        ])
    else:
        raise ValueError(f"axis must be 0, 1, or 2; got {axis}")

    scalars = plane.ravel().astype(float)
    keep = scalars > threshold
    return coords[keep], scalars[keep]


def add_alpha_slice(
    fig: n3d.PlotFigure,
    alpha: np.ndarray,
    axis: int,
    index: int,
    name: str,
) -> object | None:
    coords, scalars = slice_points(alpha, axis, index)
    if len(coords) == 0:
        return None
    return n3d.PlotSurface(
        coords,
        figure=fig,
        name=name,
        paint_by="scalars",
        scalars=scalars,
        scalars_cmap="Reds",
        scalars_clim=(0.0, ALPHA_ON),
        opacity=0.72,
        is_scalar_bar=False,
        is_reset_camera=False,
    )


def detect_loops(q5: np.ndarray) -> list[object]:
    _, director = n3d.Q_diagonalize(q5)
    defects = n3d.defect_detect(
        director,
        threshold=0.0,
        is_boundary_periodic=PERIODIC_BOUNDARY,
        planes=DETECTION_PLANES,
    )
    lines = n3d.defect_classify_into_lines(
        defects,
        box_size_periodic=[
            director.shape[0],
            np.inf,
            np.inf,
        ],
        grid_offset=np.zeros(3),
        grid_transform=np.eye(3),
    )
    return [line for line in lines if getattr(line, "kind", None) == "loop"]


def smooth_or_raw_coords(line: object) -> np.ndarray:
    if getattr(line, "calc_defect_num", 0) >= 5:
        try:
            smooth = line.act_smooth(
                window_length=5,
                min_line_length=5,
                is_window_warning=False,
            )
            coords = np.asarray(smooth.result, dtype=float)
            if coords.ndim == 2 and coords.shape[1] == 3:
                return coords
        except Exception:
            pass
    return np.asarray(line.calc_defect_coords, dtype=float)


def choose_loop(q5: np.ndarray, target: np.ndarray) -> tuple[object, np.ndarray, float]:
    loops = detect_loops(q5)
    if not loops:
        raise RuntimeError("no loop lines detected")
    candidates = []
    for line in loops:
        coords = smooth_or_raw_coords(line)
        center = coords.mean(axis=0)
        dist = float(np.linalg.norm(center - target))
        candidates.append((dist, line, center))
    _, line, center = min(candidates, key=lambda item: item[0])
    radius = float(np.sqrt(np.mean(np.sum((smooth_or_raw_coords(line) - center) ** 2, axis=1))))
    return line, center, radius


def trim_output_to_content(output: Path, padding: int = 24) -> None:
    image = Image.open(output).convert("RGB")
    mask = image.point(lambda value: 255 if value < 248 else 0).convert("L")
    bbox = mask.getbbox()
    if bbox is None:
        image.save(output)
        return
    left, upper, right, lower = bbox
    left = max(0, left - padding)
    upper = max(0, upper - padding)
    right = min(image.width, right + padding)
    lower = min(image.height, lower + padding)
    image.crop((left, upper, right, lower)).save(output)


def render(
    data_dir: Path,
    output: Path,
    step: int,
    target_center: np.ndarray,
) -> None:
    q5 = np.load(data_dir / f"Q_{step}.npy")
    nx, ny, nz = q5.shape[:3]
    loop, center, radius = choose_loop(q5, target_center)

    fig = n3d.PlotFigure(
        is_off_screen=True,
        name=f"channel loop step {step}",
        opts=n3d.OptsFigure(
            size=(2200, 360),
            bg_color=(1.0, 1.0, 1.0),
            focal_point=(nx / 2, ny / 2, nz / 2),
        ),
    )

    add_box(
        fig,
        middle_half_bounds((nx, ny, nz)),
        radius=0.12,
        color=(0.18, 0.18, 0.18),
        opacity=0.30,
        name="channel domain",
    )
    add_box(
        fig,
        ACTIVITY_BOX_BOUNDS,
        radius=0.22,
        color=(1.0, 0.0, 0.0),
        opacity=1.0,
        name="alpha field box",
    )
    loop.act_visualize(
        figure=fig,
        is_wrap=True,
        is_smooth=True,
        opts=n3d.OptsTube(
            radius=0.38,
            color=(0.0, 0.0, 0.0),
            opacity=1.0,
            sides=32,
            is_scalar_bar=False,
            is_reset_camera=False,
        ),
    )

    fig.act_view_xy()
    fig.act_commit(
        focal_point=(nx / 2, ny / 2, nz / 2),
        distance=230,
        azimuth=90,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.act_savefig(str(output), scale=2)
    trim_output_to_content(output)
    print(
        f"step={step} center={center.round(3).tolist()} radius={radius:.3f} "
        f"png={output}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data_control_box")
    parser.add_argument("--step", type=int, default=DEFAULT_STEP)
    parser.add_argument("--target-center", type=float, nargs=3, default=DEFAULT_TARGET_CENTER.tolist())
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT
        / "data_control_box"
        / "mid_channel_loop_geometry"
        / "mid_channel_loop_geometry.png",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render(
        args.data_dir,
        args.output,
        step=int(args.step),
        target_center=np.asarray(args.target_center, dtype=float),
    )


if __name__ == "__main__":
    main()
