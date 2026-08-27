#!/usr/bin/env python3
"""Render channel geometry with the selected disclination loop."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from PIL import Image, ImageDraw


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
LOCAL_NEMATICS_SRC_CANDIDATES = [
    REPO_ROOT.parent / "Nematics3D" / "src",
    REPO_ROOT / "Nematics3D" / "src",
]
for candidate in LOCAL_NEMATICS_SRC_CANDIDATES:
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

import nematics3d as n3d
from pssolver.snapshots import load_q_snapshot
from pssolver.models.active_nematics.nematics3d_adapter import director_from_Q


ALPHA_ON = 5.0
ACTIVITY_BOX_BOUNDS = ((246, 266), (4, 36), (4, 36))
OBSERVATION_BOX_BOUNDS = ((242, 270), (0, 40), (0, 40))
PERIODIC_BOUNDARY = (True, False, False)
DETECTION_PLANES = (True, True, True)
DEFAULT_STEP = 6400
DEFAULT_TARGET_CENTER = np.array([256.5, 26.4, 21.0], dtype=float)


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
    director = director_from_Q(q5)
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


def highlight_loop_projection(output: Path) -> None:
    image = Image.open(output).convert("RGB")
    width, height = image.size
    x0 = round(width * 0.40)
    x1 = round(width * 0.60)
    y0 = round(height * 0.34)
    y1 = round(height * 0.66)

    pixels = image.load()
    coords = []
    for y in range(y0, y1):
        for x in range(x0, x1):
            r, g, b = pixels[x, y]
            if r < 35 and g < 35 and b < 35:
                coords.append((x, y))
    if not coords:
        image.save(output)
        return

    xs = [item[0] for item in coords]
    ys = [item[1] for item in coords]
    padding_x = 42
    padding_y = 44
    box = (
        max(0, min(xs) - padding_x),
        max(0, min(ys) - padding_y),
        min(width - 1, max(xs) + padding_x),
        min(height - 1, max(ys) + padding_y),
    )

    draw = ImageDraw.Draw(image)
    draw.rectangle(box, outline=(210, 35, 35), width=6)
    image.save(output)


def render(
    data_dir: Path,
    output: Path,
    step: int,
    target_center: np.ndarray,
) -> None:
    q5 = load_q_snapshot(
        data_dir / f"Q_{step}.npy",
        require_S_initial=True,
    ).values
    nx, ny, nz = q5.shape[:3]
    loop, center, radius = choose_loop(q5, target_center)

    fig = n3d.PlotFigure(
        is_off_screen=True,
        name=f"channel loop step {step}",
        opts=n3d.OptsFigure(
            size=(2200, 560),
            bg_color=(1.0, 1.0, 1.0),
            focal_point=(nx / 2, ny / 2, nz / 2),
        ),
    )

    add_box(
        fig,
        ((0, nx), (0, ny), (0, nz)),
        radius=0.12,
        color=(0.18, 0.18, 0.18),
        opacity=0.30,
        name="channel domain",
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

    fig.pl.add_axes(
        line_width=3,
        x_color=(0.85, 0.10, 0.10),
        y_color=(0.10, 0.55, 0.10),
        z_color=(0.10, 0.20, 0.85),
        xlabel="x",
        ylabel="y",
        zlabel="z",
        viewport=(0.77, 0.35, 0.97, 0.55),
    )

    fig.act_view_xy()
    fig.act_commit(
        focal_point=(nx / 2, ny / 2, nz / 2),
        distance=270,
        azimuth=90,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.act_savefig(str(output), scale=2)
    highlight_loop_projection(output)
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
        default=REPO_ROOT / "data_control_box" / "loop_summary" / "alpha_geometry.png",
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
