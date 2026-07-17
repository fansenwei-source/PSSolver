#!/usr/bin/env python3
"""Render one local disclination loop as it grows and shrinks without director rods."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


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
from nematics3d.classes.bounds import OptsBounds
from nematics3d.classes.visual.plot_figure import OptsFigure, PlotFigure
from nematics3d.classes.visual.plot_rod import OptsRod


PERIODIC_BOUNDARY = (True, False, False)
DETECTION_PLANES = (True, True, True)
DEFAULT_STEPS = (6200, 6360, 7000)
DEFAULT_TARGET_CENTER = np.array([256.5, 26.4, 21.0], dtype=float)


@dataclass(frozen=True)
class LoopInfo:
    step: int
    line: object
    coords: np.ndarray
    center: np.ndarray
    radius: float
    points: int


@dataclass(frozen=True)
class RenderBox:
    center: np.ndarray
    extent: tuple[float, float, float, float, float, float]


def q5_to_tensor(q5: np.ndarray) -> np.ndarray:
    """Convert component-last 5-component Q storage into a 3x3 tensor field."""
    qxx = q5[..., 0]
    qxy = q5[..., 1]
    qxz = q5[..., 2]
    qyy = q5[..., 3]
    qyz = q5[..., 4]
    qzz = -qxx - qyy
    out = np.empty(q5.shape[:3] + (3, 3), dtype=np.float64)
    out[..., 0, 0] = qxx
    out[..., 0, 1] = qxy
    out[..., 0, 2] = qxz
    out[..., 1, 0] = qxy
    out[..., 1, 1] = qyy
    out[..., 1, 2] = qyz
    out[..., 2, 0] = qxz
    out[..., 2, 1] = qyz
    out[..., 2, 2] = qzz
    return out


def detect_loops(q5: np.ndarray) -> list[object]:
    _, director = n3d.Q_diagonalize(q5)
    defects = n3d.defect_detect(
        director,
        threshold=0.0,
        is_boundary_periodic=PERIODIC_BOUNDARY,
        planes=DETECTION_PLANES,
    )
    box_size_periodic = [
        director.shape[axis] if PERIODIC_BOUNDARY[axis] else np.inf
        for axis in range(3)
    ]
    lines = n3d.defect_classify_into_lines(
        defects,
        box_size_periodic=box_size_periodic,
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


def loop_radius(coords: np.ndarray) -> float:
    center = coords.mean(axis=0)
    delta = coords - center
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def choose_loop(step: int, q5: np.ndarray, target: np.ndarray) -> LoopInfo:
    loops = detect_loops(q5)
    if not loops:
        raise RuntimeError(f"step {step}: no loop lines detected")

    candidates: list[LoopInfo] = []
    for line in loops:
        coords = smooth_or_raw_coords(line)
        center = coords.mean(axis=0)
        candidates.append(
            LoopInfo(
                step=step,
                line=line,
                coords=coords,
                center=center,
                radius=loop_radius(coords),
                points=int(getattr(line, "calc_defect_num", len(coords))),
            )
        )
    return min(candidates, key=lambda item: float(np.linalg.norm(item.center - target)))


def sample_local_directors(
    q5: np.ndarray,
    center: np.ndarray,
    half_width: float,
    spacing: int,
    plane_offsets: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    _, director = n3d.Q_diagonalize(q5)
    nx, ny, nz = q5.shape[:3]
    y_min = max(1, int(np.floor(center[1] - half_width)))
    y_max = min(ny - 2, int(np.ceil(center[1] + half_width)))
    z_min = max(1, int(np.floor(center[2] - half_width)))
    z_max = min(nz - 2, int(np.ceil(center[2] + half_width)))

    coords = []
    orient = []
    scalars = []
    for dx in plane_offsets:
        x = int(round(center[0] + dx)) % nx
        for y in range(y_min, y_max + 1, spacing):
            for z in range(z_min, z_max + 1, spacing):
                n = director[x, y, z]
                coords.append((float(x), float(y), float(z)))
                orient.append(n)
                scalars.append(float(n[0]))
    return np.asarray(coords), np.asarray(orient), np.asarray(scalars)


def local_bounds(box: RenderBox) -> n3d.Bounds:
    x0, x1, y0, y1, z0, z1 = box.extent
    return n3d.Bounds(
        opts=OptsBounds(
            origin=(x0, y0, z0),
            axis1=(1.0, 0.0, 0.0),
            axis2=(0.0, 1.0, 0.0),
            length1=x1 - x0,
            length2=y1 - y0,
            length3=z1 - z0,
            alignment="min_corner",
        )
    )


def local_extent_from_center(
    center: np.ndarray,
    shape: tuple[int, int, int],
    half_width: float,
) -> tuple[float, float, float, float, float, float]:
    nx, ny, nz = shape
    x0 = max(0.0, center[0] - 3.5)
    y0 = max(0.0, center[1] - half_width)
    z0 = max(0.0, center[2] - half_width)
    x1 = min(float(nx), center[0] + 3.5)
    y1 = min(float(ny), center[1] + half_width)
    z1 = min(float(nz), center[2] + half_width)
    return x0, x1, y0, y1, z0, z1


def make_render_box(
    infos: list[LoopInfo],
    shape: tuple[int, int, int],
    half_width: float,
) -> RenderBox:
    """Choose one fixed local box that contains the selected loop trajectory."""
    centers = np.asarray([info.center for info in infos], dtype=float)
    center = centers.mean(axis=0)
    return RenderBox(
        center=center,
        extent=local_extent_from_center(center, shape, half_width),
    )


def render_loop(
    output: Path,
    q5: np.ndarray,
    info: LoopInfo,
    box: RenderBox,
    half_width: float,
    director_spacing: int,
    show_axes: bool,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    bounds = local_bounds(box)

    fig = PlotFigure(
        is_off_screen=True,
        name=f"loop step {info.step}",
        opts=OptsFigure(
            size=(900, 900),
            bg_color=(1.0, 1.0, 1.0),
            focal_point=tuple(box.center),
        ),
    )

    # A faint local box gives scale without drawing the full simulation domain.
    bounds.act_visualize(
        figure=fig,
        opts=n3d.OptsTube(
            radius=0.035,
            color=(0.75, 0.75, 0.75),
            opacity=0.18,
            sides=8,
            is_scalar_bar=False,
            is_reset_camera=False,
        ),
    )

    info.line.act_visualize(
        figure=fig,
        is_wrap=True,
        is_smooth=True,
        opts=n3d.OptsTube(
            radius=0.28,
            color=(0.0, 0.0, 0.0),
            opacity=1.0,
            sides=32,
            is_scalar_bar=False,
            is_reset_camera=False,
        ),
        bounds=bounds,
        clip_mode="mesh",
    )

    if False and show_axes:
        fig.pl.add_axes(
            line_width=3,
            x_color=(0.85, 0.10, 0.10),
            y_color=(0.10, 0.55, 0.10),
            z_color=(0.10, 0.20, 0.85),
            xlabel="x",
            ylabel="y",
            zlabel="z",
            viewport=(0.77, 0.03, 0.97, 0.23),
        )

    fig.act_view_isometric()
    fig.act_commit(
        focal_point=tuple(box.center),
        distance=82,
        azimuth=18,
        elevation=16,
        roll=-78,
    )
    fig.act_savefig(str(output), scale=2)


def write_manifest(path: Path, rows: list[LoopInfo], image_paths: list[Path]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["step", "image", "center_x", "center_y", "center_z", "radius", "line_points"])
        for info, image_path in zip(rows, image_paths):
            writer.writerow(
                [
                    info.step,
                    str(image_path),
                    f"{info.center[0]:.6g}",
                    f"{info.center[1]:.6g}",
                    f"{info.center[2]:.6g}",
                    f"{info.radius:.6g}",
                    info.points,
                ]
            )


def make_contact_sheet(
    image_paths: list[Path],
    infos: list[LoopInfo],
    output: Path,
) -> None:
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return

    top_margin = 72
    thumb_size = 420
    thumbs = []
    for image_path in image_paths:
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_size, thumb_size), "white")
        canvas.paste(img, ((thumb_size - img.width) // 2, (thumb_size - img.height) // 2))
        thumbs.append(canvas)

    width = thumb_size * len(thumbs)
    sheet = Image.new("RGB", (width, top_margin + thumb_size), "white")
    for idx, img in enumerate(thumbs):
        sheet.paste(img, (idx * thumb_size, top_margin))

    draw = ImageDraw.Draw(sheet)
    try:
        from PIL import ImageFont

        axis_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 46)
    except Exception:
        axis_font = None
    axis_y = 70
    left_x = thumb_size // 2
    right_x = width - thumb_size // 2
    arrow = 20
    draw.line((left_x, axis_y, right_x - arrow, axis_y), fill=(30, 30, 30), width=5)
    draw.polygon(
        [
            (right_x, axis_y),
            (right_x - arrow, axis_y - arrow // 2),
            (right_x - arrow, axis_y + arrow // 2),
        ],
        fill=(30, 30, 30),
    )
    label = "time"
    bbox = draw.textbbox((0, 0), label, font=axis_font)
    draw.text(
        ((left_x + right_x - (bbox[2] - bbox[0])) / 2, 4),
        label,
        fill=(0, 0, 0),
        font=axis_font,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data_control_box")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "data_control_box" / "loop_growth_local_clean")
    parser.add_argument("--steps", type=int, nargs="+", default=list(DEFAULT_STEPS))
    parser.add_argument("--target-center", type=float, nargs=3, default=DEFAULT_TARGET_CENTER.tolist())
    parser.add_argument("--half-width", type=float, default=13.0)
    parser.add_argument("--director-spacing", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = np.asarray(args.target_center, dtype=float)
    infos: list[LoopInfo] = []
    q_snapshots: list[np.ndarray] = []
    image_paths: list[Path] = []

    for step in args.steps:
        q_path = args.data_dir / f"Q_{step}.npy"
        if not q_path.exists():
            raise FileNotFoundError(q_path)
        q5 = np.load(q_path)
        info = choose_loop(step, q5, target)
        target = info.center
        q_snapshots.append(q5)
        infos.append(info)

    box = make_render_box(infos, q_snapshots[0].shape[:3], float(args.half_width))

    for index, (q5, info) in enumerate(zip(q_snapshots, infos)):
        image_path = args.output_dir / f"loop_local_alt_no_director_step_{info.step:05d}.png"
        render_loop(
            image_path,
            q5,
            info,
            box,
            half_width=float(args.half_width),
            director_spacing=int(args.director_spacing),
            show_axes=index == len(infos) - 1,
        )
        image_paths.append(image_path)
        print(
            f"step={info.step} image={image_path} center={info.center.round(3).tolist()} "
            f"radius={info.radius:.3f} points={info.points}"
        )

    write_manifest(args.output_dir / "loop_growth_3panel_alt_no_director_manifest.csv", infos, image_paths)
    contact = args.output_dir / "loop_growth_3panel_alt_no_director.png"
    make_contact_sheet(image_paths, infos, contact)
    print(f"manifest={args.output_dir / 'loop_growth_3panel_alt_no_director_manifest.csv'}")
    print(f"contact_sheet={contact}")


if __name__ == "__main__":
    main()
