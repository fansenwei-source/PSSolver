#!/usr/bin/env python3
"""Render the DAL initial and target loops with their local director fields."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from PIL import Image, ImageDraw, ImageFont
import pyvista as pv


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
for candidate in (
    REPO_ROOT.parent / "Nematics3D" / "src",
    REPO_ROOT / "Nematics3D" / "src",
):
    if candidate.exists():
        sys.path.insert(0, str(candidate))
        break

import nematics3d as n3d
from pssolver.snapshots import load_q_snapshot
from pssolver.models.active_nematics.nematics3d_adapter import director_from_Q
from nematics3d.classes.bounds import OptsBounds
from nematics3d.classes.visual.plot_figure import OptsFigure, PlotFigure
from nematics3d.classes.visual.plot_rod import OptsRod


PERIODIC_BOUNDARY = (True, False, False)
DETECTION_PLANES = (True, True, True)
DEFAULT_INITIAL = REPO_ROOT / "data_optimal_control_loop_translation" / "Q_initial_pure_loop.npy"
DEFAULT_TARGET = REPO_ROOT / "data_pure_splay_loop_x62" / "Q_pure_splay_loop.npy"
DEFAULT_INITIAL_METADATA = REPO_ROOT / "data_optimal_control_loop_translation" / "metadata.json"
DEFAULT_TARGET_METADATA = REPO_ROOT / "data_pure_splay_loop_x62" / "metadata.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "data_dal_pure_splay_x64_to_x62_T1_coreaware" / "visualization" / "initial_target"


@dataclass(frozen=True)
class LoopInfo:
    center: np.ndarray
    radius: float
    points: int
    physical_center: np.ndarray
    physical_radius: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial", type=Path, default=DEFAULT_INITIAL)
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument("--initial-metadata", type=Path, default=DEFAULT_INITIAL_METADATA)
    parser.add_argument("--target-metadata", type=Path, default=DEFAULT_TARGET_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--half-width", type=float, default=6.0)
    parser.add_argument("--director-spacing", type=int, default=4)
    parser.add_argument("--plane-offsets", type=int, nargs="+", default=(-4, 0, 4))
    return parser.parse_args()


def load_q(path: Path) -> np.ndarray:
    return load_q_snapshot(path, require_S_initial=True).values


def principal_director(q5: np.ndarray) -> np.ndarray:
    director = director_from_Q(q5)
    return director


def loop_from_metadata(path: Path) -> LoopInfo:
    with path.open() as handle:
        metadata = json.load(handle)
    physical_center = np.asarray(metadata["loop"]["center"], dtype=float)
    physical_radius = float(metadata["loop"]["radius"])
    spacing = np.asarray(metadata["domain"]["spacing"], dtype=float)
    center = physical_center / spacing - 0.5
    radius = physical_radius / spacing[1]
    return LoopInfo(
        center=center,
        radius=radius,
        points=160,
        physical_center=physical_center,
        physical_radius=physical_radius,
    )


def loop_tube(info: LoopInfo) -> pv.PolyData:
    angle = np.linspace(0.0, 2.0 * np.pi, info.points, endpoint=False)
    points = np.column_stack(
        (
            np.full_like(angle, info.center[0]),
            info.center[1] + info.radius * np.cos(angle),
            info.center[2] + info.radius * np.sin(angle),
        )
    )
    return pv.lines_from_points(points, close=True).tube(radius=0.38, n_sides=32)


def local_bounds(center: np.ndarray, half_width: float) -> n3d.Bounds:
    x_half_width = 8.0
    origin = center - np.array([x_half_width, half_width, half_width])
    return n3d.Bounds(
        opts=OptsBounds(
            origin=tuple(origin),
            axis1=(1.0, 0.0, 0.0),
            axis2=(0.0, 1.0, 0.0),
            length1=2.0 * x_half_width,
            length2=2.0 * half_width,
            length3=2.0 * half_width,
            alignment="min_corner",
        )
    )


def sample_directors(
    director: np.ndarray,
    center: np.ndarray,
    half_width: float,
    spacing: int,
    plane_offsets: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = director.shape[:3]
    y0 = max(1, int(np.floor(center[1] - half_width)))
    y1 = min(ny - 2, int(np.ceil(center[1] + half_width)))
    z0 = max(1, int(np.floor(center[2] - half_width)))
    z1 = min(nz - 2, int(np.ceil(center[2] + half_width)))
    coordinates: list[tuple[float, float, float]] = []
    orientations: list[np.ndarray] = []
    colors: list[float] = []
    for offset in plane_offsets:
        x = int(round(center[0] + offset)) % nx
        for y in range(y0, y1 + 1, spacing):
            for z in range(z0, z1 + 1, spacing):
                n = director[x, y, z]
                coordinates.append((float(x), float(y), float(z)))
                orientations.append(n)
                colors.append(float(n[0]))
    return np.asarray(coordinates), np.asarray(orientations), np.asarray(colors)


def render(
    q5: np.ndarray,
    info: LoopInfo,
    output: Path,
    *,
    half_width: float,
    director_spacing: int,
    plane_offsets: tuple[int, ...],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    bounds = local_bounds(info.center, half_width)
    figure = PlotFigure(
        is_off_screen=True,
        name=output.stem,
        opts=OptsFigure(
            size=(900, 900),
            bg_color=(1.0, 1.0, 1.0),
            focal_point=tuple(info.center),
        ),
    )
    bounds.act_visualize(
        figure=figure,
        opts=n3d.OptsTube(
            radius=0.035,
            color=(0.72, 0.72, 0.72),
            opacity=0.18,
            sides=8,
            is_scalar_bar=False,
            is_reset_camera=False,
        ),
    )

    director = principal_director(q5)
    coordinates, orientations, colors = sample_directors(
        director,
        info.center,
        half_width,
        director_spacing,
        plane_offsets,
    )
    n3d.PlotRod(
        coordinates,
        orientations,
        figure=figure,
        opts=OptsRod(
            length=2.9,
            radius=0.10,
            paint_by="scalars",
            scalars=colors,
            scalars_cmap="coolwarm",
            scalars_clim=(-1.0, 1.0),
            opacity=0.62,
            sides=10,
            is_scalar_bar=False,
            is_reset_camera=False,
        ),
        bounds=bounds,
        clip_mode="center",
    )
    figure.pl.add_mesh(
        loop_tube(info),
        color=(0.03, 0.03, 0.03),
        opacity=1.0,
        smooth_shading=True,
        specular=0.25,
        reset_camera=False,
    )
    figure.pl.add_axes(
        line_width=3,
        x_color=(0.85, 0.10, 0.10),
        y_color=(0.10, 0.55, 0.10),
        z_color=(0.10, 0.20, 0.85),
        xlabel="x",
        ylabel="y",
        zlabel="z",
        viewport=(0.80, 0.03, 0.98, 0.21),
    )
    figure.act_view_isometric()
    figure.act_commit(
        focal_point=tuple(info.center),
        distance=52,
        azimuth=18,
        elevation=16,
        roll=-78,
    )
    figure.act_savefig(str(output), scale=2)


def compose(initial_path: Path, target_path: Path, output: Path) -> None:
    panels = [Image.open(path).convert("RGB") for path in (initial_path, target_path)]
    panel_size = 780
    title_height = 88
    canvas = Image.new("RGB", (2 * panel_size, panel_size + title_height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
    except OSError:
        font = None
    for index, (panel, label) in enumerate(zip(panels, ("Initial: x = 64", "Target: x = 62"))):
        panel.thumbnail((panel_size, panel_size), Image.Resampling.LANCZOS)
        x0 = index * panel_size + (panel_size - panel.width) // 2
        canvas.paste(panel, (x0, title_height + (panel_size - panel.height) // 2))
        box = draw.textbbox((0, 0), label, font=font)
        text_width = box[2] - box[0]
        draw.text(
            (index * panel_size + (panel_size - text_width) / 2, 25),
            label,
            fill=(28, 32, 36),
            font=font,
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def translated_difference(initial: np.ndarray, target: np.ndarray, shift: int) -> tuple[float, float]:
    difference = np.roll(initial, shift, axis=0) - target
    return float(np.max(np.abs(difference))), float(np.sqrt(np.mean(difference**2)))


def main() -> None:
    args = parse_args()
    expected_initial_metadata = args.initial.parent / "metadata.json"
    expected_target_metadata = args.target.parent / "metadata.json"
    if args.initial_metadata.resolve() != expected_initial_metadata.resolve():
        raise ValueError(
            "--initial-metadata must be the metadata.json colocated with --initial"
        )
    if args.target_metadata.resolve() != expected_target_metadata.resolve():
        raise ValueError(
            "--target-metadata must be the metadata.json colocated with --target"
        )

    initial = load_q(args.initial)
    target = load_q(args.target)
    if initial.shape != target.shape:
        raise ValueError(f"Initial shape {initial.shape} does not match target shape {target.shape}")

    initial_info = loop_from_metadata(args.initial_metadata)
    target_info = loop_from_metadata(args.target_metadata)
    shift = int(round(target_info.center[0] - initial_info.center[0]))
    maximum_error, rms_error = translated_difference(initial, target, shift)

    output_dir = args.output_dir.resolve()
    initial_path = output_dir / "initial_loop_directors.png"
    target_path = output_dir / "target_loop_directors.png"
    comparison_path = output_dir / "initial_target_loop_directors.png"
    render(
        initial,
        initial_info,
        initial_path,
        half_width=float(args.half_width),
        director_spacing=int(args.director_spacing),
        plane_offsets=tuple(args.plane_offsets),
    )
    render(
        target,
        target_info,
        target_path,
        half_width=float(args.half_width),
        director_spacing=int(args.director_spacing),
        plane_offsets=tuple(args.plane_offsets),
    )
    compose(initial_path, target_path, comparison_path)

    print(f"initial_center_physical={initial_info.physical_center.tolist()} radius={initial_info.physical_radius:.6g}")
    print(f"target_center_physical={target_info.physical_center.tolist()} radius={target_info.physical_radius:.6g}")
    print(f"translation_grid_points={shift}")
    print(f"translated_Q_max_error={maximum_error:.9g}")
    print(f"translated_Q_rms_error={rms_error:.9g}")
    print(f"comparison={comparison_path}")


if __name__ == "__main__":
    main()
